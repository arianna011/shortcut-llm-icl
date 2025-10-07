import os
from patched_unibias import WB_logging as L
from extract_activations import load_shortcut_prompts as E
from extract_activations import BaseLLM, HuggingFaceLLM
import random
import numpy as np
import wandb
import json
import subprocess
import pandas as pd
import torch
from transformers import pipeline
from representation_engineering import repe_pipeline_registry
repe_pipeline_registry()


def prepare_shortcut_training_dataset(
    repo_path: str,
    dataset_name: str,
    shortcut_type: str,
    num_samples: int,
    prompts_selec_method: L.SelectionMethod,
    random_seed: int,
    model: BaseLLM = None,
    num_shot: int = 0,
    max_ans_tokens: int = 5,
    model_temperature: float = 0.0,
    logits_step: int = 0,
    debug: bool = False
):
    if dataset_name == "ShortcutSuite":

        clean_df_path = os.path.join(repo_path,"data","ShortcutSuite","dev_matched.tsv")
        dirty_df_path = os.path.join(repo_path,"data","ShortcutSuite",f"dev_matched_{shortcut_type}.tsv")
        df_standard = E.load_nli_shortcuts_from_tsv(clean_df_path)
        df_shortcut = E.load_nli_shortcuts_from_tsv(dirty_df_path)
        df = E.create_paired_dataset(df_standard, df_shortcut)

        if prompts_selec_method is L.SelectionMethod.RANDOM:
           selected_df = df.sample(n=num_samples, random_state=random_seed)
        else:
            assert model, "Please provide a model for prompts evaluation"
            condition = _get_selection_condition(prompts_selec_method, shortcut_type)
            selected_df = E.select_shortcut_prompts(df, E.Task.NLI, n_samples=num_samples, 
                                                    model=model, num_shot=num_shot, temperature=model_temperature,
                                                    condition=condition, max_tokens=max_ans_tokens, seed=random_seed, 
                                                    debug=debug, logits_step=logits_step)
        task = "NLI"
        columns = selected_df.columns.to_list()
        size = selected_df.shape[0]
        labels = (selected_df["gold_label"].unique()).tolist()
        description = "Training dataset generated during execution of full RepE pipeline"
        L.log_dataset_artifact(
            dataset=selected_df,
            dataset_name=dataset_name,
            task=task,
            size=size,
            columns=columns,
            labels=labels,
            shortcut=shortcut_type,
            selection_method=prompts_selec_method,
            random_seed=random_seed,
            description=description)

    else:
       raise NotImplementedError
    

def prepare_shortcut_activations(
    repo_path: str,
    drive_path: str,
    overwrite_df_artifact: bool,
    dataset_name: str,
    shortcut_type: str,
    num_samples: int,
    prompts_selec_method: L.SelectionMethod,
    random_seed: int,
    clean_instr: str,
    dirty_instr: str,
    shuffle_data: bool,
    model_wrap: BaseLLM,
    alpha_coeff: float,
    direction_method: str,
    rep_token: int = -1,
    num_shot: int = 0,
    max_ans_tokens: int = 5,
    model_temperature: float = 0.0,
    logits_step: int = 0,
    batch_size: int = 32,
    debug: bool = False
):
    api = wandb.Api()
    
    dataset_art_name = L.get_dataset_artifact_name(
        dataset_name=dataset_name,
        size=num_samples,
        shortcut=shortcut_type,
        selection_method=prompts_selec_method,
        random_seed=random_seed)
    
    # check if the needed dataset artifact already exists
    try:
      dataset_artifact = api.artifact(f"{L.WB_TEAM}/{L.WB_PROJECT_NAME}/{dataset_art_name}:latest")
    except wandb.CommError:
      dataset_artifact = None
      print(f"Artifact {dataset_art_name} not found. Starting creation...")
    
    if not dataset_artifact or overwrite_df_artifact:
       # create a new dataset artifact
        prepare_shortcut_training_dataset(repo_path=repo_path,
                                         dataset_name=dataset_name,
                                         shortcut_type=shortcut_type,
                                         num_samples=num_samples,
                                         prompts_selec_method=prompts_selec_method,
                                         random_seed=random_seed, model=model_wrap,
                                         num_shot=num_shot, max_ans_tokens=max_ans_tokens,
                                         model_temperature=model_temperature, logits_step=logits_step, debug=debug)
        try:
            dataset_artifact = api.artifact(f"{L.WB_TEAM}/{L.WB_PROJECT_NAME}/{dataset_art_name}:latest")
        except wandb.CommError:
            dataset_artifact = None
    
    assert dataset_artifact, "Failure in dataset artifact retrieval"
    df_entry = dataset_artifact.get_entry(f"{dataset_name}.json")  
    local_df_path = df_entry.download() 
    with open(local_df_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    train_data, train_labels = _format_data_nli(df, clean_instr, dirty_instr, shuffle_data)
    rep_reading_pipeline = pipeline("rep-reading", model=model_wrap.model, tokenizer=model_wrap.tokenizer)
    hidden_layers = list(range(-1, -model_wrap.model.config.num_hidden_layers, -1))

    print("Training RepReader...")
    shortcut_rep_reader = rep_reading_pipeline.get_directions(
                                                    train_data,
                                                    rep_token=rep_token,
                                                    hidden_layers=hidden_layers,
                                                    n_difference=1,
                                                    train_labels=train_labels,
                                                    direction_method=direction_method,
                                                    batch_size=batch_size)
    activations = {}
    for layer in hidden_layers:
        activations[layer] = torch.tensor(alpha_coeff * shortcut_rep_reader.directions[layer] * shortcut_rep_reader.direction_signs[layer]).to(model_wrap.model.device).half()
    
    # log the activations file
    activations_art_name = L.get_activations_artifact_name(dataset_artifact_name=dataset_art_name,
                                                           coeff=alpha_coeff,
                                                           direction_method=direction_method, 
                                                           clean_instruction=clean_instr,
                                                           dirty_instruction=dirty_instr,
                                                           shuffled_data=shuffle_data)
    activations_path = os.path.join(drive_path, "activations", f"{activations_art_name}.pt")
    torch.save(activations, activations_path)
    print(f"Activations saved at {activations_path}")

    description = "Activations generated during execution of full RepE pipeline"

    L.log_activations_artifact(
       activations_path=activations_path,
       dataset_artifact_name=dataset_art_name,
       coeff=alpha_coeff,
       rep_token=rep_token,
       hidden_layers=hidden_layers,
       direction_method=direction_method,
       clean_instruction=clean_instr,
       dirty_instruction=dirty_instr,
       shuffled_data=shuffle_data,
       description=description
    )
    

def _get_selection_condition(sel_method: L.SelectionMethod, shortcut_type: str):
   
    if sel_method is L.SelectionMethod.MODEL_FAILS_ON_SPECIFIC_LABELS and shortcut_type == "negation":
        def select_if(task, row, pred_clean, pred_dirty):
            gold = row["gold_label"]
            positive_lab = next(iter(task.reference_gen_to_labels().values()))
            return pred_clean == gold and gold == positive_lab and pred_dirty != gold
        return select_if
    elif sel_method is L.SelectionMethod.MODEL_FAILS:
        def select_if(task, row, pred_clean, pred_dirty):
            gold = row["gold_label"]
            return pred_clean == gold and pred_dirty != gold
        return select_if
    else:
      raise NotImplementedError
  
def _format_data_nli(df, clean_instr, dirty_instr, shuffle):
  c_instr = f"[INST] {clean_instr} [/INST] "
  d_instr = f"[INST] {dirty_instr} [/INST] "
  c_template = lambda prem, hyp: c_instr + f'Premise: {prem}\nHypothesis: {hyp}'
  d_template = lambda prem, hyp: d_instr + f'Premise: {prem}\nHypothesis: {hyp}'
  data = [[d_template(prem_d,hyp_d), c_template(prem_c,hyp_c)] for (prem_d,hyp_d,prem_c,hyp_c)
            in zip(df['premise_dirty'], df['hypothesis_dirty'], df['premise_clean'], df['hypothesis_clean'])]
  labels = []  # 1 = +shortcut (dirty), 0 = -shortcut (clean)
  for d in data:
        dirty = d[0]
        if shuffle:
          random.shuffle(d) # shuffling inside contrastive pairs
        labels.append([s == dirty for s in d])
  return np.concatenate(data).tolist(), labels


def RepE_evaluation(
    repo_path: str,
    drive_path: str,
    overwrite_df_artifact: bool,
    overwrite_act_artifact: bool,
    training_dataset_name: str,
    training_dataset_size: int,
    training_dataset_shortcut_type: str,
    training_dataset_sel_method: L.SelectionMethod,
    training_dataset_random_seed: int,
    activations_clean_instr: str,
    activations_dirty_instr: str,
    activations_data_shuffle: bool,
    activations_direction_method: str,
    activations_alpha_coeff: float,
    model_wrap: BaseLLM,
    eval_dataset_name: str,
    eval_num_shot: int,
    eval_intervention_layers: list[int],
    eval_resume: str,
    training_dataset_num_shot: int = 0,
    training_max_ans_tokens: int = 5,
    training_logits_step: int = 0,
    training_model_temperature: float = 0.0,
    training_batch_size: int = 32,
    training_debug: bool = False):

    api = wandb.Api()

    training_dataset_art_name = L.get_dataset_artifact_name(
        dataset_name=training_dataset_name,
        size=training_dataset_size,
        shortcut=training_dataset_shortcut_type,
        selection_method=training_dataset_sel_method,
        random_seed=training_dataset_random_seed)
  
    activations_art_name = L.get_activations_artifact_name(
        dataset_artifact_name=training_dataset_art_name,
        coeff=activations_alpha_coeff,
        direction_method=activations_direction_method,
        clean_instruction=activations_clean_instr,
        dirty_instruction=activations_dirty_instr,
        shuffled_data=activations_data_shuffle)
    
    # check if the needed acrivations artifact already exists
    try:
        activations_artifact = api.artifact(f"{L.WB_TEAM}/{L.WB_PROJECT_NAME}/{activations_art_name}:latest")
    except wandb.CommError:
        activations_artifact = None
        print(f"Artifact {activations_art_name} not found. Starting creation...")
    
    if not activations_artifact or overwrite_act_artifact:
        # create a new activations artifact
        prepare_shortcut_activations(
                                    repo_path=repo_path,
                                    drive_path=drive_path,
                                    overwrite_df_artifact=overwrite_df_artifact,
                                    dataset_name=training_dataset_name,
                                    shortcut_type=training_dataset_shortcut_type,
                                    num_samples=training_dataset_size,
                                    prompts_selec_method=training_dataset_sel_method,
                                    random_seed=training_dataset_random_seed, 
                                    clean_instr=activations_clean_instr,
                                    dirty_instr=activations_dirty_instr,
                                    direction_method=activations_direction_method,
                                    shuffle_data=activations_data_shuffle,
                                    model_wrap=model_wrap,
                                    alpha_coeff=activations_alpha_coeff,
                                    num_shot=training_dataset_num_shot,
                                    max_ans_tokens=training_max_ans_tokens,
                                    model_temperature=training_model_temperature,
                                    logits_step=training_logits_step,
                                    batch_size=training_batch_size,
                                    debug=training_debug)
                                        
        try:
            activations_artifact = api.artifact(f"{L.WB_TEAM}/{L.WB_PROJECT_NAME}/{activations_art_name}:latest")
        except wandb.CommError:
            activations_artifact = None
    
    assert activations_artifact, "Failure in activations artifact retrieval"

    cmd = [
        "python", "patched_unibias/main.py",
        "--dataset_name", eval_dataset_name,
        "--num_shot", str(eval_num_shot),
        "--RepE", "true",
        "--resume", eval_resume,
        "--activations", activations_art_name,
        "--intervention_layers", eval_intervention_layers,
        "--log_on_WB", "true"
    ]
    print("Launching evaluation command:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, text=True, capture_output=True)

    print(result.stdout)
    if result.returncode != 0:
        print("Evaluation script failed:")
        print(result.stderr)
        raise RuntimeError("Evaluation process exited with errors.")
    
