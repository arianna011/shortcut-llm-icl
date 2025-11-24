import os
from patched_unibias import WB_logging as L
from patched_unibias import prepare_dataset_test, task_labels, find_possible_ids_for_labels, ICL_evaluation
from extract_activations import load_shortcut_prompts as E
from extract_activations import BaseLLM, HuggingFaceLLM, DATASETS_TO_TASKS
import random
import numpy as np
import wandb
import json
import subprocess
import pandas as pd
import torch
import sys,subprocess
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from transformers import pipeline
from representation_engineering import repe_pipeline_registry
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
repe_pipeline_registry()

# define methods to combine the activations from multiple shortcut types
class ShortcutAggregation(Enum):
    NORMALIZED_SUM = "sum"

    def get_name(self):
        return self.value

    def combine_activations(self, acts: list[dict[int, torch.Tensor]], hidden_layers:list[int]) -> dict[int, torch.Tensor]:
       
        if self is ShortcutAggregation.NORMALIZED_SUM:

            comb_act = {}
            for l in hidden_layers:
                layer_dirs = []
                # collect all directions for this layer
                for act in acts:
                    if l not in act:
                        continue
                    d = act[l]
                    # normalize each direction
                    d_norm = F.normalize(d, p=2, dim=-1)
                    layer_dirs.append(d_norm)

                # if no shortcut provided a dir for this layer, skip
                if not layer_dirs:
                    continue
                # sum all normalized directions
                stacked = torch.stack(layer_dirs, dim=0)  # (num_shortcuts, ...)
                combined = stacked.sum(dim=0)

                # normalize the final combined direction
                combined = F.normalize(combined, p=2, dim=-1)
                comb_act[l] = combined

            return comb_act
        
        else:
            raise NotImplementedError(f"{self} is not implemented.")
        

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
    shortcut_types: Union[str, list[str]],
    num_samples: int,
    prompts_selec_method: L.SelectionMethod,
    random_seed: int,
    clean_instr: str,
    dirty_instr: str,
    shuffle_data: bool,
    model_wrap: BaseLLM,
    alpha_coeff: float,
    direction_method: str,
    aggregation_type: ShortcutAggregation = None,
    rep_token: int = -1,
    num_shot: int = 0,
    max_ans_tokens: int = 5,
    model_temperature: float = 0.0,
    logits_step: int = 0,
    batch_size: int = 32,
    debug: bool = False
):
    api = wandb.Api()
    mul_shorts = False
    if isinstance(shortcut_types, list) and len(shortcut_types) > 1:
        assert aggregation_type, "An aggregation method must be provided to combine multiple shortcut types."
        mul_shorts = True
        dataset_art_names = []
        act_list = []
        act_names_list = []

    hidden_layers = list(range(-1, -model_wrap.model.config.num_hidden_layers, -1))
    for shortcut_type in shortcut_types:
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
        if mul_shorts:
            dataset_art_names.append(dataset_art_name)
            act_list.append(activations)
            act_names_list.append(activations_art_name)
    
    if mul_shorts:
        comb_acts = aggregation_type.combine_activations(act_list, hidden_layers)
        comb_acts_art_name = L.get_combined_activations_artifact_name(act_names_list, aggregation_type.get_name())
        comb_activations_path = os.path.join(drive_path, "activations", f"{comb_acts_art_name}.pt")
        torch.save(comb_acts, comb_activations_path)
        print(f"Combined activations saved at {comb_activations_path}")

        description = "Combined activations generated during execution of full RepE pipeline"

        L.log_combined_activations_artifact(
            activations_path=comb_activations_path,
            dataset_artifact_names=dataset_art_names,
            aggregation_type_name=aggregation_type.get_name(),
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


def unibias_main(model_name: str,
                 hf_token: str,
                 eval_dataset_name: str,
                 num_shot: int,
                 RepE_enabled: bool,
                 activations_name: str,
                 intervention_layers: list[int],
                 operator: str = "linear_comb",
                 resume: bool = False,
                 new_tokens: int = 10,
                 log_on_WB: bool = True):
    
    device = torch.device("cuda:0")
    hf_token = hf_token or os.environ.get("HF_TOKEN")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,         
        bnb_4bit_quant_type="nf4",                 
        bnb_4bit_compute_dtype=torch.float16,  
    )

    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }

    if hf_token:
        model_kwargs["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token if hf_token else None)
    model.eval()

    prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_test(eval_dataset_name, num_shot=num_shot)
    ans_label_list = task_labels(eval_dataset_name)
    gt_ans_ids_list = find_possible_ids_for_labels(ans_label_list, tokenizer)

    class_labels = [DATASETS_TO_TASKS[eval_dataset_name].reference_gen_to_labels()[ans[0]] for ans in ans_label_list]

    control_method = None
    block_name = None
    if RepE_enabled:
        assert activations_name and intervention_layers, "Please provide non-empty RepE params"
        control_method = "reading_vec"
        block_name = "decoder_block"

    print("Starting evaluation...")

    # evaluate performance
    final_acc, predictions, all_label_probs, cf = ICL_evaluation(model, tokenizer, device, prompt_list, test_labels, 
                                                    gt_ans_ids_list, eval_dataset_name, 
                                                    repE=RepE_enabled, activations_art_name=activations_name, operator=operator,
                                                    gen_tokens=new_tokens, intervention_layers=intervention_layers, resume=resume)

    if log_on_WB:
        L.log_evaluation_run( model_name=model_name,
                            eval_dataset=eval_dataset_name,
                            ICL_shots=num_shot,
                            repE_active=RepE_enabled,
                            predictions=predictions,
                            gt_labels=test_labels,
                            all_label_probs=all_label_probs,
                            class_names=class_labels,
                            prompt_list=prompt_list,
                            confusion_matrix=cf,
                            activations_artifact_name=activations_name,
                            operator = operator,
                            intervention_layers=intervention_layers,
                            control_method=control_method,
                            block_name=block_name
                           )


def RepE_evaluation(
    repo_path: str,
    drive_path: str,
    overwrite_df_artifact: bool,
    overwrite_act_artifact: bool,
    training_dataset_name: str,
    training_dataset_size: int,
    training_dataset_shortcut_types: Union[str, list[str]],
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
    shortcut_aggregation: ShortcutAggregation = None,
    eval_operator: str = "linear_comb",
    training_dataset_num_shot: int = 0,
    training_max_ans_tokens: int = 5,
    training_logits_step: int = 0,
    training_model_temperature: float = 0.0,
    eval_resume: bool = False,
    training_batch_size: int = 32,
    training_debug: bool = False):

    api = wandb.Api()

    mul_shorts = False
    if isinstance(training_dataset_shortcut_types, list) and len(training_dataset_shortcut_types)>1:
        assert shortcut_aggregation, "An aggregation method must be provided to combine multiple shortcut types."
        mul_shorts = True
        train_dataset_art_names = []
        act_art_names = []
    
    for shortcut_type in training_dataset_shortcut_types:
        training_dataset_art_name = L.get_dataset_artifact_name(
            dataset_name=training_dataset_name,
            size=training_dataset_size,
            shortcut=shortcut_type,
            selection_method=training_dataset_sel_method,
            random_seed=training_dataset_random_seed)
        
        activations_art_name = L.get_activations_artifact_name(
            dataset_artifact_name=training_dataset_art_name,
            coeff=activations_alpha_coeff,
            direction_method=activations_direction_method,
            clean_instruction=activations_clean_instr,
            dirty_instruction=activations_dirty_instr,
            shuffled_data=activations_data_shuffle)
        
        if mul_shorts:
            train_dataset_art_names.append(training_dataset_art_name)
            act_art_names.append(activations_art_name)       
    
    if mul_shorts:
        activations_art_name = L.get_combined_activations_artifact_name(act_art_names, shortcut_aggregation.get_name())
    
    # check if the needed acrivations artifact already exists
    try:
        activations_artifact = api.artifact(f"{L.WB_TEAM}/{L.WB_PROJECT_NAME}/{activations_art_name}:latest")
    except wandb.CommError:
        activations_artifact = None
        print(f"Artifact {activations_art_name} not found. Starting creation...")
    
    if not activations_artifact or overwrite_act_artifact:
        # create a new activations artifact
        prepare_shortcut_activations(   repo_path=repo_path,
                                        drive_path=drive_path,
                                        overwrite_df_artifact=overwrite_df_artifact,
                                        dataset_name=training_dataset_name,
                                        shortcut_types=training_dataset_shortcut_types,
                                        aggregation_type=shortcut_aggregation,
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

    unibias_main(model_name=model_wrap.model_name,
                 hf_token=model_wrap.hf_token,
                 eval_dataset_name=eval_dataset_name,
                 num_shot=eval_num_shot,
                 RepE_enabled=True,
                 resume=eval_resume,
                 activations_name=activations_art_name,
                 operator=eval_operator,
                 intervention_layers=eval_intervention_layers,
                 log_on_WB=True
                 )
    
