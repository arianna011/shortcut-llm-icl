import json
import torch
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

WB_PROJECT_NAME = "shortcut-repe"
WB_TEAM = "paolini-1943164-sapienza-universit-di-roma"
WB_USER = "paolini-1943164"

class SelectionMethod(Enum):
   RANDOM="random"
   MODEL_FAILS="failures"
   MODEL_FAILS_ON_SPECIFIC_LABELS="shortcut_failures"

# ARTIFACTS ===================================================================================================================

def get_dataset_artifact_name(
    dataset_name: str,
    size: int,
    shortcut: str,
    selection_method: SelectionMethod,
    random_seed: int):
    return f'{dataset_name}_{size}_{shortcut}_{selection_method.value}_seed_{random_seed}'

def get_activations_artifact_name(
    dataset_artifact_name: str,
    coeff: float,
    direction_method: str,
    clean_instruction: str,
    dirty_instruction: str, 
    shuffled_data: bool = True):

    shuffle_str = ""
    instr_str = ""
    if not shuffled_data:
        shuffle_str = "_no_shuffle"
    if clean_instruction != dirty_instruction:
        instr_str = "_custom_instr"

    return f'coeff_{coeff}{shuffle_str}{instr_str}_{direction_method}_{dataset_artifact_name}'
    


def log_dataset_artifact(
    dataset: Union[str,pd.DataFrame],
    dataset_name: str,
    task: str,
    size: int,
    columns: list[str],
    labels: list[str],
    shortcut: str,
    selection_method: SelectionMethod,
    random_seed: int,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
):
    """
    Logs a small dataset (file or Pandas datafreame) as a W&B artifact.
    """

    artifact_name = get_dataset_artifact_name(dataset_name, size, shortcut, selection_method, random_seed)
    wandb.init(project=WB_PROJECT_NAME, name=f"log_{artifact_name}")

    # save dataset locally if given as dataframe
    dataset_path = Path(f"{dataset_name}.json")
    if isinstance(dataset, pd.DataFrame):
        dataset.to_json(dataset_path, orient="records", indent=2)
    else:
        dataset_path = Path(dataset)

    if metadata is None:
        metadata = {}  
    metadata.update({
        "dataset_name":dataset_name,
        "task": task,
        "n_examples": size,
        "columns": columns,
        "labels": labels,
        "shortcut_type": shortcut,
        "selection_method": selection_method,
        "random_seed": random_seed
    })

    artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=description or "Training or evaluation dataset",
        metadata=metadata
    )
    artifact.add_file(str(dataset_path))

    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"✅ Logged dataset artifact: {artifact_name}")


def log_activations_artifact(
    activations_path: str,
    dataset_artifact_name: str,
    coeff: float,
    rep_token: int,
    hidden_layers: list[int],
    direction_method: str,
    clean_instruction: str,
    dirty_instruction: str, 
    shuffled_data: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
):
    """
    Logs an activation (.pt) file as a W&B artifact and links it
    to the dataset artifact it was derived from.
    """

    artifact_name = get_activations_artifact_name(
        dataset_artifact_name, coeff, direction_method, clean_instruction, dirty_instruction, shuffled_data
    )

    run = wandb.init(project=WB_PROJECT_NAME, name=f"log_{artifact_name}")
    dataset_artifact = run.use_artifact(f"{WB_TEAM}/{WB_PROJECT_NAME}/{dataset_artifact_name}:latest")
    dataset_metadata = dataset_artifact.metadata
       
    if metadata is None:
        metadata = {}  
    metadata.update({
        "dataset_metadata": dataset_metadata,
        "alpha_coeff":coeff,
        "rep_token": rep_token,
        "hidden_layers": hidden_layers,
        "direction_method": direction_method,
        "clean_instruction": clean_instruction,
        "dirty_instruction": dirty_instruction,
        "shuffled_data": shuffled_data
    })

    artifact = wandb.Artifact(
        name=artifact_name,
        type="activations",
        description=description or "RepE reading activations (including coefficient and sign)",
        metadata=metadata,
    )
    artifact.add_file(activations_path)

    run.log_artifact(artifact)
    run.finish()
    print(f"✅ Logged activation artifact: {artifact_name}")


# LOGGING EVALUATION RUN ==============================================================================================================

def log_evaluation_run(
    model_name: str,
    eval_dataset: str,
    ICL_shots: int,
    repE_active: bool,
    predictions: list[int],
    gt_labels: list[int],
    all_label_probs: list[list[float]],
    class_names: list[str],
    prompt_list: list[str],
    confusion_matrix: np.ndarray,
    activations_artifact_name: Optional[str] = None,
    intervention_layers: Optional[list[int]] = None,
    control_method: Optional[str] = None,
    block_name: Optional[str] = None,
    tags: Optional[list] = None
):
    """
    Creates a W&B run for evaluation results using an activation artifact as input.
    """
    run_name = f"{model_name}_{eval_dataset}_{ICL_shots}_shots"

    config={
        "model": model_name,
        "dataset": eval_dataset,
        "num_shots": ICL_shots,
        "repE_enabled": repE_active}

    if repE_active:
        assert activations_artifact_name, "RepE is active, please provide the activations artifact name"
        assert intervention_layers, "RepE is active, please provide the list of hidden layers for RepE Control"
        first_l, last_l = intervention_layers[0], intervention_layers[-1]
        run_name += f'_layers_{first_l}_{last_l}'
        run_name += f'_activations_{activations_artifact_name}'

        config.update({"intervention_layers": intervention_layers, "control_method": control_method, 
                       "block_name": block_name, "activations": activations_artifact_name})  
    else:
        run_name += f'_baseline'

    if wandb.run is not None: # running inside a sweep
        run = wandb.run
    else: # intialize new standalone run
        run = wandb.init(project=WB_PROJECT_NAME, name=run_name, job_type="evaluation", config=config, tags=tags or [], 
                     settings = wandb.Settings(init_timeout=240))

    if repE_active:
        # link activations artifact as input
        try:
            activation_artifact = run.use_artifact(f"{WB_TEAM}/{WB_PROJECT_NAME}/{activations_artifact_name}:latest")
        except wandb.CommError:
            print(f"⚠️ Warning: Artifact {activations_artifact_name} not found.")

    results = {
        "accuracy": accuracy_score(gt_labels, predictions),
        "f1_macro": f1_score(gt_labels, predictions, average="macro")
    }
    wandb.log(results)

    table_columns = ["id", "prompt", "true", "pred"] + class_names
    table = wandb.Table(columns=table_columns)

    for idx, (prompt, true_idx, pred_idx, probs) in enumerate(zip(prompt_list, gt_labels, predictions, all_label_probs)):
        row = [idx, prompt, class_names[true_idx], class_names[pred_idx]] + probs
        table.add_data(*row)

    wandb.log({"predictions_table": table})

    cm_table = wandb.Table(columns=["True", "Predicted", "Count"])
    for i, true_label in enumerate(class_names):
        for j, pred_label in enumerate(class_names):
            cm_table.add_data(true_label, pred_label, int(confusion_matrix[i][j]))

    wandb.log({"confusion_matrix_table": cm_table})

    wandb.finish()
    print(f"✅ Logged evaluation run: {run_name}")
