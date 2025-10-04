import json
import torch
import wandb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional
from enum import Enum
import matplotlib.pyplot as plt

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
    hidden_layers: list[int],
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
    hidden_layers_str = "_".join(map(str, hidden_layers))

    return f'coeff_{coeff}{shuffle_str}{instr_str}_layers_{hidden_layers_str}_{direction_method}_{dataset_artifact_name}'
    


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


def log_activation_artifact(
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
        dataset_artifact_name, coeff, hidden_layers, direction_method, clean_instruction, dirty_instruction, shuffled_data
    )

    wandb.init(project=WB_PROJECT_NAME, name=f"log_{artifact_name}")
    dataset_artifact = wandb.use_artifact(f"{dataset_artifact_name}:latest")
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
    artifact.add_reference(dataset_artifact)

    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"✅ Logged activation artifact: {artifact_name}")


# LOGGING EVALUATION RUN ==============================================================================================================

def log_evaluation_run(
    eval_dataset: str,
    ICL_shots: int,
    repE_active: bool,
    accuracy: float,
    predictions: list[int],
    gt_labels: list[int],
    all_label_probs: list[list[float]],
    class_names: list[str],
    activations_artifact_name: Optional[str] = None,
    tags: Optional[list] = None
):
    """
    Creates a W&B run for evaluation results using an activation artifact as input.

    Args:
        project: W&B project name.
        name: Run name (e.g. "evaluate_layer12_rte_shortcut").
        activation_artifact_name: Activation artifact name to use as input.
        results: Dictionary of metrics to log (e.g., {"accuracy": 0.87}).
        table_data: Optional list of (id, y_true, y_pred) tuples for logging results.
        class_names: Optional class labels for confusion matrix.
        tags: Optional list of run tags.
    """
    run_name = f"{eval_dataset}_{ICL_shots}_shots"
    if repE_active:
        assert activations_artifact_name, "RepE is active, please provide the activations artifact name"
        run_name += f'_activations_{activations_artifact_name}'
    else:
        run_name += f'_baseline'

    run = wandb.init(project=WB_PROJECT_NAME, name=run_name, tags=tags or [])

    if repE_active:
        # link activations artifact as input
        activation_artifact = run.use_artifact(f"{activations_artifact_name}:latest")

    results = {
        "accuracy": accuracy,
    }
    # log metrics
    wandb.log(results)

    table_columns = ["id", "true", "pred"] + class_names
    table = wandb.Table(columns=table_columns)

    for idx, (true_idx, pred_idx, probs) in enumerate(zip(gt_labels, predictions, all_label_probs)):
        row = [idx, class_names[true_idx], class_names[pred_idx]] + probs
        table.add_data(*row)

    wandb.log({"predictions_table": table})

    y_true = [class_names[t] for t in gt_labels]
    y_pred = [class_names[p] for p in predictions]
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred, class_names=class_names
        )
    })

    wandb.finish()
    print(f"✅ Logged evaluation run: {run_name}")