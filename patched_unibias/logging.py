import json
import torch
import wandb
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, Optional

WB_PROJECT_NAME = "shortcut-repe"
WB_TEAM = "paolini-1943164-sapienza-universit-di-roma"
WB_USER = "paolini-1943164"

# ARTIFACTS ===================================================================================================================

def log_dataset_artifact(
    artifact_name: str,
    dataset: Union[str,pd.DataFrame],
    dataset_name: str,
    task: str,
    size: int,
    columns: list[str],
    labels: list[str],
    shortcut: str,
    selection_method: str,
    random_seed: int,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
):
    """
    Logs a small dataset (file or Pandas datafreame) as a W&B artifact.
    """
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
    artifact_name: str,
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
    name: str,
    activations_artifact_name: str,
    results: Dict[str, Any],
    table_data: Optional[list] = None,
    class_names: Optional[list] = None,
    tags: Optional[list] = None,
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
    run = wandb.init(project=WB_PROJECT_NAME, name=name, tags=tags or [])

    # Link activation artifact as input
    activation_artifact = run.use_artifact(f"{activations_artifact_name}:latest")
    activation_dir = activation_artifact.download()

    # Log metrics
    wandb.log(results)

    # Optionally log detailed results table
    if table_data:
        table = wandb.Table(columns=["id", "true", "pred"])
        for row in table_data:
            table.add_data(*row)
        wandb.log({"results_table": table})

        # Optionally log confusion matrix
        if class_names:
            y_true = [r[1] for r in table_data]
            y_pred = [r[2] for r in table_data]
            wandb.log({
                "confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true, preds=y_pred, class_names=class_names
                )
            })

    wandb.finish()
    print(f"✅ Logged evaluation run: {name} (input: {activations_artifact_name})")