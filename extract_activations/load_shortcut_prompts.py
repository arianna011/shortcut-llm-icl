import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import glob
#from pandasgui import show
from enum import Enum
from collections.abc import Callable
import torch
from datasets import load_dataset
import random
from extract_activations import BaseLLM, EchoLLM

class Task(Enum):
    NLI = 0

    def reference_dataset_name(self):
        if self is Task.NLI:
            return "mnli"
        
    def reference_labels(self):
        if self is Task.NLI:
            return ["yes", # entailment 
                    "maybe", # neutral
                    "no"] # contradiction
        
    def format_input(self, input: Union[str, tuple[str]]):
        if self is Task.NLI:
            assert isinstance(input, tuple)
            prem, hyp = input
            return f'Premise: {prem}\nHypothesis: {hyp}\nAnswer: '



def load_nli_shortcuts_from_tsv(paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load one or more ShortcutSuite TSV files in a single pandas dataframe
    and keep only the relevant columns:
    pairID, premise, hypothesis, gold_label, heuristic, subcase
    """
    dfs = []
    required_cols = ["pairID", "premise", "hypothesis", "gold_label", "heuristic", "subcase"]
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
        df = df.rename(columns={
            "sentence1": "premise",
            "sentence2": "hypothesis"
        })
        for col in required_cols:
            if col not in df.columns:
                df[col] = "unknown"
        df = df[required_cols]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_nli_shortcuts_from_folder(folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.tsv"))
    return load_nli_shortcuts_from_tsv(files)

def create_paired_dataset(standard_df: pd.DataFrame, shortcut_df: pd.DataFrame, id_column: str = "pairID", 
                          unknown_value: str = "unknown", clean_suffix: str = "_clean", dirty_suffix: str = "_dirty") -> pd.DataFrame:
    """
    Given two dataframes, one with standard NLP statements, the other with injected shortcuts,
    pair the examples corresponding to the same original prompt (based on a column ID) and return a merged dataframe
    """
    if id_column not in standard_df.columns or id_column not in shortcut_df.columns:
        raise ValueError("Both dataframes must contain an ID column for alignment.")
    
    paired_df = pd.merge(
        standard_df,
        shortcut_df,
        on=id_column,
        suffixes=(clean_suffix, dirty_suffix))
    
    # remove duplicated columns and columns with 'unknown' value
    for col in paired_df.columns:
        if col.endswith(clean_suffix):
            base = col[:-len(clean_suffix)]
            other = base + dirty_suffix
            if other in paired_df:
                identical = paired_df[[col, other]].nunique(axis=1).max() == 1
                all_unknown = paired_df[col].eq(unknown_value).all() and paired_df[other].eq(unknown_value).all()
                if all_unknown: 
                     paired_df = paired_df.drop(columns=[col,other])
                elif identical:
                    paired_df = paired_df.drop(columns=[other]).rename(columns={col: base})

    return paired_df

def get_ICL_context_func(task: Task, num_shot: int, seed: int = 42) -> Callable[..., str]:
    """
    Given an NLP task and a number of ICL shots, 
    build a function to get ICL prompts with demonstrations from the reference dataset for the given task

    Params:
        - task (the considered NLP task)
        - num shot (the number of examples to provide in the ICL context for each ground truth label)

    Returns:
        a function to add an ICL context to raw NLP inputs
    """
    random.seed(seed)

    task_dataset = task.reference_dataset_name()
    task_labels = task.reference_labels()
    
    if task == Task.NLI:

        examples_dataset = load_dataset("nyu-mll/glue", task_dataset, split="train", trust_remote_code=True)
        premises = examples_dataset['premise']
        hypotheses = examples_dataset['hypothesis']
        labels = examples_dataset['label']
        answers = [task_labels[i] for i in labels]
        example_pairs = list(zip(premises, hypotheses, answers))
        
        # build label-specific index pools
        label_to_indices = {lab: [] for lab in set(labels)}
        for i, lab in enumerate(labels):
            label_to_indices[lab].append(i)

        # sample demonstrations equally from each class
        sampled_indices = []
        if num_shot > 0:
            for lab in sorted(label_to_indices.keys()):
                sampled_indices.extend(random.sample(label_to_indices[lab], num_shot))

        instruction = "Given the premise, are we justified in saying the hypothesis? Answer: yes, no, or maybe.\n\n"
        demonstration = instruction
        for idx in sampled_indices:
            demo = (
                f"Premise: {example_pairs[idx][0]}\n"
                f"Hypothesis: {example_pairs[idx][1]}\n"
                f"Answer: {example_pairs[idx][2]}\n\n"
            )
            demonstration += demo

        def context_func(prem: str, hyp: str) -> str:
            return demonstration + task.format_input((prem, hyp))

        return context_func
            
    else:
        raise NotImplementedError
    

def select_shortcut_prompts(paired_dataset: pd.DataFrame, task: Task, size: int, model: BaseLLM, num_shot: int) -> pd.DataFrame:
    """
    Given a dataset containing pairs (clean, dirty) of NLP prompts with and without an injected shortcut,
    extract a desired number of prompt pairs where the input model succeed to peform the given task
    on the clean prompt but fails on the dirty one

    Params:
        - paired dataset (contains clean and dirty NLP inputs)
        - task (the considered NLP task)
        - size (the desired number of prompts to randomly select from the suitable ones)
        - model (the LLM to be tested on shortcuts)
        - num shot (parameter for ICL setting)
    
    Returns:
        a dataset of the desired size containing the selected prompts
    """

    add_context = get_ICL_context_func(task, num_shot)
    selected_rows = []

    if task == Task.NLI:

        for _, row in paired_dataset.iterrows():
            clean_prompt = add_context(row["premise_clean"], row["hypothesis_clean"])
            dirty_prompt = add_context(row["premise_dirty"], row["hypothesis_dirty"])
            gold = row["gold_label"]

            # get model predictions
            pred_clean = model.complete(clean_prompt, max_tokens=10).lower().strip()
            pred_dirty = model.complete(dirty_prompt, max_tokens=10).lower().strip()

            if pred_clean == gold and pred_dirty != gold:
                selected_rows.append(row)

        if len(selected_rows) > size:
            selected_rows = random.sample(selected_rows, size)

        return pd.DataFrame(selected_rows).reset_index(drop=True)


    else:
        raise NotImplementedError

if __name__ == '__main__':
    # df = load_nli_shortcuts_from_folder("data/ShortcutSuite/")
    # print(df.head())
    # show(df)
    df_standard = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched.tsv")
    df_shortcut = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched_negation.tsv")
    df = create_paired_dataset(df_standard, df_shortcut)
    selected_df = select_shortcut_prompts(df, Task.NLI, size=10, model=EchoLLM(), num_shot=1)
    #show(selected_df)