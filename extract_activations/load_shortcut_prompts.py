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
#from rainbow_teaming import BaseLLM, EchoLLM
from extract_activations import BaseLLM, EchoLLM
import re
from tqdm import tqdm

class Task(Enum):
    NLI = 0

    def reference_dataset_name(self):
        if self is Task.NLI:
            return "mnli"
        
    def reference_instruction(self):
        if self is Task.NLI:
            return "Given the premise, are we justified in saying the hypothesis? yes, no, or maybe.\n\n"
        
    def reference_gen_to_labels(self):
        if self is Task.NLI:
            return {"yes": "entailment",
                    "maybe": "neutral", 
                    "no": "contradiction"}
        
    def format_input(self, input: Union[str, tuple[str]]):
        if self is Task.NLI:
            assert isinstance(input, tuple)
            prem, hyp = input
            return f'Premise: {prem}\nHypothesis: {hyp}\nAnswer (choose only one: yes / no / maybe): '
    



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
    rng = random.Random(seed)

    task_dataset = task.reference_dataset_name()
    task_gen_to_labels = task.reference_gen_to_labels()
    task_instruction = task.reference_instruction()
    
    if task == Task.NLI:

        examples_dataset = load_dataset("nyu-mll/glue", task_dataset, split="train")
        premises = examples_dataset['premise']
        hypotheses = examples_dataset['hypothesis']
        labels = examples_dataset['label']
        answers = [list(task_gen_to_labels.keys())[i] for i in labels]
        example_pairs = list(zip(premises, hypotheses, answers))
        
        # build label-specific index pools
        label_to_indices = {lab: [] for lab in set(labels)}
        for i, lab in enumerate(labels):
            label_to_indices[lab].append(i)

        # sample demonstrations equally from each class
        sampled_indices = []
        if num_shot > 0:
            for lab in sorted(label_to_indices.keys()):
                sampled_indices.extend(rng.sample(label_to_indices[lab], num_shot)) 

        demonstration = task_instruction
        for idx in sampled_indices:
            demo = (
                f"Premise: {example_pairs[idx][0]}\n"
                f"Hypothesis: {example_pairs[idx][1]}\n"
                f"Answer (choose only one: yes / no / maybe): {example_pairs[idx][2]}\n\n"
            )
            demonstration += demo

        def context_func(prem: str, hyp: str) -> str:
            return demonstration + task.format_input((prem, hyp))

        return context_func
            
    else:
        raise NotImplementedError

#from Unibias
def find_possible_ids_for_labels(arg_str_list: list[str], tokenizer):

    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000

    # hold the IDs of tokens corresponding to each arg_str
    args_lower = [arg.lower() for arg in arg_str_list]
    ids_dict = {arg_str: [] for arg_str in args_lower}
    
    # Iterate over the range of IDs only once
    for id in range(vocab_size):
        decoded = tokenizer.decode(id)
        # Check each arg_str for a match
        if decoded:
            decoded = decoded.lower()
            for arg in args_lower:
                if len(arg) > 1:
                    if decoded in arg and arg[0] == decoded[0] and len(decoded)>1:
                        ids_dict[arg].append(id)
                else:
                    if decoded in arg and arg[0] == decoded[0]:
                        ids_dict[arg].append(id)

    ids_list = list(ids_dict.values())
    max_len = max(len(sublist) for sublist in ids_list)
    padded_lst = [sublist + [sublist[0]] * (max_len - len(sublist)) for sublist in ids_list]
    return padded_lst

def predict_label(answer_logit, gt_ans_ids_list):
    prediction_logit_list = []
    # for each label, take the maximum logit among all compatible tokens
    for i_id_list in gt_ans_ids_list:
        logit_i = torch.max(answer_logit[torch.tensor(i_id_list)]).item()
        prediction_logit_list.append(logit_i)
    # take the label with highest score
    prediction = torch.max(torch.tensor(prediction_logit_list), dim=0)[1]
    return prediction, prediction_logit_list

def match_gen_to_label(task: Task, model: BaseLLM, gen: str, ans_probs: list[torch.Tensor], debug: bool = False) -> str:
    
    ans = list(task.reference_gen_to_labels().keys())
    ids = find_possible_ids_for_labels(ans, model.tokenizer)
    pred_idx, _ = predict_label(ans_probs, ids)
    pred = ans[pred_idx]
    return pred

def select_shortcut_prompts(paired_dataset: pd.DataFrame, task: Task, n_samples: int, model: BaseLLM, 
                            num_shot: int, condition: Callable[[str,str],bool], temperature: float = 0.0, 
                            max_tokens: int = 5, seed: int = 42, debug: bool = False, logits_step: int=0) -> pd.DataFrame:
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
        - condition (determines whether to select or reject a pair of prompts)
    
    Returns:
        a dataset of the desired size containing the selected prompts
    """

    assert len(paired_dataset) >= n_samples
    add_context = get_ICL_context_func(task, num_shot, seed=seed)
    selected_rows = []
    count = 0

    if task == Task.NLI:

        with tqdm(total=n_samples, desc="Selecting prompts", ncols=100, dynamic_ncols=True) as pbar:
            for i, row in paired_dataset.sample(frac=1, random_state=seed).iterrows():

                if count >= n_samples:
                    break

                clean_prompt = add_context(row["premise_clean"], row["hypothesis_clean"])
                dirty_prompt = add_context(row["premise_dirty"], row["hypothesis_dirty"])

                # get model predictions
                gen_clean, answ_probs_clean = model.complete(clean_prompt, max_tokens=max_tokens, 
                                                             temperature=temperature, return_ans_probs=True, logits_step=logits_step)
                pred_clean = match_gen_to_label(task, model, gen_clean, answ_probs_clean, debug)
                gen_dirty, answ_probs_dirty = model.complete(dirty_prompt, max_tokens=max_tokens, 
                                                             temperature=temperature, return_ans_probs=True, logits_step=logits_step)
                pred_dirty = match_gen_to_label(task, model, gen_dirty, answ_probs_dirty, debug)

                if debug:
                    tqdm.write(f"\n---- Sample {i}")
                    tqdm.write(f'Clean prompt: {clean_prompt}\n {pred_clean}\n (Generation: "{gen_clean}")\n')
                    tqdm.write(f'Dirty prompt: {dirty_prompt}\n {pred_dirty}\n (Generation: "{gen_dirty}")\n')
                    tqdm.write(f"\n----------------")

                if condition(task, row, pred_clean, pred_dirty):
                    count += 1
                    row_copy = row.copy()
                    row_copy["dirty_label"] = pred_dirty
                    selected_rows.append(row_copy)
                    pbar.update(1)
                    pbar.refresh()
                    if debug: tqdm.write(f"Extracted sample {i}")
            df = pd.DataFrame(selected_rows).reset_index(drop=True)
            if len(df) < n_samples:
                tqdm.write(f"Only {len(df)} samples found (requested {n_samples}).")
            return df

    else:
        raise NotImplementedError

if __name__ == '__main__':
    # df = load_nli_shortcuts_from_folder("data/ShortcutSuite/")
    # print(df.head())
    # show(df)
    df_standard = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched.tsv")
    df_shortcut = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched_negation.tsv")
    df = create_paired_dataset(df_standard, df_shortcut)
    #selected_df = select_shortcut_prompts(df, Task.NLI, size=10, model=EchoLLM(), num_shot=1)
    #show(df)