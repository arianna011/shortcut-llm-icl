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
try:
    from rainbow_teaming import BaseLLM, EchoLLM
except Exception:
    from extract_activations import BaseLLM, EchoLLM
import re
from tqdm import tqdm


class Task(Enum):
    NLI = 0
    BINARY_NLI = 1
    CAUSAL_REASONING = 2
    QUESTION_CLASSIFICATION = 3
    SENTIMENT_CLASSIFICATION = 4

    def reference_dataset_name(self):
        if self is Task.NLI:
            return "mnli"
        elif self is Task.BINARY_NLI:
            return "rte"
        elif self is Task.CAUSAL_REASONING:
            return "copa"
        elif self is Task.QUESTION_CLASSIFICATION:
            return "trec"
        elif self is Task.SENTIMENT_CLASSIFICATION:
            return "sst2" 
        else:
            return None
        
    def reference_instruction(self):
        if self is Task.NLI:
            return "Given the premise, are we justified in saying the hypothesis? yes, no, or maybe.\n\n"
        elif self is Task.BINARY_NLI:
            return "Given the premise, are we justified in saying the hypothesis? yes or no.\n\n"
        elif self is Task.CAUSAL_REASONING:
            return "Given the premise, which is the most plausible causal alternative? 1 or 2.\n\n"
        elif self is Task.QUESTION_CLASSIFICATION:
            return 'Classify the type of the answer to the question. abbreviation, entity, description, person, location or number?\n\n'
        elif self is Task.SENTIMENT_CLASSIFICATION:
            return 'Classify the sentiment of the review. positive or negative?\n\n'
        else:
            return None
        
    def reference_gen_to_labels(self):
        if self is Task.NLI:
            return {"yes": "entailment",
                    "maybe": "neutral", 
                    "no": "contradiction"}
        elif self is Task.BINARY_NLI:
            return {"yes": "entailment",
                    "no": "not entailment"} 
        elif self is Task.CAUSAL_REASONING:
            return {"1": "1",
                    "2": "2"} 
        elif self is Task.QUESTION_CLASSIFICATION:
            return {'abbreviation': 'abbreviation', 
             'entity': 'entity', 
             'description': 'description', 
             'person': 'person', 
             'location': 'location', 
             'number': 'number'}
        elif self is Task.SENTIMENT_CLASSIFICATION:
            return {'negative': 'negative', 
             'positive': 'positive'}
        else:
            return None
        
    def format_input(self, input: Union[str, tuple[str]]):
        if self is Task.NLI:
            assert isinstance(input, tuple)
            prem, hyp = input
            return f'Premise: {prem}\nHypothesis: {hyp}\nAnswer (choose only one: yes / no / maybe): '
        elif self is Task.BINARY_NLI:
            assert isinstance(input, tuple)
            prem, hyp = input
            return f'Premise: {prem}\nHypothesis: {hyp}\nAnswer (choose only one: yes / no): '
        elif self is Task.CAUSAL_REASONING:
            assert isinstance(input, tuple)
            prem, choice1, choice2 = input
            return f'Premise: {prem}\nChoice 1: {choice1}\nChoice 2: {choice2}\nAnswer (choose only one: 1 / 2): '
        elif self is Task.QUESTION_CLASSIFICATION:
            return f'Question: ' + input + '\n' + 'Answer Type: '
        elif self is Task.SENTIMENT_CLASSIFICATION:
            return f'Review: ' + input + '\n' + 'Sentiment: '
        else:
            return None
        

DATASETS_TO_TASKS = {"mnli":Task.NLI, "rte": Task.BINARY_NLI, "copa": Task.CAUSAL_REASONING, "trec": Task.QUESTION_CLASSIFICATION, "sst2": Task.SENTIMENT_CLASSIFICATION, "cr": Task.SENTIMENT_CLASSIFICATION}     
SHORTCUT_SUITE_COLS = ["pairID", "sentence1", "sentence2", "gold_label"] 
SHORTCUT_SUITE_LABELS = ["entailment", "neutral", "contradiction"]
    

def load_nli_shortcuts_from_tsv(paths: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load one or more ShortcutSuite TSV files in a single pandas dataframe
    and keep only the relevant columns
    """
    dfs = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        df = pd.read_csv(path, sep="\t", on_bad_lines="skip")
        for col in SHORTCUT_SUITE_COLS:
            assert col in df.columns, "This method accepts only ShortcutSuite .tsv files"
        df = df[SHORTCUT_SUITE_COLS]
        df = df.rename(columns={
            "sentence1": "premise",
            "sentence2": "hypothesis"
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_nli_shortcuts_from_folder(folder: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder, "*.tsv"))
    return load_nli_shortcuts_from_tsv(files)

def create_paired_dataset(standard_df: pd.DataFrame, shortcut_df: pd.DataFrame, id_column: str = "pairID", 
                          label_column: str = "gold_label", clean_suffix: str = "_clean", dirty_suffix: str = "_dirty") -> pd.DataFrame:
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
    
    assert paired_df.shape[0] != 0, "Datasets merge resulted in empty dataset"
    
    labels_clean_col = label_column + clean_suffix
    labels_dirty_col = label_column + dirty_suffix
    paired_df = paired_df[(paired_df[labels_clean_col].isin(SHORTCUT_SUITE_LABELS)) & (paired_df[labels_dirty_col].isin(SHORTCUT_SUITE_LABELS))]
    assert (paired_df[labels_clean_col] == paired_df[labels_dirty_col]).all(), "Mismatch between the gold labels in clean and dirty datasets"
    paired_df = paired_df.drop(columns=[labels_dirty_col]).rename(columns={labels_clean_col: label_column})
   
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
    
    if task in [Task.NLI, Task.BINARY_NLI]:

        examples_dataset = load_dataset("nyu-mll/glue", task_dataset, split="train")
        try: 
            premises = examples_dataset['premise']
            hypotheses = examples_dataset['hypothesis']
        except KeyError:
            premises = examples_dataset['sentence1']
            hypotheses = examples_dataset['sentence2']
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
            demo = task.format_input(example_pairs[idx][0], example_pairs[idx][1]) + f"{example_pairs[idx][2]}\n\n"
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
    
    ans_dict = task.reference_gen_to_labels()
    ans = list(ans_dict.keys())
    ids = find_possible_ids_for_labels(ans, model.tokenizer)
    pred_idx, _ = predict_label(ans_probs, ids)
    pred = ans[pred_idx]
    return ans_dict[pred]

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

    if task in [Task.NLI, Task.BINARY_NLI]:

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

#if __name__ == '__main__':
    # df = load_nli_shortcuts_from_folder("data/ShortcutSuite/")
    # print(df.head())
    # show(df)
    # df_standard = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched.tsv")
    # df_shortcut = load_nli_shortcuts_from_tsv("data/ShortcutSuite/constituent.tsv")
    # df = create_paired_dataset(df_standard, df_shortcut)
    # print(df.columns)
    # print(df.shape)
    #selected_df = select_shortcut_prompts(df, Task.NLI, size=10, model=EchoLLM(), num_shot=1)
    #show(df)