import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import glob
from pandasgui import show


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


def create_paired_dataset(standard_df: pd.DataFrame, shortcut_df: pd.DataFrame, id_column: str = "pairID") -> pd.DataFrame:
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
        suffixes=("_standard", "_shortcut"))
    
    # remove duplicated columns and columns with 'unknown' value
    for col in paired_df.columns:
        if col.endswith("_standard"):
            base = col[:-9]
            other = base + "_shortcut"
            if other in paired_df:
                identical = paired_df[[col, other]].nunique(axis=1).max() == 1
                all_unknown = paired_df[col].eq("unknown").all() and paired_df[other].eq("unknown").all()
                if all_unknown: 
                     paired_df = paired_df.drop(columns=[col,other])
                elif identical:
                    paired_df = paired_df.drop(columns=[other]).rename(columns={col: base})

    return paired_df


if __name__ == '__main__':
    # df = load_nli_shortcuts_from_folder("data/ShortcutSuite/")
    # print(df.head())
    # show(df)
    df_standard = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched.tsv")
    df_shortcut = load_nli_shortcuts_from_tsv("data/ShortcutSuite/dev_matched_negation.tsv")
    df = create_paired_dataset(df_standard, df_shortcut)
    show(df)