"""
Load a dataset and extract data which is likely to provide a specific kind of shortcut for an LLM

Types of shortcuts considered are:
- lexical -> e.g. presence of polarity words in sentiment classification SST2
"""

from datasets import load_dataset
import re
from enum import Enum


class Sentiment(Enum):
    POSITIVE = {"good","great","excellent","amazing","awesome","fantastic","love","loved","enjoyed","wonderful","brilliant", "beautiful"}
    NEGATIVE = {"bad","terrible","awful","poor","hate","hated","boring","worst","dull","horrible","lame","stupid"}
    
    @property
    def lexicon(self):
        """Return the set of words associated with this sentiment."""
        return self.value
    
class PolarityTask(Enum):
    SENTIMENT = Sentiment


def load_lexical_shortcut_candidates(dataset_name: str, task: PolarityTask):
    ds = load_dataset(dataset_name)
    pos, neg = task.POSITIVE.lexicon, task.NEGATIVE.lexicon
    cands = [(ex["sentence"], ex["label"]) for ex in ds["train"] if has_polarity_shortcut(ex["sentence"], ex["label"], pos, neg)]
    print(f"Found {len(cands)} examples with potential lexical shortcuts")
    return cands



def has_polarity_shortcut(text, y, pos_shortcuts, neg_shortcuts):
    """
    Determine whether the input text with label y 
    contains the given polarity words (positive or negative)
    which could represent a lexical shortcut for classification
    """
    shortcuts = pos_shortcuts | neg_shortcuts
    words= set(re.findall(r"[A-Za-z']+", text.lower()))
    if not (words & shortcuts):
        return False
    pos_hit = bool(words & pos_shortcuts)
    neg_hit = bool(words & neg_shortcuts)
    if pos_hit and not neg_hit and y==1: return True
    if neg_hit and not pos_hit and y==0: return True
    # mixed or neutral: not a clear lexical shortcut
    return False