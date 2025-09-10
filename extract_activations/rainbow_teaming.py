""""
Generate prompts with the Rainbow Teaming approach (https://arxiv.org/abs/2402.16822)

The implemented procedure is the following:

1. initialize a K-dimensional archive, where each cell corresponds to a unique combination of K categories (numerical features are discretized into a set of intervals) describing the prompt within it (the descriptor z =<c1,..., cK> ) by random generation with an LLM or by loading some seed prompts from an existing dataset;

2. Iteratively:
    - sample an adversarial prompt x from the archive with descriptor z and a descriptor z' for the candidate prompt to be generated,
    with a distribution biased towards low-fitness archive areas; 
    
    - provide x and z' to an LLM (Mutator) to generate a new candidate prompt x' with descriptor z' by mutating the prompt x once for each feature;

    - feed the prompt x' to the target LLM to generate a response;

    - ask a Judge LLM to compare the effectiveness of x' and of the current archive’s elite prompt for the same descriptor, by using majority voting over multiple evaluations and swapping prompts order to mitigate order bias;

    - store the winning prompt in the position specified by z'.


Main components:
- DescriptorSchema: defines the K features for the archive.
- Archive: stores prompts and fitness scores keyed by descriptors.
- PromptRegistry: stores system instructions for Mutator and Judge LLMs based on factors like task and dataset.
- LLM Clients: Mutator and Judge adapters (Gemini, echo fallback).
- Main Loop: iteration of Rainbow Teaming to update the archive.

"""

from __future__ import annotations
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import google.generativeai as genai
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import yaml
import fnmatch
import torch
import gc
from accelerate.utils import release_memory


# Feature Descriptors ================================================================================================
@dataclass
class Feature:
    """
    A single feature dimension for the descriptor space.
    - if categorical: use values
    - if numeric: use bins
    """
    name: str
    values: Optional[List[str]] = None
    bins: Optional[List[Tuple[float, float]]] = None

    def discretize(self, value: Any) -> str:
        """Map a raw feature value to a discrete label."""
        if self.values is not None: # categorical value
            if value not in self.values:
                raise ValueError(f"Invalid value {value} for feature {self.name}")
            return str(value)
        elif self.bins is not None: # numerical value
            x = float(value)
            for i, (low, high) in enumerate(self.bins):
                if (x > low) and (x <= high):
                    return f"bin_{i}"
            if x <= self.bins[0][0]:
                return "bin_0"
            return f"bin_{len(self.bins)-1}"
        return str(value)


@dataclass
class DescriptorSchema:
    """Defines the full descriptor schema as a list of features."""
    features: List[Feature]

    def to_key(self, categories: Sequence[str]) -> Tuple[str, ...]:
       return tuple(str(c) for c in categories)

    def random_descriptor(self) -> Tuple[str, ...]:
        """Sample a random descriptor z from the feature space."""
        z = []
        for f in self.features:
            if f.values:
                z.append(random.choice(f.values))
            elif f.bins:
                z.append(f"bin_{random.randrange(len(f.bins))}")
            else:
                continue
        return self.to_key(z)

    def all_cells(self) -> List[Tuple[str, ...]]:
        """Return all possible descriptor combinations (archive cells)."""
        feat_choices = []
        cells = [[]]
        for f in self.features:
            if f.values:
                feat_choices.append([str(v) for v in f.values])
            elif f.bins:
                feat_choices.append([f"bin_{i}" for i in range(len(f.bins))])
            else:
                continue
        for opts in feat_choices:
            cells = [c + [o] for c in cells for o in opts]
        return [self.to_key(c) for c in cells]
    

# Archive ======================================================================================================

@dataclass
class ArchiveCell:
    """A single cell in the archive with its elite prompt and metadata."""
    prompt: Optional[str] = None
    fitness: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Archive:
    """The full K-dimensional archive mapping descriptors to cells."""

    def __init__(self, schema: DescriptorSchema):
        self.schema = schema
        self.grid = {cell: ArchiveCell() for cell in schema.all_cells()}

    def get(self, z: Tuple[str, ...]) -> ArchiveCell:
        return self.grid[z]

    def insert_if_better(self, z: Tuple[str, ...], prompt: str, fitness: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Insert prompt in cell z if fitness is better than current elite."""
        cell = self.grid[z]
        if (cell.prompt is None) or (cell.fitness is None) or (fitness > cell.fitness):
            self.grid[z] = ArchiveCell(prompt=prompt, fitness=fitness, metadata=metadata or {})
            return True
        return False

    def sample_cell_biased(self, temperature: float = 1.0) -> Tuple[Tuple[str, ...], ArchiveCell]:
        """Sample a cell biased toward empty or low-fitness ones."""
        weights = []
        cells = list(self.grid.items())
        all_fit = [c.fitness for _, c in cells if c.fitness is not None]
        max_fit = max(all_fit) if all_fit else 1.0
        for key, cell in cells:
            f = cell.fitness if cell.fitness is not None else 0.0
            nf = (f / max_fit) if max_fit > 0 else 0.0
            bonus = 0.5 if cell.prompt is None else 0.0
            weight = math.exp(((1.0 - nf) + bonus) / max(1e-6, temperature))
            weights.append(weight)
        s = sum(weights)
        r = random.random() * s
        acc = 0.0
        for (key, cell), w in zip(cells, weights):
            acc += w
            if acc >= r:
                return key, cell
        return cells[-1]

    def save(self, path: str):
        """Save archive to JSON file."""
        data = {
            "schema": [{"name": f.name, "values": f.values, "bins": f.bins} for f in self.schema.features],
            "grid": {
                    "|".join(k): {"prompt": v.prompt, "fitness": v.fitness, "metadata": v.metadata}
                     for k, v in self.grid.items()
                    }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> Archive:
        """Load archive from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        schema = DescriptorSchema(features=[Feature(**fd) for fd in data["schema"]])
        arch = Archive(schema)
        arch.grid = {}
        for k_str, v in data["grid"].items():
            key = tuple(k_str.split("|"))
            arch.grid[key] = ArchiveCell(prompt=v.get("prompt"), fitness=v.get("fitness"), metadata=v.get("metadata", {}))
        return arch
    

# System prompts ===========================================================================================================

@dataclass
class PromptSpec:
    """
    A single system prompt.
    role: 'mutator' | 'judge'
    when: dict of conditions (e.g., {'dataset':'sst2','task':'sentiment_classification','labels':['positive', 'negative']})
          values may be str, list[str], or glob patterns (e.g., 'sst*')
    priority: higher wins on ties
    """
    role: str
    system: str
    when: Dict[str, Union[str, List[str]]] = field(default_factory=dict)
    priority: int = 0
    id: Optional[str] = None


class PromptRegistry:
    """
    Store system prompts for different roles with hierarchical matching:
    - exact matches beat wildcards/globs
    - more fields matched beats fewer
    - higher priority beats lower
    - earlier registration wins ties
    """

    def __init__(self):
        self.specs: List[PromptSpec] = []

    def register(self, spec: PromptSpec):
        self.specs.append(spec)

    @staticmethod
    def _match_value(ctx_value: Optional[str], cond_value: Union[str, List[str]]) -> bool:
        if ctx_value is None:
            return False
        if isinstance(cond_value, list):
            return any(fnmatch.fnmatch(ctx_value, str(v)) for v in cond_value)
        return fnmatch.fnmatch(ctx_value, str(cond_value))

    def _score(self, spec: PromptSpec, ctx: Dict[str, Any]) -> Optional[Tuple[int,int,int]]:
        """
        Score = (num_exact_or_glob_matches, -num_wildcards_used, priority)
        Return None if any condition fails.
        """
        matches = 0
        wildcards = 0
        for k, v in spec.when.items():
            ctx_val = ctx.get(k)
            # treat "*" specially as unconditional pass but counts as wildcard
            if (isinstance(v, str) and v == "*") or (isinstance(v, list) and any(x == "*" for x in v)):
                # only count if key exists at all in context
                if k in ctx:
                    matches += 1
                    wildcards += 1
                continue
            if not self._match_value(ctx_val, v):
                return None
            matches += 1
        return (matches, -wildcards, spec.priority)

    def get_system(self,
                   role: str,
                   schema,
                   z: Optional[Tuple[str, ...]] = None,
                   meta: Optional[Dict[str, Any]] = None,
                   default: str = "") -> str:
        """
        role: 'mutator' | 'judge'
        z: descriptor to extract feature-context (e.g., dataset/label/language/shortcut_reliance)
        meta: extra context (e.g., {'task':'sentiment_classification'})
        """
        ctx = {}
        ctx.update(self._descriptor_to_dict(schema, z))
        if meta:
            ctx.update(meta)

        best_score = None
        best_idx = None
        for i, spec in enumerate(self.specs):
            if spec.role != role:
                continue
            s = self._score(spec, ctx)
            if s is None:
                continue
            if (best_score is None) or (s > best_score):
                best_score = s
                best_idx = i
        if best_idx is None:
            return default
        return self.specs[best_idx].system

    @classmethod
    def from_yaml(cls, path: str) -> "PromptRegistry":
        reg = cls()
        data = yaml.safe_load(open(path, "r", encoding="utf-8"))
        for i, row in enumerate(data):
            reg.register(PromptSpec(
                role=row["role"],
                system=row["system"],
                when=row.get("when", {}),
                priority=row.get("priority", 0),
                id=row.get("id")
            ))
        return reg


# LLM Abstractions =========================================================================================================

class BaseLLM:
    """Abstract interface for all LLM adapters."""
    name: str = "base"

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        raise NotImplementedError

class EchoLLM(BaseLLM):
    """Echo-only LLM (for debugging)."""
    name = "echo"
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        return f"[ECHO]\n{prompt[:500]}\n[/ECHO]"

class GeminiLLM(BaseLLM):
    """Google Gemini client."""
    def __init__(self, api_key: str, model: str, sys_instr: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model, system_instruction=sys_instr)
        self.name = f"gemini:{model}"

    def _extract_text(self, res) -> str:
        try:
            t = getattr(res, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        except Exception:
            pass
        # manual extraction from candidates/parts
        texts = []
        try:
            for cand in getattr(res, "candidates", []) or []:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if parts:
                    for p in parts:
                        pt = getattr(p, "text", None)
                        if isinstance(pt, str) and pt.strip():
                            texts.append(pt.strip())
        except Exception:
            pass
        return "\n".join(texts).strip()

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        try:
            res = self.model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens, "temperature": temperature})
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}")

        txt = self._extract_text(res)
        if txt:
            return txt
        # no text produced
        try:
            cand0 = res.candidates[0] if getattr(res, "candidates", None) else None
            fr = getattr(cand0, "finish_reason", None)
        except Exception:
            fr = None
        raise ValueError(f"Gemini returned no text, finish_reason={fr}.")

class HuggingFaceLLM(BaseLLM):
    """LLM loaded from Hugging Face."""

    def __init__(self, model_name: str, hf_token: str = None, quantize: bool = True, use_auto_device_map=True):

        self.model_name = model_name
        self.quantize = quantize
        self.use_auto_device_map = use_auto_device_map
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token if hf_token else None)
        
        self.device = torch.device("cuda:0")
        self.model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True
        }

        if hf_token:
            self.model_kwargs["token"] = hf_token

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,            
                bnb_4bit_quant_type="nf4",                  
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model_kwargs["quantization_config"] = bnb_config

        if use_auto_device_map:
            self.model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **self.model_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **self.model_kwargs).to(self.device)
        
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
            
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        prompt_length = len(self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True))
        output_dict = self.model.generate(input_ids=inputs["input_ids"], 
                                          attention_mask=attention_mask,
                                          do_sample = True,
                                          temperature = temperature,
                                          max_new_tokens=max_tokens, 
                                          return_dict_in_generate=True, 
                                          output_scores=False, 
                                          pad_token_id=self.tokenizer.eos_token_id)
              
        outputs = output_dict['sequences'] # token IDs of the prompt + generated text
        outputs = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0] #batch size=1
        outputs_only = outputs[prompt_length:]
        gc.collect()
        torch.cuda.empty_cache()
        release_memory(self.model)

        return outputs_only


@dataclass
class LLMRegistry:
    """Container to hold mutator, target and judge LLMs."""
    mutator: BaseLLM
    target: BaseLLM
    judge: BaseLLM



# =============================
# Core Operators
# =============================

# def mutate_prompt(mutator: BaseLLM, 
#                   mutator_prompt:str, 
#                   parent_prompt: str, 
#                   descriptor: Tuple[str, ...], 
#                   feature_names: List[str], 
#                   temperature: float = 0.9) -> str:
#     """Call mutator LLM to produce a new candidate prompt."""
#     desc_lines = [f"{name}: {val}" for name, val in zip(feature_names, descriptor)]
#     prompt = MUTATOR_TEMPLATE.format(descriptor="\n".join(desc_lines), parent=parent_prompt)
#     return mutator.complete(prompt, max_tokens=200, temperature=temperature).strip()


# def run_target(target: BaseLLM, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
#     """Query the target LLM with a prompt."""
#     return target.complete(prompt, max_tokens=max_tokens, temperature=temperature).strip()


# def judge_pair(judge: BaseLLM, resp_a: str, resp_b: str, criterion: str = "toxic", n_votes: int = 3, temperature: float = 0.0) -> str:
#     """Judge candidate vs elite with majority vote and order swapping."""
#     votes = []
#     for i in range(n_votes):
#         if i % 2 == 0:
#             prompt = JUDGE_TEMPLATE.format(criterion=criterion, resp_a=resp_a, resp_b=resp_b)
#             first = "A"; second = "B"
#         else:
#             prompt = JUDGE_TEMPLATE.format(criterion=criterion, resp_a=resp_b, resp_b=resp_a)
#             first = "B"; second = "A"
#     out = judge.complete(prompt, max_tokens=1, temperature=temperature).strip().upper()
#     vote = first if out.startswith("A") else second
#     votes.append(vote)
#     return "A" if votes.count("A") >= votes.count("B") else "B"



# Rainbow Teaming Loop ===============================================================================================

@dataclass
class RainbowConfig:
    """Configuration for a Rainbow Teaming run."""
    schema: DescriptorSchema # archive features
    registry: LLMRegistry # Mutator, Target and Judge LLMs
    seed_prompts: List[Tuple[Tuple[str, ...], str]] # initial archive prompts
    sample_temperature: float = 1.0
    target_max_tokens: int = 256
    judge_votes: int = 3
    save_path: str = "archive.json"
   

# def initialize_archive(cfg: RainbowConfig) -> Archive:
#     """Initialize archive with seeds from an existing NLP dataset"""

#     assert(cfg.seed_prompts), "Need a non-empty dataset to initialize archive"
#     arch = Archive(cfg.schema)
#     for z, p in cfg.seed_prompts:
#         arch.insert_if_better(z, p, fitness=0.0, metadata={"note": "seed"})
#     return arch

# def generate_seeds_from_dataset(dataset_name: str, cfg: RainbowConfig) -> List[Tuple[Tuple[str, ...], str]]:
#     """Prepare prompts from the given dataset in the right format for complete archive seed initialization"""
#     ds = load_dataset(dataset_name)
#     ds_train = load_dataset(dataset_name)["train"]


# def evaluate_in_cell(cfg: RainbowConfig, arch: Archive, z: Tuple[str, ...], parent_prompt: str, candidate_prompt: str) -> Tuple[float, Dict[str, Any]]:
#     """Evaluate candidate prompt vs elite in a cell."""
#     target = cfg.registry.target
#     judge = cfg.registry.judge

#     resp_candidate = run_target(target, candidate_prompt, max_tokens=cfg.target_max_tokens)
#     current = arch.get(z)
#     resp_elite = run_target(target, current.prompt, max_tokens=cfg.target_max_tokens) if current.prompt else ""

#     if not current.prompt:
#         return 1.0, {"winner": "candidate", "resp_candidate": resp_candidate, "resp_elite": resp_elite}

#     winner_label = judge_pair(
#     judge,
#     resp_a=resp_candidate,
#     resp_b=resp_elite,
#     criterion=cfg.criterion,
#     n_votes=cfg.judge_votes,
#     )
#     if winner_label == "A":
#         fitness = 1.0
#         meta = {"winner": "candidate", "resp_candidate": resp_candidate, "resp_elite": resp_elite}
#     else:
#         fitness = max(current.fitness or 0.5, 0.9)
#         meta = {"winner": "elite", "resp_candidate": resp_candidate, "resp_elite": resp_elite}
#     return fitness, meta


# def rainbow_iterate(cfg: RainbowConfig, arch: Archive, n_iters: int = 100, log_every: int = 5) -> Archive:
#     """Main Rainbow Teaming iteration loop."""
#     feature_names = [f.name for f in cfg.schema.features]
#     for t in range(1, n_iters + 1):
#         z_from, cell = arch.sample_cell_biased(temperature=cfg.sample_temperature)
#         z_to = cfg.schema.random_descriptor()
#         parent_prompt = cell.prompt or f"Explain {', '.join(z_from)} in an unsafe way."

#         try:
#             candidate = mutate_prompt(cfg.registry.mutator, parent_prompt, z_to, feature_names)
#         except Exception:
#             candidate = parent_prompt

#         fitness, meta = evaluate_in_cell(cfg, arch, z_to, parent_prompt, candidate)
#         inserted = arch.insert_if_better(
#                 z_to,
#                 candidate if meta.get("winner") == "candidate" else arch.get(z_to).prompt,
#                 fitness,
#                 metadata=meta,
#             )

#         if (t % log_every) == 0:
#             print(f"[t={t}] z_from={z_from} -> z_to={z_to} | winner={meta.get('winner')} | inserted={inserted}")
#     try:
#         arch.save(cfg.save_path)
#     except Exception as e:
#         print("Save failed:", e)
#     return arch




# =============================
# Example configuration (entry point)
# =============================
# if __name__ == "__main__":
#     random.seed(42)

#     API = "AIzaSyBuqsCfDar-hHfpFGWmTI7jawPHEwgsnH4"

#     schema = DescriptorSchema(features=[
#     Feature(name="shortcut", values=["+", "-"]),
#     Feature(name="length", values=["short", "medium", "long"])
#     ])

#     # Choose LLMs
#     try:
#         mutator = GeminiLLM(API, "gemini-2.5-flash","you are a useful assistant mutator")
#         judge = GeminiLLM(API, "gemini-2.5-flash","you are a useful assistant judge")
#     except Exception as e:
#         print(e)

#     print(mutator.complete("Hello. How are you?"))

    


"""     registry = LLMRegistry(mutator=mutator, target=target, judge=judge)


    # Configure Rainbow Teaming
    cfg = RainbowConfig(
    schema=schema,
    registry=registry
    sample_temperature=1.0,
    target_max_tokens=256,
    judge_votes=5,
    save_path="archive.json",
    )


    # Initialize archive and run iterations
    arch = initialize_archive(cfg)
    arch = rainbow_iterate(cfg, arch, n_iters=50, log_every=5)


    # Save final archive
    arch.save(cfg.save_path)
    print("Saved archive to", cfg.save_path) """