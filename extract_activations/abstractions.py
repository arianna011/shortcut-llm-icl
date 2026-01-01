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
import torch.nn.functional as F


class BaseLLM:
    """Abstract interface for all LLM adapters."""
    name: str = "base"

    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        raise NotImplementedError

class EchoLLM(BaseLLM):
    """Echo-only LLM (for debugging)."""
    name = "echo"
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
        return f"[ECHO]\n{prompt[:max_tokens]}\n[/ECHO]"

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

    def __init__(self, model_name: str, hf_token: str, quantize: bool = True, use_auto_device_map: bool = True):

        self.model_name = model_name
        self.hf_token = hf_token

        model_kwargs = {
            "token": hf_token,
        }

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
            )
            model_kwargs["quantization_config"] = bnb_config

        if use_auto_device_map:
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.tokenizer.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.tokenizer.bos_token_id = 1
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
    
    def complete(self, prompt: str, max_tokens: int = 512, temperature: float = 1.0, 
                 return_ans_probs=False, logits_step=0) -> tuple[str, torch.Tensor]:
            
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        answer_probs = None

        prompt_length = len(self.tokenizer.decode(
                            input_ids[0],
                            skip_special_tokens=True))
        if temperature == 0.0: 
            output_dict = self.model.generate(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            do_sample = False,
                                            max_new_tokens=max_tokens, 
                                            return_dict_in_generate=True, 
                                            output_scores=return_ans_probs, 
                                            pad_token_id=self.tokenizer.eos_token_id)
        else:
            output_dict = self.model.generate(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            do_sample = True,
                                            temperature = temperature,
                                            max_new_tokens=max_tokens, 
                                            return_dict_in_generate=True, 
                                            output_scores=return_ans_probs, 
                                            pad_token_id=self.tokenizer.eos_token_id)
              
        outputs = output_dict['sequences'] # token IDs of the prompt + generated text
        outputs = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0] #batch size=1
        outputs_only = outputs[prompt_length:]
        
        if return_ans_probs:
            scores = output_dict['scores']
            answer_logits = scores[logits_step][0]
            answer_probs = F.softmax(answer_logits, dim=-1)
        
        gc.collect()
        torch.cuda.empty_cache()
        release_memory(self.model)

        return outputs_only, answer_probs


@dataclass
class LLMRegistry:
    """Container to hold mutator, target and judge LLMs."""
    mutator: BaseLLM
    target: BaseLLM
    judge: BaseLLM