from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_ROLE_PREFIX_RE = re.compile(r"^(?:\s*(system|user|assistant)\s*[:：]?\s*)+", re.IGNORECASE)


def _clean_generation_text(text: str) -> str:
    s = (text or "").strip()
    # Some chat templates may leak role markers into plain text output.
    # Remove a few leading occurrences.
    for _ in range(3):
        s2 = _ROLE_PREFIX_RE.sub("", s)
        if s2 == s:
            break
        s = s2.strip()
    return s


@dataclass
class HFGenerator:
    model_id: str
    device: str = "auto"  # auto|cpu|cuda

    _tokenizer: Optional[object] = None
    _model: Optional[object] = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=("auto" if self.device == "auto" else None),
        )
        if self.device in {"cpu", "cuda"}:
            model = model.to(self.device)

        self._tokenizer = tokenizer
        self._model = model

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2) -> str:
        self.load()
        assert self._tokenizer is not None and self._model is not None

        tokenizer = self._tokenizer
        model = self._model

        # Qwen-style chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            # Some tokenizers require add_generation_prompt=True to append the assistant tag.
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            except TypeError:
                # Backward compatible fallback
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
        else:
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids

        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                repetition_penalty=1.05,
                pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )

        # Remove prompt tokens
        gen_ids = output_ids[0][input_ids.shape[-1] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return _clean_generation_text(text)
