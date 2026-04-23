from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMRequest:
    prompt: str
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 512


class BaseLLMBackend:
    def generate(self, req: LLMRequest) -> str:
        raise NotImplementedError


class LocalVLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
    ) -> None:
        from vllm import LLM

        self._llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=int(tensor_parallel_size),
            dtype="bfloat16",
            max_model_len=int(max_model_len),
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=bool(trust_remote_code),
            enforce_eager=bool(enforce_eager),
        )

    def generate(self, req: LLMRequest) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=float(req.temperature),
            top_p=float(req.top_p),
            max_tokens=int(req.max_tokens),
        )
        out = self._llm.generate([req.prompt], params)
        return out[0].outputs[0].text


class LocalTransformersBackend(BaseLLMBackend):
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        dtype = torch.float16 if torch_dtype == "float16" else "auto"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=bool(trust_remote_code))
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=bool(trust_remote_code),
        )
        self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def generate(self, req: LLMRequest) -> str:
        out = self._pipe(
            req.prompt,
            do_sample=bool(req.temperature > 0),
            temperature=float(req.temperature),
            top_p=float(req.top_p),
            max_new_tokens=int(req.max_tokens),
            return_full_text=False,
        )
        return out[0]["generated_text"]


class RuleBasedExtractiveBackend(BaseLLMBackend):
    """Fallback backend for offline smoke tests without heavy LLM dependencies."""

    def generate(self, req: LLMRequest) -> str:
        lines = [ln.strip() for ln in req.prompt.splitlines() if ln.strip()]
        context_lines = [ln for ln in lines if ln.startswith("[EVID") or ln.startswith("- ")]
        if not context_lines:
            return "문서에서 확인되지 않습니다"
        picked = context_lines[:3]
        return "\n".join([f"- {ln[:120]} [근거 1]" for ln in picked])


_BACKEND_CACHE: Dict[str, BaseLLMBackend] = {}


def get_backend(backend_cfg: Dict[str, Any]) -> BaseLLMBackend:
    backend = backend_cfg.get("backend", "local_vllm")
    model_name_or_path = backend_cfg.get("model_name_or_path", "")
    cache_key = f"{backend}:{model_name_or_path}:{backend_cfg.get('device', '')}:{backend_cfg.get('tensor_parallel_size', 1)}"
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    if backend == "local_vllm":
        obj = LocalVLLMBackend(
            model_name_or_path=model_name_or_path,
            tensor_parallel_size=int(backend_cfg.get("tensor_parallel_size", 1)),
            max_model_len=int(backend_cfg.get("max_model_len", 4096)),
            gpu_memory_utilization=float(backend_cfg.get("gpu_memory_utilization", 0.85)),
            trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
            enforce_eager=bool(backend_cfg.get("enforce_eager", False)),
        )
    elif backend == "local_transformers":
        obj = LocalTransformersBackend(
            model_name_or_path=model_name_or_path,
            device=str(backend_cfg.get("device", "cuda")),
            torch_dtype=str(backend_cfg.get("torch_dtype", "auto")),
            trust_remote_code=bool(backend_cfg.get("trust_remote_code", False)),
        )
    elif backend == "local_rule":
        obj = RuleBasedExtractiveBackend()
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    _BACKEND_CACHE[cache_key] = obj
    return obj
