"""vLLM backbone -- the paper's inference engine (appendix A.3).

Generation only. vLLM does not expose the teacher-forced token log-probabilities
Equation 4 needs, so ``score`` raises; build the filter's labels with the
``huggingface`` backend and run inference with this one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..config import LLMConfig
from .base import LLM, ScoredSequence, register_llm


class VLLMBackend(LLM):
    def __init__(self, config: LLMConfig) -> None:
        from vllm import LLM as _VLLM  # type: ignore

        self.config = config
        self.name = config.model
        self.revision = config.revision or "main"
        engine_kwargs: Dict[str, Any] = dict(config.options.get("engine", {}))
        engine_kwargs.setdefault("dtype", config.dtype)
        if config.revision:
            engine_kwargs.setdefault("revision", config.revision)
        self.engine = _VLLM(model=config.model, **engine_kwargs)
        self.tokenizer = self.engine.get_tokenizer()
        self.use_chat_template = bool(config.chat_template) and getattr(
            self.tokenizer, "chat_template", None
        )

    def _wrap(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        from vllm import SamplingParams  # type: ignore

        if not prompts:
            return []
        params = SamplingParams(
            temperature=float(kwargs.get("temperature", self.config.temperature)),
            top_p=self.config.top_p,
            max_tokens=int(kwargs.get("max_new_tokens", self.config.max_new_tokens)),
        )
        outputs = self.engine.generate([self._wrap(p) for p in prompts], params)
        return [o.outputs[0].text.strip() for o in outputs]

    def score(self, prompt: str, continuation: str) -> ScoredSequence:
        raise NotImplementedError(
            "vLLM does not expose teacher-forced token log-probabilities; use "
            "llm.backend=huggingface to build perplexity labels"
        )

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "revision": self.revision, "class": "VLLMBackend"}


@register_llm("vllm")
def _build_vllm(config: LLMConfig) -> LLM:
    return VLLMBackend(config)
