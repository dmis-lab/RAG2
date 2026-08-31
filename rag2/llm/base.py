"""Backbone-LLM interface.

The paper uses one LLM for three jobs (section 3.3, section 3.2):

* generating the chain-of-thought rationale used as the retrieval query,
* generating the final answer, and
* scoring rationale perplexity with and without a document, which is what
  produces the filter's training labels.

Only the third needs token-level log-probabilities, so ``score`` is optional:
an inference-only backend (a hosted API, say) can raise ``NotImplementedError``
and still drive the pipeline; only label construction will refuse to run.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

from ..config import LLMConfig


@dataclass
class ScoredSequence:
    """Token-level scores of a continuation under a prompt."""

    token_logprobs: List[float]
    num_tokens: int

    @property
    def sum_logprob(self) -> float:
        return float(sum(self.token_logprobs))

    @property
    def mean_logprob(self) -> float:
        return self.sum_logprob / self.num_tokens if self.num_tokens else 0.0


class LLM(abc.ABC):
    """A causal LM used as the RAG2 backbone."""

    name: str = ""
    revision: str = ""

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        """Greedy-decode a completion for each prompt (paper appendix A.3: T=0)."""

    def score(self, prompt: str, continuation: str) -> ScoredSequence:
        """Log-probabilities of ``continuation`` tokens conditioned on ``prompt``.

        Prompt tokens must not contribute; the returned scores cover the
        continuation only, which is what makes Equation 4's length
        normalisation well defined.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support scoring")

    def score_batch(self, pairs: Sequence[tuple]) -> List[ScoredSequence]:
        return [self.score(prompt, continuation) for prompt, continuation in pairs]

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "revision": self.revision, "class": type(self).__name__}


_REGISTRY: Dict[str, Callable[[LLMConfig], LLM]] = {}


def register_llm(key: str):
    def decorator(factory: Callable[[LLMConfig], LLM]):
        if key in _REGISTRY:
            raise ValueError(f"llm backend {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_llms() -> List[str]:
    return sorted(_REGISTRY)


def build_llm(config: LLMConfig) -> LLM:
    from . import stub as _stub  # noqa: F401  (registration)

    if config.backend in ("huggingface", "hf"):
        from . import hf as _hf  # noqa: F401
    elif config.backend == "vllm":
        from . import vllm_backend as _vllm  # noqa: F401
    elif config.backend == "openai":
        from . import openai_backend as _openai  # noqa: F401

    if config.backend not in _REGISTRY:
        raise KeyError(f"unknown llm backend {config.backend!r}; available: {available_llms()}")
    return _REGISTRY[config.backend](config)
