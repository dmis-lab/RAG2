"""Deterministic, dependency-free LLM used by unit tests and the smoke test.

It is *not* a model: it produces reproducible pseudo-text and pseudo-scores from
a hash of its input so the full pipeline -- rationale generation, perplexity
labeling, answer generation, evaluation -- can be exercised end to end on a
machine with no GPU and no torch. It must never be used for a real experiment;
``describe()`` marks it so, and the run manifest records that marker.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

from ..config import LLMConfig
from .base import LLM, ScoredSequence, register_llm


def _rand(seed: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{seed}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


class StubLLM(LLM):
    is_stub = True

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig(backend="stub", model="stub")
        self.name = self.config.model or "stub"
        self.revision = "n/a"
        self.answers: Dict[str, str] = dict(self.config.options.get("answers", {})) if config else {}
        self.generate_calls: List[str] = []
        self.score_calls: List[tuple] = []

    def generate(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        out: List[str] = []
        for prompt in prompts:
            self.generate_calls.append(prompt)
            letter = self._choice_for(prompt)
            out.append(
                "Summarizing the available information, the presentation is consistent "
                "with the described condition. Therefore, the answer is "
                f"({letter})."
            )
        return out

    def _choice_for(self, prompt: str) -> str:
        for key, letter in self.answers.items():
            if key and key in prompt:
                return letter
        return "ABCD"[int(_rand(prompt, "choice") * 4) % 4]

    def score(self, prompt: str, continuation: str) -> ScoredSequence:
        self.score_calls.append((prompt, continuation))
        tokens = continuation.split() or [""]
        base = 0.5 + 2.0 * _rand(prompt, "base")
        logprobs = [
            -(base + 0.5 * _rand(prompt, f"tok{i}")) for i in range(len(tokens))
        ]
        return ScoredSequence(token_logprobs=logprobs, num_tokens=len(tokens))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self.revision,
            "class": "StubLLM",
            "WARNING": "stub backend - not a real model, results are meaningless",
        }


@register_llm("stub")
def _build_stub(config: LLMConfig) -> LLM:
    return StubLLM(config)
