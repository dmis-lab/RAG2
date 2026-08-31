"""OpenAI backbone, for the paper's GPT-4o rows.

Generation only. The Chat Completions API does not return teacher-forced
log-probabilities for a supplied continuation, so perplexity labels cannot be
built with this backend -- which is consistent with the paper, whose filter is
trained on open-weight backbones.

The served model snapshot is not pinned by the paper ("the latest version"); the
snapshot actually served is read back from the response and recorded in the run
manifest.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from ..config import LLMConfig
from .base import LLM, ScoredSequence, register_llm


class OpenAIBackend(LLM):
    def __init__(self, config: LLMConfig) -> None:
        from openai import OpenAI  # type: ignore

        self.config = config
        self.name = config.model
        self.revision = config.revision or ""
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._served_model: Optional[str] = None

    def generate(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(kwargs.get("temperature", self.config.temperature)),
                max_tokens=int(kwargs.get("max_new_tokens", self.config.max_new_tokens)),
            )
            self._served_model = getattr(response, "model", self._served_model)
            outputs.append((response.choices[0].message.content or "").strip())
        return outputs

    def score(self, prompt: str, continuation: str) -> ScoredSequence:
        raise NotImplementedError(
            "the OpenAI API does not score a supplied continuation; build "
            "perplexity labels with an open-weight backbone"
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": self._served_model or self.revision or "unpinned",
            "class": "OpenAIBackend",
            "note": "paper does not pin a GPT-4o snapshot; served snapshot recorded above",
        }


@register_llm("openai")
def _build_openai(config: LLMConfig) -> LLM:
    return OpenAIBackend(config)
