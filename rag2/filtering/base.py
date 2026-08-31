"""The evidence-filter interface.

This is the **one** seam the later thesis work needs: an alternative filter
implements :class:`EvidenceFilter`, registers itself, and the rest of the
pipeline is untouched. Nothing outside this package directory knows how filtering
is done.

A filter sees, per question, the reranked candidate snippets and returns one
:class:`~rag2.schema.FilterDecision` per candidate. It must not mutate the
candidates.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import FilterConfig
from ..prompts import DEFAULT_PROMPTS, PromptSet
from ..schema import Evidence, FilterDecision, Question


class EvidenceFilter(abc.ABC):
    """Decide, per snippet, whether it reaches the answer generator."""

    name: str = ""

    @abc.abstractmethod
    def decide(
        self, question: Question, candidates: Sequence[Evidence]
    ) -> List[FilterDecision]:
        """Return one decision per candidate, in the same order."""

    def apply(
        self, question: Question, candidates: Sequence[Evidence]
    ) -> tuple[List[Evidence], List[FilterDecision]]:
        """Run :meth:`decide` and return ``(kept, decisions)``."""
        decisions = self.decide(question, candidates)
        if len(decisions) != len(candidates):
            raise ValueError(
                f"{type(self).__name__}.decide returned {len(decisions)} decisions "
                f"for {len(candidates)} candidates"
            )
        kept = [c for c, d in zip(candidates, decisions) if d.keep]
        return kept, decisions

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "class": type(self).__name__}


_REGISTRY: Dict[str, Callable[..., EvidenceFilter]] = {}


def register_filter(key: str):
    def decorator(factory: Callable[..., EvidenceFilter]):
        if key in _REGISTRY:
            raise ValueError(f"filter {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_filters() -> List[str]:
    return sorted(_REGISTRY)


def build_filter(config: FilterConfig, prompts: Optional[PromptSet] = None) -> EvidenceFilter:
    from . import passthrough as _passthrough  # noqa: F401  (registration)

    if config.kind == "rag2_perplexity":
        from . import rag2_filter as _rag2_filter  # noqa: F401

    if config.kind not in _REGISTRY:
        raise KeyError(f"unknown filter {config.kind!r}; available: {available_filters()}")
    return _REGISTRY[config.kind](config, prompts or DEFAULT_PROMPTS)
