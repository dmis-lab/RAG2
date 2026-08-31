"""Filters that keep everything -- the paper's "RAG2 w/o filter" ablation.

Figure 3 plots this as its own curve, so it is a first-class configuration, not
a debugging convenience: ``filter.kind: passthrough``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..config import FilterConfig
from ..prompts import LABEL_HELPFUL, PromptSet
from ..schema import Evidence, FilterDecision, Question
from .base import EvidenceFilter, register_filter


class PassthroughFilter(EvidenceFilter):
    """Keep every candidate ("RAG2 w/o filter")."""

    name = "passthrough"

    def decide(self, question: Question, candidates: Sequence[Evidence]) -> List[FilterDecision]:
        return [FilterDecision(keep=True, label=LABEL_HELPFUL) for _ in candidates]


class NoEvidenceFilter(EvidenceFilter):
    """Drop every candidate -- the closed-book "no RAG" row of Table 2."""

    name = "no_evidence"

    def decide(self, question: Question, candidates: Sequence[Evidence]) -> List[FilterDecision]:
        from ..prompts import LABEL_NOT_HELPFUL

        return [FilterDecision(keep=False, label=LABEL_NOT_HELPFUL) for _ in candidates]


@register_filter("passthrough")
def _build_passthrough(config: FilterConfig, prompts: Optional[PromptSet] = None) -> EvidenceFilter:
    return PassthroughFilter()


@register_filter("no_evidence")
def _build_no_evidence(config: FilterConfig, prompts: Optional[PromptSet] = None) -> EvidenceFilter:
    return NoEvidenceFilter()
