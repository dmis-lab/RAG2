"""Stage 1: rationale-based query formulation (paper section 3.3).

The backbone LLM is prompted with the paper's chain-of-thought prompt and the
resulting rationale becomes the retrieval query. The paper is explicit that the
initial query is *not* concatenated: "We search for document snippets solely
using the rationale, excluding the initial query."
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .llm.base import LLM
from .prompts import DEFAULT_PROMPTS, PromptSet
from .schema import Question


def generate_rationales(
    llm: LLM,
    questions: Sequence[Question],
    prompts: Optional[PromptSet] = None,
    batch_size: int = 8,
    progress: Optional[callable] = None,
) -> Dict[str, str]:
    """Return ``{qid: rationale}`` for every question."""
    prompts = prompts or DEFAULT_PROMPTS
    rendered = [prompts.render_rationale_prompt(q) for q in questions]
    out: Dict[str, str] = {}
    for start in range(0, len(rendered), max(batch_size, 1)):
        chunk = rendered[start : start + max(batch_size, 1)]
        completions = llm.generate(chunk)
        for question, completion in zip(questions[start : start + len(chunk)], completions):
            out[question.qid] = completion.strip()
        if progress:
            progress(min(start + len(chunk), len(rendered)), len(rendered))
    return out


def retrieval_query(rationale: str, question: Question, use_rationale: bool = True) -> str:
    """The string handed to the retriever.

    ``use_rationale=False`` reproduces the paper's ``MedCPT`` baseline row, which
    retrieves with the initial query instead.
    """
    if use_rationale and rationale.strip():
        return rationale.strip()
    return question.question.strip()
