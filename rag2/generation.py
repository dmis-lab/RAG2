"""Stage 4: answer generation.

The backbone LLM answers the initial query conditioned on the snippets the
filter kept (Figure 1). Greedy decoding at temperature 0 (appendix A.3).

The prompt itself is a **documented assumption** -- the paper and the release
never publish it; see rag2/prompts.py and docs/rag2_reproduction.md section 3.3.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from .config import GenerationConfig
from .llm.base import LLM
from .prompts import DEFAULT_PROMPTS, PromptSet
from .schema import Evidence, Question


def build_answer_prompts(
    questions: Sequence[Question],
    evidence_per_question: Sequence[Sequence[Evidence]],
    prompts: Optional[PromptSet] = None,
) -> List[str]:
    prompts = prompts or DEFAULT_PROMPTS
    if len(questions) != len(evidence_per_question):
        raise ValueError("questions and evidence lists must be the same length")
    return [
        prompts.render_answer_prompt(question, list(evidences))
        for question, evidences in zip(questions, evidence_per_question)
    ]


def generate_answers(
    llm: LLM,
    questions: Sequence[Question],
    evidence_per_question: Sequence[Sequence[Evidence]],
    config: Optional[GenerationConfig] = None,
    prompts: Optional[PromptSet] = None,
    batch_size: int = 8,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Return one generation per question."""
    config = config or GenerationConfig()
    rendered = build_answer_prompts(questions, evidence_per_question, prompts)
    outputs: List[str] = []
    step = max(batch_size, 1)
    for start in range(0, len(rendered), step):
        chunk = rendered[start : start + step]
        outputs.extend(
            llm.generate(
                chunk,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
        )
        if progress:
            progress(min(start + step, len(rendered)), len(rendered))
    return [o.strip() for o in outputs]
