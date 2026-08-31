"""Prompt templates for the original RAG2 system.

Every template here is a versioned constant so that a prompt edit shows up in the
run manifest as a changed hash (see rag2/experiment.py). Provenance of each
template is recorded inline and in docs/rag2_reproduction.md section 3.

Legend:
  [S] specified by the paper / released repository, reproduced verbatim
  [A] not specified; reconstructed by documented assumption
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .schema import Evidence, Question, format_question, stable_hash

# --------------------------------------------------------------------------
# [S] Rationale generation. Paper section 3.3, quoted verbatim (following Kim
#     et al., 2024). The paper prints this prompt in full.
# --------------------------------------------------------------------------
RATIONALE_PROMPT = (
    "The following are multiple choice questions about medical knowledge. Solve them in "
    "a step-by-step fashion, starting by summarizing the available information. Output "
    "your explanation and single option from the given options as the final answer.\n\n"
    "Here is the question: {question}"
)

# --------------------------------------------------------------------------
# [S] Filter input. Not printed in the paper, but recovered verbatim from the
#     released training artifact classifier/data/medqa/llama3_cot/5%-train.json,
#     where every record uses exactly this wording and these separators.
#     Note it carries the INITIAL question, not the rationale (Figure 1).
# --------------------------------------------------------------------------
FILTER_PROMPT = (
    "Given the following evidence, determine whether it helps answer the provided question."
    "\n\nEvidence: {evidence}"
    "\n\nQuestion: {question}"
)

LABEL_HELPFUL = "[HELPFUL]"
LABEL_NOT_HELPFUL = "[NOT_HELPFUL]"
FILTER_LABELS = (LABEL_HELPFUL, LABEL_NOT_HELPFUL)

# --------------------------------------------------------------------------
# [A] Answer generation. NOT published by the paper or the repository. Figure 1
#     shows only the structure: prompt = snippets + initial query. This
#     reconstruction reuses the paper's own chain-of-thought prompt with an
#     evidence block prepended, because the paper states the same LLM does both
#     rationale generation and QA (section 3.3) and Meerkat was instruction-tuned
#     on that prompt. See docs/rag2_reproduction.md section 3.3 -- this is the
#     largest prompt-level assumption in the reproduction.
# --------------------------------------------------------------------------
ANSWER_PROMPT_WITH_EVIDENCE = (
    "The following are multiple choice questions about medical knowledge. Solve them in "
    "a step-by-step fashion, starting by summarizing the available information. Output "
    "your explanation and single option from the given options as the final answer.\n\n"
    "Here are the retrieved documents: {evidence}\n\n"
    "Here is the question: {question}"
)

# [A] Closed-book fallback: identical to the rationale prompt, which is what the
#     paper's "no RAG" rows use (Table 2, 0-shot).
ANSWER_PROMPT_NO_EVIDENCE = RATIONALE_PROMPT

# [A] How a kept snippet is rendered inside the evidence block. The paper does
#     not specify; a rank prefix is used so the ordering is legible in logs.
EVIDENCE_ITEM_TEMPLATE = "[{rank}] {text}"
EVIDENCE_JOIN = "\n"

PROMPT_VERSION = "rag2-original-v1"


@dataclass(frozen=True)
class PromptSet:
    """The full set of templates used by one run, hashable for the manifest."""

    rationale: str = RATIONALE_PROMPT
    filter_input: str = FILTER_PROMPT
    answer_with_evidence: str = ANSWER_PROMPT_WITH_EVIDENCE
    answer_no_evidence: str = ANSWER_PROMPT_NO_EVIDENCE
    evidence_item: str = EVIDENCE_ITEM_TEMPLATE
    evidence_join: str = EVIDENCE_JOIN
    option_format: str = "{letter}) {text}"
    version: str = PROMPT_VERSION

    # -- rendering ---------------------------------------------------------
    def render_question(self, question: Question) -> str:
        return format_question(question, self.option_format)

    def render_rationale_prompt(self, question: Question) -> str:
        return self.rationale.format(question=self.render_question(question))

    def render_filter_prompt(self, question: Question, evidence: Evidence | str) -> str:
        """Build the Flan-T5 filter input for one (question, snippet) pair.

        Only ``Evidence.text`` is used: provenance metadata never reaches the
        filter (docs/rag2_reproduction.md section 4.1).
        """
        text = evidence.text if isinstance(evidence, Evidence) else str(evidence)
        return self.filter_input.format(
            evidence=text.strip(), question=self.render_question(question)
        )

    def render_evidence_block(self, evidences: Sequence[Evidence | str]) -> str:
        items: List[str] = []
        for rank, ev in enumerate(evidences, start=1):
            text = ev.text if isinstance(ev, Evidence) else str(ev)
            items.append(self.evidence_item.format(rank=rank, text=text.strip()))
        return self.evidence_join.join(items)

    def render_answer_prompt(
        self, question: Question, evidences: Sequence[Evidence | str]
    ) -> str:
        rendered_q = self.render_question(question)
        if not evidences:
            return self.answer_no_evidence.format(question=rendered_q)
        return self.answer_with_evidence.format(
            evidence=self.render_evidence_block(evidences), question=rendered_q
        )

    # -- provenance --------------------------------------------------------
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": self.version,
                "rationale": self.rationale,
                "filter_input": self.filter_input,
                "answer_with_evidence": self.answer_with_evidence,
                "answer_no_evidence": self.answer_no_evidence,
                "evidence_item": self.evidence_item,
                "evidence_join": self.evidence_join,
                "option_format": self.option_format,
            }
        )


DEFAULT_PROMPTS = PromptSet()
