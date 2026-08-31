"""Driver that turns cached candidates into filter training data.

For each (question, snippet) pair the base LLM is run three ways:

1. **closed book** -- answer the question with no evidence  -> correct w/o retrieval,
   and the rationale that both perplexity terms score;
2. **with the snippet** -- answer the question given that one snippet -> correct w/ retrieval;
3. **teacher-forced scoring** of the rationale under both prompts -> Delta-PPL.

Those three results are exactly the inputs Figure 2's decision tree consumes.
The tree, the tau threshold and the training-file schema live in
``rag2.filter_training.labeling``.

Cost note: this is one closed-book generation per question plus one generation
and two scoring passes per (question, snippet) pair. On the paper's setup --
MedQA + MedMCQA training splits at ~10 snippets each -- that is the dominant
compute of the whole reproduction. ``filter_training.label_top_k`` bounds it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..evaluation import extract_choice, is_correct
from ..llm.base import LLM
from ..prompts import DEFAULT_PROMPTS, PromptSet
from ..schema import CandidateSet, Evidence, Question
from .labeling import LabelingObservation, LabeledPair, label_observations
from .perplexity import compute_perplexity_pair


def _answer_correct(
    llm: LLM,
    question: Question,
    evidences: Sequence[Evidence],
    prompts: PromptSet,
) -> Tuple[bool, str, Optional[str]]:
    prompt = prompts.render_answer_prompt(question, evidences)
    generation = llm.generate([prompt])[0]
    prediction = extract_choice(generation, question.options)
    correct = bool(is_correct(prediction, question.answer))
    return correct, generation, prediction


def build_observations(
    llm: LLM,
    questions: Sequence[Question],
    candidate_sets: Mapping[str, CandidateSet],
    prompts: Optional[PromptSet] = None,
    top_k: int = 10,
    ppl_target: str = "rationale",
    ppl_rationale: str = "no_retrieval",
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[LabelingObservation], Dict[str, Any]]:
    """Run the base LLM over every (question, snippet) pair.

    Returns the observations plus per-question diagnostics (the closed-book
    generation and correctness), which the caller writes to a sidecar so the
    labels are auditable without re-running the LLM.
    """
    prompts = prompts or DEFAULT_PROMPTS
    if ppl_rationale != "no_retrieval":
        raise ValueError(
            f"unsupported ppl_rationale {ppl_rationale!r}: the reproduction scores the "
            "closed-book rationale in both terms so Delta-PPL isolates the document's "
            "effect on one fixed string (docs/rag2_reproduction.md section 5.4)"
        )

    observations: List[LabelingObservation] = []
    diagnostics: Dict[str, Any] = {}

    for position, question in enumerate(questions):
        candidate_set = candidate_sets.get(question.qid)
        if candidate_set is None:
            continue
        if question.answer is None:
            raise ValueError(
                f"question {question.qid} has no gold answer; filter labels need the "
                "training split's labels (paper section 4.2)"
            )

        # (1) closed book -- correctness and the rationale both terms score.
        correct_without, rationale, prediction_without = _answer_correct(llm, question, [], prompts)
        diagnostics[question.qid] = {
            "closed_book_generation": rationale,
            "closed_book_prediction": prediction_without,
            "correct_without_retrieval": correct_without,
            "gold": question.answer,
            "snippets": [],
        }

        for index, evidence in enumerate(candidate_set.top(top_k)):
            # (2) with this one snippet -- the paper labels snippets individually.
            correct_with, generation_with, prediction_with = _answer_correct(
                llm, question, [evidence], prompts
            )
            # (3) Delta-PPL of the same rationale, with and without the snippet.
            ppl = compute_perplexity_pair(
                llm, question, rationale, evidence, prompts=prompts, target=ppl_target
            )
            observations.append(
                LabelingObservation(
                    qid=question.qid,
                    snippet_index=index,
                    correct_without_retrieval=correct_without,
                    correct_with_retrieval=correct_with,
                    delta_ppl=ppl.delta,
                    evidence=evidence,
                    question=question,
                    ppl=ppl,
                )
            )
            diagnostics[question.qid]["snippets"].append(
                {
                    "snippet_index": index,
                    "source": evidence.source,
                    "doc_id": evidence.doc_id,
                    "prediction_with_retrieval": prediction_with,
                    "correct_with_retrieval": correct_with,
                    **ppl.to_dict(),
                }
            )
        if progress:
            progress(position + 1, len(questions))

    return observations, diagnostics


def build_training_data(
    llm: LLM,
    questions: Sequence[Question],
    candidate_sets: Mapping[str, CandidateSet],
    dataset_name: str,
    prompts: Optional[PromptSet] = None,
    top_k: int = 10,
    tau_percentile: float = 25.0,
    tau_scope: str = "global",
    ppl_target: str = "rationale",
    ppl_rationale: str = "no_retrieval",
    drop_undecided: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[LabeledPair], Dict[str, Any]]:
    """End-to-end: observations -> tau -> Figure 2 labels."""
    observations, diagnostics = build_observations(
        llm,
        questions,
        candidate_sets,
        prompts=prompts,
        top_k=top_k,
        ppl_target=ppl_target,
        ppl_rationale=ppl_rationale,
        progress=progress,
    )
    pairs, stats = label_observations(
        observations,
        dataset_name=dataset_name,
        prompts=prompts,
        top_percent=tau_percentile,
        tau_scope=tau_scope,
        drop_undecided=drop_undecided,
    )
    stats["diagnostics"] = diagnostics
    return pairs, stats
