"""Perplexity-based label construction for the RAG2 filter (paper section 3.2, Figure 2).

The released ``classifier/data/preprocess.py`` is an **empty file**, in this
repository and upstream alike, so this module reconstructs the annotation
procedure from Figure 2 and section 3.2. The decision tree is transcribed
exactly:

    correct w/o retrieval?
      |-- yes -> correct w/ retrieval?
      |            |-- yes -> Delta-PPL >= tau ? [HELPFUL] : DISCARD
      |            '-- no  -> [NOT_HELPFUL]
      '-- no  -> correct w/ retrieval?
                   |-- yes -> [HELPFUL]
                   '-- no  -> Delta-PPL >= tau ? [NOT_HELPFUL] : DISCARD

Read plainly: a snippet that raises confidence in a correct answer is helpful;
one that raises confidence in a wrong answer is harmful; one that flips
correctness settles the label by itself; one that changes neither correctness nor
confidence is dropped rather than labelled.

``tau`` is the top-25% quantile of the Delta-PPL distribution (section 3.2:
"setting the threshold value tau to the top 25% of perplexity differentials
consistently yielded the best performance and was therefore fixed across all our
experiments"). Whether that quantile is taken globally or per question is not
stated; ``tau_scope`` selects, defaulting to global. See
docs/rag2_reproduction.md section 5.5.

Labels are produced per (question, snippet) pair, each snippet evaluated
individually -- stated in the paper's Limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL, DEFAULT_PROMPTS, PromptSet
from ..schema import Evidence, Question
from .perplexity import PerplexityPair, top_percent_threshold

DISCARD = "[DISCARD]"


@dataclass
class LabelingObservation:
    """Everything Figure 2's decision tree consumes for one (question, snippet)."""

    qid: str
    snippet_index: int
    correct_without_retrieval: bool
    correct_with_retrieval: bool
    delta_ppl: float
    evidence: Optional[Evidence] = None
    question: Optional[Question] = None
    ppl: Optional[PerplexityPair] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def decide_label(
    correct_without_retrieval: bool,
    correct_with_retrieval: bool,
    lower_perplexity: bool,
) -> str:
    """Figure 2, transcribed. Returns ``[HELPFUL]``, ``[NOT_HELPFUL]`` or ``[DISCARD]``.

    ``lower_perplexity`` is the outcome of the ``Delta-PPL >= tau`` test.
    """
    if correct_without_retrieval:
        if correct_with_retrieval:
            return LABEL_HELPFUL if lower_perplexity else DISCARD
        return LABEL_NOT_HELPFUL
    if correct_with_retrieval:
        return LABEL_HELPFUL
    return LABEL_NOT_HELPFUL if lower_perplexity else DISCARD


def compute_tau(
    observations: Sequence[LabelingObservation],
    top_percent: float = 25.0,
    scope: str = "global",
) -> Dict[str, float]:
    """Threshold(s) for the ``Delta-PPL >= tau`` test.

    Returns ``{"__global__": tau}`` for global scope, or ``{qid: tau}`` per
    question. Global is the default: the paper describes tau as a single fixed
    value, and per-question quantiles over ~10 snippets are very noisy.
    """
    if scope == "global":
        return {"__global__": top_percent_threshold([o.delta_ppl for o in observations], top_percent)}
    if scope == "per_question":
        grouped: Dict[str, List[float]] = {}
        for observation in observations:
            grouped.setdefault(observation.qid, []).append(observation.delta_ppl)
        return {qid: top_percent_threshold(vals, top_percent) for qid, vals in grouped.items()}
    raise ValueError(f"unknown tau_scope {scope!r}; expected 'global' or 'per_question'")


def tau_for(taus: Mapping[str, float], qid: str) -> float:
    return taus.get(qid, taus.get("__global__", float("nan")))


@dataclass
class LabeledPair:
    """One training record for the Flan-T5 filter."""

    id: str
    question: str  # the rendered filter input (prompt + evidence + question)
    answer: str  # [HELPFUL] or [NOT_HELPFUL]
    dataset_name: str
    # Provenance, kept out of the model input. Written to a sidecar, not the
    # training file, so the training file matches the release schema exactly.
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_training_record(self) -> Dict[str, str]:
        """The four-field schema of classifier/data/medqa/llama3_cot/5%-train.json."""
        return {
            "id": self.id,
            "answer": self.answer,
            "dataset_name": self.dataset_name,
            "question": self.question,
        }


def label_observations(
    observations: Sequence[LabelingObservation],
    dataset_name: str,
    prompts: Optional[PromptSet] = None,
    top_percent: float = 25.0,
    tau_scope: str = "global",
    drop_undecided: bool = True,
    id_prefix: str = "",
) -> Tuple[List[LabeledPair], Dict[str, Any]]:
    """Apply the Figure 2 tree to every observation.

    Returns the labelled pairs and a stats dict (tau, label counts, discard rate)
    that the caller writes into the run manifest.
    """
    prompts = prompts or DEFAULT_PROMPTS
    taus = compute_tau(observations, top_percent=top_percent, scope=tau_scope)

    pairs: List[LabeledPair] = []
    counts = {LABEL_HELPFUL: 0, LABEL_NOT_HELPFUL: 0, DISCARD: 0}

    for index, observation in enumerate(observations):
        tau = tau_for(taus, observation.qid)
        lower_perplexity = observation.delta_ppl >= tau  # Equation 3 uses >=
        label = decide_label(
            observation.correct_without_retrieval,
            observation.correct_with_retrieval,
            lower_perplexity,
        )
        counts[label] = counts.get(label, 0) + 1
        if label == DISCARD and drop_undecided:
            continue
        if observation.question is None or observation.evidence is None:
            raise ValueError(
                "labeling needs the Question and Evidence to render the filter input"
            )
        prefix = id_prefix or dataset_name
        pairs.append(
            LabeledPair(
                id=f"{prefix}_{index}",
                question=prompts.render_filter_prompt(observation.question, observation.evidence),
                answer=label,
                dataset_name=dataset_name,
                provenance={
                    "qid": observation.qid,
                    "snippet_index": observation.snippet_index,
                    "source": observation.evidence.source,
                    "doc_id": observation.evidence.doc_id,
                    "passage_id": observation.evidence.passage_id,
                    "delta_ppl": observation.delta_ppl,
                    "tau": tau,
                    "correct_without_retrieval": observation.correct_without_retrieval,
                    "correct_with_retrieval": observation.correct_with_retrieval,
                    **({"ppl": observation.ppl.to_dict()} if observation.ppl else {}),
                },
            )
        )

    total = len(observations)
    stats = {
        "num_observations": total,
        "tau_scope": tau_scope,
        "tau_percentile": top_percent,
        "tau": taus if tau_scope == "per_question" else taus.get("__global__"),
        "label_counts": counts,
        "num_labeled": len(pairs),
        "discard_rate": (counts.get(DISCARD, 0) / total) if total else 0.0,
    }
    return pairs, stats
