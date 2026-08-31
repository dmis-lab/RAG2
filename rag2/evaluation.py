"""Evaluation: answer extraction, accuracy, filter metrics, open-ended metrics.

The paper reports **accuracy** on the three multiple-choice benchmarks (Table 2)
and, for the ClinicalQA25 side experiment, **ROUGE-L F1** and **BERTScore F1**
(appendix A.4.1, both defined there explicitly).

Answer extraction -- documented assumption
------------------------------------------
Neither the paper nor the release states how the final option is read out of a
free-form chain-of-thought generation. The prompt asks the model to "Output your
explanation and single option from the given options as the final answer", and
the paper's own worked example ends "Therefore, the answer is (C) Intubation"
(Figure 4). The patterns below are tried in order and the **last** match in the
generation wins, since the CoT prompt puts the answer at the end. A generation
that matches nothing counts as incorrect (``evaluation.unparsed_as_incorrect``),
never as an abstention. See docs/rag2_reproduction.md section 6.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL
from .schema import PipelineResult

#: Ordered extraction patterns. Each must expose the option letter as group 1.
DEFAULT_EXTRACTION_PATTERNS: List[str] = [
    r"(?:the\s+)?answer\s+is\s*:?\s*\(?\s*([A-Za-z])\s*[\)\.:,]",
    r"(?:the\s+)?answer\s+is\s*:?\s*\(?\s*([A-Za-z])\s*$",
    r"\banswer\s*:\s*\(?\s*([A-Za-z])\b",
    r"\boption\s*\(?\s*([A-Za-z])\s*\)?",
    r"^\s*\(?\s*([A-Za-z])\s*[\)\.]\s",
    r"\(\s*([A-Za-z])\s*\)",
]

_LETTER_ONLY = re.compile(r"\b([A-Za-z])\b")


def extract_choice(
    text: str,
    options: Optional[Mapping[str, str]] = None,
    patterns: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Pull the selected option letter out of a generation.

    Patterns are tried in order; within a pattern the **last** match wins. If no
    pattern matches, falls back to the last standalone letter that is a valid
    option, then to matching an option's *text* verbatim.
    """
    if not text:
        return None
    valid = {k.upper() for k in options} if options else None
    body = text.strip()

    for pattern in (patterns if patterns is not None else DEFAULT_EXTRACTION_PATTERNS):
        matches = list(re.finditer(pattern, body, flags=re.IGNORECASE | re.MULTILINE))
        for match in reversed(matches):
            letter = match.group(1).upper()
            if valid is None or letter in valid:
                return letter

    for match in reversed(list(_LETTER_ONLY.finditer(body))):
        letter = match.group(1).upper()
        if valid and letter in valid:
            return letter

    if options:
        lowered = body.lower()
        hits = [
            (lowered.rfind(str(option_text).strip().lower()), letter)
            for letter, option_text in options.items()
            if str(option_text).strip() and str(option_text).strip().lower() in lowered
        ]
        if hits:
            return max(hits)[1]
    return None


def is_correct(prediction: Optional[str], gold: Optional[str], unparsed_as_incorrect: bool = True) -> Optional[bool]:
    """Compare an extracted choice with the gold letter."""
    if gold is None:
        return None
    if prediction is None:
        return False if unparsed_as_incorrect else None
    return prediction.upper() == gold.upper()


def accuracy(results: Sequence[PipelineResult]) -> Dict[str, Any]:
    """Overall accuracy in percent, as Table 2 reports it."""
    scored = [r for r in results if r.correct is not None]
    correct = sum(1 for r in scored if r.correct)
    unparsed = sum(1 for r in results if r.prediction is None)
    return {
        "num_examples": len(results),
        "num_scored": len(scored),
        "num_correct": correct,
        "accuracy": (100.0 * correct / len(scored)) if scored else 0.0,
        "num_unparsed": unparsed,
    }


def accuracy_by(results: Sequence[PipelineResult], key: str) -> Dict[str, Dict[str, Any]]:
    """Accuracy broken down by a metadata key (e.g. MMLU-Med ``subject``)."""
    groups: Dict[str, List[PipelineResult]] = {}
    for result in results:
        groups.setdefault(str(result.metadata.get(key, "unknown")), []).append(result)
    return {name: accuracy(items) for name, items in sorted(groups.items())}


def evidence_report(results: Sequence[PipelineResult]) -> Dict[str, Any]:
    """How the filter behaved: kept counts and per-corpus distribution.

    Diagnostics only -- nothing here feeds back into the pipeline.
    """
    kept_counts = [len(r.kept) for r in results]
    sources: Counter = Counter()
    for result in results:
        for evidence in result.kept:
            sources[evidence.source or "unknown"] += 1
    candidates = sum(len(r.candidates) for r in results)
    kept = sum(kept_counts)
    return {
        "num_candidates_total": candidates,
        "num_kept_total": kept,
        "keep_rate": (kept / candidates) if candidates else 0.0,
        "mean_kept_per_question": (kept / len(results)) if results else 0.0,
        "questions_with_no_evidence": sum(1 for c in kept_counts if c == 0),
        "kept_by_source": dict(sources.most_common()),
    }


# ---------------------------------------------------------------------------
# Filter-level metrics -- reproduces classifier/utils.py
# ---------------------------------------------------------------------------
def filter_metrics(gold: Sequence[str], predictions: Sequence[str]) -> Dict[str, Any]:
    """Overall + per-class accuracy over ``[HELPFUL]`` / ``[NOT_HELPFUL]``.

    Same quantities ``classifier/utils.py`` writes to ``final_eval_results.json``
    and ``final_eval_results_perClass.json``.
    """
    if len(gold) != len(predictions):
        raise ValueError(f"gold/prediction length mismatch: {len(gold)} vs {len(predictions)}")
    total = len(gold)
    correct = sum(1 for g, p in zip(gold, predictions) if g == p)
    out: Dict[str, Any] = {"final_acc_score": (100.0 * correct / total) if total else 0.0}
    per_class: Dict[str, Any] = {}
    for label in (LABEL_HELPFUL, LABEL_NOT_HELPFUL):
        gold_num = sum(1 for g in gold if g == label)
        pred_num = sum(1 for p in predictions if p == label)
        hits = sum(1 for g, p in zip(gold, predictions) if g == p == label)
        per_class[f"{label} acc"] = (100.0 * hits / gold_num) if gold_num else -1
        per_class[f"{label} pred num"] = pred_num
        per_class[f"{label} gold num"] = gold_num
    out["per_class"] = per_class
    return out


# ---------------------------------------------------------------------------
# Open-ended metrics -- paper appendix A.4.1 (ClinicalQA25)
# ---------------------------------------------------------------------------
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            current.append(previous[j] + 1 if token_a == token_b else max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """ROUGE-L precision / recall / F1 over whitespace tokens (appendix A.4.1)."""
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    lcs = _lcs_length(cand_tokens, ref_tokens)
    precision = lcs / len(cand_tokens) if cand_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def bertscore(
    candidates: Sequence[str], references: Sequence[str], model: str = "roberta-large"
) -> Dict[str, float]:
    """BERTScore P/R/F1 via the ``bert-score`` package (appendix A.4.1).

    The paper does not state which backbone it scored with; ``roberta-large`` is
    the package default and is recorded in the run manifest.
    """
    from bert_score import score as _score  # lazy: optional dependency

    precision, recall, f1 = _score(list(candidates), list(references), model_type=model, lang="en")
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "model": model,
    }


def open_ended_metrics(
    candidates: Sequence[str],
    references: Sequence[str],
    metrics: Sequence[str] = ("rouge_l",),
    bertscore_model: str = "roberta-large",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "rouge_l" in metrics:
        scores = [rouge_l(c, r) for c, r in zip(candidates, references)]
        out["rouge_l"] = {
            key: sum(s[key] for s in scores) / len(scores) if scores else 0.0
            for key in ("precision", "recall", "f1")
        }
    if "bertscore" in metrics:
        out["bertscore"] = bertscore(candidates, references, model=bertscore_model)
    return out
