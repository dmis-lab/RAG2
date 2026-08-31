"""Core data structures for the RAG2 reproduction.

These types are deliberately dependency-free (stdlib only) so that the
orchestration, caching, labeling and evaluation logic can be imported and tested
without torch/faiss/transformers present.

Provenance policy (see docs/rag2_reproduction.md section 4.1): ``Evidence``
carries document/passage identifiers, the source corpus and free-form metadata
such as publication information whenever the corpus supplies them. **No baseline
component reads any of it.** Only ``Evidence.text`` reaches the reranker, the
filter or the generator. This is asserted by tests/test_metadata_isolation.py.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional

OPTION_LETTERS = ("A", "B", "C", "D")


def _clean(value: Any) -> Any:
    """Make a value JSON-round-trippable, leaving plain containers alone."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return str(value)


@dataclass
class Question:
    """A single multiple-choice QA item.

    ``options`` maps an option letter to its text. The paper's three benchmarks
    are all four-option (paper section 4.1), but nothing here hard-codes four.
    """

    qid: str
    question: str
    options: Dict[str, str]
    answer: Optional[str] = None
    dataset: str = ""
    split: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.options = {str(k).strip().upper(): str(v) for k, v in self.options.items()}
        if self.answer is not None:
            self.answer = str(self.answer).strip().upper()

    @property
    def option_letters(self) -> List[str]:
        return sorted(self.options)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Question":
        return cls(
            qid=str(payload["qid"]),
            question=str(payload["question"]),
            options=dict(payload["options"]),
            answer=payload.get("answer"),
            dataset=payload.get("dataset", ""),
            split=payload.get("split", ""),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class Evidence:
    """One retrieved snippet plus its provenance.

    Attributes
    ----------
    text:
        The snippet text. **The only field any baseline component reads.**
    source:
        Corpus identifier the snippet came from (``pubmed``, ``pmc``, ``cpg``,
        ``textbook``, ...). Used for balanced retrieval bookkeeping and for
        reporting; never fed to a model.
    doc_id / passage_id:
        Stable identifiers within the source corpus, preserved so a snippet can
        be traced back to its document.
    metadata:
        Free-form provenance, e.g. ``{"publication_date": "2019-04-01",
        "journal": ...}``. Preserved verbatim; **not read by the baseline**.
    """

    text: str
    source: str = ""
    doc_id: Optional[str] = None
    passage_id: Optional[str] = None
    corpus_index: Optional[int] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Evidence":
        if isinstance(payload, str):  # tolerate the release's bare-string format
            return cls(text=payload)
        return cls(
            text=str(payload.get("text", "")),
            source=payload.get("source", ""),
            doc_id=payload.get("doc_id"),
            passage_id=payload.get("passage_id"),
            corpus_index=payload.get("corpus_index"),
            retrieval_score=payload.get("retrieval_score"),
            rerank_score=payload.get("rerank_score"),
            rank=payload.get("rank"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class CandidateSet:
    """Retrieved + reranked candidates for one question.

    This is the unit that gets cached (see rag2/cache.py) so that the same
    candidate evidence can be replayed through different filters.
    """

    qid: str
    rationale: str = ""
    candidates: List[Evidence] = field(default_factory=list)
    retrieval_query: str = ""
    rerank_query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def top(self, k: Optional[int]) -> List[Evidence]:
        return list(self.candidates) if k is None else list(self.candidates[:k])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "rationale": self.rationale,
            "retrieval_query": self.retrieval_query,
            "rerank_query": self.rerank_query,
            "candidates": [c.to_dict() for c in self.candidates],
            "metadata": _clean(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateSet":
        return cls(
            qid=str(payload["qid"]),
            rationale=payload.get("rationale", ""),
            candidates=[Evidence.from_dict(c) for c in payload.get("candidates", [])],
            retrieval_query=payload.get("retrieval_query", ""),
            rerank_query=payload.get("rerank_query", ""),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class FilterDecision:
    """Per-snippet output of an EvidenceFilter."""

    keep: bool
    label: str
    score: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PipelineResult:
    """Everything one question produced, end to end."""

    qid: str
    rationale: str = ""
    candidates: List[Evidence] = field(default_factory=list)
    decisions: List[FilterDecision] = field(default_factory=list)
    kept: List[Evidence] = field(default_factory=list)
    generation: str = ""
    prediction: Optional[str] = None
    gold: Optional[str] = None
    correct: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "rationale": self.rationale,
            "candidates": [c.to_dict() for c in self.candidates],
            "decisions": [d.to_dict() for d in self.decisions],
            "kept": [c.to_dict() for c in self.kept],
            "generation": self.generation,
            "prediction": self.prediction,
            "gold": self.gold,
            "correct": self.correct,
            "metadata": _clean(self.metadata),
        }


def stable_hash(payload: Any, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable payload."""
    blob = json.dumps(_clean(payload), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def format_options(options: Mapping[str, str], template: str = "{letter}) {text}") -> str:
    """Serialise answer options the way the released filter data does.

    The paper never states the option format; the released training artifact
    (classifier/data/medqa/llama3_cot/5%-train.json) uses ``A) ... B) ...``
    appended inline after the question stem, which is what this reproduces.
    See docs/rag2_reproduction.md section 3.1.
    """
    return " ".join(
        template.format(letter=letter, text=options[letter]) for letter in sorted(options)
    )


def format_question(question: Question, template: str = "{letter}) {text}") -> str:
    """Question stem with its options appended inline."""
    options = format_options(question.options, template)
    stem = question.question.strip()
    return f"{stem} {options}".strip() if options else stem

