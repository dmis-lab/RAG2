"""Persist and replay retrieved + reranked candidate evidence.

Retrieval over the paper's 564 GB corpus is by far the most expensive stage, and
comparing filters is only meaningful when every filter sees the *same* candidate
set. So candidates are written once and replayed thereafter:

    scripts/02_retrieve.py   --> cache/candidates/<name>.jsonl  (+ .meta.json)
    scripts/05_run_pipeline.py --candidates cache/.../<name>.jsonl

The sidecar records the retrieval fingerprint (``Config.retrieval_fingerprint``),
which covers the corpora, the encoder/reranker checkpoints, the candidate depth,
the rerank-query choice, the shard-merge policy, and the rationale LLM. Replaying
a cache whose fingerprint does not match the current config raises unless
``cache.allow_config_mismatch`` is set -- so "only the filter changed" is a
checked claim, not an assumption.

Evidence provenance (document id, passage id, source, publication information) is
written through verbatim.

The format is JSONL, one :class:`~rag2.schema.CandidateSet` per line, so a large
run streams rather than loading whole.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from .schema import CandidateSet

CACHE_FORMAT_VERSION = 1


def _meta_path(path: str) -> str:
    return f"{os.path.splitext(path)[0]}.meta.json"


@dataclass
class CacheMetadata:
    retrieval_fingerprint: str
    format_version: int = CACHE_FORMAT_VERSION
    created_at: str = ""
    num_questions: int = 0
    candidates_per_question: Optional[int] = None
    extra: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "created_at": self.created_at,
            "num_questions": self.num_questions,
            "candidates_per_question": self.candidates_per_question,
            "extra": self.extra or {},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CacheMetadata":
        return cls(
            retrieval_fingerprint=payload.get("retrieval_fingerprint", ""),
            format_version=int(payload.get("format_version", CACHE_FORMAT_VERSION)),
            created_at=payload.get("created_at", ""),
            num_questions=int(payload.get("num_questions", 0)),
            candidates_per_question=payload.get("candidates_per_question"),
            extra=dict(payload.get("extra", {})),
        )


class CandidateCacheError(RuntimeError):
    pass


def save_candidates(
    path: str,
    candidate_sets: Iterable[CandidateSet],
    retrieval_fingerprint: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> CacheMetadata:
    """Write candidates to ``path`` (JSONL) plus a ``.meta.json`` sidecar."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    count = 0
    widths: set = set()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        for candidate_set in candidate_sets:
            handle.write(json.dumps(candidate_set.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            widths.add(len(candidate_set.candidates))
    os.replace(tmp, path)

    metadata = CacheMetadata(
        retrieval_fingerprint=retrieval_fingerprint,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        num_questions=count,
        candidates_per_question=next(iter(widths)) if len(widths) == 1 else None,
        extra=dict(extra or {}),
    )
    with open(_meta_path(path), "w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2)
    return metadata


def read_metadata(path: str) -> Optional[CacheMetadata]:
    meta_path = _meta_path(path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as handle:
        return CacheMetadata.from_dict(json.load(handle))


def iter_candidates(path: str) -> Iterator[CandidateSet]:
    """Stream candidate sets from a cache file."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield CandidateSet.from_dict(json.loads(line))


def load_candidates(
    path: str,
    expected_fingerprint: Optional[str] = None,
    allow_config_mismatch: bool = False,
) -> List[CandidateSet]:
    """Load a cache, verifying it was built under the expected retrieval config.

    Raises :class:`CandidateCacheError` on a fingerprint mismatch unless
    ``allow_config_mismatch`` is set -- the guard that makes replaying the same
    candidates through a different filter an auditable claim.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"candidate cache not found: {path}")
    metadata = read_metadata(path)
    if expected_fingerprint and metadata and metadata.retrieval_fingerprint:
        if metadata.retrieval_fingerprint != expected_fingerprint and not allow_config_mismatch:
            raise CandidateCacheError(
                f"candidate cache {path} was built under retrieval fingerprint "
                f"{metadata.retrieval_fingerprint!r} but the current config resolves to "
                f"{expected_fingerprint!r}. Rebuild the cache, or pass "
                f"cache.allow_config_mismatch=true if the difference is intentional."
            )
    if expected_fingerprint and metadata is None and not allow_config_mismatch:
        raise CandidateCacheError(
            f"candidate cache {path} has no .meta.json sidecar, so it cannot be "
            f"verified against the current retrieval config; pass "
            f"cache.allow_config_mismatch=true to load it anyway"
        )
    return list(iter_candidates(path))


def index_by_qid(candidate_sets: Sequence[CandidateSet]) -> Dict[str, CandidateSet]:
    return {cs.qid: cs for cs in candidate_sets}


def default_cache_path(cache_dir: str, experiment: str, split: str, fingerprint: str) -> str:
    """Cache filename carrying the fingerprint, so two configs cannot collide."""
    stem = f"{experiment}.{split}.{fingerprint}".replace("/", "_")
    return os.path.join(cache_dir, f"{stem}.jsonl")
