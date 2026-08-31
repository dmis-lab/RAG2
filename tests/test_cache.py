"""Candidate caching: the mechanism that lets one candidate set be replayed."""

import json
import os

import pytest

from rag2.cache import (
    CandidateCacheError,
    default_cache_path,
    index_by_qid,
    iter_candidates,
    load_candidates,
    read_metadata,
    save_candidates,
)
from rag2.config import load_config
from rag2.schema import CandidateSet, Evidence


def _candidate_sets(n=3, per=4):
    return [
        CandidateSet(
            qid=f"q{i}",
            rationale=f"rationale {i}",
            retrieval_query=f"rationale {i}",
            rerank_query=f"question {i}",
            candidates=[
                Evidence(
                    text=f"snippet {i}-{j}",
                    source=["pubmed", "pmc", "cpg", "textbook"][j % 4],
                    doc_id=f"doc-{i}-{j}",
                    passage_id=f"doc-{i}-{j}-p0",
                    retrieval_score=1.0 / (j + 1),
                    rerank_score=float(per - j),
                    rank=j + 1,
                    metadata={"publication_date": f"20{10 + j}-01-01", "journal": "J"},
                )
                for j in range(per)
            ],
        )
        for i in range(n)
    ]


def test_round_trip_preserves_every_field(tmp_path):
    path = str(tmp_path / "candidates.jsonl")
    original = _candidate_sets()
    save_candidates(path, original, "fp-abc")
    restored = load_candidates(path, expected_fingerprint="fp-abc")

    assert len(restored) == len(original)
    for before, after in zip(original, restored):
        assert after.qid == before.qid
        assert after.rationale == before.rationale
        assert after.retrieval_query == before.retrieval_query
        assert after.rerank_query == before.rerank_query
        assert len(after.candidates) == len(before.candidates)
        for e1, e2 in zip(before.candidates, after.candidates):
            assert (e2.text, e2.source, e2.doc_id, e2.passage_id, e2.rank) == (
                e1.text, e1.source, e1.doc_id, e1.passage_id, e1.rank
            )
            assert e2.retrieval_score == pytest.approx(e1.retrieval_score)
            assert e2.rerank_score == pytest.approx(e1.rerank_score)


def test_publication_metadata_survives_the_cache(tmp_path):
    """Provenance is carried for later thesis use; it must not be dropped here."""
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(1, 2), "fp")
    restored = load_candidates(path, "fp")
    assert restored[0].candidates[0].metadata["publication_date"] == "2010-01-01"
    assert restored[0].candidates[0].metadata["journal"] == "J"


def test_metadata_sidecar_records_the_fingerprint(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(3, 4), "fp-xyz", extra={"corpora": ["cpg"]})
    metadata = read_metadata(path)
    assert metadata.retrieval_fingerprint == "fp-xyz"
    assert metadata.num_questions == 3
    assert metadata.candidates_per_question == 4
    assert metadata.extra["corpora"] == ["cpg"]
    assert metadata.created_at


def test_mismatched_fingerprint_is_refused(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(), "fp-one")
    with pytest.raises(CandidateCacheError, match="fp-one"):
        load_candidates(path, expected_fingerprint="fp-two")


def test_mismatch_can_be_overridden_explicitly(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(), "fp-one")
    assert len(load_candidates(path, "fp-two", allow_config_mismatch=True)) == 3


def test_cache_without_a_sidecar_is_refused(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(), "fp")
    os.remove(str(tmp_path / "c.meta.json"))
    with pytest.raises(CandidateCacheError, match="no .meta.json"):
        load_candidates(path, expected_fingerprint="fp")
    assert len(load_candidates(path, "fp", allow_config_mismatch=True)) == 3


def test_missing_cache_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_candidates(str(tmp_path / "nope.jsonl"), "fp")


def test_streaming_matches_eager_loading(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(5, 2), "fp")
    assert [c.qid for c in iter_candidates(path)] == [c.qid for c in load_candidates(path, "fp")]


def test_index_by_qid():
    sets = _candidate_sets(3)
    assert set(index_by_qid(sets)) == {"q0", "q1", "q2"}


def test_top_k_selects_from_a_deeper_cache():
    """A cache built at depth 32 must serve a k=8 experiment without re-retrieving."""
    candidate_set = _candidate_sets(1, 32)[0]
    assert len(candidate_set.top(8)) == 8
    assert len(candidate_set.top(None)) == 32
    assert [e.rank for e in candidate_set.top(3)] == [1, 2, 3]


def test_retrieval_fingerprint_is_insensitive_to_final_top_k():
    """Changing k must not invalidate the cache; changing the corpus must."""
    base = load_config(None, {"retrieval": {"final_top_k": 8}})
    same = load_config(None, {"retrieval": {"final_top_k": 32}})
    assert base.retrieval_fingerprint() == same.retrieval_fingerprint()

    different = load_config(None, {"retrieval": {"candidates_per_corpus": 50}})
    assert different.retrieval_fingerprint() != base.retrieval_fingerprint()


def test_retrieval_fingerprint_tracks_the_documented_discrepancy_switches():
    base = load_config(None, {})
    for override in ({"rerank_query": "rationale"}, {"shard_merge": "concat"}):
        changed = load_config(None, {"retrieval": override})
        assert changed.retrieval_fingerprint() != base.retrieval_fingerprint(), override


def test_default_cache_path_embeds_the_fingerprint():
    path = default_cache_path("cache", "medqa-llama3", "test", "abc123")
    assert path.endswith("medqa-llama3.test.abc123.jsonl")


def test_cache_file_is_valid_jsonl(tmp_path):
    path = str(tmp_path / "c.jsonl")
    save_candidates(path, _candidate_sets(3), "fp")
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 3
    assert set(rows[0]) == {"qid", "rationale", "retrieval_query", "rerank_query", "candidates", "metadata"}
