"""Balanced retrieval, sharding and reranking."""

import pytest

numpy = pytest.importorskip("numpy")

from rag2.config import RetrievalConfig
from rag2.corpora.base import Corpus, InMemoryCorpus, decode_passage
from rag2.retrieval.balanced import balanced_retrieve, corpus_distribution
from rag2.retrieval.index import ShardSearcher, search_corpus
from rag2.retrieval.rerank import IdentityReranker, rerank_candidates
from rag2.schema import Evidence

DIM = 8


def _corpus(name, n=10, seed=0):
    rng = numpy.random.default_rng(seed)
    passages = [
        {"id": f"{name}-{i}", "text": f"{name} passage {i}", "publication_date": f"20{10 + i:02d}-01-01"}
        for i in range(n)
    ]
    return InMemoryCorpus(name, passages, rng.normal(size=(n, DIM)).astype("float32"))


class ShardedCorpus(Corpus):
    """Same vectors as ``inner`` but exposed as several shards."""

    def __init__(self, inner, num_shards):
        self.name = inner.name
        self.inner = inner
        self.matrix = list(inner.embedding_shards())[0][1]
        self.num_shards = num_shards

    def __len__(self):
        return len(self.inner)

    def passage(self, index):
        return self.inner.passage(index)

    def embedding_shards(self):
        size = len(self.inner) // self.num_shards
        for s in range(self.num_shards):
            start = s * size
            stop = len(self.inner) if s == self.num_shards - 1 else start + size
            yield start, self.matrix[start:stop]


def _queries(n=3, seed=99):
    return numpy.random.default_rng(seed).normal(size=(n, DIM)).astype("float32")


def test_search_returns_exact_inner_product_top_k():
    corpus = _corpus("cpg")
    queries = _queries(2)
    scores, indices = search_corpus(corpus, queries, top_k=4)

    matrix = list(corpus.embedding_shards())[0][1]
    for q in range(queries.shape[0]):
        expected = numpy.argsort(-(matrix @ queries[q]))[:4]
        assert list(indices[q]) == list(expected)
        assert scores[q] == pytest.approx(matrix[expected] @ queries[q], rel=1e-5)


def test_scores_are_descending():
    scores, _ = search_corpus(_corpus("cpg"), _queries(3), top_k=5)
    for row in scores:
        assert list(row) == sorted(row, reverse=True)


def test_top_k_larger_than_the_corpus_is_clamped():
    _, indices = search_corpus(_corpus("cpg", n=3), _queries(1), top_k=10)
    assert indices.shape[1] == 3


def test_shard_merge_by_score_equals_a_single_index():
    """Sharding is a memory workaround; it must not change the result."""
    corpus = _corpus("pubmed", n=12, seed=3)
    queries = _queries(4)
    single_scores, single_indices = search_corpus(corpus, queries, top_k=5)
    sharded_scores, sharded_indices = search_corpus(
        ShardedCorpus(corpus, 4), queries, top_k=5, shard_merge="score"
    )
    assert (single_indices == sharded_indices).all()
    assert sharded_scores == pytest.approx(single_scores, rel=1e-5)


def test_shard_merge_concat_reproduces_the_release_imbalance():
    """retriever/main.py concatenates per-shard top-k, over-representing PubMed."""
    corpus = _corpus("pubmed", n=12, seed=3)
    _, indices = search_corpus(ShardedCorpus(corpus, 4), _queries(2), top_k=5, shard_merge="concat")
    # Four shards of 3 rows each, so each contributes min(5, 3) = 3 candidates.
    assert indices.shape[1] == 12


def test_balanced_retrieval_takes_the_same_count_from_every_corpus():
    """Paper section 3.4: 'an equal number of documents from each corpus'."""
    corpora = [_corpus(name, seed=i) for i, name in enumerate(("pubmed", "pmc", "cpg", "textbook"))]
    config = RetrievalConfig(embedding_dim=DIM, candidates_per_corpus=3)
    pooled = balanced_retrieve(corpora, _queries(2), config)

    assert len(pooled) == 2
    for candidates in pooled:
        assert len(candidates) == 12
        assert corpus_distribution(candidates) == {"cpg": 3, "pmc": 3, "pubmed": 3, "textbook": 3}


def test_balanced_retrieval_attaches_scores_and_provenance():
    corpora = [_corpus("cpg", seed=1)]
    pooled = balanced_retrieve(corpora, _queries(1), RetrievalConfig(embedding_dim=DIM, candidates_per_corpus=2))
    evidence = pooled[0][0]
    assert evidence.source == "cpg"
    assert evidence.doc_id.startswith("cpg-")
    assert evidence.retrieval_score is not None
    assert "publication_date" in evidence.metadata


def test_rerank_sorts_descending_and_truncates():
    candidates = [Evidence(text=f"s{i}", source="cpg") for i in range(5)]

    class Scorer:
        def score(self, query, snippets):
            return [0.0, 3.0, 1.0, 4.0, 2.0]

    ranked = rerank_candidates(Scorer(), "q", candidates, top_k=3)
    assert [e.text for e in ranked] == ["s3", "s1", "s4"]
    assert [e.rank for e in ranked] == [1, 2, 3]
    assert [e.rerank_score for e in ranked] == [4.0, 3.0, 2.0]


def test_rerank_can_reorder_across_corpora():
    """Balance is a property of the pool, not of the final top-k."""
    candidates = [
        Evidence(text="pubmed a", source="pubmed"),
        Evidence(text="cpg a", source="cpg"),
        Evidence(text="cpg b", source="cpg"),
    ]

    class Scorer:
        def score(self, query, snippets):
            return [0.1, 9.0, 8.0]

    ranked = rerank_candidates(Scorer(), "q", candidates, top_k=2)
    assert corpus_distribution(ranked) == {"cpg": 2}


def test_rerank_handles_no_candidates():
    assert rerank_candidates(IdentityReranker(), "q", []) == []


def test_identity_reranker_preserves_pool_order():
    candidates = [Evidence(text=f"s{i}") for i in range(4)]
    assert [e.text for e in rerank_candidates(IdentityReranker(), "q", candidates)] == [
        "s0", "s1", "s2", "s3"
    ]


def test_faiss_backend_is_refused_when_faiss_is_absent():
    from rag2.retrieval.index import faiss_available

    if faiss_available():
        pytest.skip("faiss is installed")
    with pytest.raises(RuntimeError, match="faiss is not installed"):
        ShardSearcher(numpy.zeros((2, DIM), dtype="float32"), 0, backend="faiss")


def test_decode_passage_accepts_bare_strings():
    """retriever/retrieve.py assumes article entries are plain strings."""
    evidence = decode_passage("just text", "cpg", 7)
    assert evidence.text == "just text"
    assert evidence.source == "cpg"
    assert evidence.corpus_index == 7


def test_decode_passage_prefixes_the_title_and_keeps_the_rest_as_metadata():
    evidence = decode_passage(
        {"id": "d1", "title": "A Title", "content": "body text", "year": 2004, "journal": "J"},
        "pmc",
        3,
    )
    assert evidence.text == "A Title. body text"
    assert evidence.doc_id == "d1"
    assert evidence.metadata == {"year": 2004, "journal": "J"}
