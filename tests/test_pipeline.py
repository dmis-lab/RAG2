"""End-to-end wiring, driven by the stub backend."""

import pytest

from rag2.cache import index_by_qid, load_candidates, save_candidates
from rag2.config import Config, DatasetConfig, RetrievalConfig
from rag2.corpora.base import InMemoryCorpus
from rag2.filtering.passthrough import NoEvidenceFilter, PassthroughFilter
from rag2.filtering.rag2_filter import ScriptedFilter
from rag2.llm.stub import StubLLM
from rag2.pipeline import build_corpora, run_filter_and_generate, run_retrieval
from rag2.rationale import generate_rationales, retrieval_query
from rag2.retrieval.rerank import IdentityReranker
from rag2.schema import CandidateSet, Evidence, Question

numpy = pytest.importorskip("numpy")
DIM = 8


class StubEncoder:
    def encode(self, queries):
        rng = numpy.random.default_rng(len(queries))
        return rng.normal(size=(len(queries), DIM)).astype("float32")


def _questions(n=3):
    return [
        Question(f"q{i}", f"Vignette {i}?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "ABCD"[i % 4])
        for i in range(n)
    ]


def _corpora(per=3):
    rng = numpy.random.default_rng(0)
    return [
        InMemoryCorpus(
            name,
            [{"id": f"{name}-{j}", "text": f"{name} snippet {j}", "publication_date": f"200{j}-01-01"}
             for j in range(per)],
            rng.normal(size=(per, DIM)).astype("float32"),
        )
        for name in ("pubmed", "pmc", "cpg", "textbook")
    ]


def _config(final_top_k=4):
    config = Config()
    config.experiment.name = "unit"
    config.dataset = DatasetConfig(loader="inline", name="unit")
    config.retrieval = RetrievalConfig(
        embedding_dim=DIM, candidates_per_corpus=2, final_top_k=final_top_k
    )
    config.llm.backend = "stub"
    config.filter.kind = "scripted"
    return config


def test_rationale_is_the_retrieval_query_not_the_question():
    """Paper 3.3: search 'solely using the rationale, excluding the initial query'."""
    question = _questions(1)[0]
    assert retrieval_query("a generated rationale", question) == "a generated rationale"
    # ...and the question is the fallback when no rationale exists.
    assert retrieval_query("", question) == question.question


def test_generate_rationales_covers_every_question():
    questions = _questions(5)
    rationales = generate_rationales(StubLLM(), questions, batch_size=2)
    assert set(rationales) == {q.qid for q in questions}
    assert all(rationales.values())


def test_run_retrieval_produces_a_balanced_pool_and_ranks_it():
    config = _config()
    questions = _questions()
    sets = run_retrieval(
        config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
        reranker=IdentityReranker(),
    )
    assert [s.qid for s in sets] == [q.qid for q in questions]
    for candidate_set in sets:
        assert len(candidate_set.candidates) == 8  # 2 per corpus x 4 corpora
        assert candidate_set.metadata["pool_by_source"] == {
            "cpg": 2, "pmc": 2, "pubmed": 2, "textbook": 2
        }
        assert [e.rank for e in candidate_set.candidates] == list(range(1, 9))


def test_rerank_query_choice_is_recorded_and_honoured():
    config, questions = _config(), _questions(1)
    initial = run_retrieval(
        config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
        reranker=IdentityReranker(),
    )
    assert "Vignette 0?" in initial[0].rerank_query

    config.retrieval.rerank_query = "rationale"
    rationale_run = run_retrieval(
        config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
        reranker=IdentityReranker(),
    )
    assert rationale_run[0].rerank_query == rationale_run[0].rationale


def test_unknown_rerank_query_is_rejected():
    config = _config()
    config.retrieval.rerank_query = "both"
    with pytest.raises(ValueError, match="unknown retrieval.rerank_query"):
        run_retrieval(
            config, _questions(1), llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
            reranker=IdentityReranker(),
        )


def test_final_top_k_selects_from_a_deeper_cache_without_re_retrieving():
    """The k sweep of Figure 3 must not need a new retrieval pass."""
    config, questions = _config(), _questions(1)
    sets = run_retrieval(
        config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
        reranker=IdentityReranker(),
    )
    cached = index_by_qid(sets)
    llm = StubLLM()

    for k in (1, 2, 4, 8):
        config.retrieval.final_top_k = k
        results = run_filter_and_generate(config, questions, cached, PassthroughFilter(), llm)
        assert len(results[0].candidates) == k
        assert len(results[0].kept) == k


def test_cache_replay_gives_the_same_filter_input_as_a_live_run(tmp_path):
    """The guarantee the thesis depends on: identical candidates, whatever filter."""
    config, questions = _config(), _questions(2)
    sets = run_retrieval(
        config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
        reranker=IdentityReranker(),
    )
    path = str(tmp_path / "c.jsonl")
    fingerprint = config.retrieval_fingerprint()
    save_candidates(path, sets, fingerprint)
    replayed = index_by_qid(load_candidates(path, fingerprint))

    def texts(mapping):
        return {qid: [e.text for e in cs.candidates] for qid, cs in mapping.items()}

    assert texts(replayed) == texts(index_by_qid(sets))

    seen_live, seen_replay = [], []
    run_filter_and_generate(
        config, questions, index_by_qid(sets),
        ScriptedFilter(lambda r, q, e: seen_live.append(r) or 1.0), StubLLM(),
    )
    run_filter_and_generate(
        config, questions, replayed,
        ScriptedFilter(lambda r, q, e: seen_replay.append(r) or 1.0), StubLLM(),
    )
    assert seen_live == seen_replay


def test_two_filters_see_identical_candidates():
    config, questions = _config(), _questions(2)
    sets = index_by_qid(
        run_retrieval(
            config, questions, llm=StubLLM(), corpora=_corpora(), encoder=StubEncoder(),
            reranker=IdentityReranker(),
        )
    )
    a = run_filter_and_generate(config, questions, sets, PassthroughFilter(), StubLLM())
    b = run_filter_and_generate(
        config, questions, sets, ScriptedFilter(lambda r, q, e: 0.0), StubLLM()
    )
    assert [[e.text for e in r.candidates] for r in a] == [[e.text for e in r.candidates] for r in b]
    assert all(len(r.kept) == len(r.candidates) for r in a)
    assert all(r.kept == [] for r in b)


def test_empty_filter_output_falls_back_to_closed_book():
    config, questions = _config(), _questions(2)
    sets = {q.qid: CandidateSet(qid=q.qid, candidates=[Evidence(text="s", source="cpg", rank=1)]) for q in questions}
    results = run_filter_and_generate(config, questions, sets, NoEvidenceFilter(), StubLLM())
    assert all(r.kept == [] for r in results)
    assert all(r.metadata["empty_fallback"] == "no_evidence" for r in results)
    assert all(r.generation for r in results)


def test_keep_top1_fallback():
    config, questions = _config(), _questions(1)
    config.filter.on_empty = "keep_top1"
    sets = {
        questions[0].qid: CandidateSet(
            qid=questions[0].qid,
            candidates=[Evidence(text=f"s{i}", source="cpg", rank=i + 1) for i in range(3)],
        )
    }
    results = run_filter_and_generate(config, questions, sets, NoEvidenceFilter(), StubLLM())
    assert [e.text for e in results[0].kept] == ["s0"]
    assert results[0].metadata["empty_fallback"] == "keep_top1"


def test_unknown_on_empty_is_rejected():
    config, questions = _config(), _questions(1)
    config.filter.on_empty = "retry"
    sets = {questions[0].qid: CandidateSet(qid=questions[0].qid, candidates=[Evidence(text="s")])}
    with pytest.raises(ValueError, match="unknown filter.on_empty"):
        run_filter_and_generate(config, questions, sets, NoEvidenceFilter(), StubLLM())


def test_questions_without_candidates_still_produce_a_result():
    config, questions = _config(), _questions(2)
    results = run_filter_and_generate(config, questions, {}, PassthroughFilter(), StubLLM())
    assert len(results) == 2
    assert all(r.candidates == [] and r.generation for r in results)


def test_predictions_and_correctness_are_filled_in():
    config, questions = _config(), _questions(4)
    sets = {q.qid: CandidateSet(qid=q.qid, candidates=[Evidence(text="s", source="cpg")]) for q in questions}
    results = run_filter_and_generate(config, questions, sets, PassthroughFilter(), StubLLM())
    for result, question in zip(results, questions):
        assert result.prediction in set("ABCD")
        assert result.gold == question.answer
        assert result.correct == (result.prediction == question.answer)


def test_build_corpora_requires_a_configuration():
    config = _config()
    config.retrieval.corpora = []
    with pytest.raises(ValueError, match="retrieval.corpora is empty"):
        build_corpora(config)


def test_results_serialise_and_round_trip():
    config, questions = _config(), _questions(1)
    sets = {questions[0].qid: CandidateSet(
        qid=questions[0].qid,
        candidates=[Evidence(text="s", source="cpg", doc_id="d1", metadata={"publication_date": "1999"})],
    )}
    results = run_filter_and_generate(config, questions, sets, PassthroughFilter(), StubLLM())
    payload = results[0].to_dict()
    assert payload["kept"][0]["metadata"]["publication_date"] == "1999"
    assert set(payload) >= {"qid", "candidates", "decisions", "kept", "generation", "prediction", "gold", "correct"}
