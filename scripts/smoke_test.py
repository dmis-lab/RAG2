#!/usr/bin/env python
"""End-to-end smoke test on a tiny synthetic corpus.

Exercises every stage of the original RAG2 pipeline -- rationale generation,
balanced multi-corpus retrieval, reranking, candidate caching and replay,
perplexity labeling, filtering, answer generation, evaluation -- using the stub
LLM and an exact numpy MIPS index, so it runs in seconds with no GPU, no model
downloads, and no corpus.

It checks that the parts *fit together*, not that the system is accurate: the
stub LLM's outputs are hashes, so its accuracy is meaningless by construction.
A tiny synthetic corpus is built here only because the real medical dataset is
not ready -- it is not a substitute dataset and nothing else uses it.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rag2.cache import index_by_qid, load_candidates, save_candidates
from rag2.config import Config, DatasetConfig, RetrievalConfig
from rag2.corpora.base import InMemoryCorpus
from rag2.datasets.base import InMemoryDataset
from rag2.evaluation import accuracy, evidence_report
from rag2.filter_training.build_labels import build_training_data
from rag2.filtering.rag2_filter import ScriptedFilter
from rag2.llm.stub import StubLLM
from rag2.pipeline import run_filter_and_generate, run_retrieval
from rag2.retrieval.rerank import IdentityReranker
from rag2.schema import Question

CORPORA = ("pubmed", "pmc", "cpg", "textbook")


def build_fixtures(num_questions: int = 6, per_corpus: int = 8, dim: int = 16):
    """A synthetic dataset + four synthetic corpora with aligned embeddings."""
    import numpy as np

    rng = np.random.default_rng(0)
    questions = [
        Question(
            qid=f"q{i}",
            question=f"Synthetic clinical vignette {i}. Which is the best next step?",
            options={"A": f"option a{i}", "B": f"option b{i}", "C": f"option c{i}", "D": f"option d{i}"},
            answer="ABCD"[i % 4],
            dataset="smoke",
            split="test",
            metadata={"subject": f"subject{i % 2}"},
        )
        for i in range(num_questions)
    ]

    corpora = []
    for name in CORPORA:
        passages = [
            {
                "id": f"{name}-doc{j}",
                "passage_id": f"{name}-doc{j}-p0",
                "text": f"[{name}] Synthetic evidence passage {j} about clinical management.",
                # Provenance the baseline must never read.
                "publication_date": f"{2000 + j}-01-01",
                "journal": f"Journal of {name}",
            }
            for j in range(per_corpus)
        ]
        embeddings = rng.normal(size=(per_corpus, dim)).astype("float32")
        corpora.append(InMemoryCorpus(name, passages, embeddings))
    return questions, corpora, dim


class StubEncoder:
    """Deterministic query encoder standing in for MedCPT."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, queries):
        import numpy as np

        rng = np.random.default_rng(abs(hash(tuple(queries))) % (2 ** 32))
        return rng.normal(size=(len(queries), self.dim)).astype("float32")


def keyword_filter_score(rendered_prompt: str, question, evidence) -> float:
    """Deterministic stand-in for the trained Flan-T5 filter.

    Keeps snippets from the two smaller corpora. Arbitrary but fixed -- the point
    is to exercise the EvidenceFilter contract, not to imitate the real filter.
    """
    return 1.0 if evidence.source in ("cpg", "textbook") else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="keep the temporary run directory")
    args = parser.parse_args()

    workdir = tempfile.mkdtemp(prefix="rag2-smoke-")
    failures = []

    def check(label, condition, detail=""):
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {label}{(' -- ' + detail) if detail else ''}")
        if not condition:
            failures.append(label)

    try:
        questions, corpora, dim = build_fixtures()
        dataset = InMemoryDataset(questions, name="smoke", version="synthetic-v1")

        config = Config()
        config.experiment.name = "smoke"
        config.experiment.output_dir = os.path.join(workdir, "run")
        config.dataset = DatasetConfig(loader="inline", name="smoke", version="synthetic-v1", split="test")
        config.retrieval = RetrievalConfig(
            embedding_dim=dim,
            candidates_per_corpus=3,
            # Filter the whole pool, so every corpus reaches the filter and the
            # "kept" checks below are not vacuous.
            final_top_k=12,
            rerank_query="initial",
        )
        config.llm.backend = "stub"
        config.llm.model = "stub"
        config.filter.kind = "scripted"
        llm = StubLLM(config.llm)

        print("stage 1-2: rationale generation, balanced retrieval, reranking")
        candidate_sets = run_retrieval(
            config,
            dataset.questions(),
            llm=llm,
            corpora=corpora,
            encoder=StubEncoder(dim),
            reranker=IdentityReranker(),
        )
        check("one candidate set per question", len(candidate_sets) == len(questions))
        check(
            "balanced pool: 3 candidates from each of 4 corpora",
            all(len(cs.candidates) == 12 for cs in candidate_sets),
            f"first={len(candidate_sets[0].candidates)}",
        )
        sources = {c.source for cs in candidate_sets for c in cs.candidates}
        check("every corpus represented", sources == set(CORPORA), str(sorted(sources)))
        check("rationales generated", all(cs.rationale for cs in candidate_sets))
        check("ranks assigned by the reranker", candidate_sets[0].candidates[0].rank == 1)

        print("stage 2b: cache write and replay")
        cache_path = os.path.join(workdir, "candidates.jsonl")
        fingerprint = config.retrieval_fingerprint()
        save_candidates(cache_path, candidate_sets, fingerprint)
        replayed = load_candidates(cache_path, expected_fingerprint=fingerprint)
        check("cache round-trips every question", len(replayed) == len(candidate_sets))
        original = candidate_sets[0].candidates[0]
        restored = replayed[0].candidates[0]
        check(
            "provenance survives the cache",
            (restored.doc_id, restored.source, restored.metadata.get("publication_date"))
            == (original.doc_id, original.source, original.metadata.get("publication_date")),
            f"{restored.doc_id} / {restored.metadata.get('publication_date')}",
        )

        print("stage 3a: perplexity labeling (Figure 2)")
        pairs, stats = build_training_data(
            llm,
            dataset.questions(),
            index_by_qid(replayed),
            dataset_name="smoke_stub",
            top_k=4,
            tau_percentile=config.filter_training.tau_percentile,
            tau_scope=config.filter_training.tau_scope,
        )
        check("labels produced", len(pairs) > 0, f"{len(pairs)} of {stats['num_observations']} observations")
        check(
            "labels are the two filter tokens",
            {p.answer for p in pairs} <= {"[HELPFUL]", "[NOT_HELPFUL]"},
        )
        check(
            "training records match the release schema",
            set(pairs[0].to_training_record()) == {"id", "answer", "dataset_name", "question"},
        )
        check(
            "filter input uses the released template",
            pairs[0].question.startswith(
                "Given the following evidence, determine whether it helps answer the provided question."
            ),
        )

        print("stage 3b-4: filtering and answer generation")
        evidence_filter = ScriptedFilter(keyword_filter_score, config.prompt_set())
        results = run_filter_and_generate(
            config, dataset.questions(), index_by_qid(replayed), evidence_filter, llm
        )
        check("one result per question", len(results) == len(questions))
        check(
            "filter kept the small-corpus snippets",
            all(len(r.kept) == 6 for r in results),
            f"first kept={len(results[0].kept)} of {len(results[0].candidates)}",
        )
        check(
            "filter dropped the large-corpus snippets",
            all(all(e.source in ("cpg", "textbook") for e in r.kept) for r in results),
        )
        check(
            "one decision per candidate",
            all(len(r.decisions) == len(r.candidates) for r in results),
        )
        check("answers generated", all(r.generation for r in results))
        check("predictions extracted", all(r.prediction in set("ABCD") for r in results))

        print("filter swap: the pipeline runs unchanged with a different filter")
        from rag2.filtering.passthrough import NoEvidenceFilter, PassthroughFilter

        passthrough = run_filter_and_generate(
            config, dataset.questions(), index_by_qid(replayed), PassthroughFilter(), llm
        )
        check(
            "passthrough keeps every candidate ('RAG2 w/o filter')",
            all(len(r.kept) == len(r.candidates) for r in passthrough),
        )
        empty = run_filter_and_generate(
            config, dataset.questions(), index_by_qid(replayed), NoEvidenceFilter(), llm
        )
        check(
            "all-filtered-out falls back to closed-book generation",
            all(r.kept == [] and r.metadata["empty_fallback"] == "no_evidence" for r in empty)
            and all(r.generation for r in empty),
        )
        config.filter.on_empty = "keep_top1"
        top1 = run_filter_and_generate(
            config, dataset.questions(), index_by_qid(replayed), NoEvidenceFilter(), llm
        )
        check(
            "filter.on_empty=keep_top1 restores the top candidate",
            all(len(r.kept) == 1 and r.kept[0].rank == 1 for r in top1),
        )
        config.filter.on_empty = "no_evidence"

        print("stage 5: evaluation")
        metrics = accuracy(results)
        report = evidence_report(results)
        check("accuracy computed", 0.0 <= metrics["accuracy"] <= 100.0, f"{metrics['accuracy']:.1f}%")
        check("evidence report computed", report["num_candidates_total"] > 0, str(report["kept_by_source"]))

        print("metadata isolation: publication dates never reach a model")
        prompts = config.prompt_set()
        rendered = "\n".join(
            [prompts.render_filter_prompt(questions[0], e) for e in results[0].kept]
            + [prompts.render_answer_prompt(questions[0], results[0].kept)]
        )
        dates = [
            e.metadata["publication_date"]
            for e in results[0].kept
            if e.metadata.get("publication_date")
        ]
        check(
            "no publication date appears in any model input",
            bool(dates) and not any(d in rendered for d in dates),
            f"{len(dates)} dates carried as metadata",
        )

        print()
        if failures:
            print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed: {failures}")
            return 1
        print("SMOKE TEST PASSED -- all stages wired correctly")
        print("(stub LLM: the accuracy number above is meaningless by construction)")
        return 0
    finally:
        if args.keep:
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
