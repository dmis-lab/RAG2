"""Stage orchestration for the original RAG2 pipeline.

Two entry points, deliberately separate so the expensive half runs once:

``run_retrieval``  stages 1-2: rationale generation, balanced retrieval,
                   reranking. Produces :class:`~rag2.schema.CandidateSet`\\ s,
                   which ``rag2.cache`` persists.
``run_filter_and_generate``  stages 3-4: filtering and answer generation, driven
                   from candidates (fresh or replayed from cache).

Splitting there is what lets the identical candidate evidence be replayed through
different filters -- the filter is the only thing that changes between such runs,
and the cache's fingerprint check enforces it.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

from .config import Config
from .corpora.base import Corpus, build_corpus
from .evaluation import extract_choice, is_correct
from .filtering.base import EvidenceFilter
from .generation import generate_answers
from .llm.base import LLM
from .prompts import PromptSet
from .rationale import generate_rationales, retrieval_query
from .retrieval.balanced import balanced_retrieve, corpus_distribution
from .retrieval.rerank import rerank_candidates
from .schema import CandidateSet, Evidence, PipelineResult, Question

ProgressFn = Optional[Callable[..., None]]


def build_corpora(config: Config) -> List[Corpus]:
    if not config.retrieval.corpora:
        raise ValueError(
            "retrieval.corpora is empty: the paper retrieves evenly from four corpora "
            "(PubMed, PMC, CPG, Textbook -- appendix Table A1). Configure them, or "
            "supply candidates from a cache."
        )
    return [build_corpus(corpus) for corpus in config.retrieval.corpora]


def run_retrieval(
    config: Config,
    questions: Sequence[Question],
    llm: Optional[LLM] = None,
    corpora: Optional[Sequence[Corpus]] = None,
    encoder: Optional[Any] = None,
    reranker: Optional[Any] = None,
    prompts: Optional[PromptSet] = None,
    rationales: Optional[Mapping[str, str]] = None,
    progress: ProgressFn = None,
) -> List[CandidateSet]:
    """Stages 1-2: rationale query formulation, balanced retrieval, reranking.

    ``encoder``/``reranker`` may be injected (the smoke test does), otherwise the
    MedCPT models named in the config are loaded.
    """
    prompts = prompts or config.prompt_set()
    retrieval_cfg = config.retrieval

    # -- stage 1: rationales as queries (paper section 3.3) ---------------
    if rationales is None:
        if llm is None:
            raise ValueError("run_retrieval needs an LLM (or precomputed rationales)")
        rationales = generate_rationales(
            llm, questions, prompts=prompts, batch_size=config.llm.batch_size, progress=progress
        )
    queries = [retrieval_query(rationales.get(q.qid, ""), q) for q in questions]

    # -- stage 2a: encode + balanced MIPS (paper section 3.4) -------------
    if encoder is None:
        from .retrieval.encoder import build_query_encoder

        encoder = build_query_encoder(retrieval_cfg)
    embeddings = encoder.encode(queries)

    corpora = list(corpora) if corpora is not None else build_corpora(config)
    pooled = balanced_retrieve(corpora, embeddings, retrieval_cfg)

    # -- stage 2b: cross-encoder rerank -----------------------------------
    if reranker is None:
        from .retrieval.rerank import build_reranker

        reranker = build_reranker(retrieval_cfg)

    out: List[CandidateSet] = []
    for index, question in enumerate(questions):
        rationale = rationales.get(question.qid, "")
        # Paper (Figure 1 caption, section 3.4) cross-encodes the INITIAL query;
        # retriever/main.py passes the rationale. retrieval.rerank_query selects.
        if retrieval_cfg.rerank_query == "initial":
            rerank_q = prompts.render_question(question)
        elif retrieval_cfg.rerank_query == "rationale":
            rerank_q = rationale or prompts.render_question(question)
        else:
            raise ValueError(
                f"unknown retrieval.rerank_query {retrieval_cfg.rerank_query!r}; "
                "expected 'initial' or 'rationale'"
            )
        ranked = rerank_candidates(reranker, rerank_q, pooled[index], top_k=None)
        out.append(
            CandidateSet(
                qid=question.qid,
                rationale=rationale,
                candidates=ranked,
                retrieval_query=queries[index],
                rerank_query=rerank_q,
                metadata={
                    "pooled_size": len(pooled[index]),
                    "pool_by_source": corpus_distribution(pooled[index]),
                },
            )
        )
    return out


def run_filter_and_generate(
    config: Config,
    questions: Sequence[Question],
    candidate_sets: Mapping[str, CandidateSet],
    evidence_filter: EvidenceFilter,
    llm: LLM,
    prompts: Optional[PromptSet] = None,
    progress: ProgressFn = None,
) -> List[PipelineResult]:
    """Stages 3-4: rationale-guided filtering, then answer generation.

    Only the top ``retrieval.final_top_k`` candidates are filtered, matching the
    paper's top-k sweep (Figure 3): the cache may hold a deeper pool, and ``k``
    selects from it without re-retrieving.
    """
    prompts = prompts or config.prompt_set()
    top_k = config.retrieval.final_top_k

    results: List[PipelineResult] = []
    evidence_for_generation: List[List[Evidence]] = []

    for question in questions:
        candidate_set = candidate_sets.get(question.qid)
        candidates = candidate_set.top(top_k) if candidate_set else []
        kept, decisions = evidence_filter.apply(question, candidates)

        # The paper does not say what happens when everything is filtered out;
        # docs/rag2_reproduction.md section 5.6 records the choice.
        fallback = None
        if not kept and candidates:
            if config.filter.on_empty == "keep_top1":
                kept = [candidates[0]]
                fallback = "keep_top1"
            elif config.filter.on_empty == "no_evidence":
                fallback = "no_evidence"
            else:
                raise ValueError(
                    f"unknown filter.on_empty {config.filter.on_empty!r}; "
                    "expected 'no_evidence' or 'keep_top1'"
                )

        results.append(
            PipelineResult(
                qid=question.qid,
                rationale=candidate_set.rationale if candidate_set else "",
                candidates=list(candidates),
                decisions=decisions,
                kept=list(kept),
                gold=question.answer,
                metadata={
                    "num_candidates": len(candidates),
                    "num_kept": len(kept),
                    "final_top_k": top_k,
                    "empty_fallback": fallback,
                    "kept_by_source": corpus_distribution(kept),
                    **{k: v for k, v in question.metadata.items() if k == "subject"},
                },
            )
        )
        evidence_for_generation.append(list(kept))

    generations = generate_answers(
        llm,
        questions,
        evidence_for_generation,
        config=config.generation,
        prompts=prompts,
        batch_size=config.llm.batch_size,
        progress=progress,
    )

    patterns = config.evaluation.extraction_patterns or None
    for result, question, generation in zip(results, questions, generations):
        result.generation = generation
        result.prediction = extract_choice(generation, question.options, patterns)
        result.correct = is_correct(
            result.prediction, question.answer, config.evaluation.unparsed_as_incorrect
        )
    return results
