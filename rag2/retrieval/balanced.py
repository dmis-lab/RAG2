"""Balanced multi-corpus retrieval (paper section 3.4).

"This approach extracts an equal number of documents from each corpus, ensuring
that all corpora are represented more evenly compared to existing methods."

So: search each corpus independently for ``candidates_per_corpus`` snippets,
then pool. The pool is what the reranker sees; the balance is a property of the
*pool*, not of the final top-k (which the reranker is free to draw entirely from
one corpus).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..config import RetrievalConfig
from ..corpora.base import Corpus
from ..schema import Evidence
from .index import search_corpus


def balanced_retrieve(
    corpora: Sequence[Corpus],
    query_embeddings: Any,
    config: RetrievalConfig,
    backend: str = "auto",
    progress: Optional[callable] = None,
) -> List[List[Evidence]]:
    """Return, per query, the pooled candidates from every corpus.

    Candidates are decoded to :class:`Evidence`, carrying source, document id,
    passage id and any corpus metadata (publication information included).
    """
    n_queries = int(query_embeddings.shape[0])
    pooled: List[List[Evidence]] = [[] for _ in range(n_queries)]

    for corpus in corpora:
        scores, indices = search_corpus(
            corpus,
            query_embeddings,
            top_k=config.candidates_per_corpus,
            backend=backend,
            shard_merge=config.shard_merge,
        )
        for q in range(n_queries):
            for score, index in zip(scores[q].tolist(), indices[q].tolist()):
                evidence = corpus.passage(int(index))
                evidence.retrieval_score = float(score)
                pooled[q].append(evidence)
        if progress:
            progress(corpus.name, len(corpus))

    return pooled


def corpus_distribution(evidences: Sequence[Evidence]) -> Dict[str, int]:
    """How many kept snippets came from each corpus.

    Reported per run so the balance claim of section 3.4 is auditable; it feeds
    nothing back into the pipeline.
    """
    counts: Dict[str, int] = {}
    for evidence in evidences:
        counts[evidence.source or "unknown"] = counts.get(evidence.source or "unknown", 0) + 1
    return dict(sorted(counts.items()))
