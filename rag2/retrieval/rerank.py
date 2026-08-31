"""MedCPT cross-encoder reranking (paper section 3.4; retriever/rerank.py).

The pooled candidates are scored by cross-encoding the query with each snippet
and sorted by the resulting logit; the top ``final_top_k`` survive.

Two things to note, both recorded in docs/rag2_reproduction.md section 4.3:

* **Which query.** The paper says the *initial* query, twice (Figure 1 caption
  and section 3.4). The released ``retriever/main.py`` passes the rationale file.
  The default here follows the paper; ``retrieval.rerank_query: rationale``
  reproduces the code path.
* **No ``[SEP]`` insertion.** SciSpacy sentence splitting is applied when
  encoding for MIPS but not for the reranker (retriever/README.md).

Unlike the release, candidates are scored in batches, so a large
``candidates_per_corpus`` does not build one giant padded tensor.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..config import RetrievalConfig
from ..schema import Evidence


class MedCPTCrossEncoder:
    def __init__(self, config: RetrievalConfig) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.config = config
        device = config.device
        self.device = torch.device(
            device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.reranker)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.reranker)
        self.model.eval()
        self.model.to(self.device)

    def score(self, query: str, snippets: Sequence[str]) -> List[float]:
        import torch

        if not snippets:
            return []
        size = self.config.rerank_batch_size or 32
        out: List[float] = []
        for start in range(0, len(snippets), size):
            pairs = [[query, s] for s in snippets[start : start + size]]
            with torch.no_grad():
                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.config.rerank_max_length,
                ).to(self.device)
                logits = self.model(**encoded).logits.squeeze(dim=1)
            out.extend(float(v) for v in logits.detach().cpu().tolist())
        return out


def rerank_candidates(
    scorer: Any,
    query: str,
    candidates: Sequence[Evidence],
    top_k: Optional[int] = None,
) -> List[Evidence]:
    """Score, sort descending, truncate. ``scorer`` needs a ``score`` method.

    Ranks are 1-based and written onto the returned Evidence objects.
    """
    if not candidates:
        return []
    scores = scorer.score(query, [c.text for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    if top_k is not None:
        order = order[:top_k]
    out: List[Evidence] = []
    for rank, index in enumerate(order, start=1):
        evidence = candidates[index]
        evidence.rerank_score = float(scores[index])
        evidence.rank = rank
        out.append(evidence)
    return out


class IdentityReranker:
    """Passthrough scorer for tests and for ``retrieval.reranker: none``.

    Preserves the pooled order by handing out descending scores, so the balanced
    pool's own ordering decides the top-k.
    """

    def score(self, query: str, snippets: Sequence[str]) -> List[float]:
        return [float(len(snippets) - i) for i in range(len(snippets))]


def build_reranker(config: RetrievalConfig):
    """Instantiate the configured reranker.

    ``retrieval.reranker: none`` keeps the balanced pool's own order, which is
    what the "Balanced Retrieval" (un-reranked) curve of Figure A3 needs, and
    what smoke runs use to avoid a model download.
    """
    if config.reranker in ("none", "identity", "stub"):
        return IdentityReranker()
    return MedCPTCrossEncoder(config)
