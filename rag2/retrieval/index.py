"""Maximum-inner-product search over one corpus.

The release builds a ``faiss.IndexFlatIP(768)`` per corpus and searches it
(``retriever/retrieve.py``) -- an exact, brute-force inner-product index, no ANN
approximation. This module reproduces that, generalised over corpora and with
shard handling that preserves balanced retrieval.

Two backends:

``faiss``  the release's ``IndexFlatIP``. Used whenever faiss is importable.
``numpy``  an exact brute-force fallback computing the same inner products.
           Mathematically identical to ``IndexFlatIP`` (ties may break
           differently); it exists so the pipeline and its smoke test run on a
           machine without faiss. Real experiments should use faiss.

Sharding
--------
Very large corpora (the paper's PubMed is 69.7M passages / 400GB) cannot be held
in one index. ``retriever/main.py`` handles this by searching each shard group
and *concatenating* the per-group top-k, which yields ``n_groups * k`` PubMed
candidates against ``k`` from every other corpus -- breaking the balance the
paper's section 3.4 is about. Here shards are merged **by score** so a corpus
contributes exactly ``k`` candidates however many shards it is stored in. Set
``retrieval.shard_merge: concat`` to reproduce the release's behaviour.
"""

from __future__ import annotations

from typing import Any, List, Tuple

try:  # pragma: no cover - presence depends on the environment
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _HAS_FAISS = False


def faiss_available() -> bool:
    return _HAS_FAISS


class ShardSearcher:
    """Exact inner-product search over one shard of embeddings."""

    def __init__(self, matrix: Any, offset: int, backend: str = "auto") -> None:
        import numpy as np

        self.offset = int(offset)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.backend = "faiss" if (backend in ("auto", "faiss") and _HAS_FAISS) else "numpy"
        if backend == "faiss" and not _HAS_FAISS:
            raise RuntimeError("retrieval backend 'faiss' requested but faiss is not installed")
        self._index = None
        if self.backend == "faiss":
            self._index = faiss.IndexFlatIP(self.matrix.shape[1])
            self._index.add(self.matrix)

    def search(self, queries: Any, k: int) -> Tuple[Any, Any]:
        """Return ``(scores, global_indices)``, each ``(n_queries, k)``."""
        import numpy as np

        k = min(k, self.matrix.shape[0])
        if k <= 0:
            empty = np.zeros((queries.shape[0], 0), dtype=np.float32)
            return empty, empty.astype(np.int64)
        if self._index is not None:
            scores, indices = self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        else:
            scores_full = np.asarray(queries, dtype=np.float32) @ self.matrix.T
            part = np.argpartition(-scores_full, kth=k - 1, axis=1)[:, :k]
            part_scores = np.take_along_axis(scores_full, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            indices = np.take_along_axis(part, order, axis=1)
            scores = np.take_along_axis(part_scores, order, axis=1)
        return scores.astype("float32"), (indices.astype("int64") + self.offset)

    def free(self) -> None:
        self._index = None
        self.matrix = None  # type: ignore[assignment]


def search_corpus(
    corpus,
    queries: Any,
    top_k: int,
    backend: str = "auto",
    shard_merge: str = "score",
    query_batch_size: int = 1024,
) -> Tuple[Any, Any]:
    """Search every shard of ``corpus`` and return ``(scores, indices)``.

    With ``shard_merge='score'`` the result is the true global top-``top_k`` for
    the corpus. With ``'concat'`` it is the per-shard top-``top_k`` stacked, which
    reproduces ``retriever/main.py``'s PubMed behaviour.
    """
    import numpy as np

    n_queries = queries.shape[0]
    all_scores: List[Any] = []
    all_indices: List[Any] = []

    for offset, matrix in corpus.embedding_shards():
        searcher = ShardSearcher(matrix, offset, backend=backend)
        shard_scores: List[Any] = []
        shard_indices: List[Any] = []
        for start in range(0, n_queries, max(query_batch_size, 1)):
            batch = queries[start : start + max(query_batch_size, 1)]
            s, i = searcher.search(batch, top_k)
            shard_scores.append(s)
            shard_indices.append(i)
        searcher.free()
        if shard_scores:
            all_scores.append(np.vstack(shard_scores))
            all_indices.append(np.vstack(shard_indices))

    if not all_scores:
        return (
            np.zeros((n_queries, 0), dtype="float32"),
            np.zeros((n_queries, 0), dtype="int64"),
        )

    scores = np.concatenate(all_scores, axis=1)
    indices = np.concatenate(all_indices, axis=1)
    if shard_merge == "concat" or len(all_scores) == 1:
        return scores, indices

    keep = min(top_k, scores.shape[1])
    part = np.argpartition(-scores, kth=keep - 1, axis=1)[:, :keep]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    chosen = np.take_along_axis(part, order, axis=1)
    return np.take_along_axis(scores, chosen, axis=1), np.take_along_axis(indices, chosen, axis=1)
