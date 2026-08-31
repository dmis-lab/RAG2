"""MedCPT query encoding (paper section 3.4; retriever/query_encode.py).

Reproduces the release's semantics exactly:

* checkpoint ``ncbi/MedCPT-Query-Encoder``,
* embedding = the CLS position of the last hidden state
  (``last_hidden_state[:, 0, :]``),
* truncation at 512 tokens,

while fixing three defects in the released implementation: a hard-coded
``cuda:7``, a batch size pinned to 1, and an O(n^2) ``np.vstack`` inside the
encoding loop.

The optional SciSpacy ``[SEP]`` insertion follows MedCPT and is off by default,
as in the release (``--use_spacy False``).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..config import RetrievalConfig


def insert_sep(texts: Sequence[str], model_name: str = "en_core_sci_scibert") -> List[str]:
    """Join sentence boundaries with ``[SEP]``, following MedCPT.

    Mirrors ``retriever/query_encode.py``: single-character "sentences" are
    dropped and the final sentence gets no trailing separator.
    """
    import spacy  # lazy: scispacy is an optional dependency

    nlp = spacy.load(model_name)
    out: List[str] = []
    for doc in nlp.pipe(list(texts)):
        sentences = [s.text for s in doc.sents if len(s.text) > 1]
        out.append(" [SEP] ".join(sentences) if sentences else doc.text)
    return out


class MedCPTQueryEncoder:
    def __init__(self, config: RetrievalConfig) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        device = config.device
        self.device = torch.device(
            device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.query_encoder)
        self.model = AutoModel.from_pretrained(config.query_encoder)
        self.model.eval()
        self.model.to(self.device)

    def encode(self, queries: Sequence[str], batch_size: Optional[int] = None):
        """Return a ``(n, dim)`` float32 array of CLS embeddings."""
        import numpy as np
        import torch

        if not queries:
            return np.zeros((0, self.config.embedding_dim), dtype=np.float32)
        texts = list(queries)
        if self.config.use_scispacy_sep:
            texts = insert_sep(texts, self.config.scispacy_model)

        size = batch_size or self.config.encode_batch_size or 32
        chunks: List[Any] = []
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            with torch.no_grad():
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.config.query_max_length,
                ).to(self.device)
                # CLS pooling, exactly as retriever/query_encode.py does.
                embeds = self.model(**encoded).last_hidden_state[:, 0, :]
            chunks.append(embeds.detach().cpu().numpy().astype("float32"))
        return np.vstack(chunks)


class HashQueryEncoder:
    """Deterministic stand-in for MedCPT, for smoke runs and tests.

    Embeds a query as a fixed pseudo-random unit vector derived from a hash of
    its text. It is *not* a semantic encoder -- retrieval driven by it is
    meaningless -- but it exercises the exact MIPS/rerank/cache path with no
    model download. Selected with ``retrieval.query_encoder: stub``.
    """

    is_stub = True

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def encode(self, queries: Sequence[str], batch_size: Optional[int] = None):
        import hashlib

        import numpy as np

        dim = self.config.embedding_dim
        rows = []
        for query in queries:
            digest = hashlib.sha256(query.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            vector = np.random.default_rng(seed).normal(size=(dim,)).astype("float32")
            norm = float(np.linalg.norm(vector)) or 1.0
            rows.append(vector / norm)
        return np.vstack(rows) if rows else np.zeros((0, dim), dtype="float32")


def build_query_encoder(config: RetrievalConfig):
    """Instantiate the configured query encoder."""
    if config.query_encoder in ("stub", "hash"):
        return HashQueryEncoder(config)
    return MedCPTQueryEncoder(config)
