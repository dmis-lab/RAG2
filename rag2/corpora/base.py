"""The evidence-corpus plug-in interface.

A corpus exposes (a) N passages addressable by index and (b) N aligned MedCPT
embedding vectors, so that a FAISS index built over the embeddings decodes back
to the right passage -- the alignment invariant the release states in
``retriever/README.md``.

Provenance (document id, passage id, source, publication information) is read
here and attached to every :class:`rag2.schema.Evidence`. It is carried for the
thesis's later use and for traceability; **no baseline component consumes it**.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple

from ..config import CorpusConfig
from ..schema import Evidence


class Corpus(abc.ABC):
    """A single retrievable corpus (``pubmed``, ``pmc``, ``cpg``, ``textbook``, ...)."""

    name: str = ""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Total number of passages."""

    @abc.abstractmethod
    def passage(self, index: int) -> Evidence:
        """Decode a global passage index into an Evidence with provenance."""

    @abc.abstractmethod
    def embedding_shards(self) -> Iterator[Tuple[int, Any]]:
        """Yield ``(offset, matrix)`` pairs covering every passage in order.

        ``matrix`` is a float32 array of shape ``(n_i, dim)``; ``offset`` is the
        global index of its first row. Yielding more than one shard lets very
        large corpora (the paper's PubMed is 69.7M passages) be indexed in
        pieces without materialising the whole matrix.
        """

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "size": len(self)}


_REGISTRY: Dict[str, Callable[[CorpusConfig], Corpus]] = {}


def register_corpus(key: str):
    def decorator(factory: Callable[[CorpusConfig], Corpus]):
        if key in _REGISTRY:
            raise ValueError(f"corpus loader {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_corpora() -> List[str]:
    return sorted(_REGISTRY)


def build_corpus(config: CorpusConfig) -> Corpus:
    from . import json_corpus as _json_corpus  # noqa: F401  (registration)

    if config.loader not in _REGISTRY:
        raise KeyError(f"unknown corpus loader {config.loader!r}; available: {available_corpora()}")
    return _REGISTRY[config.loader](config)


class InMemoryCorpus(Corpus):
    """Passages and embeddings supplied directly. Used by tests and smoke runs."""

    def __init__(
        self,
        name: str,
        passages: Sequence[Any],
        embeddings: Any = None,
        text_fields: Sequence[str] = ("text", "content"),
    ) -> None:
        self.name = name
        self._passages = list(passages)
        self._embeddings = embeddings
        self._text_fields = tuple(text_fields)

    def __len__(self) -> int:
        return len(self._passages)

    def passage(self, index: int) -> Evidence:
        return decode_passage(self._passages[index], self.name, index, self._text_fields)

    def embedding_shards(self):
        if self._embeddings is None:
            return iter(())
        return iter([(0, self._embeddings)])


DEFAULT_TEXT_FIELDS = ("text", "content", "contents", "body", "abstract", "passage")
DEFAULT_ID_FIELDS = ("doc_id", "id", "pmid", "PMID", "pmcid", "docid", "document_id")
DEFAULT_PASSAGE_ID_FIELDS = ("passage_id", "chunk_id", "snippet_id", "pid")


def decode_passage(
    raw: Any,
    source: str,
    index: int,
    text_fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    id_fields: Sequence[str] = DEFAULT_ID_FIELDS,
    passage_id_fields: Sequence[str] = DEFAULT_PASSAGE_ID_FIELDS,
    title_field: str = "title",
) -> Evidence:
    """Turn a raw corpus record into an :class:`Evidence`.

    Accepts both shapes the release and common MedCPT dumps use: a bare string
    (what ``retriever/retrieve.py`` assumes) or a dict. For a dict, the text is
    taken from the first present ``text_fields`` entry (optionally prefixed with
    the title), identifiers from ``id_fields``/``passage_id_fields``, and
    **every remaining key is preserved verbatim in ``metadata``** -- which is
    how publication information survives into the cache.
    """
    if isinstance(raw, str):
        return Evidence(text=raw, source=source, corpus_index=index)
    if not isinstance(raw, dict):
        return Evidence(text=str(raw), source=source, corpus_index=index)

    text = ""
    for key in text_fields:
        if raw.get(key):
            text = str(raw[key])
            break
    title = raw.get(title_field)
    if title and text and not text.startswith(str(title)):
        text = f"{title}. {text}"
    elif title and not text:
        text = str(title)

    doc_id = next((str(raw[k]) for k in id_fields if raw.get(k) is not None), None)
    passage_id = next((str(raw[k]) for k in passage_id_fields if raw.get(k) is not None), None)
    consumed = set(text_fields) | set(id_fields) | set(passage_id_fields) | {title_field}
    metadata = {k: v for k, v in raw.items() if k not in consumed}
    return Evidence(
        text=text,
        source=source,
        doc_id=doc_id,
        passage_id=passage_id,
        corpus_index=index,
        metadata=metadata,
    )
