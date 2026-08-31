"""Corpus backed by the article/embedding file layout of the original release.

``retriever/README.md`` specifies, per corpus, a set of ``.json`` article files
and ``.npy`` embedding files whose rows align by index. This loader reads exactly
that layout, so an index built for the released ``retriever/`` drops straight in:

    embeddings/pubmed/PubMed_Embeds_{0..37}.npy   articles/pubmed/PubMed_Articles_{0..37}.json
    embeddings/pmc/PMC_{Main,Abs}_Embeds.npy      articles/pmc/PMC_{Main,Abs}_Articles.json
    embeddings/cpg/CPG_Total_Embeds.npy           articles/cpg/CPG_Total_Articles.json
    embeddings/textbook/Textbook_Total_Embeds.npy articles/textbook/Textbook_Total_Articles.json

Files are listed explicitly in the config (``articles``/``embeddings``) rather
than reconstructed from a hard-coded per-corpus naming scheme, which is what made
``retriever/retrieve.py`` need five near-identical functions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..config import CorpusConfig
from ..schema import Evidence
from .base import Corpus, decode_passage, register_corpus

# The release's per-corpus filenames, for convenience when a config only names
# the corpus. Source: retriever/README.md.
RELEASE_LAYOUT: Dict[str, Dict[str, List[str]]] = {
    "pubmed": {
        "articles": [f"PubMed_Articles_{i}.json" for i in range(38)],
        "embeddings": [f"PubMed_Embeds_{i}.npy" for i in range(38)],
    },
    "pmc": {
        "articles": ["PMC_Main_Articles.json", "PMC_Abs_Articles.json"],
        "embeddings": ["PMC_Main_Embeds.npy", "PMC_Abs_Embeds.npy"],
    },
    "cpg": {"articles": ["CPG_Total_Articles.json"], "embeddings": ["CPG_Total_Embeds.npy"]},
    "textbook": {
        "articles": ["Textbook_Total_Articles.json"],
        "embeddings": ["Textbook_Total_Embeds.npy"],
    },
    "statpearls": {
        "articles": ["Statpearls_Total_Articles.json"],
        "embeddings": ["Statpearls_Total_Embeds.npy"],
    },
}


class JsonDirCorpus(Corpus):
    def __init__(self, config: CorpusConfig) -> None:
        self.name = config.name
        self.config = config
        layout = RELEASE_LAYOUT.get(config.name, {})
        self.article_files = [
            _join(config.articles_dir, f) for f in (config.articles or layout.get("articles", []))
        ]
        self.embedding_files = [
            _join(config.embeddings_dir, f)
            for f in (config.embeddings or layout.get("embeddings", []))
        ]
        if not self.article_files:
            raise ValueError(
                f"corpus {config.name!r}: no article files configured and no default layout "
                f"known for that name (known: {sorted(RELEASE_LAYOUT)}). Set "
                f"retrieval.corpora[].articles / .embeddings explicitly."
            )
        if len(self.embedding_files) != len(self.article_files):
            raise ValueError(
                f"corpus {config.name!r}: {len(self.article_files)} article file(s) but "
                f"{len(self.embedding_files)} embedding file(s); they must correspond "
                f"one-to-one so retrieved indices decode to the right passage"
            )
        missing = [f for f in self.article_files + self.embedding_files if not os.path.exists(f)]
        if missing:
            defaulted = not (config.articles or config.embeddings)
            hint = (
                " These filenames came from the released layout for a corpus named "
                f"{config.name!r} (retriever/README.md). If your files are named or sharded "
                "differently, list them explicitly under retrieval.corpora[].articles and "
                ".embeddings."
                if defaulted
                else ""
            )
            raise FileNotFoundError(
                f"corpus {config.name!r}: {len(missing)} configured file(s) not found, "
                f"first: {missing[0]}.{hint}"
            )
        options = dict(config.options or {})
        self.text_fields = tuple(options.get("text_fields", ()))or None
        self.lazy = bool(options.get("lazy", True))
        self._offsets: Optional[List[int]] = None
        self._articles: Dict[int, List[Any]] = {}
        self._total: Optional[int] = None

    # -- article side ------------------------------------------------------
    def _load_file(self, file_index: int) -> List[Any]:
        if file_index not in self._articles:
            path = self.article_files[file_index]
            with open(path, "r", encoding="utf-8") as handle:
                self._articles[file_index] = json.load(handle)
            if self.lazy:
                # Keep at most two files resident so PubMed's 38 shards do not
                # all end up in memory during decoding.
                for key in list(self._articles):
                    if key != file_index and len(self._articles) > 2:
                        del self._articles[key]
        return self._articles[file_index]

    def _build_offsets(self) -> List[int]:
        if self._offsets is None:
            offsets = [0]
            for i in range(len(self.article_files)):
                offsets.append(offsets[-1] + len(self._load_file(i)))
            self._offsets = offsets
            self._total = offsets[-1]
        return self._offsets

    def __len__(self) -> int:
        self._build_offsets()
        assert self._total is not None
        return self._total

    def passage(self, index: int) -> Evidence:
        offsets = self._build_offsets()
        if index < 0 or index >= offsets[-1]:
            raise IndexError(f"{self.name}: passage index {index} out of range (0..{offsets[-1]-1})")
        file_index = _bisect(offsets, index)
        local = index - offsets[file_index]
        raw = self._load_file(file_index)[local]
        kwargs = {"text_fields": self.text_fields} if self.text_fields else {}
        return decode_passage(raw, self.name, index, **kwargs)

    # -- embedding side ----------------------------------------------------
    def embedding_shards(self) -> Iterator[Tuple[int, Any]]:
        import numpy as np  # imported lazily: keeps stdlib-only paths importable

        offset = 0
        for path in self.embedding_files:
            matrix = np.load(path, mmap_mode="r")
            matrix = np.asarray(matrix, dtype=np.float32)
            yield offset, matrix
            offset += matrix.shape[0]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "article_files": len(self.article_files),
            "embedding_files": len(self.embedding_files),
        }


def _join(directory: str, filename: str) -> str:
    return filename if not directory or os.path.isabs(filename) else os.path.join(directory, filename)


def _bisect(offsets: List[int], index: int) -> int:
    lo, hi = 0, len(offsets) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if offsets[mid] <= index:
            lo = mid
        else:
            hi = mid
    return lo


@register_corpus("json_dir")
def _build_json_dir(config: CorpusConfig) -> Corpus:
    return JsonDirCorpus(config)
