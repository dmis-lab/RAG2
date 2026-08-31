"""The dataset plug-in interface.

The medical corpus/benchmark being prepared in parallel plugs in here: implement
``QADataset`` (or reuse ``JsonQADataset`` with a field map) and register it. No
other part of the pipeline needs to change.

A dataset yields :class:`rag2.schema.Question` objects. Everything the loader
knows but the pipeline does not need -- publication information, source ids,
annotator notes -- goes into ``Question.metadata`` and is carried through
untouched.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Iterator, List, Sequence

from ..config import DatasetConfig
from ..schema import Question


class QADataset(abc.ABC):
    """A split of multiple-choice medical QA items."""

    #: Human-readable dataset name, recorded in the run manifest.
    name: str = ""
    #: Dataset version/revision, recorded in the run manifest.
    version: str = ""

    @abc.abstractmethod
    def questions(self) -> List[Question]:
        """Return every question in the requested split, in a stable order."""

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions())

    def __len__(self) -> int:
        return len(self.questions())

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version, "size": len(self)}


_REGISTRY: Dict[str, Callable[[DatasetConfig], QADataset]] = {}


def register_dataset(key: str) -> Callable[[Callable[[DatasetConfig], QADataset]], Callable]:
    def decorator(factory: Callable[[DatasetConfig], QADataset]):
        if key in _REGISTRY:
            raise ValueError(f"dataset loader {key!r} already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)


def build_dataset(config: DatasetConfig) -> QADataset:
    """Instantiate the loader named by ``config.loader``."""
    from . import jsonl as _jsonl  # noqa: F401  (registration side effects)
    from . import benchmarks as _benchmarks  # noqa: F401

    if config.loader not in _REGISTRY:
        raise KeyError(
            f"unknown dataset loader {config.loader!r}; available: {available_datasets()}"
        )
    dataset = _REGISTRY[config.loader](config)
    if config.limit:
        dataset = LimitedDataset(dataset, config.limit)
    return dataset


class LimitedDataset(QADataset):
    """Truncating wrapper, used by ``dataset.limit`` for smoke runs."""

    def __init__(self, inner: QADataset, limit: int) -> None:
        self.inner = inner
        self.limit = limit
        self.name = inner.name
        self.version = inner.version

    def questions(self) -> List[Question]:
        return self.inner.questions()[: self.limit]


class InMemoryDataset(QADataset):
    """Questions supplied directly. Used by tests and the smoke pipeline."""

    def __init__(self, questions: Sequence[Question], name: str = "inline", version: str = "") -> None:
        self._questions = list(questions)
        self.name = name
        self.version = version

    def questions(self) -> List[Question]:
        return list(self._questions)


@register_dataset("inline")
def _build_inline(config: DatasetConfig) -> QADataset:
    raw = config.options.get("questions", [])
    return InMemoryDataset(
        [Question.from_dict(q) for q in raw], name=config.name or "inline", version=config.version
    )
