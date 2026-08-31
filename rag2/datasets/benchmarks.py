"""Loaders for the three benchmarks the paper evaluates on.

MedQA (Jin et al., 2021), MedMCQA (Pal et al., 2022) and MMLU-Med (Hendrycks
et al., 2021 / the six biomedical subjects). All three are four-option multiple
choice (paper section 4.1, Table 1).

These loaders read the datasets' *published* file formats from local paths. The
paper's corpora and benchmark files are not redistributed with this repository;
point ``dataset.path`` at your own copy. Split sizes are asserted against
Table 1 when ``options.assert_size`` is set, so a wrong file is caught early.

Note on MedMCQA (documented assumption, see docs/rag2_reproduction.md 7.1): the
official test split is unlabelled, and the paper's 6,150 test items match the
*validation* split. ``split_map`` therefore maps ``test -> dev`` by default.
"""

from __future__ import annotations

import os
from typing import Dict, List

from ..config import DatasetConfig
from ..schema import Question
from .base import QADataset, register_dataset
from .jsonl import normalise_answer, normalise_options, read_records

# Paper Table 1.
EXPECTED_SIZES = {
    "medqa": {"train": 10178, "validation": 1272, "test": 1273},
    "medmcqa": {"train": 182822, "validation": 4183, "test": 6150},
    "mmlu_med": {"test": 1089},
}

# Paper section 4.1 (section A.2 says "human genetics"; the MMLU subject id is
# medical_genetics -- recorded as a discrepancy in the reproduction doc).
MMLU_MED_SUBJECTS = (
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
)


class _BenchmarkDataset(QADataset):
    key = ""
    default_fields: Dict[str, str] = {}
    default_answer_style = "letter"
    default_split_map: Dict[str, str] = {}

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.name = config.name or self.key
        self.version = config.version
        self._cache: List[Question] | None = None

    def _split_file(self) -> str:
        split_map = {**self.default_split_map, **(self.config.split_map or {})}
        target = split_map.get(self.config.split, self.config.split)
        path = self.config.path
        if os.path.isdir(path):
            for suffix in (".jsonl", ".json"):
                candidate = os.path.join(path, f"{target}{suffix}")
                if os.path.exists(candidate):
                    return candidate
            raise FileNotFoundError(f"no {target}.json[l] under {path}")
        return path

    def _check_size(self, questions: List[Question]) -> None:
        if not self.config.options.get("assert_size"):
            return
        expected = EXPECTED_SIZES.get(self.key, {}).get(self.config.split)
        if expected is not None and len(questions) != expected:
            raise ValueError(
                f"{self.key}/{self.config.split}: expected {expected} items "
                f"(paper Table 1), found {len(questions)}"
            )

    def questions(self) -> List[Question]:
        if self._cache is None:
            self._cache = self._load()
            self._check_size(self._cache)
        return list(self._cache)

    def _load(self) -> List[Question]:  # pragma: no cover - overridden
        raise NotImplementedError


class MedQADataset(_BenchmarkDataset):
    """MedQA-USMLE 4-option. Published records carry ``question``, ``options``
    (a letter dict) and ``answer_idx`` (a letter)."""

    key = "medqa"

    def _load(self) -> List[Question]:
        questions: List[Question] = []
        for i, record in enumerate(read_records(self._split_file())):
            options = normalise_options(record["options"])
            gold = record.get("answer_idx", record.get("answer"))
            style = "letter" if str(gold).strip().upper() in options else "text"
            questions.append(
                Question(
                    qid=str(record.get("id", f"medqa-{self.config.split}-{i}")),
                    question=str(record["question"]),
                    options=options,
                    answer=normalise_answer(gold, options, style),
                    dataset=self.name,
                    split=self.config.split,
                    metadata={k: record[k] for k in ("meta_info",) if k in record},
                )
            )
        return questions


class MedMCQADataset(_BenchmarkDataset):
    """MedMCQA. Published records carry ``question``, ``opa``..``opd`` and a
    0-based ``cop``."""

    key = "medmcqa"
    default_split_map = {"test": "dev"}

    def _load(self) -> List[Question]:
        questions: List[Question] = []
        for i, record in enumerate(read_records(self._split_file())):
            if all(k in record for k in ("opa", "opb", "opc", "opd")):
                options = {"A": record["opa"], "B": record["opb"], "C": record["opc"], "D": record["opd"]}
            else:
                options = normalise_options(record["options"])
            gold = record.get("cop", record.get("answer"))
            style = "index" if isinstance(gold, int) else "letter"
            questions.append(
                Question(
                    qid=str(record.get("id", f"medmcqa-{self.config.split}-{i}")),
                    question=str(record["question"]),
                    options={k: str(v) for k, v in options.items()},
                    answer=normalise_answer(gold, options, style),
                    dataset=self.name,
                    split=self.config.split,
                    metadata={k: record[k] for k in ("subject_name", "topic_name", "choice_type") if k in record},
                )
            )
        return questions


class MMLUMedDataset(_BenchmarkDataset):
    """The six biomedical MMLU subjects, test split only (no training data --
    paper section 4.2 uses the MedMCQA-trained filter here)."""

    key = "mmlu_med"

    def _load(self) -> List[Question]:
        subjects = list(self.config.options.get("subjects", MMLU_MED_SUBJECTS))
        base = self.config.path
        questions: List[Question] = []
        for subject in subjects:
            path = base
            if os.path.isdir(base):
                for suffix in (".jsonl", ".json"):
                    candidate = os.path.join(base, f"{subject}{suffix}")
                    if os.path.exists(candidate):
                        path = candidate
                        break
                else:
                    raise FileNotFoundError(f"no file for MMLU subject {subject} under {base}")
            for i, record in enumerate(read_records(path)):
                if os.path.isdir(base) is False and record.get("subject") not in (None, subject):
                    continue
                options = normalise_options(record.get("options", record.get("choices")))
                gold = record.get("answer", record.get("answer_idx"))
                style = "index" if isinstance(gold, int) else "letter"
                questions.append(
                    Question(
                        qid=str(record.get("id", f"mmlu-{subject}-{i}")),
                        question=str(record["question"]),
                        options=options,
                        answer=normalise_answer(gold, options, style),
                        dataset=self.name,
                        split="test",
                        metadata={"subject": record.get("subject", subject)},
                    )
                )
            if not os.path.isdir(base):
                break  # a single combined file already holds every subject
        return questions


@register_dataset("medqa")
def _build_medqa(config: DatasetConfig) -> QADataset:
    return MedQADataset(config)


@register_dataset("medmcqa")
def _build_medmcqa(config: DatasetConfig) -> QADataset:
    return MedMCQADataset(config)


@register_dataset("mmlu_med")
def _build_mmlu_med(config: DatasetConfig) -> QADataset:
    return MMLUMedDataset(config)
