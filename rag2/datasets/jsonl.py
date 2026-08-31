"""Generic JSON / JSONL dataset loader -- the main plug-in point.

Point ``dataset.path`` at a ``.json`` (list of objects) or ``.jsonl`` file and
describe its field names via ``dataset.options.fields``. Nothing about the
medical dataset's internal schema needs to leak into the pipeline.

Example config::

    dataset:
      loader: jsonl
      name: my-medical-benchmark
      version: "2026-08"
      path: data/medical/test.jsonl
      options:
        fields:
          qid: id
          question: question
          options: options        # dict, or list -> lettered A,B,C,D...
          answer: answer_idx
        answer_style: index       # "letter" | "index" | "text"
        metadata_fields: [source, publication_date, topic]
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

from ..config import DatasetConfig
from ..schema import OPTION_LETTERS, Question
from .base import QADataset, register_dataset

DEFAULT_FIELDS = {
    "qid": "id",
    "question": "question",
    "options": "options",
    "answer": "answer",
}


def read_records(path: str) -> List[Dict[str, Any]]:
    """Read a ``.json`` list or a ``.jsonl`` stream into a list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset file not found: {path}")
    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        for key in ("data", "questions", "examples"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"expected a list of records in {path}")
    return list(payload)


def normalise_options(raw: Any) -> Dict[str, str]:
    """Accept ``{"A": ...}``, ``{"op_a": ...}`` or ``[...]`` and return a letter map."""
    if isinstance(raw, Mapping):
        keys = list(raw)
        # {"A": ...} / {"a": ...}
        if all(len(str(k).strip()) == 1 for k in keys):
            return {str(k).strip().upper(): str(v) for k, v in raw.items()}
        # {"op_a": ...} / {"opa": ...}
        out: Dict[str, str] = {}
        for key, value in raw.items():
            letter = str(key).strip().upper()[-1]
            out[letter] = str(value)
        return out
    if isinstance(raw, (list, tuple)):
        return {OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else chr(ord("A") + i): str(v)
                for i, v in enumerate(raw)}
    raise ValueError(f"cannot interpret options: {raw!r}")


def normalise_answer(raw: Any, options: Mapping[str, str], style: str) -> Optional[str]:
    """Map a gold answer to its option letter."""
    if raw is None:
        return None
    letters = sorted(options)
    if style == "index":
        idx = int(raw)
        # tolerate 1-based indices
        if idx not in range(len(letters)) and 1 <= idx <= len(letters):
            idx -= 1
        return letters[idx]
    if style == "text":
        target = str(raw).strip()
        for letter, text in options.items():
            if str(text).strip() == target:
                return letter
        raise ValueError(f"gold answer text {target!r} matches no option")
    letter = str(raw).strip().upper()
    if letter in options:
        return letter
    # tolerate "(A)" / "A)" / "answer is A"
    for candidate in letters:
        if letter.endswith(candidate) or letter.startswith(candidate):
            return candidate
    raise ValueError(f"gold answer {raw!r} is not one of {letters}")


class JsonQADataset(QADataset):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.name = config.name or os.path.basename(config.path)
        self.version = config.version
        options = dict(config.options or {})
        self.fields = {**DEFAULT_FIELDS, **dict(options.get("fields", {}))}
        self.answer_style = options.get("answer_style", "letter")
        self.metadata_fields = list(options.get("metadata_fields", []))
        self.keep_all_metadata = bool(options.get("keep_all_metadata", False))
        self._path = self._resolve_path()
        self._cache: Optional[List[Question]] = None

    def _resolve_path(self) -> str:
        path = self.config.path
        mapped = (self.config.split_map or {}).get(self.config.split)
        if mapped:
            path = mapped if os.path.isabs(mapped) or os.sep in mapped else os.path.join(
                os.path.dirname(path) or ".", mapped
            )
        return path

    def questions(self) -> List[Question]:
        if self._cache is not None:
            return list(self._cache)
        questions: List[Question] = []
        for i, record in enumerate(read_records(self._path)):
            options = normalise_options(record[self.fields["options"]])
            answer_key = self.fields["answer"]
            answer = normalise_answer(record.get(answer_key), options, self.answer_style)
            qid = str(record.get(self.fields["qid"], i))
            consumed = set(self.fields.values())
            if self.keep_all_metadata:
                metadata = {k: v for k, v in record.items() if k not in consumed}
            else:
                metadata = {k: record[k] for k in self.metadata_fields if k in record}
            questions.append(
                Question(
                    qid=qid,
                    question=str(record[self.fields["question"]]),
                    options=options,
                    answer=answer,
                    dataset=self.name,
                    split=self.config.split,
                    metadata=metadata,
                )
            )
        self._cache = questions
        return list(questions)


@register_dataset("jsonl")
def _build_jsonl(config: DatasetConfig) -> QADataset:
    return JsonQADataset(config)


@register_dataset("json")
def _build_json(config: DatasetConfig) -> QADataset:
    return JsonQADataset(config)
