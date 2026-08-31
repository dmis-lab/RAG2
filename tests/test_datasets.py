"""The dataset plug-in interface -- where the medical dataset will attach."""

import json

import pytest

from rag2.config import DatasetConfig
from rag2.datasets.base import InMemoryDataset, available_datasets, build_dataset
from rag2.datasets.jsonl import normalise_answer, normalise_options
from rag2.schema import Question


def _write(tmp_path, name, payload):
    path = tmp_path / name
    if name.endswith(".jsonl"):
        path.write_text("\n".join(json.dumps(r) for r in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_registry_lists_the_paper_benchmarks_and_the_generic_loader():
    build_dataset(DatasetConfig(loader="inline"))  # triggers registration
    assert {"medqa", "medmcqa", "mmlu_med", "jsonl", "inline"} <= set(available_datasets())


def test_unknown_loader_is_reported_with_the_available_ones():
    with pytest.raises(KeyError, match="unknown dataset loader"):
        build_dataset(DatasetConfig(loader="not_a_loader"))


def test_generic_loader_maps_arbitrary_field_names(tmp_path):
    path = _write(
        tmp_path,
        "custom.json",
        [{"uid": "x1", "stem": "Which?", "choices": ["a", "b", "c", "d"], "gold": 2}],
    )
    dataset = build_dataset(
        DatasetConfig(
            loader="jsonl",
            name="custom",
            version="v1",
            path=path,
            options={
                "fields": {"qid": "uid", "question": "stem", "options": "choices", "answer": "gold"},
                "answer_style": "index",
            },
        )
    )
    question = dataset.questions()[0]
    assert question.qid == "x1"
    assert question.question == "Which?"
    assert question.options == {"A": "a", "B": "b", "C": "c", "D": "d"}
    assert question.answer == "C"
    assert dataset.describe() == {"name": "custom", "version": "v1", "size": 1}


def test_generic_loader_preserves_named_metadata(tmp_path):
    """Publication information rides along as metadata, unread by the baseline."""
    path = _write(
        tmp_path,
        "m.jsonl",
        [
            {
                "id": "q1",
                "question": "Q?",
                "options": ["a", "b", "c", "d"],
                "answer": "A",
                "publication_date": "2021-05-01",
                "source": "guideline",
                "ignored": 1,
            }
        ],
    )
    dataset = build_dataset(
        DatasetConfig(
            loader="jsonl",
            path=path,
            options={"metadata_fields": ["publication_date", "source"]},
        )
    )
    question = dataset.questions()[0]
    assert question.metadata == {"publication_date": "2021-05-01", "source": "guideline"}


def test_generic_loader_can_keep_all_unconsumed_fields(tmp_path):
    path = _write(
        tmp_path,
        "m.json",
        [{"id": "q1", "question": "Q?", "options": ["a", "b"], "answer": "A", "extra": 7}],
    )
    dataset = build_dataset(
        DatasetConfig(loader="jsonl", path=path, options={"keep_all_metadata": True})
    )
    assert dataset.questions()[0].metadata == {"extra": 7}


def test_limit_truncates_for_smoke_runs(tmp_path):
    path = _write(
        tmp_path,
        "m.json",
        [{"id": f"q{i}", "question": "Q?", "options": ["a", "b"], "answer": "A"} for i in range(10)],
    )
    dataset = build_dataset(DatasetConfig(loader="jsonl", path=path, limit=3))
    assert len(dataset) == 3


def test_medqa_loader_reads_the_published_shape(tmp_path):
    _write(
        tmp_path,
        "test.json",
        [
            {
                "question": "Which?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer_idx": "C",
                "meta_info": "step1",
            }
        ],
    )
    dataset = build_dataset(DatasetConfig(loader="medqa", path=str(tmp_path), split="test"))
    question = dataset.questions()[0]
    assert question.answer == "C"
    assert question.metadata["meta_info"] == "step1"


def test_medmcqa_loader_reads_opa_opd_and_zero_based_cop(tmp_path):
    _write(
        tmp_path,
        "dev.json",
        [{"id": "m1", "question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 1,
          "subject_name": "Physiology"}],
    )
    dataset = build_dataset(
        DatasetConfig(loader="medmcqa", path=str(tmp_path), split="test", split_map={"test": "dev"})
    )
    question = dataset.questions()[0]
    assert question.answer == "B"
    assert question.metadata["subject_name"] == "Physiology"


def test_medmcqa_maps_test_to_dev_by_default(tmp_path):
    """The official test split is unlabelled; the paper's 6,150 items are dev."""
    _write(tmp_path, "dev.json", [{"question": "Q?", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 0}])
    dataset = build_dataset(DatasetConfig(loader="medmcqa", path=str(tmp_path), split="test"))
    assert len(dataset) == 1


def test_mmlu_med_reads_the_six_biomedical_subjects(tmp_path):
    from rag2.datasets.benchmarks import MMLU_MED_SUBJECTS

    assert len(MMLU_MED_SUBJECTS) == 6
    for subject in MMLU_MED_SUBJECTS:
        _write(tmp_path, f"{subject}.json", [{"question": f"{subject}?", "choices": ["a", "b", "c", "d"], "answer": 0}])
    dataset = build_dataset(DatasetConfig(loader="mmlu_med", path=str(tmp_path), split="test"))
    questions = dataset.questions()
    assert len(questions) == 6
    assert {q.metadata["subject"] for q in questions} == set(MMLU_MED_SUBJECTS)


def test_assert_size_catches_the_wrong_file(tmp_path):
    """Guards against silently evaluating on the wrong split (paper Table 1)."""
    _write(tmp_path, "test.json", [{"question": "Q?", "options": {"A": "a", "B": "b"}, "answer_idx": "A"}])
    with pytest.raises(ValueError, match="expected 1273"):
        build_dataset(
            DatasetConfig(loader="medqa", path=str(tmp_path), split="test", options={"assert_size": True})
        ).questions()


def test_option_normalisation_accepts_the_shapes_in_the_wild():
    assert normalise_options(["a", "b"]) == {"A": "a", "B": "b"}
    assert normalise_options({"a": "x", "b": "y"}) == {"A": "x", "B": "y"}
    assert normalise_options({"opa": "x", "opb": "y"}) == {"A": "x", "B": "y"}


def test_answer_normalisation_styles():
    options = {"A": "alpha", "B": "beta"}
    assert normalise_answer("B", options, "letter") == "B"
    assert normalise_answer(1, options, "index") == "B"
    assert normalise_answer("beta", options, "text") == "B"
    assert normalise_answer(None, options, "letter") is None


def test_answer_normalisation_rejects_an_out_of_range_gold():
    with pytest.raises(ValueError):
        normalise_answer("Z", {"A": "a", "B": "b"}, "letter")


def test_in_memory_dataset_is_iterable():
    questions = [Question(f"q{i}", "Q?", {"A": "a", "B": "b"}, "A") for i in range(3)]
    dataset = InMemoryDataset(questions, name="unit", version="v0")
    assert [q.qid for q in dataset] == ["q0", "q1", "q2"]
    assert len(dataset) == 3
