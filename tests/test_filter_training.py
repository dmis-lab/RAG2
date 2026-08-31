"""Training-data construction and the training command."""

import json
import os

import pytest

from rag2.config import FilterTrainingConfig
from rag2.filter_training.build_labels import build_observations, build_training_data
from rag2.filter_training.train import build_eval_command, build_train_command, write_training_file
from rag2.llm.stub import StubLLM
from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL
from rag2.schema import CandidateSet, Evidence, Question

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _questions(n=4):
    return [
        Question(f"q{i}", f"Vignette {i}?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
        for i in range(n)
    ]


def _candidates(questions, per=3):
    return {
        q.qid: CandidateSet(
            qid=q.qid,
            candidates=[
                Evidence(text=f"snippet {q.qid}-{j}", source="cpg", doc_id=f"d{j}",
                         metadata={"publication_date": "2005-01-01"})
                for j in range(per)
            ],
        )
        for q in questions
    }


def test_observations_cover_every_question_snippet_pair():
    questions = _questions(3)
    observations, diagnostics = build_observations(StubLLM(), questions, _candidates(questions, 3), top_k=3)
    assert len(observations) == 9
    assert {o.qid for o in observations} == {q.qid for q in questions}
    assert {o.snippet_index for o in observations} == {0, 1, 2}
    assert set(diagnostics) == {q.qid for q in questions}


def test_top_k_bounds_the_labeling_cost():
    questions = _questions(2)
    observations, _ = build_observations(StubLLM(), questions, _candidates(questions, 10), top_k=4)
    assert len(observations) == 8


def test_diagnostics_record_what_the_labels_were_derived_from():
    questions = _questions(1)
    _, diagnostics = build_observations(StubLLM(), questions, _candidates(questions, 2), top_k=2)
    entry = diagnostics["q0"]
    assert entry["closed_book_generation"]
    assert entry["closed_book_prediction"] in set("ABCD")
    assert entry["gold"] == "A"
    assert len(entry["snippets"]) == 2
    assert {"delta_ppl", "ppl_with", "ppl_without"} <= set(entry["snippets"][0])


def test_a_question_without_a_gold_answer_is_rejected():
    """Filter labels need correctness, so the training split's labels are required."""
    questions = [Question("q0", "Q?", {"A": "a", "B": "b"}, None)]
    with pytest.raises(ValueError, match="no gold answer"):
        build_observations(StubLLM(), questions, _candidates(questions, 1), top_k=1)


def test_unsupported_ppl_rationale_is_rejected():
    questions = _questions(1)
    with pytest.raises(ValueError, match="unsupported ppl_rationale"):
        build_observations(
            StubLLM(), questions, _candidates(questions, 1), top_k=1, ppl_rationale="with_retrieval"
        )


def test_questions_absent_from_the_cache_are_skipped():
    questions = _questions(3)
    partial = _candidates(questions[:1], 2)
    observations, _ = build_observations(StubLLM(), questions, partial, top_k=2)
    assert {o.qid for o in observations} == {"q0"}


def test_build_training_data_produces_release_shaped_records():
    questions = _questions(6)
    pairs, stats = build_training_data(
        StubLLM(), questions, _candidates(questions, 3), dataset_name="unit_stub", top_k=3
    )
    assert stats["num_observations"] == 18
    assert stats["tau"] is not None
    assert all(p.answer in (LABEL_HELPFUL, LABEL_NOT_HELPFUL) for p in pairs)
    for pair in pairs:
        record = pair.to_training_record()
        assert set(record) == {"id", "answer", "dataset_name", "question"}
        assert record["question"].startswith("Given the following evidence")
        assert "\n\nEvidence: " in record["question"]
        assert "\n\nQuestion: " in record["question"]


def test_provenance_is_written_beside_the_training_data_not_inside_it():
    questions = _questions(4)
    pairs, _ = build_training_data(
        StubLLM(), questions, _candidates(questions, 2), dataset_name="unit", top_k=2
    )
    assert pairs
    provenance = pairs[0].provenance
    assert {"qid", "snippet_index", "delta_ppl", "tau", "source", "doc_id"} <= set(provenance)
    assert "publication_date" not in json.dumps(pairs[0].to_training_record())


def test_labeling_is_deterministic_for_a_fixed_stub():
    questions = _questions(4)
    candidates = _candidates(questions, 2)
    first, _ = build_training_data(StubLLM(), questions, candidates, dataset_name="unit", top_k=2)
    second, _ = build_training_data(StubLLM(), questions, candidates, dataset_name="unit", top_k=2)
    assert [(p.id, p.answer) for p in first] == [(p.id, p.answer) for p in second]


def test_write_training_file_round_trips(tmp_path):
    questions = _questions(4)
    pairs, _ = build_training_data(
        StubLLM(), questions, _candidates(questions, 2), dataset_name="unit", top_k=2
    )
    path = write_training_file(str(tmp_path / "train.json"), pairs)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert len(payload) == len(pairs)
    assert set(payload[0]) == {"id", "answer", "dataset_name", "question"}


def test_written_file_matches_the_released_artifacts_schema(tmp_path):
    """Same four keys and types as classifier/data/medqa/llama3_cot/5%-train.json."""
    release_path = os.path.join(REPO_ROOT, "classifier", "data", "medqa", "llama3_cot", "5%-train.json")
    with open(release_path, "r", encoding="utf-8") as handle:
        released = json.load(handle)

    questions = _questions(4)
    pairs, _ = build_training_data(
        StubLLM(), questions, _candidates(questions, 2), dataset_name="unit", top_k=2
    )
    path = write_training_file(str(tmp_path / "train.json"), pairs)
    with open(path, "r", encoding="utf-8") as handle:
        ours = json.load(handle)

    assert set(ours[0]) == set(released[0])
    assert {type(v) for v in ours[0].values()} == {type(v) for v in released[0].values()} == {str}


def test_train_command_carries_the_papers_hyperparameters():
    command = build_train_command(
        "base", "train.json", "out", FilterTrainingConfig(), seed=42
    )
    text = " ".join(command)
    assert "classifier/run_classifier.py" in text  # the authors' own script
    assert "--learning_rate 3e-05" in text
    assert "--num_train_epochs 40" in text
    assert "--per_device_train_batch_size 16" in text
    assert "--max_seq_length 512" in text
    assert "--doc_stride 128" in text
    assert "--checkpointing_steps epoch" in text
    assert "--do_train" in text
    assert "--question_column question --answer_column answer" in text
    assert "--seed 42" in text  # the release left this unset


def test_train_command_adds_validation_only_when_asked():
    without = build_train_command("b", "t.json", "o", FilterTrainingConfig())
    assert "--validation_file" not in " ".join(without)
    assert "--seed" not in " ".join(without)

    with_val = build_train_command("b", "t.json", "o", FilterTrainingConfig(), validation_file="v.json")
    assert "--validation_file v.json" in " ".join(with_val)
    assert "--val_column validation" in " ".join(with_val)


def test_eval_command_uses_the_release_eval_mode():
    text = " ".join(build_eval_command("ckpt", "val.json", "out", FilterTrainingConfig()))
    assert "--do_eval" in text
    assert "--per_device_eval_batch_size 16" in text
