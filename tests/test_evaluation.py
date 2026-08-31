"""Answer extraction and metrics."""

import pytest

from rag2.evaluation import (
    accuracy,
    accuracy_by,
    evidence_report,
    extract_choice,
    filter_metrics,
    is_correct,
    open_ended_metrics,
    rouge_l,
)
from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL
from rag2.schema import Evidence, PipelineResult

OPTIONS = {"A": "BiPAP", "B": "Chest tube placement", "C": "Intubation", "D": "Needle decompression"}


@pytest.mark.parametrize(
    "generation,expected",
    [
        # The paper's own worked example, Figure 4.
        ("... Therefore, the answer is (C) Intubation.", "C"),
        ("Therefore, the answer is (A) BiPAP.", "A"),
        ("The answer is D", "D"),
        ("Answer: B", "B"),
        ("the answer is: (D)", "D"),
        ("I would choose option A here.", "A"),
        ("Considering everything, (B).", "B"),
        ("A) BiPAP is the best next step", "A"),
    ],
)
def test_extract_choice_handles_common_shapes(generation, expected):
    assert extract_choice(generation, OPTIONS) == expected


def test_last_answer_wins():
    """The CoT prompt puts the decision at the end, so later statements win."""
    text = "At first the answer is (B). On reflection, the answer is (C)."
    assert extract_choice(text, OPTIONS) == "C"


def test_falls_back_to_matching_option_text():
    assert extract_choice("The best next step is Needle decompression.", OPTIONS) == "D"


def test_returns_none_when_nothing_matches():
    assert extract_choice("I am not sure about this case.", OPTIONS) is None
    assert extract_choice("", OPTIONS) is None


def test_letters_outside_the_option_set_are_ignored():
    assert extract_choice("The answer is (F)", OPTIONS) is None


def test_custom_patterns_are_honoured():
    assert extract_choice("FINAL>>C<<", OPTIONS, patterns=[r"FINAL>>([A-D])<<"]) == "C"


def test_unparsed_counts_as_incorrect_by_default():
    assert is_correct(None, "A") is False
    assert is_correct(None, "A", unparsed_as_incorrect=False) is None
    assert is_correct("a", "A") is True
    assert is_correct("B", "A") is False
    assert is_correct("A", None) is None


def _result(qid, correct, prediction="A", kept=0, candidates=4, subject="anatomy"):
    return PipelineResult(
        qid=qid,
        candidates=[Evidence(text="s", source="cpg") for _ in range(candidates)],
        kept=[Evidence(text="s", source="cpg") for _ in range(kept)],
        prediction=prediction,
        gold="A",
        correct=correct,
        metadata={"subject": subject},
    )


def test_accuracy_is_reported_in_percent():
    results = [_result("q1", True), _result("q2", False), _result("q3", True), _result("q4", True)]
    metrics = accuracy(results)
    assert metrics["accuracy"] == pytest.approx(75.0)
    assert metrics["num_correct"] == 3
    assert metrics["num_scored"] == 4


def test_accuracy_counts_unparsed_generations():
    results = [_result("q1", True), _result("q2", False, prediction=None)]
    assert accuracy(results)["num_unparsed"] == 1


def test_accuracy_on_an_empty_run_does_not_divide_by_zero():
    assert accuracy([])["accuracy"] == 0.0


def test_accuracy_by_subject():
    results = [
        _result("q1", True, subject="anatomy"),
        _result("q2", False, subject="anatomy"),
        _result("q3", True, subject="genetics"),
    ]
    by_subject = accuracy_by(results, "subject")
    assert by_subject["anatomy"]["accuracy"] == pytest.approx(50.0)
    assert by_subject["genetics"]["accuracy"] == pytest.approx(100.0)


def test_evidence_report_summarises_filtering():
    results = [_result("q1", True, kept=2), _result("q2", True, kept=0)]
    report = evidence_report(results)
    assert report["num_candidates_total"] == 8
    assert report["num_kept_total"] == 2
    assert report["keep_rate"] == pytest.approx(0.25)
    assert report["questions_with_no_evidence"] == 1
    assert report["kept_by_source"] == {"cpg": 2}


def test_filter_metrics_match_the_release_quantities():
    """classifier/utils.py reports overall accuracy plus per-class acc and counts."""
    gold = [LABEL_HELPFUL, LABEL_HELPFUL, LABEL_NOT_HELPFUL, LABEL_NOT_HELPFUL]
    predictions = [LABEL_HELPFUL, LABEL_NOT_HELPFUL, LABEL_NOT_HELPFUL, LABEL_NOT_HELPFUL]
    metrics = filter_metrics(gold, predictions)
    assert metrics["final_acc_score"] == pytest.approx(75.0)
    assert metrics["per_class"][f"{LABEL_HELPFUL} acc"] == pytest.approx(50.0)
    assert metrics["per_class"][f"{LABEL_NOT_HELPFUL} acc"] == pytest.approx(100.0)
    assert metrics["per_class"][f"{LABEL_HELPFUL} pred num"] == 1
    assert metrics["per_class"][f"{LABEL_NOT_HELPFUL} gold num"] == 2


def test_filter_metrics_report_minus_one_for_an_absent_class():
    """Matches the release's sentinel for a class with no gold examples."""
    metrics = filter_metrics([LABEL_HELPFUL], [LABEL_HELPFUL])
    assert metrics["per_class"][f"{LABEL_NOT_HELPFUL} acc"] == -1


def test_filter_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        filter_metrics([LABEL_HELPFUL], [LABEL_HELPFUL, LABEL_HELPFUL])


def test_rouge_l_matches_the_appendix_definition():
    """Appendix A.4.1: P = LCS/|C|, R = LCS/|R|, F1 harmonic."""
    scores = rouge_l("the cat sat on the mat", "the cat sat")
    assert scores["precision"] == pytest.approx(3 / 6)
    assert scores["recall"] == pytest.approx(3 / 3)
    assert scores["f1"] == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_rouge_l_uses_a_subsequence_not_a_substring():
    assert rouge_l("a x b x c", "a b c")["recall"] == pytest.approx(1.0)


def test_rouge_l_handles_empty_input():
    assert rouge_l("", "reference")["f1"] == 0.0


def test_open_ended_metrics_average_rouge_l():
    metrics = open_ended_metrics(["a b c", "d e"], ["a b c", "d e"], metrics=("rouge_l",))
    assert metrics["rouge_l"]["f1"] == pytest.approx(1.0)
    assert "bertscore" not in metrics
