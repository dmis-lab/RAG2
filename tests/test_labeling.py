"""The Figure 2 decision tree and the tau threshold.

These are the specification of the original filter's labels, so they are pinned
literally: any change to decide() has to change this table too.
"""


import pytest

from rag2.filter_training.labeling import (
    DISCARD,
    LabelingObservation,
    compute_tau,
    decide_label,
    label_observations,
    tau_for,
)
from rag2.filter_training.perplexity import percentile, top_percent_threshold
from rag2.prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL
from rag2.schema import Evidence, Question

# (correct w/o retrieval, correct w/ retrieval, Delta-PPL >= tau) -> label
FIGURE_2 = [
    (True, True, True, LABEL_HELPFUL),
    (True, True, False, DISCARD),
    (True, False, True, LABEL_NOT_HELPFUL),
    (True, False, False, LABEL_NOT_HELPFUL),
    (False, True, True, LABEL_HELPFUL),
    (False, True, False, LABEL_HELPFUL),
    (False, False, True, LABEL_NOT_HELPFUL),
    (False, False, False, DISCARD),
]


@pytest.mark.parametrize("without,with_,lower,expected", FIGURE_2)
def test_decision_tree_matches_figure_2(without, with_, lower, expected):
    assert decide_label(without, with_, lower) == expected


def test_correctness_flip_decides_regardless_of_perplexity():
    """The two flip branches of Figure 2 do not consult Delta-PPL at all."""
    assert decide_label(True, False, True) == decide_label(True, False, False)
    assert decide_label(False, True, True) == decide_label(False, True, False)


def test_tau_selects_the_top_25_percent():
    deltas = [float(i) for i in range(100)]
    tau = top_percent_threshold(deltas, 25.0)
    assert sum(1 for d in deltas if d >= tau) == 25


def test_tau_percentile_matches_numpy():
    numpy = pytest.importorskip("numpy")
    values = [float(v) for v in numpy.random.default_rng(7).normal(size=500)]
    assert percentile(values, 75.0) == pytest.approx(float(numpy.percentile(values, 75)))


def test_tau_rejects_out_of_range_percentages():
    with pytest.raises(ValueError):
        top_percent_threshold([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        top_percent_threshold([1.0, 2.0], 101.0)


def _observation(qid, index, without, with_, delta):
    question = Question(qid, "Stem?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")
    return LabelingObservation(
        qid=qid,
        snippet_index=index,
        correct_without_retrieval=without,
        correct_with_retrieval=with_,
        delta_ppl=delta,
        evidence=Evidence(text=f"snippet {qid}-{index}", source="cpg", doc_id=f"d{index}"),
        question=question,
    )


def test_tau_scope_global_vs_per_question():
    observations = [
        _observation("q1", 0, True, True, 10.0),
        _observation("q1", 1, True, True, 1.0),
        _observation("q2", 0, True, True, 0.5),
        _observation("q2", 1, True, True, 0.1),
    ]
    global_tau = compute_tau(observations, 25.0, "global")
    assert set(global_tau) == {"__global__"}

    per_question = compute_tau(observations, 25.0, "per_question")
    assert set(per_question) == {"q1", "q2"}
    # q2's own threshold is far below the global one, so its best snippet clears
    # its per-question bar but not the global bar.
    assert tau_for(per_question, "q2") < tau_for(global_tau, "q2")


def test_tau_scope_rejects_unknown_value():
    with pytest.raises(ValueError):
        compute_tau([_observation("q1", 0, True, True, 1.0)], 25.0, "per_corpus")


def test_label_observations_drops_discards_and_reports_stats():
    observations = [
        _observation("q1", 0, True, True, 10.0),   # helpful (top quartile)
        _observation("q1", 1, True, True, 0.0),    # discard
        _observation("q1", 2, True, False, 0.0),   # not helpful
        _observation("q1", 3, False, True, 0.0),   # helpful
    ]
    pairs, stats = label_observations(observations, dataset_name="unit")
    assert stats["num_observations"] == 4
    assert stats["label_counts"][DISCARD] == 1
    assert len(pairs) == 3
    assert stats["discard_rate"] == pytest.approx(0.25)
    assert {p.answer for p in pairs} == {LABEL_HELPFUL, LABEL_NOT_HELPFUL}


def test_label_observations_can_keep_discards():
    observations = [_observation("q1", i, True, True, float(i)) for i in range(8)]
    pairs, stats = label_observations(observations, dataset_name="unit", drop_undecided=False)
    assert stats["label_counts"][DISCARD] > 0
    assert len(pairs) == len(observations)
    assert DISCARD in {p.answer for p in pairs}


def test_tau_is_a_quantile_of_the_observed_deltas():
    """With a single observation tau equals it, so the test trivially passes --
    a property of quantile thresholding worth pinning so it is not mistaken for
    a bug later."""
    pairs, stats = label_observations(
        [_observation("q1", 0, True, True, -1.0)], dataset_name="unit"
    )
    assert stats["tau"] == pytest.approx(-1.0)
    assert [p.answer for p in pairs] == [LABEL_HELPFUL]


def test_labeled_pair_matches_release_schema():
    observations = [_observation("q1", 0, False, True, 0.0)]
    pairs, _ = label_observations(observations, dataset_name="llama3_5%")
    record = pairs[0].to_training_record()
    # Exactly the four fields of classifier/data/medqa/llama3_cot/5%-train.json.
    assert set(record) == {"id", "answer", "dataset_name", "question"}
    assert record["dataset_name"] == "llama3_5%"
    assert all(isinstance(v, str) for v in record.values())


def test_provenance_is_kept_out_of_the_training_record():
    observations = [_observation("q1", 0, False, True, 0.0)]
    pairs, _ = label_observations(observations, dataset_name="unit")
    assert pairs[0].provenance["doc_id"] == "d0"
    assert "doc_id" not in pairs[0].to_training_record()
