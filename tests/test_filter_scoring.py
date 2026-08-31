"""The filter's scoring rule and the EvidenceFilter contract."""

import math

import pytest

from rag2.config import FilterConfig
from rag2.filtering.base import EvidenceFilter, build_filter
from rag2.filtering.passthrough import NoEvidenceFilter, PassthroughFilter
from rag2.filtering.rag2_filter import (
    KEEP_THRESHOLD,
    ScriptedFilter,
    decisions_from_probabilities,
    helpful_probability,
)
from rag2.prompts import DEFAULT_PROMPTS, LABEL_HELPFUL, LABEL_NOT_HELPFUL
from rag2.schema import Evidence, FilterDecision, Question


def _question():
    return Question("q1", "Stem?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")


def _candidates(n=3):
    return [Evidence(text=f"snippet {i}", source="cpg", doc_id=f"d{i}") for i in range(n)]


def test_helpful_probability_is_a_two_way_softmax():
    """The release softmaxes over the two label logits only, not the vocabulary."""
    assert helpful_probability(1.0, 1.0) == pytest.approx(0.5)
    assert helpful_probability(3.0, 1.0) == pytest.approx(1 / (1 + math.exp(-2.0)))
    assert helpful_probability(1.0, 3.0) == pytest.approx(1 - helpful_probability(3.0, 1.0))


def test_helpful_probability_is_numerically_stable():
    assert helpful_probability(1000.0, -1000.0) == pytest.approx(1.0)
    assert helpful_probability(-1000.0, 1000.0) == pytest.approx(0.0)


def test_helpful_probability_matches_torch_softmax():
    """Pins the pure helper against the exact torch call the release makes."""
    torch = pytest.importorskip("torch")
    logits = [(2.0, -1.0), (-0.5, 0.5), (10.0, 9.9)]
    for helpful, not_helpful in logits:
        stacked = torch.stack([torch.tensor([helpful]), torch.tensor([not_helpful])])
        expected = float(torch.nn.functional.softmax(stacked, dim=0)[0][0])
        assert helpful_probability(helpful, not_helpful) == pytest.approx(expected, abs=1e-6)


def test_keep_threshold_is_argmax_over_the_two_labels():
    decisions = decisions_from_probabilities([0.51, 0.49, KEEP_THRESHOLD])
    assert [d.keep for d in decisions] == [True, False, True]
    assert [d.label for d in decisions] == [LABEL_HELPFUL, LABEL_NOT_HELPFUL, LABEL_HELPFUL]


def test_apply_returns_one_decision_per_candidate():
    kept, decisions = ScriptedFilter(lambda *_: 1.0).apply(_question(), _candidates(4))
    assert len(decisions) == 4
    assert len(kept) == 4


def test_apply_rejects_a_filter_that_returns_the_wrong_number_of_decisions():
    class Broken(EvidenceFilter):
        name = "broken"

        def decide(self, question, candidates):
            return [FilterDecision(keep=True, label=LABEL_HELPFUL)]

    with pytest.raises(ValueError, match="1 decisions"):
        Broken().apply(_question(), _candidates(3))


def test_filter_does_not_mutate_candidates():
    candidates = _candidates(3)
    before = [(c.text, c.source, c.doc_id, c.rank) for c in candidates]
    ScriptedFilter(lambda *_: 1.0).apply(_question(), candidates)
    assert [(c.text, c.source, c.doc_id, c.rank) for c in candidates] == before


def test_scripted_filter_sees_the_rendered_filter_prompt():
    seen = []

    def score(rendered, question, evidence):
        seen.append(rendered)
        return 1.0

    ScriptedFilter(score, DEFAULT_PROMPTS).apply(_question(), _candidates(2))
    assert len(seen) == 2
    assert all(s.startswith("Given the following evidence") for s in seen)
    assert "snippet 0" in seen[0]


def test_passthrough_is_the_no_filter_ablation():
    kept, decisions = PassthroughFilter().apply(_question(), _candidates(5))
    assert len(kept) == 5
    assert all(d.keep and d.label == LABEL_HELPFUL for d in decisions)


def test_no_evidence_filter_drops_everything():
    kept, decisions = NoEvidenceFilter().apply(_question(), _candidates(5))
    assert kept == []
    assert all(not d.keep for d in decisions)


def test_empty_candidate_list_is_handled():
    assert ScriptedFilter(lambda *_: 1.0).apply(_question(), []) == ([], [])


def test_registry_builds_the_configured_filter():
    assert isinstance(build_filter(FilterConfig(kind="passthrough")), PassthroughFilter)
    with pytest.raises(KeyError, match="unknown filter"):
        build_filter(FilterConfig(kind="scaf"))


def test_rag2_filter_refuses_to_run_without_a_checkpoint():
    """The paper's trained filter is not distributed; the error must say so."""
    with pytest.raises(ValueError, match="not distributed"):
        build_filter(FilterConfig(kind="rag2_perplexity", checkpoint=""))
