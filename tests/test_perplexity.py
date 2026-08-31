"""Equation 3 and Equation 4."""

import math

import pytest

from rag2.filter_training.perplexity import (
    PerplexityPair,
    compute_perplexity_pair,
    perplexity_from_scores,
)
from rag2.llm.base import LLM, ScoredSequence
from rag2.prompts import DEFAULT_PROMPTS
from rag2.schema import Evidence, Question


def test_perplexity_is_exp_of_negative_mean_logprob():
    scored = ScoredSequence(token_logprobs=[-1.0, -2.0, -3.0], num_tokens=3)
    assert perplexity_from_scores(scored) == pytest.approx(math.exp(2.0))


def test_perplexity_is_length_normalised():
    """Equation 4 divides by L, so repeating a sequence leaves PPL unchanged."""
    short = ScoredSequence(token_logprobs=[-1.5, -2.5], num_tokens=2)
    long = ScoredSequence(token_logprobs=[-1.5, -2.5] * 5, num_tokens=10)
    assert perplexity_from_scores(short) == pytest.approx(perplexity_from_scores(long))


def test_empty_continuation_is_infinite_perplexity():
    assert perplexity_from_scores(ScoredSequence([], 0)) == float("inf")


def test_delta_is_without_minus_with():
    """Equation 3: positive Delta-PPL means the document raised confidence."""
    pair = PerplexityPair(ppl_without=10.0, ppl_with=4.0)
    assert pair.delta == pytest.approx(6.0)
    assert PerplexityPair(ppl_without=4.0, ppl_with=10.0).delta < 0


class RecordingLLM(LLM):
    """Returns a fixed score per prompt and records what it was asked."""

    def __init__(self, scores):
        self.scores = scores
        self.seen = []

    def generate(self, prompts, **kwargs):
        return ["" for _ in prompts]

    def score(self, prompt, continuation):
        self.seen.append((prompt, continuation))
        value = self.scores[len(self.seen) - 1]
        tokens = max(len(continuation.split()), 1)
        return ScoredSequence(token_logprobs=[value / tokens] * tokens, num_tokens=tokens)


def _question():
    return Question("q1", "Stem?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")


def test_rationale_target_scores_the_same_string_under_both_prompts():
    """Delta-PPL must isolate the document's effect on one fixed sequence."""
    llm = RecordingLLM([-4.0, -2.0])
    evidence = Evidence(text="a snippet", source="cpg")
    pair = compute_perplexity_pair(
        llm, _question(), "the rationale text", evidence, DEFAULT_PROMPTS, target="rationale"
    )
    assert len(llm.seen) == 2
    # Same continuation in both terms...
    assert llm.seen[0][1] == llm.seen[1][1] == "the rationale text"
    # ...different conditioning: only the second prompt carries the snippet.
    assert "a snippet" not in llm.seen[0][0]
    assert "a snippet" in llm.seen[1][0]
    # The second call has the higher log-probability, so PPL dropped: delta > 0.
    assert pair.delta > 0


def test_query_target_reproduces_the_literal_reading_of_equation_4():
    llm = RecordingLLM([-4.0, -2.0])
    compute_perplexity_pair(
        llm, _question(), "unused rationale", Evidence(text="a snippet"), DEFAULT_PROMPTS, target="query"
    )
    # Under the literal reading the scored tokens are the query, not the rationale.
    assert "Stem?" in llm.seen[0][1]
    assert "unused rationale" not in llm.seen[0][1]


def test_unknown_target_is_rejected():
    with pytest.raises(ValueError, match="unknown ppl target"):
        compute_perplexity_pair(
            RecordingLLM([-1.0, -1.0]), _question(), "r", Evidence(text="s"), target="answer"
        )


def test_scored_sequence_aggregates():
    scored = ScoredSequence(token_logprobs=[-1.0, -3.0], num_tokens=2)
    assert scored.sum_logprob == pytest.approx(-4.0)
    assert scored.mean_logprob == pytest.approx(-2.0)
