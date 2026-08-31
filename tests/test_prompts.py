"""Prompt templates, pinned against the paper and the released data."""

import json
import os


from rag2.prompts import (
    ANSWER_PROMPT_WITH_EVIDENCE,
    DEFAULT_PROMPTS,
    FILTER_PROMPT,
    LABEL_HELPFUL,
    LABEL_NOT_HELPFUL,
    RATIONALE_PROMPT,
    PromptSet,
)
from rag2.schema import Evidence, Question, format_question

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_SAMPLE = os.path.join(
    REPO_ROOT, "classifier", "data", "medqa", "llama3_cot", "5%-train.json"
)


def _question():
    return Question(
        "q1",
        "A 5-year-old boy presents with conjunctivitis. Which is most likely?",
        {"A": "Metapneumovirus", "B": "Influenza virus", "C": "Rhinovirus", "D": "Adenovirus"},
        "D",
    )


def test_rationale_prompt_is_the_paper_text_verbatim():
    """Paper section 3.3 prints this prompt in full."""
    assert RATIONALE_PROMPT.startswith(
        "The following are multiple choice questions about medical knowledge. Solve them in "
        "a step-by-step fashion, starting by summarizing the available information. Output "
        "your explanation and single option from the given options as the final answer."
    )
    assert "Here is the question: {question}" in RATIONALE_PROMPT


def test_options_are_serialised_inline_as_in_the_released_data():
    rendered = format_question(_question())
    assert rendered.endswith("A) Metapneumovirus B) Influenza virus C) Rhinovirus D) Adenovirus")


def test_filter_prompt_matches_the_released_training_artifact():
    """The paper never prints the filter prompt; the release's data fixes it."""
    with open(RELEASE_SAMPLE, "r", encoding="utf-8") as handle:
        released = json.load(handle)

    for record in released:
        body = record["question"]
        head, _, rest = body.partition("\n\nEvidence: ")
        evidence, _, question = rest.partition("\n\nQuestion: ")
        rebuilt = FILTER_PROMPT.format(evidence=evidence, question=question)
        assert rebuilt == body, "our template does not reconstruct the released record"
        assert head == (
            "Given the following evidence, determine whether it helps answer the provided question."
        )
        assert record["answer"] in (LABEL_HELPFUL, LABEL_NOT_HELPFUL)


def test_filter_prompt_carries_the_initial_question_not_the_rationale():
    """Figure 1 shows the filter's prompt as 'Snippet + Initial Query'."""
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), Evidence(text="a snippet"))
    assert "a snippet" in rendered
    assert "A 5-year-old boy" in rendered
    assert rendered.count("\n\n") == 2


def test_filter_prompt_uses_only_the_evidence_text():
    evidence = Evidence(
        text="a snippet",
        source="pubmed",
        doc_id="PMID123",
        metadata={"publication_date": "1998-07-01", "journal": "Nature"},
    )
    rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), evidence)
    for leaked in ("PMID123", "1998-07-01", "Nature", "pubmed"):
        assert leaked not in rendered


def test_answer_prompt_has_evidence_and_question_slots():
    assert "{evidence}" in ANSWER_PROMPT_WITH_EVIDENCE
    assert "{question}" in ANSWER_PROMPT_WITH_EVIDENCE


def test_answer_prompt_falls_back_to_closed_book_without_evidence():
    closed = DEFAULT_PROMPTS.render_answer_prompt(_question(), [])
    assert "retrieved documents" not in closed
    assert closed == DEFAULT_PROMPTS.render_rationale_prompt(_question())


def test_evidence_block_is_rank_ordered():
    rendered = DEFAULT_PROMPTS.render_answer_prompt(
        _question(), [Evidence(text="first"), Evidence(text="second")]
    )
    assert "[1] first" in rendered and "[2] second" in rendered
    assert rendered.index("[1] first") < rendered.index("[2] second")


def test_prompt_fingerprint_changes_when_a_template_changes():
    """A silent prompt edit must show up in the run manifest."""
    baseline = DEFAULT_PROMPTS.fingerprint()
    assert PromptSet().fingerprint() == baseline
    assert PromptSet(rationale="something else").fingerprint() != baseline


def test_config_can_override_a_template():
    from rag2.config import load_config

    config = load_config(None, {"prompts": {"evidence_item": "- {text}"}})
    prompts = config.prompt_set()
    assert prompts.evidence_item == "- {text}"
    assert prompts.rationale == RATIONALE_PROMPT  # untouched fields keep the constant
    assert prompts.fingerprint() != DEFAULT_PROMPTS.fingerprint()
