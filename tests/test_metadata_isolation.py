"""Publication dates and other provenance must not influence the baseline.

The thesis will use evidence currency later. For the baseline to be a clean
comparison point, that information has to be *carried* but never *consulted*:
no model input may contain it, and reordering or stripping it must not change a
single decision. These tests are the guarantee.
"""

import copy


from rag2.config import Config, DatasetConfig, RetrievalConfig
from rag2.filtering.rag2_filter import ScriptedFilter
from rag2.llm.stub import StubLLM
from rag2.pipeline import run_filter_and_generate
from rag2.prompts import DEFAULT_PROMPTS
from rag2.schema import CandidateSet, Evidence, Question

DATE_FIELDS = ("publication_date", "year", "pub_date", "date", "retrieved_at")


def _question():
    return Question("q1", "Which is best?", {"A": "a", "B": "b", "C": "c", "D": "d"}, "A")


def _candidates():
    return [
        Evidence(
            text=f"Clinical evidence snippet number {i}.",
            source=["pubmed", "cpg"][i % 2],
            doc_id=f"PMID{1000 + i}",
            passage_id=f"PMID{1000 + i}-p0",
            rank=i + 1,
            metadata={
                "publication_date": f"{1990 + i * 7}-03-15",
                "year": 1990 + i * 7,
                "journal": "The Lancet",
            },
        )
        for i in range(4)
    ]


def _config():
    config = Config()
    config.dataset = DatasetConfig(loader="inline", name="unit")
    config.retrieval = RetrievalConfig(embedding_dim=8, candidates_per_corpus=2, final_top_k=4)
    config.llm.backend = "stub"
    config.filter.kind = "scripted"
    return config


def test_filter_prompt_contains_no_provenance():
    for evidence in _candidates():
        rendered = DEFAULT_PROMPTS.render_filter_prompt(_question(), evidence)
        for field in DATE_FIELDS:
            assert str(evidence.metadata.get(field, "\0")) not in rendered
        assert evidence.doc_id not in rendered
        assert evidence.source not in rendered
        assert "Lancet" not in rendered


def test_answer_prompt_contains_no_provenance():
    candidates = _candidates()
    rendered = DEFAULT_PROMPTS.render_answer_prompt(_question(), candidates)
    for evidence in candidates:
        assert str(evidence.metadata["publication_date"]) not in rendered
        assert str(evidence.metadata["year"]) not in rendered
        assert evidence.doc_id not in rendered
    assert "Lancet" not in rendered


def test_filter_decisions_are_unchanged_when_dates_are_stripped():
    question = _question()
    evidence_filter = ScriptedFilter(lambda rendered, q, e: len(rendered) % 7 / 7.0)

    with_dates = _candidates()
    without_dates = copy.deepcopy(with_dates)
    for evidence in without_dates:
        evidence.metadata = {}

    a = evidence_filter.decide(question, with_dates)
    b = evidence_filter.decide(question, without_dates)
    assert [(d.keep, d.label, d.score) for d in a] == [(d.keep, d.label, d.score) for d in b]


def test_filter_decisions_are_unchanged_when_dates_are_permuted():
    question = _question()
    evidence_filter = ScriptedFilter(lambda rendered, q, e: len(rendered) % 7 / 7.0)

    original = _candidates()
    shuffled = copy.deepcopy(original)
    dates = [e.metadata["publication_date"] for e in shuffled]
    for evidence, date in zip(shuffled, reversed(dates)):
        evidence.metadata["publication_date"] = date

    assert [(d.keep, d.score) for d in evidence_filter.decide(question, original)] == [
        (d.keep, d.score) for d in evidence_filter.decide(question, shuffled)
    ]


def test_full_pipeline_output_is_unchanged_when_dates_are_stripped():
    config = _config()
    question = _question()
    llm = StubLLM(config.llm)
    evidence_filter = ScriptedFilter(lambda rendered, q, e: 1.0 if "number 1" in rendered else 0.0)

    with_dates = {question.qid: CandidateSet(qid=question.qid, candidates=_candidates())}
    stripped = copy.deepcopy(with_dates)
    for evidence in stripped[question.qid].candidates:
        evidence.metadata = {}

    a = run_filter_and_generate(config, [question], with_dates, evidence_filter, llm)
    b = run_filter_and_generate(config, [question], stripped, evidence_filter, llm)

    assert a[0].generation == b[0].generation
    assert a[0].prediction == b[0].prediction
    assert [e.text for e in a[0].kept] == [e.text for e in b[0].kept]


def test_provenance_still_reaches_the_output():
    """Carried, not consulted: the results must retain what the corpus supplied."""
    config = _config()
    question = _question()
    candidate_sets = {question.qid: CandidateSet(qid=question.qid, candidates=_candidates())}
    results = run_filter_and_generate(
        config, [question], candidate_sets, ScriptedFilter(lambda *_: 1.0), StubLLM(config.llm)
    )
    kept = results[0].kept[0]
    assert kept.doc_id == "PMID1000"
    assert kept.metadata["publication_date"] == "1990-03-15"
    assert kept.metadata["journal"] == "The Lancet"


def _docstring_line_spans(source: str):
    """Line ranges occupied by module/class/function docstrings."""
    import ast

    spans = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(
            first.value.value, str
        ):
            spans.append((first.lineno, first.end_lineno or first.lineno))
    return spans


def test_no_baseline_module_reads_a_date_field():
    """A guard against date handling creeping into the baseline.

    Comments and docstrings may *discuss* publication metadata -- this file's own
    documentation does. Executable code, including string literals used as dict
    keys, may not name it: that is how a filter would start reading dates.
    """
    import io
    import os
    import re
    import tokenize

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag2")
    # No word boundaries: a violation is just as likely to be spelled
    # ``compute_recency`` or ``_pub_date`` as bare.
    pattern = re.compile(r"publication_date|pub_date|recency|currency|timestamp|published_at")
    offenders = []
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, "r", encoding="utf-8") as handle:
                source = handle.read()
            spans = _docstring_line_spans(source)
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    continue
                if token.type == tokenize.STRING and any(
                    lo <= token.start[0] <= hi for lo, hi in spans
                ):
                    continue  # a docstring, not code
                if pattern.search(token.string):
                    offenders.append(
                        f"{os.path.relpath(path, root)}:{token.start[0]}: {token.line.strip()}"
                    )
    assert not offenders, "baseline code references publication/recency fields:\n" + "\n".join(offenders)


def test_the_date_guard_would_catch_a_real_violation():
    """The guard above is only worth having if it fires. Prove it does."""
    import io
    import re
    import tokenize

    # No word boundaries: a violation is just as likely to be spelled
    # ``compute_recency`` or ``_pub_date`` as bare.
    pattern = re.compile(r"publication_date|pub_date|recency|currency|timestamp|published_at")

    def scan(source: str):
        spans = _docstring_line_spans(source)
        hits = []
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            if token.type == tokenize.STRING and any(lo <= token.start[0] <= hi for lo, hi in spans):
                continue
            if pattern.search(token.string):
                hits.append(token.string)
        return hits

    # Code that reads a date -- must be caught, however it is spelled.
    assert scan("x = evidence.metadata['publication_date']\n")
    assert scan("weight = compute_recency(evidence)\n")
    assert scan("if evidence.pub_date > cutoff:\n    pass\n")
    # Documentation that merely mentions it -- must not be caught.
    assert not scan('"""We carry publication_date as metadata only."""\n')
    assert not scan("# publication_date is never read\nx = 1\n")
