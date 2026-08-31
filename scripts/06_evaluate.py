#!/usr/bin/env python
"""Stage 5: score a run and compare it with the paper's reported numbers.

Reads a ``predictions.jsonl`` written by ``05_run_pipeline.py``, recomputes the
metrics, and prints the delta against Table 2. It does **not** tune anything --
per the reproduction brief, divergences are recorded, not engineered away.

    python scripts/06_evaluate.py --predictions runs/medqa-llama3/predictions.jsonl \\
        --paper llama3:medqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

from rag2.evaluation import accuracy, accuracy_by, evidence_report
from rag2.experiment import write_json
from rag2.schema import Evidence, FilterDecision, PipelineResult

# Paper Table 2, accuracy in percent: (no RAG, + RAG2).
PAPER_RESULTS = {
    "llama3:medqa": (57.7, 64.6),
    "llama3:medmcqa": (53.5, 59.4),
    "llama3:mmlu_med": (69.5, 74.8),
    "meerkat:medqa": (71.2, 75.6),
    "meerkat:medmcqa": (60.8, 63.0),
    "meerkat:mmlu_med": (73.8, 78.7),
    "gpt4o:medqa": (88.5, 91.1),
    "gpt4o:medmcqa": (76.7, 77.2),
    "gpt4o:mmlu_med": (92.8, 92.5),
}


def load_results(path: str):
    results = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            results.append(
                PipelineResult(
                    qid=payload["qid"],
                    rationale=payload.get("rationale", ""),
                    candidates=[Evidence.from_dict(c) for c in payload.get("candidates", [])],
                    decisions=[FilterDecision(**d) for d in payload.get("decisions", [])],
                    kept=[Evidence.from_dict(c) for c in payload.get("kept", [])],
                    generation=payload.get("generation", ""),
                    prediction=payload.get("prediction"),
                    gold=payload.get("gold"),
                    correct=payload.get("correct"),
                    metadata=payload.get("metadata", {}),
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="predictions.jsonl from stage 4")
    parser.add_argument("--paper", default="", help=f"paper row to compare against: {sorted(PAPER_RESULTS)}")
    parser.add_argument("--by", default="", help="report accuracy grouped by a metadata key")
    parser.add_argument("--out", default="", help="where to write the report JSON")
    args = parser.parse_args()

    results = load_results(args.predictions)
    report = {
        "predictions": os.path.abspath(args.predictions),
        "accuracy": accuracy(results),
        "evidence": evidence_report(results),
    }
    if args.by:
        report["accuracy_by"] = {args.by: accuracy_by(results, args.by)}

    observed = report["accuracy"]["accuracy"]
    print(f"accuracy: {observed:.1f}% ({report['accuracy']['num_correct']}/{report['accuracy']['num_scored']})")
    print(f"unparsed generations: {report['accuracy']['num_unparsed']}")
    print(f"evidence: {report['evidence']}")

    if args.paper:
        if args.paper not in PAPER_RESULTS:
            print(f"unknown paper row {args.paper!r}; known: {sorted(PAPER_RESULTS)}", file=sys.stderr)
            return 2
        no_rag, rag2 = PAPER_RESULTS[args.paper]
        report["paper"] = {"row": args.paper, "no_rag": no_rag, "rag2": rag2, "delta": observed - rag2}
        print(f"paper {args.paper}: no-RAG {no_rag} / RAG2 {rag2}  ->  reproduction delta {observed - rag2:+.1f}")
        print(
            "A gap is expected and must be explained, not tuned away. Work through "
            "docs/rag2_reproduction.md section 11 in order (corpus, reconstructed filter "
            "checkpoint, answer prompt, rerank query, tau population, model versions)."
        )

    path = args.out or os.path.splitext(args.predictions)[0] + ".report.json"
    write_json(path, report)
    print(f"wrote report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
