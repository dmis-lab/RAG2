#!/usr/bin/env python
"""Stages 3-4: filter the cached candidates, then generate answers.

Replays candidate evidence from the cache written by ``02_retrieve.py``, so
retrieval is not repeated and every filter configuration sees the identical
candidate set. The cache's fingerprint is checked against the current retrieval
config, so "only the filter changed" is verified rather than assumed.

    python scripts/05_run_pipeline.py -c configs/medqa_llama3.yaml \\
        --candidates cache/candidates/medqa-llama3.test.<fp>.jsonl
"""

from __future__ import annotations

import argparse
import os

from _common import add_common_args, prepare_run, progress_printer, resolve_config

from rag2.cache import index_by_qid, load_candidates
from rag2.datasets.base import build_dataset
from rag2.evaluation import accuracy, accuracy_by, evidence_report
from rag2.experiment import write_json, write_jsonl
from rag2.filtering.base import build_filter
from rag2.llm.base import build_llm
from rag2.pipeline import run_filter_and_generate, run_retrieval


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--candidates", default="", help="candidate cache; omit to retrieve now")
    parser.add_argument("--by", default="", help="also report accuracy grouped by this metadata key")
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = prepare_run(config, "pipeline")
    prompts = config.prompt_set()

    dataset = build_dataset(config.dataset)
    questions = dataset.questions()
    print(f"dataset {dataset.describe()}")

    llm = build_llm(config.llm)
    print(f"llm {llm.describe()}")

    if args.candidates:
        candidate_sets = index_by_qid(
            load_candidates(
                args.candidates,
                expected_fingerprint=config.retrieval_fingerprint(),
                allow_config_mismatch=config.cache.allow_config_mismatch,
            )
        )
        print(f"replaying {len(candidate_sets)} cached candidate sets from {args.candidates}")
    else:
        print("no --candidates given: running retrieval now (slow; prefer the cache)")
        candidate_sets = index_by_qid(
            run_retrieval(config, questions, llm=llm, prompts=prompts, progress=progress_printer("retrieval"))
        )

    evidence_filter = build_filter(config.filter, prompts)
    print(f"filter {evidence_filter.describe()}")

    results = run_filter_and_generate(
        config,
        questions,
        candidate_sets,
        evidence_filter,
        llm,
        prompts=prompts,
        progress=progress_printer("generation"),
    )

    write_jsonl(os.path.join(output_dir, "predictions.jsonl"), (r.to_dict() for r in results))
    metrics = {
        "accuracy": accuracy(results),
        "evidence": evidence_report(results),
        "filter": evidence_filter.describe(),
        "final_top_k": config.retrieval.final_top_k,
    }
    if args.by:
        metrics["accuracy_by"] = {args.by: accuracy_by(results, args.by)}
    write_json(os.path.join(output_dir, "metrics.json"), metrics)

    print(f"accuracy: {metrics['accuracy']['accuracy']:.1f}% "
          f"({metrics['accuracy']['num_correct']}/{metrics['accuracy']['num_scored']})")
    print(f"evidence: {metrics['evidence']}")
    print(f"wrote results -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
