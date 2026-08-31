#!/usr/bin/env python
"""Stage 3a: build the perplexity-based training labels for the filter.

Paper section 3.2 and Figure 2. The released ``classifier/data/preprocess.py`` is
an empty file, so this reconstructs the annotation procedure: for every
(question, snippet) pair the base LLM is asked the question with and without the
snippet, the rationale's perplexity is scored both ways, and Figure 2's decision
tree assigns [HELPFUL] / [NOT_HELPFUL] / discard.

Runs on the *training* split (paper 4.2: filter data comes from MedQA and
MedMCQA training QA pairs).

    python scripts/03_build_filter_labels.py -c configs/medqa_llama3.yaml \\
        -o dataset.split=train --candidates cache/candidates/<train cache>.jsonl
"""

from __future__ import annotations

import argparse
import os

from _common import add_common_args, prepare_run, progress_printer, resolve_config

from rag2.cache import index_by_qid, load_candidates
from rag2.datasets.base import build_dataset
from rag2.experiment import write_json
from rag2.filter_training.build_labels import build_training_data
from rag2.filter_training.train import write_training_file
from rag2.llm.base import build_llm


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--candidates", required=True, help="candidate cache for the training split")
    parser.add_argument("--out", default="", help="training JSON path (default: <output_dir>/filter_train.json)")
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = prepare_run(config, "filter_labels")
    training = config.filter_training

    dataset = build_dataset(config.dataset)
    questions = dataset.questions()
    print(f"dataset {dataset.describe()} split={config.dataset.split}")
    if config.dataset.split == "test":
        print(
            "WARNING: building filter labels on the test split. The paper trains the "
            "filter on MedQA/MedMCQA *training* QA pairs (section 4.2)."
        )

    candidate_sets = index_by_qid(
        load_candidates(
            args.candidates,
            expected_fingerprint=config.retrieval_fingerprint(),
            allow_config_mismatch=config.cache.allow_config_mismatch,
        )
    )
    print(f"loaded {len(candidate_sets)} cached candidate sets")

    llm = build_llm(config.llm)
    print(f"llm {llm.describe()}")

    pairs, stats = build_training_data(
        llm,
        questions,
        candidate_sets,
        dataset_name=f"{dataset.name}_{config.llm.model.split('/')[-1]}",
        prompts=config.prompt_set(),
        top_k=training.label_top_k,
        tau_percentile=training.tau_percentile,
        tau_scope=training.tau_scope,
        ppl_target=training.ppl_target,
        ppl_rationale=training.ppl_rationale,
        drop_undecided=training.drop_undecided,
        progress=progress_printer("labeling"),
    )

    path = args.out or os.path.join(output_dir, "filter_train.json")
    write_training_file(path, pairs)
    write_json(
        os.path.join(output_dir, "filter_train.provenance.json"),
        [{"id": p.id, **p.provenance} for p in pairs],
    )
    diagnostics = stats.pop("diagnostics", {})
    write_json(os.path.join(output_dir, "filter_label_diagnostics.json"), diagnostics)
    write_json(os.path.join(output_dir, "filter_label_stats.json"), stats)

    print(f"tau ({training.tau_scope}, top {training.tau_percentile}%): {stats['tau']}")
    print(f"label counts: {stats['label_counts']}")
    print(f"wrote {len(pairs)} labeled pairs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
