#!/usr/bin/env python
"""Stage 3b: train the Flan-T5 filter on the perplexity-based labels.

Training itself runs through the authors' own ``classifier/run_classifier.py``,
kept unmodified; this script adds the ``[HELPFUL]`` / ``[NOT_HELPFUL]`` tokens and
supplies the paper's hyperparameters (appendix A.3: lr 3e-5, 40 epochs, batch 16)
from the config.

    # once: extend google/flan-t5-large with the two label tokens
    python scripts/04_train_filter.py -c configs/medqa_llama3.yaml \\
        --init-tokens --token-dir runs/filter-base

    # then train
    python scripts/04_train_filter.py -c configs/medqa_llama3.yaml \\
        --model runs/filter-base \\
        --train-file runs/medqa-llama3/filter_train.json \\
        --filter-output-dir runs/filter-medqa-llama3
"""

from __future__ import annotations

import argparse
import json
import os

from _common import REPO_ROOT, add_common_args, prepare_run, resolve_config

from rag2.experiment import write_json
from rag2.filter_training.train import (
    add_label_tokens,
    build_train_command,
    evaluate_filter_checkpoint,
    run_command,
)


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--init-tokens", action="store_true", help="add the two label tokens and exit")
    parser.add_argument("--token-dir", default="", help="where --init-tokens writes the extended model")
    parser.add_argument("--model", default="", help="base model with label tokens already added")
    parser.add_argument("--train-file", default="", help="labeled training JSON from stage 3a")
    parser.add_argument("--validation-file", default="", help="labeled validation JSON")
    parser.add_argument("--filter-output-dir", default="", help="where checkpoints are written")
    parser.add_argument("--select", action="store_true", help="score epoch checkpoints and pick the best")
    parser.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = prepare_run(config, "filter_training")

    if args.init_tokens:
        token_dir = args.token_dir or os.path.join(output_dir, "filter-base")
        info = add_label_tokens(config.filter.base_model, token_dir)
        write_json(os.path.join(output_dir, "label_tokens.json"), info)
        print(json.dumps(info, indent=2))
        return 0

    if not args.train_file:
        parser.error("--train-file is required (build it with scripts/03_build_filter_labels.py)")
    model = args.model or args.token_dir
    if not model:
        parser.error("--model is required: pass the directory produced by --init-tokens")

    filter_dir = args.filter_output_dir or os.path.join(output_dir, "filter")
    command = build_train_command(
        model_name_or_path=model,
        train_file=os.path.abspath(args.train_file),
        output_dir=os.path.abspath(filter_dir),
        training=config.filter_training,
        seed=config.experiment.seed,
        validation_file=os.path.abspath(args.validation_file) if args.validation_file else "",
    )
    write_json(os.path.join(output_dir, "filter_train_command.json"), {"command": command})
    if args.dry_run:
        print(" ".join(command))
        return 0

    code = run_command(command, cwd=REPO_ROOT)
    if code != 0:
        print(f"training failed with exit code {code}")
        return code

    if args.select and args.validation_file:
        with open(args.validation_file, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        # The paper selects "a few candidate models from the validation set"
        # without stating a rule; we score every epoch checkpoint and take the
        # best filter accuracy (filter_training.select_by).
        checkpoints = sorted(
            os.path.join(filter_dir, name)
            for name in os.listdir(filter_dir)
            if name.startswith("epoch_")
        ) + [filter_dir]
        scores = []
        for checkpoint in checkpoints:
            try:
                scores.append(evaluate_filter_checkpoint(checkpoint, records, config.filter))
            except Exception as error:  # a partially written checkpoint
                print(f"skipping {checkpoint}: {error}")
        if scores:
            scores.sort(key=lambda s: s["final_acc_score"], reverse=True)
            write_json(os.path.join(output_dir, "filter_checkpoint_selection.json"), scores)
            print(f"best checkpoint: {scores[0]['checkpoint']} ({scores[0]['final_acc_score']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
