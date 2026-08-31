#!/usr/bin/env python
"""Stage 1: generate chain-of-thought rationales to use as retrieval queries.

Paper section 3.3. The rationale replaces the question as the retrieval query;
the initial query is deliberately *not* concatenated.

    python scripts/01_generate_rationales.py -c configs/medqa_llama3.yaml
"""

from __future__ import annotations

import argparse
import os

from _common import add_common_args, prepare_run, progress_printer, resolve_config

from rag2.datasets.base import build_dataset
from rag2.experiment import write_json
from rag2.llm.base import build_llm
from rag2.rationale import generate_rationales


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--out", default="", help="output JSON path (default: <output_dir>/rationales.json)")
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = prepare_run(config, "rationales")

    dataset = build_dataset(config.dataset)
    questions = dataset.questions()
    print(f"dataset {dataset.describe()}")

    llm = build_llm(config.llm)
    print(f"llm {llm.describe()}")

    rationales = generate_rationales(
        llm,
        questions,
        prompts=config.prompt_set(),
        batch_size=config.llm.batch_size,
        progress=progress_printer("rationale"),
    )

    path = args.out or os.path.join(output_dir, "rationales.json")
    write_json(path, rationales)
    print(f"wrote {len(rationales)} rationales -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
