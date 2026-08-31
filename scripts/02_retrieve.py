#!/usr/bin/env python
"""Stage 2: balanced retrieval + reranking, cached for replay.

Paper section 3.4. Retrieves an equal number of candidates from every configured
corpus, pools them, and reranks with the MedCPT cross-encoder.

The output is the cache that later filter runs replay, so retrieval over the
paper's 564 GB corpus happens once:

    python scripts/02_retrieve.py -c configs/medqa_llama3.yaml
    python scripts/05_run_pipeline.py -c configs/medqa_llama3.yaml \\
        --candidates cache/candidates/<...>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

from _common import add_common_args, prepare_run, progress_printer, resolve_config

from rag2.cache import default_cache_path, save_candidates
from rag2.datasets.base import build_dataset
from rag2.experiment import write_json
from rag2.llm.base import build_llm
from rag2.pipeline import run_retrieval
from rag2.retrieval.balanced import corpus_distribution


def main() -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--rationales", default="", help="reuse a rationales.json from stage 1")
    parser.add_argument("--out", default="", help="cache path (default: derived from the config)")
    args = parser.parse_args()

    config = resolve_config(args)
    output_dir = prepare_run(config, "retrieval")

    dataset = build_dataset(config.dataset)
    questions = dataset.questions()
    print(f"dataset {dataset.describe()}")

    rationales = None
    llm = None
    if args.rationales:
        with open(args.rationales, "r", encoding="utf-8") as handle:
            rationales = json.load(handle)
        print(f"reusing {len(rationales)} rationales from {args.rationales}")
    else:
        llm = build_llm(config.llm)
        print(f"llm {llm.describe()}")

    candidate_sets = run_retrieval(
        config,
        questions,
        llm=llm,
        rationales=rationales,
        prompts=config.prompt_set(),
        progress=progress_printer("retrieval"),
    )

    fingerprint = config.retrieval_fingerprint()
    path = args.out or default_cache_path(
        config.cache.dir, config.experiment.name, config.dataset.split, fingerprint
    )
    metadata = save_candidates(
        path,
        candidate_sets,
        retrieval_fingerprint=fingerprint,
        extra={
            "dataset": dataset.describe(),
            "candidates_per_corpus": config.retrieval.candidates_per_corpus,
            "corpora": [c.name for c in config.retrieval.corpora],
            "rerank_query": config.retrieval.rerank_query,
            "shard_merge": config.retrieval.shard_merge,
        },
    )
    print(f"cached {metadata.num_questions} candidate sets -> {path}")

    pooled = {}
    for candidate_set in candidate_sets:
        for source, count in corpus_distribution(candidate_set.candidates).items():
            pooled[source] = pooled.get(source, 0) + count
    write_json(os.path.join(output_dir, "retrieval_stats.json"), {"pool_by_source": pooled})
    print(f"pooled candidates by source: {pooled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
