#!/usr/bin/env python
"""Write a tiny synthetic dataset + corpora so the CLI can be exercised offline.

This exists **only** so the stage scripts can be run end to end before the real
medical dataset is ready. It is not a substitute benchmark: the questions are
nonsense and the "evidence" is generated text. Nothing but ``configs/smoke.yaml``
refers to it.

    python scripts/make_smoke_fixture.py --out data/smoke
"""

from __future__ import annotations

import argparse
import json
import os

from _common import REPO_ROOT  # noqa: F401  (sys.path bootstrap)

CORPORA = {
    "pubmed": ("PubMed_Articles_0.json", "PubMed_Embeds_0.npy"),
    "pmc": ("PMC_Main_Articles.json", "PMC_Main_Embeds.npy"),
    "cpg": ("CPG_Total_Articles.json", "CPG_Total_Embeds.npy"),
    "textbook": ("Textbook_Total_Articles.json", "Textbook_Total_Embeds.npy"),
}


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/smoke")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--passages", type=int, default=12)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    root = os.path.abspath(args.out)
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(0)

    for split, count in (("train", args.questions), ("validation", 4), ("test", args.questions)):
        records = [
            {
                "id": f"smoke-{split}-{i}",
                "question": f"Synthetic vignette {i} ({split}). Which is the best next step?",
                "options": {"A": f"option a{i}", "B": f"option b{i}", "C": f"option c{i}", "D": f"option d{i}"},
                "answer_idx": "ABCD"[i % 4],
            }
            for i in range(count)
        ]
        with open(os.path.join(root, f"{split}.json"), "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)

    for corpus, (articles_name, embeds_name) in CORPORA.items():
        articles_dir = os.path.join(root, "articles", corpus)
        embeddings_dir = os.path.join(root, "embeddings", corpus)
        os.makedirs(articles_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)

        passages = [
            {
                "id": f"{corpus}-doc{j}",
                "passage_id": f"{corpus}-doc{j}-p0",
                "title": f"{corpus.upper()} document {j}",
                "text": (
                    f"Synthetic {corpus} evidence passage {j}. It discusses clinical "
                    f"management considerations for the described presentation."
                ),
                # Provenance carried as metadata; the baseline never reads it.
                "publication_date": f"{1995 + j}-06-01",
                "journal": f"Journal of {corpus}",
            }
            for j in range(args.passages)
        ]
        with open(os.path.join(articles_dir, articles_name), "w", encoding="utf-8") as handle:
            json.dump(passages, handle, indent=2)

        embeddings = rng.normal(size=(args.passages, args.dim)).astype("float32")
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        np.save(os.path.join(embeddings_dir, embeds_name), embeddings)

    print(f"wrote smoke fixture -> {root}")
    print("NOTE: synthetic data for wiring checks only; it is not a benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
