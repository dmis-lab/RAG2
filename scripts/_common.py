"""Shared CLI plumbing for the stage scripts."""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag2.config import Config, load_config, merge_overrides  # noqa: E402
from rag2.experiment import build_manifest, set_all_seeds, write_manifest  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("-c", "--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. -o retrieval.final_top_k=32 (repeatable)",
    )
    parser.add_argument("--output-dir", default="", help="override experiment.output_dir")
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    config = load_config(args.config, merge_overrides(args.override))
    if getattr(args, "output_dir", ""):
        config.experiment.output_dir = args.output_dir
    return config


def prepare_run(config: Config, stage: str) -> str:
    """Seed, create the output directory, write the manifest; return the directory."""
    output_dir = config.resolved_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    seed_info = set_all_seeds(config.experiment.seed)
    manifest = build_manifest(config, stage, extra={"seeding": seed_info}, repo_dir=REPO_ROOT)
    write_manifest(os.path.join(output_dir, f"manifest.{stage}.json"), manifest)
    return output_dir


def progress_printer(label: str):
    def report(done, total=None):
        if total is None:
            print(f"[{label}] {done}", flush=True)
        else:
            print(f"[{label}] {done}/{total}", flush=True)

    return report
