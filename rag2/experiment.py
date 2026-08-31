"""Run manifests: everything needed to re-run or audit an experiment.

Written next to every run's outputs. Records the resolved config, the git state,
the model/checkpoint identities, the dataset identity and version, seeds, prompt
hashes, package versions, hardware and timings -- the reproducibility checklist
of docs/rag2_reproduction.md section 10.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from .config import Config

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "numpy",
    "faiss",
    "sentencepiece",
    "vllm",
    "spacy",
    "scispacy",
)


def git_state(repo_dir: str = ".") -> Dict[str, Any]:
    def _run(*args: str) -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_dir, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    except Exception:
        info["torch_cuda_available"] = None
        info["gpus"] = []
    return info


def set_all_seeds(seed: int) -> Dict[str, Any]:
    """Seed every RNG we can reach. Recorded in the manifest.

    The release left ``--seed`` unset, so exact reproduction of the authors'
    checkpoint is impossible in principle; ours is at least reproducible with
    itself.
    """
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    seeded = ["python.random"]
    try:
        import numpy as np

        np.random.seed(seed)
        seeded.append("numpy")
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        seeded.append("torch")
    except Exception:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
        seeded.append("transformers")
    except Exception:
        pass
    return {"seed": seed, "seeded": seeded}


def build_manifest(
    config: Config,
    stage: str,
    extra: Optional[Mapping[str, Any]] = None,
    repo_dir: str = ".",
) -> Dict[str, Any]:
    prompts = config.prompt_set()
    return {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment": config.experiment.name,
        "seed": config.experiment.seed,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "retrieval_fingerprint": config.retrieval_fingerprint(),
        "prompt_version": prompts.version,
        "prompt_fingerprint": prompts.fingerprint(),
        "git": git_state(repo_dir),
        "packages": package_versions(),
        "hardware": hardware(),
        "command": " ".join(sys.argv),
        **dict(extra or {}),
    }


def write_manifest(path: str, manifest: Mapping[str, Any]) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
    return path


def write_json(path: str, payload: Any) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def write_jsonl(path: str, rows: Any) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path
