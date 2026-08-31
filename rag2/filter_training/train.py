"""Training the Flan-T5 filter.

The authors' own training script -- ``classifier/run_classifier.py``, kept
unmodified in this repository -- is what actually trains the model, so the
training loop, the seq2seq cross-entropy objective and the preprocessing are
theirs, not a re-implementation. This module only

* adds ``[HELPFUL]`` / ``[NOT_HELPFUL]`` to a base Flan-T5 checkpoint (replacing
  ``classifier/model/token_add.ipynb``, which loads a T5 with
  ``AutoModelForCausalLM`` -- wrong class for a seq2seq model), and
* builds the argv for ``run_classifier.py`` from the config, so the paper's
  hyperparameters (lr 3e-5, 40 epochs, batch 16, max_seq_length 512,
  doc_stride 128 -- appendix A.3 and ``run/run_large_train_xl_000.sh``) come from
  one place instead of a shell script.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

from ..config import FilterConfig, FilterTrainingConfig
from ..prompts import LABEL_HELPFUL, LABEL_NOT_HELPFUL

CLASSIFIER_SCRIPT = os.path.join("classifier", "run_classifier.py")


def add_label_tokens(base_model: str, output_dir: str) -> Dict[str, Any]:
    """Extend a Flan-T5 checkpoint with the two label tokens and save it.

    Replaces ``classifier/model/token_add.ipynb``. Uses ``AutoModelForSeq2SeqLM``,
    which is what ``classifier/utils.py`` loads at training time.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    new_tokens = [LABEL_HELPFUL, LABEL_NOT_HELPFUL]
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in new_tokens}
    if len(set(ids.values())) != len(new_tokens):
        raise RuntimeError(f"label tokens did not get distinct ids: {ids}")
    return {
        "base_model": base_model,
        "output_dir": output_dir,
        "num_added_tokens": num_added,
        "vocab_size": len(tokenizer),
        "label_token_ids": ids,
    }


def write_training_file(path: str, records: Sequence[Any]) -> str:
    """Write the four-field training JSON the release's script reads."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = [r.to_training_record() if hasattr(r, "to_training_record") else dict(r) for r in records]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def build_train_command(
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    training: FilterTrainingConfig,
    seed: Optional[int] = None,
    validation_file: str = "",
    python_executable: Optional[str] = None,
    script: str = CLASSIFIER_SCRIPT,
) -> List[str]:
    """argv for ``classifier/run_classifier.py`` in training mode."""
    command = [
        python_executable or sys.executable,
        script,
        "--model_name_or_path", model_name_or_path,
        "--train_file", train_file,
        "--question_column", "question",
        "--answer_column", "answer",
        "--do_train",
        "--train_column", "train",
        "--checkpointing_steps", training.checkpointing_steps,
        "--learning_rate", str(training.learning_rate),
        "--max_seq_length", str(training.max_seq_length),
        "--doc_stride", str(training.doc_stride),
        "--max_answer_length", str(training.max_answer_length),
        "--per_device_train_batch_size", str(training.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(training.gradient_accumulation_steps),
        "--weight_decay", str(training.weight_decay),
        "--lr_scheduler_type", training.lr_scheduler_type,
        "--num_warmup_steps", str(training.num_warmup_steps),
        "--num_train_epochs", str(training.num_train_epochs),
        "--output_dir", output_dir,
        "--overwrite_cache",
    ]
    if validation_file:
        command += ["--validation_file", validation_file, "--val_column", "validation"]
    # The release leaves --seed unset (None). We always seed and record it.
    if seed is not None:
        command += ["--seed", str(seed)]
    return command


def build_eval_command(
    model_name_or_path: str,
    validation_file: str,
    output_dir: str,
    training: FilterTrainingConfig,
    python_executable: Optional[str] = None,
    script: str = CLASSIFIER_SCRIPT,
) -> List[str]:
    """argv for ``classifier/run_classifier.py`` in evaluation mode.

    Note (docs/rag2_reproduction.md section 5.7): this path emits one prediction
    per overflow *feature* while zipping against per-*example* ids, so its
    accuracy is unreliable whenever an input exceeds ``max_seq_length``. Prefer
    ``rag2.filtering.rag2_filter`` plus ``rag2.evaluation.filter_metrics`` for
    reported filter accuracy; this command exists to reproduce the release.
    """
    return [
        python_executable or sys.executable,
        script,
        "--model_name_or_path", model_name_or_path,
        "--validation_file", validation_file,
        "--question_column", "question",
        "--answer_column", "answer",
        "--do_eval",
        "--val_column", "validation",
        "--max_seq_length", str(training.max_seq_length),
        "--doc_stride", str(training.doc_stride),
        "--max_answer_length", str(training.max_answer_length),
        "--per_device_eval_batch_size", str(training.per_device_eval_batch_size),
        "--output_dir", output_dir,
    ]


def run_command(command: Sequence[str], cwd: Optional[str] = None) -> int:
    """Run a built command, streaming its output."""
    print("+ " + " ".join(command), flush=True)
    return subprocess.call(list(command), cwd=cwd)


def evaluate_filter_checkpoint(
    checkpoint: str,
    records: Sequence[Dict[str, str]],
    filter_config: FilterConfig,
) -> Dict[str, Any]:
    """Score a validation file with the reproduction's own inference path.

    Used for checkpoint selection (``filter_training.select_by: val_accuracy``),
    which the paper describes only as "we selected a few candidate models from
    the validation set".
    """
    from ..evaluation import filter_metrics
    from ..filtering.rag2_filter import RAG2PerplexityFilter

    config = FilterConfig(**{**filter_config.__dict__, "checkpoint": checkpoint})
    model = RAG2PerplexityFilter(config)
    probabilities = model.score_pairs([r["question"] for r in records])
    predictions = [LABEL_HELPFUL if p >= 0.5 else LABEL_NOT_HELPFUL for p in probabilities]
    metrics = filter_metrics([r["answer"] for r in records], predictions)
    metrics["checkpoint"] = checkpoint
    return metrics
