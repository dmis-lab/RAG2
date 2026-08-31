"""Typed experiment configuration for the RAG2 reproduction.

All tunables live in YAML under ``configs/`` and are parsed into these
dataclasses; nothing is hard-coded at a call site. Every field records whether
the paper specifies it ([S]) or whether it is a documented assumption ([A]) --
see docs/rag2_reproduction.md for the full accounting.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, get_type_hints

try:  # PyYAML is in environment.yml; fall back to JSON-only configs without it.
    import yaml
except ImportError:  # pragma: no cover - exercised only on minimal installs
    yaml = None

from .schema import stable_hash

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "rag2-baseline"
    seed: int = 42  # [A] the release left --seed unset (None); we always seed
    output_dir: str = "runs/{name}"
    notes: str = ""


@dataclass
class DatasetConfig:
    """Which QA dataset to run. The medical dataset plugs in here.

    ``loader`` names an entry in the rag2.datasets registry. ``path`` and
    ``split_map`` are loader-specific.
    """

    loader: str = "jsonl"  # "medqa" | "medmcqa" | "mmlu_med" | "jsonl" | "inline"
    name: str = ""
    version: str = ""  # [A] recorded in the manifest; set it for every real run
    path: str = ""
    split: str = "test"
    split_map: Dict[str, str] = field(default_factory=dict)
    limit: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusConfig:
    """One evidence corpus in the balanced-retrieval set."""

    name: str = ""
    loader: str = "json_dir"  # "json_dir" (the release layout) | "jsonl" | "inline"
    articles: List[str] = field(default_factory=list)
    embeddings: List[str] = field(default_factory=list)
    articles_dir: str = ""
    embeddings_dir: str = ""
    # [A] the paper says "sliding window with overlap" but gives no sizes
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    # [S] ncbi/MedCPT-* checkpoints, paper section 3.4 + retriever/*.py
    query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    article_encoder: str = "ncbi/MedCPT-Article-Encoder"
    reranker: str = "ncbi/MedCPT-Cross-Encoder"
    embedding_dim: int = 768  # [S] retriever/retrieve.py
    query_max_length: int = 512  # [S] retriever/query_encode.py
    rerank_max_length: int = 512  # [S] retriever/rerank.py
    # [A] per-corpus retrieval depth is never stated; 100 is the release default
    candidates_per_corpus: int = 100
    # [S] final k is swept over {1,2,4,8,16,32} and selected on validation
    final_top_k: int = 8
    # [S] balanced retrieval: equal candidates from every corpus
    corpora: List[CorpusConfig] = field(default_factory=list)
    # [D] paper says rerank with the INITIAL query; the release code uses the
    #     rationale. Default follows the paper.
    rerank_query: str = "initial"  # "initial" | "rationale"
    # [D] sharding PubMed and concatenating per-shard top-k breaks balance;
    #     "score" merges shards and keeps exactly candidates_per_corpus.
    shard_merge: str = "score"  # "score" | "concat"
    shard_size: int = 0  # 0 = single index per corpus (no sharding)
    # [A] SciSpacy [SEP] insertion is off in the release
    use_scispacy_sep: bool = False
    scispacy_model: str = "en_core_sci_scibert"
    encode_batch_size: int = 32
    rerank_batch_size: int = 32
    device: str = "auto"


@dataclass
class LLMConfig:
    """The backbone LLM: rationale generation, answer generation, and (during
    filter-label construction) perplexity scoring. The paper uses one model for
    all three (section 3.3)."""

    backend: str = "huggingface"  # "huggingface" | "vllm" | "openai" | "stub"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"  # [S] paper footnote 2
    revision: str = ""  # pinned per run; recorded in the manifest
    dtype: str = "bfloat16"
    device: str = "auto"
    max_new_tokens: int = 512  # [A] not stated by the paper
    temperature: float = 0.0  # [S] appendix A.3 "greedy decoding ... temperature 0"
    top_p: float = 1.0
    max_input_tokens: int = 0  # 0 = model default
    chat_template: bool = True  # instruct models are prompted through their template
    batch_size: int = 8
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Inference-time configuration of the original RAG2 filter."""

    kind: str = "rag2_perplexity"  # registry key; "passthrough" = RAG2 w/o filter
    base_model: str = "google/flan-t5-large"  # [S] paper section 4.2 (770M)
    checkpoint: str = ""  # trained filter; empty = must be trained first
    max_seq_length: int = 512  # [S] run/run_large_train_xl_000.sh
    doc_stride: int = 128  # [S] same
    # [D] the release's eval path desynchronises features and examples when an
    #     input overflows; "truncate" scores one window per pair.
    overflow: str = "truncate"  # "truncate" | "stride"
    batch_size: int = 32
    device: str = "auto"
    # [A] the paper is silent on the all-filtered-out case
    on_empty: str = "no_evidence"  # "no_evidence" | "keep_top1"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterTrainingConfig:
    """Construction of the perplexity-based training labels, and training."""

    # -- labeling (paper section 3.2, Figure 2) --
    tau_percentile: float = 25.0  # [S] "top 25% of perplexity differentials"
    tau_scope: str = "global"  # [A] "global" | "per_question"
    ppl_target: str = "rationale"  # [A] prose says rationale; Eq. 4 literally says query
    ppl_rationale: str = "no_retrieval"  # [A] score the same string in both terms
    label_top_k: int = 10  # [A] snippets per question sent through labeling
    drop_undecided: bool = True  # [S] Figure 2 has explicit [Discard] leaves
    # -- training (paper appendix A.3 + run/run_large_train_xl_000.sh) --
    learning_rate: float = 3e-5  # [S]
    num_train_epochs: int = 40  # [S]
    per_device_train_batch_size: int = 16  # [S]
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1  # [S] script default
    weight_decay: float = 0.0  # [S] script default
    lr_scheduler_type: str = "linear"  # [S] script default
    num_warmup_steps: int = 0  # [S] script default
    max_seq_length: int = 512  # [S]
    doc_stride: int = 128  # [S]
    max_answer_length: int = 30  # [S] script default
    checkpointing_steps: str = "epoch"  # [S]
    select_by: str = "val_accuracy"  # [A] "a few candidate models from the validation set"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512  # [A]
    temperature: float = 0.0  # [S]
    stop: List[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    metric: str = "accuracy"  # [S] Table 2
    # [A] answer extraction is never specified; patterns are ordered, last match wins
    extraction_patterns: List[str] = field(default_factory=list)
    unparsed_as_incorrect: bool = True  # [A]
    open_ended_metrics: List[str] = field(default_factory=list)  # rouge_l | bertscore
    bertscore_model: str = "roberta-large"  # [A] bert-score package default


@dataclass
class CacheConfig:
    dir: str = "cache/candidates"
    enabled: bool = True
    allow_config_mismatch: bool = False


@dataclass
class PromptConfig:
    """Overrides for rag2.prompts.PromptSet. Empty string = use the constant."""

    rationale: str = ""
    filter_input: str = ""
    answer_with_evidence: str = ""
    answer_no_evidence: str = ""
    evidence_item: str = ""
    evidence_join: str = ""
    option_format: str = ""


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    filter_training: FilterTrainingConfig = field(default_factory=FilterTrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)

    # -- helpers -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def retrieval_fingerprint(self) -> str:
        """Hash of everything that determines the cached candidate set.

        Deliberately excludes ``final_top_k`` (a cache built at depth 32 can
        serve a k=8 experiment) and everything downstream of retrieval.
        """
        payload = asdict(self.retrieval)
        payload.pop("final_top_k", None)
        payload.pop("device", None)
        payload.pop("encode_batch_size", None)
        payload.pop("rerank_batch_size", None)
        return stable_hash(
            {
                "retrieval": payload,
                "dataset": {
                    "loader": self.dataset.loader,
                    "name": self.dataset.name,
                    "version": self.dataset.version,
                    "split": self.dataset.split,
                },
                "rationale_llm": {
                    "backend": self.llm.backend,
                    "model": self.llm.model,
                    "revision": self.llm.revision,
                    "temperature": self.llm.temperature,
                    "max_new_tokens": self.llm.max_new_tokens,
                },
                "prompt_version": self.prompts.rationale or "default",
            }
        )

    def resolved_output_dir(self) -> str:
        return self.experiment.output_dir.format(name=self.experiment.name)

    def prompt_set(self):
        from .prompts import PromptSet

        base = PromptSet()
        overrides = {
            k: v for k, v in asdict(self.prompts).items() if isinstance(v, str) and v
        }
        return PromptSet(**{**{f.name: getattr(base, f.name) for f in fields(base)}, **overrides})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolved_hints(cls: type) -> Dict[str, Any]:
    """Field name -> resolved type. Needed because ``from __future__ import
    annotations`` leaves ``Field.type`` as a string."""
    try:
        return get_type_hints(cls)
    except Exception:  # pragma: no cover - defensive
        return {}


def _from_mapping(cls: Type[T], payload: Mapping[str, Any], path: str = "") -> T:
    """Build a (possibly nested) dataclass from a mapping, rejecting unknown keys."""
    if not is_dataclass(cls):
        return payload  # type: ignore[return-value]
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"unknown config key(s) under {where}: {sorted(unknown)}")
    hints = _resolved_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, value in payload.items():
        sub_path = f"{path}.{name}" if path else name
        hint = hints.get(name)
        if is_dataclass(hint) and isinstance(value, Mapping):
            kwargs[name] = _from_mapping(hint, value, sub_path)
        elif name == "corpora" and isinstance(value, list):
            kwargs[name] = [
                _from_mapping(CorpusConfig, c, f"{sub_path}[]") if isinstance(c, Mapping) else c
                for c in value
            ]
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[arg-type]


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs (pip install pyyaml)")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(
    path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None
) -> Config:
    """Load a config file, following any ``base:`` chain, then apply overrides.

    ``base`` is resolved relative to the including file, so
    ``configs/medqa_llama3.yaml`` can say ``base: default.yaml``.
    """
    payload: Dict[str, Any] = {}
    if path:
        payload = _load_with_base(path, seen=set())
    if overrides:
        payload = _deep_merge(payload, overrides)
    return _from_mapping(Config, payload)


def _load_with_base(path: str, seen: set) -> Dict[str, Any]:
    real = os.path.realpath(path)
    if real in seen:
        raise ValueError(f"circular config base chain at {path}")
    seen.add(real)
    payload = load_yaml(path)
    base = payload.pop("base", None)
    if not base:
        return payload
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(real), base)
    return _deep_merge(_load_with_base(base_path, seen), payload)


def parse_override(text: str) -> Dict[str, Any]:
    """Parse a ``section.key=value`` CLI override into a nested dict."""
    if "=" not in text:
        raise ValueError(f"override must be key=value, got {text!r}")
    key, _, raw = text.partition("=")
    if yaml is not None:
        value = yaml.safe_load(raw)
    else:  # pragma: no cover
        value = raw
    node: Dict[str, Any] = {}
    cursor = node
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return node


def merge_overrides(items: List[str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        merged = _deep_merge(merged, parse_override(item))
    return merged
