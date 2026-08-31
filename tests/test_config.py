"""Configuration loading, inheritance and provenance."""

import glob
import os

import pytest

from rag2.config import Config, load_config, merge_overrides, parse_override

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")


def test_defaults_match_the_papers_stated_hyperparameters():
    config = Config()
    # Appendix A.3 and run/run_large_train_xl_000.sh.
    assert config.filter_training.learning_rate == pytest.approx(3e-5)
    assert config.filter_training.num_train_epochs == 40
    assert config.filter_training.per_device_train_batch_size == 16
    assert config.filter_training.max_seq_length == 512
    assert config.filter_training.doc_stride == 128
    # Paper section 3.2.
    assert config.filter_training.tau_percentile == 25.0
    # Paper section 4.2 / footnote 2.
    assert config.filter.base_model == "google/flan-t5-large"
    assert config.llm.model == "meta-llama/Meta-Llama-3-8B-Instruct"
    # retriever/*.py.
    assert config.retrieval.query_encoder == "ncbi/MedCPT-Query-Encoder"
    assert config.retrieval.reranker == "ncbi/MedCPT-Cross-Encoder"
    assert config.retrieval.embedding_dim == 768
    assert config.retrieval.query_max_length == 512
    assert config.retrieval.rerank_max_length == 512
    # Appendix A.3: greedy decoding at temperature 0.
    assert config.llm.temperature == 0.0
    assert config.generation.temperature == 0.0


def test_defaults_follow_the_paper_where_the_code_disagrees():
    config = Config()
    # Paper says the reranker cross-encodes the initial query (Figure 1, 3.4);
    # retriever/main.py passes the rationale.
    assert config.retrieval.rerank_query == "initial"
    # The release concatenates per-shard PubMed top-k, breaking balance.
    assert config.retrieval.shard_merge == "score"


def test_every_shipped_config_loads():
    files = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    assert files, "no configs found"
    for path in files:
        config = load_config(path)
        assert isinstance(config, Config), path
        assert config.experiment.name, path


def test_base_chain_is_inherited_and_overridden():
    config = load_config(os.path.join(CONFIG_DIR, "medqa_llama3.yaml"))
    # from default.yaml
    assert config.retrieval.query_encoder == "ncbi/MedCPT-Query-Encoder"
    # from corpora.example.yaml
    assert [c.name for c in config.retrieval.corpora] == ["pubmed", "pmc", "cpg", "textbook"]
    # from the file itself: Figure 3 puts Llama-3-8B's MedQA peak at k=32
    assert config.retrieval.final_top_k == 32


def test_ablation_configs_encode_the_documented_variants():
    no_filter = load_config(os.path.join(CONFIG_DIR, "ablation_no_filter.yaml"))
    assert no_filter.filter.kind == "passthrough"

    release = load_config(os.path.join(CONFIG_DIR, "ablation_release_retrieval.yaml"))
    assert release.retrieval.rerank_query == "rationale"
    assert release.retrieval.shard_merge == "concat"


def test_unknown_keys_are_rejected_rather_than_silently_ignored():
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(None, {"retrieval": {"top_k": 5}})
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(None, {"nonsense": {}})


def test_cli_overrides_are_parsed_and_typed():
    assert parse_override("retrieval.final_top_k=32") == {"retrieval": {"final_top_k": 32}}
    assert parse_override("filter.on_empty=keep_top1") == {"filter": {"on_empty": "keep_top1"}}
    assert parse_override("cache.enabled=false") == {"cache": {"enabled": False}}
    assert parse_override("filter_training.tau_percentile=12.5")["filter_training"][
        "tau_percentile"
    ] == pytest.approx(12.5)


def test_override_requires_a_value():
    with pytest.raises(ValueError, match="key=value"):
        parse_override("retrieval.final_top_k")


def test_overrides_merge_without_clobbering_siblings():
    merged = merge_overrides(["retrieval.final_top_k=4", "retrieval.rerank_query=rationale"])
    config = load_config(None, merged)
    assert config.retrieval.final_top_k == 4
    assert config.retrieval.rerank_query == "rationale"
    assert config.retrieval.query_encoder == "ncbi/MedCPT-Query-Encoder"


def test_fingerprint_changes_with_the_config():
    base = load_config(None, {})
    assert load_config(None, {}).fingerprint() == base.fingerprint()
    assert load_config(None, {"filter": {"on_empty": "keep_top1"}}).fingerprint() != base.fingerprint()


def test_output_dir_is_templated_on_the_experiment_name():
    config = load_config(None, {"experiment": {"name": "abc"}})
    assert config.resolved_output_dir() == "runs/abc"


def test_corpora_entries_become_typed_objects():
    config = load_config(None, {"retrieval": {"corpora": [{"name": "cpg", "articles_dir": "a"}]}})
    corpus = config.retrieval.corpora[0]
    assert corpus.name == "cpg"
    assert corpus.articles_dir == "a"
    assert corpus.loader == "json_dir"


def test_circular_base_chain_is_detected(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("base: b.yaml\n", encoding="utf-8")
    b.write_text("base: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(str(a))
