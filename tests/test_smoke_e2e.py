"""End-to-end smoke coverage, run through the real entry points."""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("numpy")
pytest.importorskip("yaml")


def _run(*args, cwd=REPO_ROOT):
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"{args}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result.stdout


def test_smoke_script_passes():
    """scripts/smoke_test.py exercises every stage with in-memory fixtures."""
    output = _run(os.path.join("scripts", "smoke_test.py"))
    assert "SMOKE TEST PASSED" in output
    assert "FAIL" not in output


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """Drive the actual stage scripts over the synthetic fixture."""
    workdir = tmp_path_factory.mktemp("cli")
    data_dir = workdir / "data"
    scripts = os.path.join(REPO_ROOT, "scripts")

    _run(os.path.join(scripts, "make_smoke_fixture.py"), "--out", str(data_dir))

    overrides = [
        "-o", f"dataset.path={data_dir}",
        "-o", f"experiment.output_dir={workdir / 'runs'}/{{name}}",
        "-o", f"cache.dir={workdir / 'cache'}",
    ]
    # The fixture writes one shard per corpus, so PubMed and PMC need explicit
    # filenames rather than the released 38-shard / two-file layout.
    layout = {
        "pubmed": (["PubMed_Articles_0.json"], ["PubMed_Embeds_0.npy"]),
        "pmc": (["PMC_Main_Articles.json"], ["PMC_Main_Embeds.npy"]),
        "cpg": (["CPG_Total_Articles.json"], ["CPG_Total_Embeds.npy"]),
        "textbook": (["Textbook_Total_Articles.json"], ["Textbook_Total_Embeds.npy"]),
    }
    corpora = [
        {
            "name": name,
            "loader": "json_dir",
            "articles_dir": str(data_dir / "articles" / name),
            "embeddings_dir": str(data_dir / "embeddings" / name),
            "articles": articles,
            "embeddings": embeddings,
        }
        for name, (articles, embeddings) in layout.items()
    ]
    overrides += ["-o", f"retrieval.corpora={json.dumps(corpora)}"]

    config = os.path.join("configs", "smoke.yaml")
    _run(os.path.join(scripts, "02_retrieve.py"), "-c", config, *overrides)

    caches = sorted((workdir / "cache").glob("*.jsonl"))
    assert len(caches) == 1, caches
    _run(
        os.path.join(scripts, "05_run_pipeline.py"),
        "-c", config, *overrides, "--candidates", str(caches[0]),
    )
    return workdir, config, overrides, str(caches[0])


def test_cli_writes_a_cache_with_a_verifiable_sidecar(cli_run):
    workdir, _, _, cache = cli_run
    with open(os.path.splitext(cache)[0] + ".meta.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["num_questions"] == 8
    assert metadata["retrieval_fingerprint"]
    assert metadata["extra"]["corpora"] == ["pubmed", "pmc", "cpg", "textbook"]


def test_cli_retrieval_is_balanced_across_corpora(cli_run):
    workdir, _, _, cache = cli_run
    with open(cache, "r", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    counts = {}
    for candidate in first["candidates"]:
        counts[candidate["source"]] = counts.get(candidate["source"], 0) + 1
    assert counts == {"pubmed": 3, "pmc": 3, "cpg": 3, "textbook": 3}


def test_cli_writes_predictions_and_metrics(cli_run):
    workdir, _, _, _ = cli_run
    run_dir = workdir / "runs" / "smoke"
    with open(run_dir / "predictions.jsonl", "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert len(rows) == 8
    assert all(row["generation"] for row in rows)
    assert all(row["prediction"] in list("ABCD") for row in rows)

    with open(run_dir / "metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert 0.0 <= metrics["accuracy"]["accuracy"] <= 100.0
    assert metrics["evidence"]["num_candidates_total"] == 96


def test_cli_writes_a_manifest_with_full_provenance(cli_run):
    workdir, _, _, _ = cli_run
    with open(workdir / "runs" / "smoke" / "manifest.pipeline.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key in (
        "config", "config_fingerprint", "retrieval_fingerprint", "prompt_fingerprint",
        "prompt_version", "seed", "git", "packages", "hardware", "command", "created_at",
    ):
        assert key in manifest, key
    assert manifest["seed"] == 42
    assert manifest["seeding"]["seed"] == 42


def test_cli_refuses_a_cache_built_under_a_different_retrieval_config(cli_run):
    workdir, config, overrides, cache = cli_run
    result = subprocess.run(
        [
            sys.executable, os.path.join("scripts", "05_run_pipeline.py"),
            "-c", config, *overrides,
            "-o", "retrieval.candidates_per_corpus=5",
            "--candidates", cache,
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode != 0
    assert "CandidateCacheError" in result.stderr


def test_cli_builds_filter_labels_in_the_release_schema(cli_run):
    workdir, config, overrides, _ = cli_run
    scripts = os.path.join(REPO_ROOT, "scripts")
    train_overrides = overrides + ["-o", "dataset.split=train"]
    _run(os.path.join(scripts, "02_retrieve.py"), "-c", config, *train_overrides)

    caches = sorted(p for p in (workdir / "cache").glob("*.jsonl") if ".train." in p.name)
    assert caches
    _run(
        os.path.join(scripts, "03_build_filter_labels.py"),
        "-c", config, *train_overrides, "--candidates", str(caches[-1]),
    )

    run_dir = workdir / "runs" / "smoke"
    with open(run_dir / "filter_train.json", "r", encoding="utf-8") as handle:
        records = json.load(handle)
    assert records
    for record in records:
        assert set(record) == {"id", "answer", "dataset_name", "question"}
        assert record["answer"] in ("[HELPFUL]", "[NOT_HELPFUL]")

    with open(run_dir / "filter_label_stats.json", "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    assert stats["tau_percentile"] == 25.0
    assert stats["tau_scope"] == "global"
    assert stats["num_observations"] == 24  # 8 questions x label_top_k 3

    # Provenance is written beside the training data, never inside it.
    with open(run_dir / "filter_train.provenance.json", "r", encoding="utf-8") as handle:
        provenance = json.load(handle)
    assert len(provenance) == len(records)
    assert {"delta_ppl", "tau", "source", "doc_id"} <= set(provenance[0])


def test_cli_emits_the_paper_comparison(cli_run):
    workdir, _, _, _ = cli_run
    output = _run(
        os.path.join(REPO_ROOT, "scripts", "06_evaluate.py"),
        "--predictions", str(workdir / "runs" / "smoke" / "predictions.jsonl"),
        "--paper", "llama3:medqa",
    )
    assert "paper llama3:medqa: no-RAG 57.7 / RAG2 64.6" in output
    assert "not tuned away" in output
