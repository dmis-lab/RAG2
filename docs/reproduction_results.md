# Reproduction results

Record of what the reproduced baseline actually scores, and why it differs from
the paper. **Numbers go in here as measured.** Per the task brief, the baseline
is not adjusted to close a gap with the paper; a gap is a finding to explain, not
a bug to tune out.

Status: **not yet run.** The medical dataset and its MedCPT-embedded corpus are
being prepared separately (see `docs/rag2_reproduction.md` section 4.1). The
pipeline, the filter and the evaluation are implemented and pass their tests;
what is missing is the data.

---

## How to fill this in

```bash
# 1. rationales + balanced retrieval + reranking, cached once
python scripts/02_retrieve.py -c configs/medqa_llama3.yaml
python scripts/02_retrieve.py -c configs/medqa_llama3.yaml -o dataset.split=train

# 2. perplexity labels (Figure 2) on the training split, then the filter
python scripts/03_build_filter_labels.py -c configs/medqa_llama3.yaml \
    -o dataset.split=train --candidates cache/candidates/<train cache>.jsonl
python scripts/04_train_filter.py -c configs/medqa_llama3.yaml --init-tokens \
    --token-dir runs/filter-base
python scripts/04_train_filter.py -c configs/medqa_llama3.yaml \
    --model runs/filter-base \
    --train-file runs/medqa-llama3/filter_train.json \
    --validation-file runs/medqa-llama3/filter_val.json \
    --filter-output-dir runs/filter-medqa-llama3 --select

# 3. filter + generate, then score
python scripts/05_run_pipeline.py -c configs/medqa_llama3.yaml \
    --candidates cache/candidates/<test cache>.jsonl
python scripts/06_evaluate.py \
    --predictions runs/medqa-llama3/predictions.jsonl --paper llama3:medqa
```

Every run writes `manifest.<stage>.json` next to its outputs, carrying the
resolved config, git commit, model revisions, dataset version, seeds, prompt
hashes and package versions. **Cite the manifest fingerprint for every number
recorded below**, so a figure can be traced back to the exact configuration that
produced it.

---

## Main results

Accuracy (%). Paper column is Table 2. Fill the reproduction column as runs
complete; leave a row blank rather than estimating it.

| Backbone | Benchmark | k | Paper (no RAG) | Paper (+ RAG²) | Reproduced | Δ | Manifest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3-8B-Instruct | MedQA | 32 | 57.7 | 64.6 | | | |
| Llama-3-8B-Instruct | MedMCQA | 16 | 53.5 | 59.4 | | | |
| Llama-3-8B-Instruct | MMLU-Med | ? | 69.5 | 74.8 | | | |
| Meerkat-7B | MedQA | 2 | 71.2 | 75.6 | | | |
| Meerkat-7B | MedMCQA | 8 | 60.8 | 63.0 | | | |
| Meerkat-7B | MMLU-Med | ? | 73.8 | 78.7 | | | |

`k` is the final top-k the paper's Figure 3 shows each number at; `?` marks a
value not recoverable from the published figures, which must be re-selected on
validation.

Record the no-RAG baseline too (`configs/*.yaml` with
`-o filter.kind=no_evidence`): the paper's claim is an *improvement over* the
backbone, so a reproduction that misses both numbers by the same margin is a
different finding from one that misses only the RAG² number.

## Ablations

| Ablation | Config | Paper | Reproduced | Notes |
| --- | --- | --- | --- | --- |
| RAG² w/o filter (Figure 3) | `ablation_no_filter.yaml` | 63.4 @ k=32, MedQA/Llama-3 | | |
| Balanced retrieval, top-1, no filter (Table 4) | `-o retrieval.final_top_k=1 -o filter.kind=passthrough` | 55.3 MedQA / 51.3 MedMCQA (Llama-3) | | |
| Filtering at top-1 with the initial query (Table 3) | `-o retrieval.final_top_k=1` | 58.6 MedQA / 55.8 MedMCQA (Llama-3) | | |
| Released retrieval behaviour | `ablation_release_retrieval.yaml` | not reported | | measures the two code/paper discrepancies |

## Filter-level results

The paper does not report filter accuracy, so there is no target — but record it,
because a filter that keeps almost everything or almost nothing explains a
downstream gap immediately.

| Split | τ | Label counts (H / NH / discard) | Filter accuracy | Per-class (H / NH) | Keep rate at k |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Discrepancies observed

For each gap, name which item from `docs/rag2_reproduction.md` section 11 it is
attributable to, what was tested, and what the evidence was. Ordered as that
section orders them:

1. **Corpus.** _(expected dominant term: the reproduction runs on a different
   corpus than the paper's 564 GB Self-BioRAG corpus)_
2. **Reconstructed filter checkpoint.**
3. **Answer-generation prompt / answer extraction.**
4. **Rerank query (initial vs. rationale).**
5. **τ population (global vs. per-question); the unexplained "5%" artifact.**
6. **Model versions.**
7. **Decoding nondeterminism.**
8. **PubMed shard balance.**

## Resources that stayed unavailable

Carry forward anything from `docs/rag2_reproduction.md` section 8 that could not
be reconstructed, so the limitations section of the thesis can cite it directly.
