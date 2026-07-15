# [NAACL 2025] RAG²: Rationale-Guided Retrieval-Augmented Generation for Medical Question Answering

[![Paper (NAACL 2025)](https://img.shields.io/badge/Paper-NAACL%202025-b31b1b.svg)](https://aclanthology.org/2025.naacl-long.635/)
[![arXiv](https://img.shields.io/badge/arXiv-2411.00300-b31b1b.svg)](https://arxiv.org/abs/2411.00300)

**Paper** | [Rationale-Guided Retrieval Augmented Generation for Medical Question Answering](https://aclanthology.org/2025.naacl-long.635/)

**Authors:** Jiwoong Sohn\*, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim†, Jaewoo Kang†

---

## Abstract

Large language models (LLM) hold significant potential for applications in biomedicine, but they struggle with hallucinations and outdated knowledge. While retrieval-augmented generation (RAG) is generally employed to address these issues, it also has its own set of challenges: (1) LLMs are vulnerable to irrelevant or incorrect context, (2) medical queries are often not well-targeted for helpful information, and (3) retrievers are prone to bias toward the specific source corpus they were trained on. In this study, we present RAG² (RAtionale-Guided RAG), a new framework for enhancing the reliability of RAG in biomedical contexts. RAG² incorporates three key innovations: a small filtering model trained on perplexity-based labels of rationales, which selectively augments informative snippets of documents while filtering out distractors; LLM-generated rationales as queries to improve the utility of retrieved snippets; a structure designed to retrieve snippets evenly from a comprehensive set of four biomedical corpora, effectively mitigating retriever bias. Our experiments demonstrate that RAG² improves the state-of-the-art LLMs of varying sizes, with improvements of up to 6.1%, and it outperforms the previous best medical RAG model by up to 5.6% across three medical question-answering benchmarks.

## Pipeline

The method runs in four stages; each stage's output is the next stage's input:

    question
      -> [1] rationale generation (LLM): produce a chain-of-thought rationale; use it as the retrieval query
      -> [2] retrieval + reranking (MedCPT + FAISS): retrieve snippets evenly from the 4 corpora, rerank to top-k
      -> [3] filtering (Flan-T5): label each (question, snippet) [HELPFUL] or [NOT_HELPFUL]; keep only [HELPFUL]
      -> [4] answer generation (LLM): answer the question conditioned on the filtered evidence

## What's included and what's not

**This repository is not a full, one-command reproduction of the paper.** The
biomedical corpora we used cannot be redistributed (size and licensing), and the
exact filtering-model checkpoint is no longer available. What we release is the
**code and representative artifacts**, so the method and data formats are fully
specified and you can rerun the pipeline with your own corpora/model.

**Included:**

- The retriever pipeline (`retriever/`): MedCPT query encoding, balanced
  multi-corpus MIPS, and cross-encoder reranking.
- The filtering-model code (`classifier/`): training/evaluation of the Flan-T5
  filter, the `[HELPFUL]`/`[NOT_HELPFUL]` token setup, and metrics.
- A **representative training-data artifact** in the exact schema we used:
  [`classifier/data/medqa/llama3_cot/5%-train.json`](classifier/data/medqa/llama3_cot/5%-train.json),
  a sample of the perplexity-labeled (question, evidence) pairs used to train the
  filter.

**Not included (and why):**

- **The full four biomedical corpora** (PubMed, PMC, CPG, Textbook) and their
  pre-computed embeddings. These are multi-GB and partly license-restricted, so
  we cannot host them. The retriever expects you to supply the per-corpus `.npy`
  embeddings (`retriever/embeddings/<corpus>/`) and article `.json` files
  (`retriever/articles/<corpus>/`); see [`retriever/README.md`](retriever/README.md)
  for the expected filenames. Retrieval and reranking use **MedCPT**
  (Jin et al., 2023).
- **The exact trained filtering-model checkpoint** from the paper. It is not
  available for distribution; use `classifier/` to train an equivalent filter on
  your own labeled data.

**Known rough edges** (this is a research release): a few scripts hard-code
GPU device indices (`cuda:7` in `retriever/query_encode.py`, device `2` in
`retriever/rerank.py`) and dataset paths; adjust them to your environment. These
are documented in the per-component READMEs.

## Repository structure

```
RAG2/
├── environment.yml            # Single conda env (rag2) for both components
├── retriever/                 # Balanced multi-corpus retrieval + reranking
│   ├── main.py                # Entry point: encode → MIPS over 4 corpora → rerank
│   ├── query_encode.py        # MedCPT-Query-Encoder; optional SciSpacy [SEP] insertion
│   ├── retrieve.py            # FAISS index build/search and decoding per corpus
│   ├── rerank.py              # MedCPT-Cross-Encoder reranking to final top-k
│   ├── embeddings/            # Pre-computed corpus embeddings (.npy) — see below
│   ├── articles/              # Corpus article text (.json) — see below
│   ├── input/                 # Query files (questions / rationales)
│   └── output/                # Retrieved + reranked evidence
├── classifier/                # Rationale-guided filtering model
│   ├── run_classifier.py      # Train / evaluate the Flan-T5 filtering model
│   ├── utils.py               # Model loading, preprocessing, metrics
│   ├── data/                  # Training/eval data (see classifier/README.md)
│   ├── model/token_add.ipynb  # Add [HELPFUL] / [NOT_HELPFUL] special tokens
│   └── run/                   # Example launch scripts
├── retriever/README.md        # Retriever setup & usage
└── classifier/README.md       # Filtering model setup & usage
```

## Installation

> Before you start: this is **not** a full reproduction — the corpora used in the
> paper are not redistributed due to size and licensing. See
> [What's included and what's not](#whats-included-and-whats-not) above for
> what you need to supply.

The retriever and the classifier share a single `rag2` environment defined in [`environment.yml`](environment.yml). It covers everything both components need: `pytorch`, `faiss` (GPU), `transformers`, `datasets`, `nltk`, `numpy`, `accelerate` (for filter training), and `scispacy` + `en_core_sci_scibert` (optional, for `[SEP]` insertion).

```bash
conda env create -f environment.yml
conda activate rag2
```

## Quick start

The full pipeline runs in four stages. Each stage produces a file consumed by the next.

### 1. Generate rationales (queries)

Use your LLM of choice (e.g. Llama-3) to generate a chain-of-thought rationale for each question. Store the rationales as the retrieval query file under `retriever/input/`. In this repo the query files follow the `*_llama_cot.json` naming convention (rationales generated by Llama-3).

### 2. Retrieve evidence

See **[`retriever/README.md`](retriever/README.md)** for corpus/embedding setup. Then:

```bash
cd retriever
python main.py \
  --input_path  input/medqa/medqa_llama_cot.json \
  --output_path output/medqa/evidence_medqa_llama_cot.json \
  --top_k 10
```

This encodes queries with MedCPT, runs balanced MIPS over the four corpora, and reranks the pooled candidates with the MedCPT cross-encoder to produce the final top-k evidence.

### 3. Filter evidence

Train (or reuse) the Flan-T5 filtering model that labels each (question, snippet) pair as `[HELPFUL]` or `[NOT_HELPFUL]`, then keep only the helpful snippets. See **[`classifier/README.md`](classifier/README.md)**.

### 4. Generate the answer

Prompt your LLM with the question plus the filtered evidence to produce the final answer.

## Data & corpora

RAG² retrieves from four biomedical corpora (PubMed, PMC, CPG, Textbook), pulling snippets evenly across sources to reduce single-corpus retriever bias. Per corpus you provide the **article text** (`.json`, under `retriever/articles/<corpus>/`) and the **pre-computed embeddings** (`.npy`, under `retriever/embeddings/<corpus>/`); expected filenames are in [`retriever/README.md`](retriever/README.md). Embeddings are produced with `ncbi/MedCPT-Article-Encoder` (**MedCPT**; Jin et al., 2023). As noted in [What's included and what's not](#whats-included-and-whats-not), the corpora themselves are not shipped here.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{sohn-etal-2025-rationale,
    title     = "Rationale-Guided Retrieval Augmented Generation for Medical Question Answering",
    author    = "Sohn, Jiwoong and Park, Yein and Yoon, Chanwoong and Park, Sihyeon and
                 Hwang, Hyeon and Sung, Mujeen and Kim, Hyunjae and Kang, Jaewoo",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Long Papers)",
    year      = "2025",
    url       = "https://aclanthology.org/2025.naacl-long.635/"
}
```

## Acknowledgements

This work builds on [MedCPT](https://github.com/ncbi/MedCPT) for retrieval and reranking. The filtering-model training code (`classifier/`) is adapted in part from [Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG).
