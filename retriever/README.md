# RAG² Retriever

Balanced, multi-corpus dense retrieval with reranking. This module encodes queries with
[MedCPT](https://github.com/ncbi/MedCPT), runs FAISS maximum-inner-product search (MIPS) over four
biomedical corpora, and reranks the pooled candidates with the MedCPT cross-encoder to produce a
final top-k evidence set per query.

## Contents

- `main.py` — orchestrates the full retrieve → rerank pipeline.
- `query_encode.py` — encodes queries with `ncbi/MedCPT-Query-Encoder`; optional SciSpacy `[SEP]` insertion.
- `retrieve.py` — builds a FAISS `IndexFlatIP` per corpus, searches, and decodes indices back to article text.
- `rerank.py` — reranks pooled candidates with `ncbi/MedCPT-Cross-Encoder`.

## Setup

### 1. Environment

```bash
conda env create -f ../environment.yml
conda activate rag2
```

This is the shared `rag2` environment used across the repo. `faiss-gpu` and CUDA-enabled PyTorch
are required. The optional `en_core_sci_scibert` SciSpacy model (installed via the environment
file) is only needed when `--use_spacy True`.

### 2. Corpus files

For each corpus, place the article text under `articles/<corpus>/` and the pre-computed embeddings
under `embeddings/<corpus>/`. Embeddings are 768-dim float vectors produced by
`ncbi/MedCPT-Article-Encoder`. Expected filenames:

| Corpus     | `embeddings/<corpus>/`                                   | `articles/<corpus>/`                                     |
| ---------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `pubmed`   | `PubMed_Embeds_0.npy` … `PubMed_Embeds_37.npy` (38 chunks) | `PubMed_Articles_0.json` … `PubMed_Articles_37.json`     |
| `pmc`      | `PMC_Main_Embeds.npy`, `PMC_Abs_Embeds.npy`              | `PMC_Main_Articles.json`, `PMC_Abs_Articles.json`        |
| `cpg`      | `CPG_Total_Embeds.npy`                                   | `CPG_Total_Articles.json`                                |
| `textbook` | `Textbook_Total_Embeds.npy`                              | `Textbook_Total_Articles.json`                           |

> An optional `statpearls` corpus is also supported by `retrieve.py`
> (`Statpearls_Total_Embeds.npy` / `Statpearls_Total_Articles.json`).

Each embedding row must align, by index, with the corresponding article entry so that retrieved
FAISS indices decode back to the correct text.

### 3. Query (input) files

Place query files under `input/<benchmark>/`. A query file is a JSON list. There are two supported
formats:

- **Inference / evaluation queries** — a JSON list of query strings (rationales generated in the
  previous pipeline stage). This is the default (`--instruction_preprocess False`).
- **Instruction (training) set** — a JSON list of objects with `instruction` and `input` fields.
  Enable with `--instruction_preprocess True`; the two fields are concatenated (optionally with
  SciSpacy `[SEP]` insertion) into a single query.

## Usage

```bash
python main.py \
  --input_path  input/medqa/medqa_llama_cot.json \
  --output_path output/medqa/evidence_medqa_llama_cot.json \
  --embeddings_dir embeddings \
  --articles_dir   articles \
  --top_k 10
```

The output is a JSON list where entry `i` holds the final top-k reranked evidence snippets for
query `i`.

### Arguments

| Argument | Flag | Default | Description |
| --- | --- | --- | --- |
| Embeddings dir | `-e`, `--embeddings_dir` | `embeddings` | Root directory of per-corpus embeddings. |
| Articles dir | `-a`, `--articles_dir` | `articles` | Root directory of per-corpus article text. |
| Input path | `-i`, `--input_path` | `input/medqa/medqa_llama_cot.json` | Query file (see formats above). |
| Corpus | `-c`, `--corpus` | `['cpg','textbook','pmc','pubmed']` | List of corpora (all four are currently run unconditionally in `main.py`). |
| Top-k | `-k`, `--top_k` | `100` | Candidates retrieved per corpus / final reranked count. |
| Instruction preprocess | `-inst`, `--instruction_preprocess` | `False` | Use the `instruction`+`input` training-set format. |
| Output path | `-o`, `--output_path` | `output/medqa/evidence_medqa_llama_cot.json` | Where to write reranked evidence. |
| Use SciSpacy | `-spc`, `--use_spacy` | `False` | Insert `[SEP]` between sentences using `en_core_sci_scibert`. |
| PubMed group size | `-pmdn`, `--pubmed_group_num` | `38` | PubMed chunks concatenated per MIPS step. |

## How balanced retrieval works

- **PubMed** is split into 38 embedding chunks. They are processed in groups of size
  `--pubmed_group_num`; the paper groups the 38 chunks as 10/10/10/8 and retrieves `top_k` from each
  group (≈40 candidates total) to bound peak memory while covering the whole corpus.
- **PMC, CPG, Textbook** each contribute `top_k` candidates.
- All candidates are pooled and reranked with the MedCPT cross-encoder; the top-k after reranking
  are returned. Retrieving evenly across corpora is what mitigates single-corpus retriever bias.

## Notes

- Following MedCPT (Jin et al., 2023), SciSpacy `[SEP]` insertion is applied when **encoding** for
  MIPS but **not** for the reranker. It is time-intensive, so it is left as an opt-in
  (`--use_spacy`) design choice.
- GPU device indices are currently hard-coded (`cuda:7` in `query_encode.py`, device `2` in
  `rerank.py`). Adjust them to match your machine before running.
