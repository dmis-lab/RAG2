# Reproducing the original RAG² system

**Scope of this document.** This is the reference specification for the
reproduction of the *original* RAG² system (Sohn et al., NAACL 2025,
[2025.naacl-long.635](https://aclanthology.org/2025.naacl-long.635/); arXiv
2411.00300). It records, component by component:

* **[S]** — what the paper or the released repository **specifies explicitly**;
* **[A]** — what is **ambiguous** or under-specified, and which reading this
  reproduction adopts;
* **[U]** — what is **unavailable** (resource, checkpoint, corpus, code) and
  whether it can be reconstructed;
* **[D]** — **discrepancies** between the paper text and the released code.

Nothing here is invented silently: every choice that is not forced by the paper
or the release is tagged **[A]** and carries a rationale plus the config key that
controls it.

This document covers the original system **only**. No thesis contribution (SCAF,
FRB-PAIRS, recency-bias analysis, evidence currency / temporal weighting,
contested evidence, abstention, entailment-based filtering) is described,
implemented, or accommodated here beyond the two structural affordances the
thesis will need later and which do not change baseline behaviour:

1. the evidence filter sits behind a swappable interface
   (`rag2.filtering.base.EvidenceFilter`), and
2. retrieval/reranking candidates are cached so the identical candidate set can
   be replayed through any filter.

Publication dates and other provenance fields are **carried as metadata only**
and are never read by any baseline component. This is asserted by
`tests/test_metadata_isolation.py`.

---

## 0. Sources inspected

| Source | Status |
| --- | --- |
| `2025.naacl-long.635.pdf` (15 pages, incl. Appendix A) | Read in full |
| `README.md`, `retriever/README.md`, `classifier/README.md` | Read in full |
| `retriever/{main,query_encode,retrieve,rerank}.py` | Read in full |
| `classifier/{run_classifier,utils}.py`, `classifier/model/token_add.ipynb` | Read in full |
| `classifier/data/medqa/llama3_cot/5%-train.json` | Read (5 records) |
| `classifier/data/preprocess.py` | **Empty file (0 bytes)** |
| `classifier/run/run_large_train_xl_000.sh` | Read in full |
| `environment.yml` | Read in full |
| Upstream `dmis-lab/RAG2` @ `86add43` | Cloned and diffed — **byte-identical** to this repo; full history reviewed |

No supplementary material beyond Appendix A of the PDF exists. The upstream
repository contains no additional code: its git history (18 commits) was
reviewed, and the only file ever deleted (`main.py` at the repository root) was
always empty.

---

## 1. System architecture  **[S]**

The paper (§3, Figure 1) specifies a four-stage, single-pass pipeline. Each
stage's output is the next stage's input; there is no iteration and no second
retrieval round.

```
question x
  │
  ├─[1] Rationale-based query formulation  (§3.3)
  │      base LLM + chain-of-thought prompt  ──►  rationale r
  │
  ├─[2] Balanced retrieval + reranking      (§3.4)
  │      MedCPT query encoder ⨯ FAISS MIPS over 4 corpora, k each
  │      pooled candidates ──► MedCPT cross-encoder rerank ──► top-k snippets
  │
  ├─[3] Rationale-guided filtering          (§3.2)
  │      Flan-T5-large scores each (question, snippet) pair
  │      keep only [HELPFUL]
  │
  └─[4] Answer generation
         base LLM(question + kept snippets) ──► answer
```

Explicitly specified properties:

* **[S]** The **same LLM** is used for rationale generation and for QA (§3.3).
* **[S]** Retrieval uses the **rationale only**, *not* the initial query — "We
  search for document snippets solely using the rationale, excluding the initial
  query. Including both … exceeds the maximum length of the retriever and …
  leads to suboptimal performance" (§3.3).
* **[S]** Single-pass generation; no ensembling in the reported numbers (§4.3).
* **[S]** The filter judges **one snippet at a time** — "the Flan-T5 model can
  filter only one snippet at a time due to its limited context length"
  (Limitations).

### 1.1 Reproduction layout

The original release is left **untouched** under `retriever/` and `classifier/`
so it remains citable as the authors published it. The reproduction lives in a
separate package:

```
rag2/                       reproduction package (this work)
  config.py                 typed config, loaded from YAML
  schema.py                 Question / Evidence / CandidateSet (+ provenance metadata)
  prompts.py                verbatim + reconstructed prompts, versioned
  datasets/                 pluggable QA dataset interface  ← medical dataset plugs in here
  corpora/                  pluggable evidence corpus interface
  llm/                      LLM backends (HF, vLLM, stub) behind one ABC
  retrieval/                MedCPT encoder, FAISS balanced MIPS, cross-encoder rerank
  cache.py                  candidate save / replay
  filtering/                EvidenceFilter ABC + RAG² Flan-T5 filter + passthrough
  filter_training/          ΔPPL computation, Figure-2 labeling, training-data export
  generation.py             answer generation
  evaluation.py             accuracy / ROUGE-L / BERTScore
  pipeline.py               stage orchestration
configs/                    experiment configuration (YAML)
scripts/                    one CLI per stage
tests/                      unit tests + dependency-free e2e smoke test
```

---

## 2. Models and checkpoints

| Role | Identifier | Evidence |
| --- | --- | --- |
| Query encoder | `ncbi/MedCPT-Query-Encoder` | **[S]** `retriever/query_encode.py:52` |
| Article encoder (offline) | `ncbi/MedCPT-Article-Encoder` | **[S]** `README.md` "Data & corpora" |
| Reranker | `ncbi/MedCPT-Cross-Encoder` | **[S]** `retriever/rerank.py:19-20` |
| Filtering model | Flan-T5-**large**, 770M | **[S]** paper §4.2: "a filtering model based on Flan-T5-large, which has only 770 million parameters" |
| Backbone LLM (open) | `meta-llama/Meta-Llama-3-8B-Instruct` | **[S]** paper §4.2 footnote 2 (explicit URL) |
| Backbone LLM (medical) | Meerkat-7B (Kim et al., 2024) | **[S]** paper §4.2 — **[A]** see below |
| Backbone LLM (commercial) | GPT-4o | **[S]** paper §4.2; version/date unspecified |

Notes and gaps:

* **[A] Flan-T5-large HF id.** The paper names "Flan-T5-large"; the repo's
  `token_add.ipynb` leaves the base model string empty. The canonical id is
  `google/flan-t5-large`. Config key: `filter.base_model`.
* **[A] Meerkat-7B HF id.** The paper cites Kim et al. (2024) but gives no
  repository path. The model released under that paper is
  `dmis-lab/llama-3-meerkat-8b-v1.0` / `dmis-lab/meerkat-7b-v1.0`; the paper's
  description ("initialized using the Mistral-7B weights … instruction-tuned
  with rationales generated by GPT-4 … then fine-tuned on MedQA and MedMCQA")
  matches **`dmis-lab/meerkat-7b-v1.0`**, which is what this reproduction
  configures. **This should be confirmed against the Meerkat paper before the
  final run.** Config key: `llm.model`.
* **[A] GPT-4o version.** "the latest version" as of submission; no snapshot
  string. Not pinned — any GPT-4o run must record the served snapshot in the
  run manifest. The reproduction does not require GPT-4o.
* **[U] The trained filtering-model checkpoint is not available.** The repo
  README states it "is not available for distribution". It **can be
  reconstructed** by re-running the labeling + training procedure (§5–§6 below)
  — that is exactly what `scripts/03_build_filter_labels.py` and
  `scripts/04_train_filter.py` do. The reconstructed filter is an
  approximation: the base LLM's generations, and therefore the labels, will not
  be bit-identical to the authors' (see §11).
* **[S] Special tokens.** `[HELPFUL]` and `[NOT_HELPFUL]` are added to the
  Flan-T5 tokenizer and the embedding matrix is resized
  (`classifier/model/token_add.ipynb`). **[D]** That notebook loads the base
  model with `AutoModelForCausalLM`, which is wrong for T5; the training script
  uses `AutoModelForSeq2SeqLM` (`classifier/utils.py:43`). The reproduction
  uses `AutoModelForSeq2SeqLM` throughout.

---

## 3. Prompts

### 3.1 Rationale generation — **[S], verbatim**

Given in full in §3.3 of the paper (following Kim et al., 2024):

```
The following are multiple choice questions about medical knowledge. Solve them in
a step-by-step fashion, starting by summarizing the available information. Output
your explanation and single option from the given options as the final answer.

Here is the question: [initial_query]
```

`[initial_query]` is replaced with the initial query `x`. **[A]** The paper does
not state how the question stem and the four options are serialised into
`[initial_query]`. The released training data settles it: options are appended
inline in the form `A) … B) … C) … D) …` (see
`classifier/data/medqa/llama3_cot/5%-train.json`, e.g. `… Which of the following
is the most likely cause of this patient's symptoms? A) Metapneumovirus B)
Influenza virus C) Rhinovirus D) Adenovirus`). The reproduction uses that exact
serialisation. Config key: `prompts.option_format`.

### 3.2 Filter input — **[S], recovered verbatim from the released data**

The paper does not print the filter prompt, but every record in the released
training artifact uses one fixed template:

```
Given the following evidence, determine whether it helps answer the provided question.

Evidence: {snippet}

Question: {question_with_options}
```

Target string: `[HELPFUL]` or `[NOT_HELPFUL]`.

* **[S]** Separators are exactly `\n\n` (verified in the released JSON).
* **[S]** The **initial question**, not the rationale, appears in the filter
  input. This is consistent with Figure 1, where the filtering model's prompt
  box reads "Snippet + Initial Query".

### 3.3 Answer generation — **[U] not specified**

Neither the paper nor the release contains the answer-generation prompt. Figure
1 shows only its structure: `Prompt = [Snippet₁ … Snippet_k] + [Initial Query]`
→ LLM. **[A]** The reproduction reuses the paper's chain-of-thought prompt
(§3.1) with a retrieved-evidence block prepended, because the paper states the
same LLM does rationale generation and QA and the Meerkat backbone was
instruction-tuned on that prompt:

```
The following are multiple choice questions about medical knowledge. Solve them in
a step-by-step fashion, starting by summarizing the available information. Output
your explanation and single option from the given options as the final answer.

Here are the retrieved documents: [evidence_block]

Here is the question: [initial_query]
```

`[evidence_block]` is the kept snippets joined by `\n`, each prefixed by its
rank. Config keys: `prompts.answer_template`, `prompts.evidence_join`. **This is
the single largest prompt-level assumption in the reproduction** and is the
first thing to vary if accuracy does not match §11.

**[A] Zero-shot.** Table 2 marks Llama-3-8B-Instruct and GPT-4o as 0-shot and
gives no shot count for the `+ RAG²` rows; the reproduction runs RAG² zero-shot
for all backbones.

---

## 4. Retrieval and reranking

### 4.1 Corpus — **[S] identified, [U] not redistributable**

Appendix A.3 and Table A1 specify the corpus as **the Self-BioRAG corpus**
(Jeong et al., 2024a), *not* MedRAG's MedCorp:

| Corpus | # docs | # passages | Index size |
| --- | --- | --- | --- |
| PubMed (abstracts) | 36.5M | 69.7M | 400 GB |
| PMC (full text) | 1.1M | 46.3M | 160 GB |
| CPG (clinical practice guidelines, Chen et al. 2023, 8 of 16 public) | 35.7k | 607.0k | 3.5 GB |
| Textbooks (18) | 18 | 134.0k | 0.7 GB |
| **Total** | **37.6M** | **116.7M** | **564.2 GB** |

* **[S]** Chunking uses "a sliding window mechanism with overlap" (A.3).
* **[U]** The window size and stride are **not given**. Config keys:
  `corpus.chunk_size`, `corpus.chunk_overlap`; left unset by default so the
  supplied medical dataset governs its own chunking.
* **[U]** Neither the corpora nor the precomputed MedCPT embeddings are
  distributed (repo README). The reproduction therefore treats the corpus as a
  **plug-in** (`rag2/corpora/`) and does not ship one.
* **[S]** The on-disk layout the release expects is fully specified in
  `retriever/README.md` (`PubMed_Embeds_{0..37}.npy` /
  `PubMed_Articles_{0..37}.json`, `PMC_{Main,Abs}_*`, `CPG_Total_*`,
  `Textbook_Total_*`), and `rag2/corpora/json_corpus.py` reads exactly that
  layout so an existing index drops straight in.

### 4.2 Retrieval procedure — **[S]**

* **[S]** Dense MIPS, inner product, `faiss.IndexFlatIP`, 768-dim float32
  (`retriever/retrieve.py`). Exact (flat) index — no ANN approximation.
* **[S]** Query embedding = **CLS token** of the MedCPT query encoder's last
  hidden state (`last_hidden_state[:, 0, :]`), truncation at **512** tokens
  (`retriever/query_encode.py:64-73`).
* **[S]** Balanced retrieval: an **equal number of candidates per corpus**
  (§3.4), pooled, then reranked.
* **[A] Optional `[SEP]` insertion.** `query_encode.py` can split the query into
  sentences with SciSpacy `en_core_sci_scibert` and join them with ` [SEP] `,
  following MedCPT. It defaults to **off** in the release
  (`--use_spacy False`) and is only wired into the *instruction/training* query
  path, never the inference path. The reproduction keeps it off by default
  (`retrieval.use_scispacy_sep: false`) and applies it to both paths when on.

**[D] PubMed sharding breaks balance.** `retriever/main.py:57` retrieves
`top_k` from *each* PubMed shard group. With the release default
`--pubmed_group_num 38` there is a single group and PubMed contributes exactly
`top_k` — balanced, as the paper describes. But `retriever/README.md` documents
the paper's actual run as grouping the 38 shards **10/10/10/8**, which yields
`4 × top_k` PubMed candidates against `top_k` from each other corpus, i.e. a 4:1
over-representation of PubMed in the pre-rerank pool — the opposite of the
paper's stated intent. This reproduction shards for memory but **merges shards
by inner-product score and keeps exactly `top_k` PubMed candidates**, which
matches the paper's text and the single-group behaviour of the code. Config key:
`retrieval.shard_merge: score` (set to `concat` to reproduce the release's
unbalanced behaviour).

### 4.3 Reranking — **[S] procedure, [D] query**

* **[S]** MedCPT cross-encoder over `[query, snippet]` pairs, logit as relevance
  score, descending sort, take top-k (`retriever/rerank.py:17-49`).
* **[S]** Truncation at **512** tokens, `padding=True` (`rerank.py:27-33`).
* **[S]** SciSpacy `[SEP]` insertion is **not** applied for the reranker
  (`retriever/README.md`, "Notes").
* **[D] Which query is cross-encoded?** The paper says the **initial query**,
  twice: "a reranker is used to rerank the retrieved snippets by cross-encoding
  the **initial query** and each snippet" (Figure 1 caption) and "encodes the
  **original query** along with each document" (§3.4). The released
  `retriever/main.py:124` passes `input_list` — the *rationale* file — to
  `rr.combine_query_evidence`, so the code reranks with the rationale. The
  reproduction **follows the paper** (initial query) by default and exposes
  `retrieval.rerank_query: initial | rationale` to reproduce the code path. This
  discrepancy must be re-checked if reranked ordering does not match.

**[D] One `top_k` for two roles.** `retriever/main.py` uses a single `--top_k`
(default 100) both as per-corpus retrieval depth and as the final reranked
count. The paper distinguishes them: the final k is swept over
{1, 2, 4, 8, 16, 32} (Figure 3). The reproduction separates
`retrieval.candidates_per_corpus` from `retrieval.final_top_k`.

**[U] Per-corpus retrieval depth is not stated.** The paper never says how many
candidates were retrieved per corpus before reranking. Default here:
`candidates_per_corpus: 100` (the release's `--top_k` default). Documented
assumption.

---

## 5. Evidence filtering — the perplexity-based filter

This is the component the later thesis replaces, so it is specified here in
maximum detail.

### 5.1 Filter model — **[S]**

Flan-T5-large (770M), seq2seq, with `[HELPFUL]` / `[NOT_HELPFUL]` added as
**single** tokens and the embedding matrix resized.

### 5.2 Training objective — **[S]**

Standard seq2seq cross-entropy on the target token sequence
(`classifier/run_classifier.py`: `outputs = model(**batch); loss = outputs.loss`).
There is no classification head and no custom loss.

### 5.3 Training labels — **[S] rule, [U] code**

The labeling rule is fully specified by Figure 2 as a decision tree over three
binary tests: *correct without retrieval*, *correct with retrieval*, and
*ΔPPL ≥ τ* ("lower perplexity"):

| Correct w/o retrieval | Correct w/ retrieval | ΔPPL ≥ τ | Label |
| --- | --- | --- | --- |
| Yes | Yes | Yes | `[HELPFUL]` |
| Yes | Yes | No  | *discard* |
| Yes | No  | –   | `[NOT_HELPFUL]` |
| No  | Yes | –   | `[HELPFUL]` |
| No  | No  | Yes | `[NOT_HELPFUL]` |
| No  | No  | No  | *discard* |

Read plainly: a snippet that raises confidence in a *correct* answer is helpful;
one that raises confidence in a *wrong* answer is harmful; one that flips
correctness decides the label on its own; one that changes neither correctness
nor confidence is dropped from training rather than labelled.

**[U]** `classifier/data/preprocess.py`, which would have produced this data, is
an **empty file** in both this repo and upstream. The rule above is
reconstructed from Figure 2 and §3.2 and is implemented in
`rag2/filter_training/labeling.py`, with the truth table asserted verbatim in
`tests/test_labeling.py`.

**[S]** Labels are produced **per (question, snippet) pair**, each snippet
evaluated individually — stated explicitly in Limitations ("we evaluated each
snippet individually").

**[S]** Filter training data comes from the **MedQA and MedMCQA training
splits** (§4.2). MMLU-Med has no training split, so the MedMCQA-trained filter
is used for it (§4.2, Figure A1).

### 5.4 Perplexity calculation — **[S] formula, [A] which tokens**

Equation 4:

```
PPL(x)    = exp( −(1/L) · Σ_{i=0}^{L−1} log P(x_i | x_<i) )
PPL(x, d) = exp( −(1/L) · Σ_{i=0}^{L−1} log P(x_i | x_<i, d) )
```

**[A]** As literally written, the summed tokens are the *query* `x`. That
contradicts the surrounding prose in three places — the abstract ("perplexity-based
labels **of rationales**"), §2.1 ("we measure perplexity differences in the
**rationales generated by the base LLM**"), and Figure 2, where the "Lower
Perplexity" decision node is tagged **Rationale** — and it would make ΔPPL a
statement about the query rather than about the model's reasoning. The
reproduction therefore scores the **rationale tokens**, conditioned on the prompt
with and without the document, i.e. the summed index runs over the rationale and
the conditioning context differs between the two terms. Config key:
`filter_training.ppl_target: rationale | query`.

Implementation details, all **[A]** (the paper gives none):

* Which rationale? The **same** rationale in both terms — the one generated
  without retrieval — so that ΔPPL isolates the document's effect on the *same*
  string. Scoring two different generations would confound length and content.
  Config key: `filter_training.ppl_rationale: no_retrieval`.
* Prompt tokens are masked out of the loss; only rationale tokens contribute.
* Length normalisation is by rationale token count `L`, per Equation 4.
* Teacher forcing under the base LLM in eval mode, `float32` accumulation of
  log-probabilities.

### 5.5 Perplexity difference and threshold — **[S] value, [A] population**

* **[S]** `ΔPPL = PPL(x) − PPL(x, d)`; positive means the document *lowered*
  perplexity (raised confidence). Equation 3.
* **[S]** `τ` is "a threshold set to select the top percentage of perplexity
  differentials", fixed at the **top 25%** across all experiments: "setting the
  threshold value τ to the top 25% of perplexity differentials consistently
  yielded the best performance and was therefore fixed across all our
  experiments" (§3.2).
* **[A]** Over **which population** the top 25% is taken is not stated —
  globally over all (question, snippet) pairs in the split, or per question. The
  reproduction takes it **globally per dataset split**, because τ is described as
  a single fixed value and per-question quantiles over k≈10 snippets would be
  extremely noisy. Config keys: `filter_training.tau_percentile: 25`,
  `filter_training.tau_scope: global | per_question`.
* **[A]** τ is computed as the 75th percentile of the ΔPPL distribution
  (linear interpolation), and the test is `ΔPPL ≥ τ`, matching Equation 3's `≥`.
* **[A]** The released artifact is named `5%-train.json` with ids like
  `llama3_5%_23600`. What "5%" denotes is **not documented anywhere**. It is not
  a 5% subsample in the obvious sense (ids run past 23,000). It may be a
  different threshold setting than the paper's 25%, or a sampling ratio over a
  much larger pool. The reproduction follows the **paper's stated 25%** and
  flags this as an open discrepancy.

### 5.6 Filter inference — **[S], exactly**

From `classifier/run_classifier.py:696-712`:

1. Call `model.generate(..., return_dict_in_generate=True, output_scores=True)`
   and take `scores[0]` — the logits of the **first** decoded token.
2. Slice out the two label-token columns via
   `tokenizer.convert_tokens_to_ids('[HELPFUL]')` and `'[NOT_HELPFUL]'`.
3. Softmax over **those two logits only**; `argmax` gives the label.
4. Keep the snippet iff the label is `[HELPFUL]`.

The reproduction computes the same two logits directly from a single forward
pass with the decoder start token, which is numerically identical to step 1 and
avoids a generate() call per snippet. `tests/test_filter_scoring.py` pins the
equivalence.

**[A]** The paper does not say what happens when **all** snippets for a question
are filtered out. The reproduction falls back to **no-evidence generation**
(closed-book) for that question, since RAG² is reported as never scoring below
its no-RAG baseline and the alternative (forcing top-1 through) would defeat the
filter. Config key: `filter.on_empty: no_evidence | keep_top1`.

### 5.7 Input formatting, tokenization, truncation — **[S]**

* **[S]** `max_seq_length = 512` (`run/run_large_train_xl_000.sh`; the script's
  own default is 384).
* **[S]** `doc_stride = 128` with `return_overflowing_tokens=True`
  (`classifier/utils.py:99-107`) — long inputs produce **multiple overlapping
  features**, each inheriting the same label (`utils.py:131-137`).
* **[D]** At **evaluation** time the script produces one prediction per
  *feature* but zips them against per-*example* ids and gold answers
  (`run_classifier.py:737`). When any input overflows 512 tokens the two lists
  desynchronise and the reported accuracy is wrong. The reproduction scores
  **one feature per pair** (first window, truncation without overflow) at
  inference and documents this as a deliberate correction of a bug in the
  release. Config key: `filter.overflow: truncate | stride`.
* **[S]** `max_answer_length = 30` (script default); labels are single tokens so
  this is not binding.
* **[A]** The paper does not say what is truncated when a (snippet, question)
  pair exceeds 512 tokens. The reproduction truncates from the right (the
  tokenizer's default), which drops the tail of the question — matching the
  release. Flagged because snippet-length distribution will differ with a
  different corpus.

### 5.8 Filter training hyperparameters — **[S]**

| Parameter | Value | Source |
| --- | --- | --- |
| Learning rate | `3e-5` | **[S]** Appendix A.3 + `run_large_train_xl_000.sh` |
| Epochs | `40` | **[S]** Appendix A.3 + run script |
| Per-device train batch size | `16` | **[S]** Appendix A.3 + run script |
| `max_seq_length` | `512` | **[S]** run script |
| `doc_stride` | `128` | **[S]** run script |
| Optimizer | `AdamW` | **[S]** `run_classifier.py:528` |
| LR schedule | `linear`, `num_warmup_steps=0` | **[S]** script defaults, not overridden |
| Weight decay | `0.0` | **[S]** script default, not overridden |
| Gradient accumulation | `1` | **[S]** script default, not overridden |
| Checkpointing | every epoch | **[S]** run script |
| Hardware | 1× H100 80GB (paper); 1× RTX 3090 24GB claimed sufficient | **[S]** A.3, §4.2 |

* **[U] Random seed.** `--seed` defaults to `None` in `run_classifier.py`, so the
  authors' runs were **unseeded**. The reproduction seeds everything
  (`experiment.seed`, default `42`) and records it; exact-match to the authors'
  checkpoint is therefore impossible in principle.
* **[S] Checkpoint selection.** "We selected a few candidate models from the
  validation set, as performance converged after certain epochs, and evaluated
  them on the test set" (A.3). **[A]** The exact selection rule is not given;
  the reproduction selects the epoch with the best filter accuracy on the
  validation split and records all epoch scores. Config key:
  `filter_training.select_by: val_accuracy`.

---

## 6. Answer generation

* **[S]** Backbone LLM answers conditioned on the initial query plus the kept
  snippets (Figure 1).
* **[S]** Greedy decoding, **temperature 0**; vLLM used for throughput; residual
  nondeterminism acknowledged (A.3, "Inference").
* **[U]** Prompt template — see §3.3, **[A]**.
* **[U]** `max_new_tokens` is not stated. Default here: `512`
  (`generation.max_new_tokens`), enough for a CoT rationale plus a final answer.
* **[U]** Answer-extraction rule is not stated. The prompt asks the model to
  "Output your explanation and single option from the given options as the final
  answer", and the paper's own example ends "Therefore, the answer is (C)
  Intubation" (Figure 4). **[A]** The reproduction extracts the **last** option
  letter matched by an ordered list of patterns (`the answer is (X)`,
  `answer: X`, a trailing bare `(X)`, …) and falls back to the last standalone
  A–D token; unmatched generations count as incorrect. Implemented in
  `rag2/evaluation.py:extract_choice` (default patterns in
  `DEFAULT_EXTRACTION_PATTERNS`, overridable via the `evaluation.extraction_patterns`
  config key), pinned by `tests/test_evaluation.py`.

### 6.1 Top-k — **[S] grid, [S] selected values**

Figure 3 sweeps k ∈ {1, 2, 4, 8, 16, 32}; "MedRAG and RAG² use the optimal top-k
values, determined through validation" (§4.2). The k that produces each Table 2
number is recoverable from Figure 3:

| Backbone | MedQA | MedMCQA |
| --- | --- | --- |
| Llama-3-8B-Instruct | **64.6** @ k=32 | **59.4** @ k=16 |
| Meerkat-7B | **75.6** @ k=2 | **63.0** @ k=8 |

**[U]** The optimal k for MMLU-Med and for GPT-4o is not recoverable from the
figures. Config key: `retrieval.final_top_k`, set per experiment config.

---

## 7. Datasets and evaluation

### 7.1 Benchmarks — **[S]**

| Dataset | Train | Val | Test |
| --- | --- | --- | --- |
| MedQA (US, 4-option) | 10,178 | 1,272 | 1,273 |
| MedMCQA | 182,822 | 4,183 | 6,150 |
| MMLU-Med | – | – | 1,089 |

* **[S]** All are 4-option multiple choice.
* **[S]** MMLU-Med = six subjects: clinical knowledge, medical genetics,
  anatomy, professional medicine, college biology, college medicine.
  **[D]** Appendix A.2 lists "human genetics" where §4.1 lists "medical
  genetics"; the MMLU subject is `medical_genetics`.
* **[A]** MedMCQA's official test split is unlabelled; the 6,150 figure matches
  its **validation** split being used as test, which is the standard convention
  in this literature (and matches MedRAG). The reproduction follows it and
  records the choice. Config key: `dataset.split_map`.

### 7.2 Metric — **[S]**

**Accuracy** on multiple choice (Table 2, unit: %). For the open-ended
ClinicalQA25 side experiment (A.4): **ROUGE-L F1** and **BERTScore F1**, both
defined explicitly in A.4.1. **[U]** The BERTScore backbone model is not stated;
default here `roberta-large` (the `bert-score` package default), recorded in
config.

### 7.3 Filter-level metrics — **[S]**

`classifier/utils.py` computes overall accuracy plus per-class accuracy and
predicted/gold counts for `[HELPFUL]` / `[NOT_HELPFUL]`. Reproduced in
`rag2/evaluation.py:filter_metrics`.

---

## 8. What is unavailable, and whether it can be reconstructed

| Resource | Status | Reconstructable? |
| --- | --- | --- |
| Trained Flan-T5-large filter checkpoint | **[U]** not distributed | **Yes**, by re-running §5 labeling + training. Not bit-identical (unseeded original, different LLM generations). |
| Four biomedical corpora + MedCPT embeddings (564 GB) | **[U]** not distributed | **Yes in principle** (PubMed/PMC/textbooks are public; only 8 of 16 CPG sources are public per Chen et al. 2023), at substantial cost. Out of scope here: the medical dataset is being prepared separately and plugs into `rag2/corpora/`. |
| `classifier/data/preprocess.py` (labeling code) | **[U]** empty file | **Yes**, from Figure 2 + §3.2 — implemented in `rag2/filter_training/labeling.py`. |
| Rationale files (`*_llama_cot.json`) | **[U]** not distributed | **Yes**, by running §3.1 with the base LLM. |
| Answer-generation prompt | **[U]** never published | **No** — reconstructed by assumption (§3.3). |
| Sliding-window chunk size/stride | **[U]** never published | **No** — left to the supplied dataset. |
| Random seeds | **[U]** unseeded | **No**. |
| Meaning of "5%" in the released artifact filename | **[U]** undocumented | **No**. |
| Per-corpus retrieval depth before reranking | **[U]** never published | **No** — assumed 100. |
| Optimal top-k for MMLU-Med and GPT-4o | **[U]** not in figures | **No** — must be re-selected on validation. |
| GPT-4o snapshot | **[U]** not pinned | **No**. |

---

## 9. Assessment of the existing code

| File | Verdict |
| --- | --- |
| `retriever/query_encode.py` | **Reusable logic.** CLS pooling and 512-token truncation are correct. Defects: hard-coded `cuda:7`; batch size of 1 (`range(0, len, 1)` then `[i:i+1]`) makes encoding needlessly slow; `xq = np.vstack(queries)` is rebuilt inside the loop, i.e. O(n²) work. Reimplemented in `rag2/retrieval/encoder.py` with the same semantics. |
| `retriever/retrieve.py` | **Reusable logic**, but five near-identical `*_index_create` / `*_decode` function pairs hard-code corpus filenames. Generalised to one corpus-agnostic path in `rag2/retrieval/index.py`; the original filename layout is preserved by `rag2/corpora/json_corpus.py`. |
| `retriever/rerank.py` | **Reusable.** Scoring is correct. Defects: hard-coded device `2`; one tokenizer/model load per call is fine but the whole candidate list for a query is padded into a single batch, which OOMs at large `candidates_per_corpus`. Reimplemented with batching in `rag2/retrieval/rerank.py`. |
| `retriever/main.py` | **Needs correction.** Single `top_k` for two distinct roles; PubMed shard concatenation breaks balance (§4.2 **[D]**); reranks with the rationale rather than the initial query (§4.3 **[D]**); no caching; no metadata retained (snippets are bare strings, so document id / source / date are lost). Replaced by `rag2/pipeline.py` + `scripts/02_retrieve.py`. |
| `classifier/run_classifier.py` | **Reusable for training** — kept and used as-is for §5.8, since it is the authors' own script. Its `--do_eval` path has the feature/example desync bug (§5.7 **[D]**), so the reproduction does not use it for inference. |
| `classifier/utils.py` | **Reusable.** Preprocessing and metrics match the paper. |
| `classifier/model/token_add.ipynb` | **Needs correction** — `AutoModelForCausalLM` on a T5 checkpoint (§2). Replaced by `scripts/04_train_filter.py --init-tokens`. |
| `classifier/data/preprocess.py` | **Empty** — must be written from scratch (§5.3). |
| `environment.yml` | **Reusable but dated** (torch 2.1.0/CUDA 12.1, transformers 4.36.2, faiss-gpu 1.7.2). Kept verbatim as the original environment record; the reproduction adds `requirements.txt` with the same pins plus what it needs. |

The original `retriever/` and `classifier/` trees are **left unmodified**, so the
authors' release stays citable; the reproduction is additive.

---

## 10. Reproducibility record

Every run writes a manifest (`rag2/experiment.py:write_manifest`) capturing:
resolved config, git commit + dirty flag, model ids and revision hashes, dataset
id and version, seeds, prompt template hashes, package versions, hardware, wall
clock, and output digests. Prompts are versioned constants in `rag2/prompts.py`
and hashed into the manifest so a prompt edit is never silent.

Cached candidates (`rag2/cache.py`) carry the retrieval config hash. Replaying a
cache built under a different retrieval config raises unless
`--allow-config-mismatch` is passed, which is what lets the same candidate set be
fed to different filters with an auditable guarantee that only the filter changed.

---

## 11. Reference results to reproduce

Table 2 (accuracy, %), RAG² rows and their no-RAG baselines:

| Backbone | | MedQA | MedMCQA | MMLU-Med | Average |
| --- | --- | --- | --- | --- | --- |
| Llama-3-8B-Instruct | no RAG | 57.7 | 53.5 | 69.5 | 60.2 |
| | **+ RAG²** | **64.6** | **59.4** | **74.8** | **66.3** |
| Meerkat-7B | no RAG | 71.2 | 60.8 | 73.8 | 68.6 |
| | **+ RAG²** | **75.6** | **63.0** | **78.7** | **72.4** |
| GPT-4o | no RAG | 88.5 | 76.7 | 92.8 | 86.0 |
| | **+ RAG²** | **91.1** | **77.2** | **92.5** | **86.9** |

Secondary targets: balanced retrieval alone, top-1, no filter (Table 4) —
Llama-3-8B 55.3 / 51.3 / 65.8, Meerkat-7B 71.8 / 57.9 / 74.0. Filtering-method
ablation at top-1 with the *initial* query (Table 3) — Llama-3-8B + RAG² filter
58.6 (MedQA) / 55.8 (MedMCQA).

**Expected sources of divergence**, to be checked in order if numbers do not
match:

1. **Corpus.** The reproduction will run on a different medical corpus than the
   564 GB Self-BioRAG corpus. This alone can move accuracy by several points and
   is the dominant term.
2. **Reconstructed filter checkpoint.** Different base-LLM generations → different
   ΔPPL → different labels → a different filter. Unseeded original.
3. **Answer-generation prompt** (§3.3 **[A]**) and answer extraction (§6 **[A]**).
4. **Rerank query**: initial vs. rationale (§4.3 **[D]**).
5. **τ population**: global vs. per-question (§5.5 **[A]**); and the unexplained
   "5%" artifact.
6. **Model versions.** `Meta-Llama-3-8B-Instruct` and Meerkat weights may have
   been revised; the manifest records the resolved revision hash.
7. **Decoding nondeterminism**, acknowledged by the authors even at
   temperature 0 (A.3).
8. **PubMed shard balance** if `shard_merge: concat` is used (§4.2 **[D]**).

Per the task brief, the baseline is **not** to be tuned to close a gap with the
paper. `scripts/06_evaluate.py` prints the delta against the table above and
writes a `*.report.json` beside the predictions; those numbers and their
explanations are transcribed into `docs/reproduction_results.md`, not engineered
away.

---

## 12. Open questions for the authors

1. What does "5%" denote in `classifier/data/medqa/llama3_cot/5%-train.json`,
   given the paper's τ = top 25%?
2. Is ΔPPL computed over the rationale tokens (as the prose says) or the query
   tokens (as Equation 4 literally says)?
3. Is τ's top-25% taken globally or per question?
4. Was the reranker given the initial query (paper) or the rationale (code)?
5. What was the exact answer-generation prompt?
6. What were the sliding-window chunk size and stride?
7. How many candidates were retrieved per corpus before reranking?
8. Which Meerkat-7B checkpoint, and which GPT-4o snapshot?
