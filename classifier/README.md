# RAG² Filtering Model (Classifier)

The filtering model is the core of RAG²'s **rationale-guided filtering**. Given a question and a
retrieved evidence snippet, it decides whether the snippet is `[HELPFUL]` or `[NOT_HELPFUL]` for
answering the question. Only `[HELPFUL]` snippets are passed to the LLM at answer-generation time,
which removes distractors and improves reliability.

It is a small **Flan-T5** seq2seq model fine-tuned with 🤗 Accelerate. Training targets are the two
special tokens `[HELPFUL]` / `[NOT_HELPFUL]`; the supervision comes from **perplexity-based labels**
of rationales, as described in the paper.

## Contents

- `run_classifier.py` — training / evaluation entry point.
- `utils.py` — model loading, feature preprocessing, and accuracy metrics.
- `model/token_add.ipynb` — adds the `[HELPFUL]` / `[NOT_HELPFUL]` tokens to a base model + tokenizer.
- `data/` — training / evaluation data (see format below).
- `run/` — example launch scripts.

## Setup

The classifier uses the shared `rag2` environment (see the top-level README);
it already includes `accelerate` for training:

```bash
conda env create -f ../environment.yml
conda activate rag2
```

### Add the special tokens

Before training, extend a base Flan-T5 model and tokenizer with the two label tokens. Open
`model/token_add.ipynb`, set the base model name and a save directory, and run it:

```python
new_tokens = ["[HELPFUL]", "[NOT_HELPFUL]"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
```

Point `--model_name_or_path` at the resulting directory when training.

## Data format

Each split is a JSON list of objects. A minimal example (see
`data/medqa/llama3_cot/5%-train.json`):

```json
{
  "id": "llama3_5%_23600",
  "answer": "[NOT_HELPFUL]",
  "dataset_name": "llama3_5%",
  "question": "Given the following evidence, determine whether it helps answer the provided question.\n\nEvidence: ...\n\nQuestion: ..."
}
```

| Field | Description |
| --- | --- |
| `id` | Unique example identifier. |
| `question` | The model input: an instruction wrapping the (evidence, question) pair. |
| `answer` | Target label token: `[HELPFUL]` or `[NOT_HELPFUL]`. |
| `dataset_name` | Source/split tag, used for bookkeeping in evaluation output. |

The column names are configurable via `--question_column` and `--answer_column`.

## Training

```bash
cd classifier
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --model_name_or_path model/updated_flan_t5_model \
  --train_file data/medqa/llama3_cot/5%-train.json \
  --question_column question \
  --answer_column answer \
  --do_train \
  --train_column train \
  --checkpointing_steps epoch \
  --learning_rate 3e-5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_device_train_batch_size 16 \
  --num_train_epochs 40 \
  --output_dir outputs/medqa_llama3_5pct \
  --overwrite_cache
```

`run/run_large_train_xl_000.sh` is a template for the command above.

> **Note on paths:** the example script uses paths such as `classifier/model/data/...` and an
> absolute `/classifier/...` `--model_name_or_path`. Adjust these to your working directory — the
> sample data actually ships at `classifier/data/medqa/llama3_cot/5%-train.json`.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --model_name_or_path outputs/medqa_llama3_5pct \
  --validation_file data/medqa/llama3_cot/validation.json \
  --question_column question \
  --answer_column answer \
  --do_eval \
  --val_column validation \
  --per_device_eval_batch_size 16 \
  --output_dir outputs/medqa_llama3_5pct
```

Evaluation scores the two filter tokens directly: for each example it takes a softmax over the
`[HELPFUL]` / `[NOT_HELPFUL]` token logits, predicts the higher-probability label, and compares it
against the gold `answer`. It writes per-example predictions (`dict_id_pred_results.json`), overall
accuracy (`final_eval_results.json`), and per-class accuracy for `[HELPFUL]` / `[NOT_HELPFUL]`
(`final_eval_results_perClass.json`) to `--output_dir`.

## Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model_name_or_path` | — | Base/fine-tuned Flan-T5 (with the added label tokens). |
| `--train_file` / `--validation_file` | — | JSON data files. |
| `--question_column` / `--answer_column` | `question` / `answers` | Input / target columns. |
| `--do_train` / `--do_eval` | off | Select mode. |
| `--max_seq_length` | `384` | Max input length (truncated with `--doc_stride` overflow). |
| `--doc_stride` | `128` | Stride for long-context overflow chunks. |
| `--learning_rate` | `5e-5` | AdamW learning rate. |
| `--num_train_epochs` | `2` | Training epochs. |
| `--per_device_train_batch_size` | `8` | Train batch size per device. |
| `--checkpointing_steps` | `None` | `epoch` or an integer step count. |
| `--output_dir` | — | Where checkpoints, logs, and results are written. |

Multi-GPU training is handled by 🤗 Accelerate; run `accelerate config` once, then launch with
`accelerate launch run_classifier.py ...` if you want distributed training.
