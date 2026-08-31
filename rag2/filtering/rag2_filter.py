"""The original RAG2 rationale-guided filter (paper section 3.2).

A Flan-T5-large seq2seq model, extended with ``[HELPFUL]`` / ``[NOT_HELPFUL]``
as single tokens, fine-tuned on the perplexity-derived labels of
``rag2.filter_training.labeling``. At inference it scores one (question, snippet)
pair at a time -- the paper's Limitations note that "the Flan-T5 model can filter
only one snippet at a time due to its limited context length".

Scoring, reproducing ``classifier/run_classifier.py`` lines 696-712 exactly
-------------------------------------------------------------------------
The release calls ``model.generate(..., output_scores=True)`` and keeps
``scores[0]`` -- the logits of the **first** decoded token -- then takes a
softmax over just the two label-token columns and argmaxes. Since the first
decoded position is fully determined by the encoder output and the decoder start
token, a single forward pass with ``decoder_input_ids=[[pad]]`` produces the
identical logits at far lower cost. That is what ``score_pairs`` does; the
equivalence is pinned by tests/test_filter_scoring.py, and
``options.use_generate: true`` restores the literal generate() path.

Truncation
----------
``max_seq_length=512``, matching the release's run script. The release's *eval*
path additionally sets ``doc_stride=128`` with ``return_overflowing_tokens``,
which emits one prediction per overflow window while zipping against per-example
ids -- desynchronising predictions from gold whenever an input overflows (see
docs/rag2_reproduction.md section 5.7). Inference here scores one window per pair
(``filter.overflow: truncate``); ``stride`` reproduces the release's windowing
and aggregates windows by max helpfulness probability rather than desynchronising.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import math

from ..config import FilterConfig
from ..prompts import DEFAULT_PROMPTS, LABEL_HELPFUL, LABEL_NOT_HELPFUL, PromptSet
from ..schema import Evidence, FilterDecision, Question
from .base import EvidenceFilter, register_filter

#: A snippet is kept when P([HELPFUL]) wins the two-way softmax, i.e. >= 0.5 --
#: equivalent to the release's ``np.argmax`` over the two label probabilities
#: (classifier/run_classifier.py:713).
KEEP_THRESHOLD = 0.5


def helpful_probability(helpful_logit: float, not_helpful_logit: float) -> float:
    """P([HELPFUL]) from the two label logits.

    Numerically stable two-way softmax; the scalar counterpart of the batched
    torch op in :meth:`RAG2PerplexityFilter.score_pairs`, kept pure so the
    scoring rule is testable without torch.
    """
    shift = max(helpful_logit, not_helpful_logit)
    numerator = math.exp(helpful_logit - shift)
    denominator = numerator + math.exp(not_helpful_logit - shift)
    return numerator / denominator


def decisions_from_probabilities(
    probabilities: Sequence[float], threshold: float = KEEP_THRESHOLD
) -> List[FilterDecision]:
    """Turn P([HELPFUL]) values into keep/drop decisions."""
    return [
        FilterDecision(
            keep=p >= threshold,
            label=LABEL_HELPFUL if p >= threshold else LABEL_NOT_HELPFUL,
            score=float(p),
        )
        for p in probabilities
    ]


class RAG2PerplexityFilter(EvidenceFilter):
    """Flan-T5 filter trained on perplexity-derived labels."""

    name = "rag2_perplexity"

    def __init__(self, config: FilterConfig, prompts: Optional[PromptSet] = None) -> None:
        self.config = config
        self.prompts = prompts or DEFAULT_PROMPTS
        if not config.checkpoint:
            raise ValueError(
                "filter.checkpoint is empty: the paper's trained filter checkpoint is "
                "not distributed (see docs/rag2_reproduction.md section 2). Train one with "
                "scripts/04_train_filter.py, or set filter.kind=passthrough to run the "
                "'RAG2 w/o filter' ablation."
            )
        checkpoint = config.checkpoint

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = config.device
        self.device = torch.device(
            device if device and device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

        self.helpful_id = self.tokenizer.convert_tokens_to_ids(LABEL_HELPFUL)
        self.not_helpful_id = self.tokenizer.convert_tokens_to_ids(LABEL_NOT_HELPFUL)
        unk = getattr(self.tokenizer, "unk_token_id", None)
        if None in (self.helpful_id, self.not_helpful_id) or unk in (
            self.helpful_id,
            self.not_helpful_id,
        ):
            raise ValueError(
                f"checkpoint {checkpoint!r} does not have {LABEL_HELPFUL} / "
                f"{LABEL_NOT_HELPFUL} as single tokens; run "
                "scripts/04_train_filter.py --init-tokens first"
            )
        self.use_generate = bool(config.options.get("use_generate", False))

    # -- scoring -----------------------------------------------------------
    def score_pairs(self, inputs: Sequence[str]) -> List[float]:
        """P([HELPFUL]) for each rendered filter input.

        The softmax is over the two label logits only, as in the release.
        """
        import torch

        if not inputs:
            return []
        size = self.config.batch_size or 32
        probabilities: List[float] = []
        for start in range(0, len(inputs), size):
            batch = list(inputs[start : start + size])
            stride_kwargs: Dict[str, Any] = {}
            if self.config.overflow == "stride":
                stride_kwargs = {
                    "stride": self.config.doc_stride,
                    "return_overflowing_tokens": True,
                }
            encoded = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=True,
                return_tensors="pt",
                **stride_kwargs,
            )
            mapping = encoded.pop("overflow_to_sample_mapping", None)
            encoded = {k: v.to(self.device) for k, v in encoded.items() if k != "offset_mapping"}
            with torch.no_grad():
                logits = self._first_token_logits(encoded)
            pair = torch.stack(
                [logits[:, self.helpful_id], logits[:, self.not_helpful_id]], dim=0
            )
            probs = torch.nn.functional.softmax(pair, dim=0)[0].detach().cpu().tolist()
            if mapping is None:
                probabilities.extend(float(p) for p in probs)
            else:
                # One prediction per example: take the most helpful window.
                per_example: Dict[int, float] = {}
                for window, sample in enumerate(mapping.tolist()):
                    per_example[sample] = max(per_example.get(sample, 0.0), float(probs[window]))
                probabilities.extend(per_example[i] for i in range(len(batch)))
        return probabilities

    def _first_token_logits(self, encoded: Dict[str, Any]):
        """Logits at the first decoding position.

        Identical to ``generate(..., output_scores=True).scores[0]`` but computed
        with one forward pass.
        """
        import torch

        if self.use_generate:
            generated = self.model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_length=self.config.options.get("max_answer_length", 30),
                return_dict_in_generate=True,
                output_scores=True,
            )
            return generated.scores[0]
        batch = encoded["input_ids"].shape[0]
        start_id = (
            self.model.config.decoder_start_token_id
            if self.model.config.decoder_start_token_id is not None
            else self.model.config.pad_token_id
        )
        decoder_input_ids = torch.full(
            (batch, 1), start_id, dtype=torch.long, device=encoded["input_ids"].device
        )
        outputs = self.model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits[:, 0, :]

    # -- EvidenceFilter ----------------------------------------------------
    def decide(self, question: Question, candidates: Sequence[Evidence]) -> List[FilterDecision]:
        if not candidates:
            return []
        inputs = [self.prompts.render_filter_prompt(question, c) for c in candidates]
        return decisions_from_probabilities(self.score_pairs(inputs))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "checkpoint": self.config.checkpoint,
            "base_model": self.config.base_model,
            "max_seq_length": self.config.max_seq_length,
            "overflow": self.config.overflow,
            "use_generate": self.use_generate,
        }


class ScriptedFilter(EvidenceFilter):
    """Filter driven by a supplied scoring callable. Used by tests and the smoke
    run so the filter *interface* can be exercised without Flan-T5 weights."""

    name = "scripted"

    def __init__(
        self, score_fn, prompts: Optional[PromptSet] = None, threshold: float = KEEP_THRESHOLD
    ) -> None:
        self.score_fn = score_fn
        self.prompts = prompts or DEFAULT_PROMPTS
        self.threshold = threshold

    def decide(self, question: Question, candidates: Sequence[Evidence]) -> List[FilterDecision]:
        scores = [
            float(self.score_fn(self.prompts.render_filter_prompt(question, candidate), question, candidate))
            for candidate in candidates
        ]
        return decisions_from_probabilities(scores, self.threshold)


@register_filter("rag2_perplexity")
def _build_rag2_filter(config: FilterConfig, prompts: Optional[PromptSet] = None) -> EvidenceFilter:
    return RAG2PerplexityFilter(config, prompts)
