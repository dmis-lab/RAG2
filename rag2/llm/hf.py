"""HuggingFace ``transformers`` backbone.

Covers the paper's open-weight backbones -- Llama-3-8B-Instruct
(``meta-llama/Meta-Llama-3-8B-Instruct``, paper footnote 2) and Meerkat-7B --
for generation and for the token-level scoring that Equation 4 needs.

torch/transformers are imported lazily so that the rest of the package stays
importable (and testable) without them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from ..config import LLMConfig
from .base import LLM, ScoredSequence, register_llm


def _resolve_device(name: str):
    import torch

    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(name: str):
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": None,
    }.get(name, torch.float32)


class HFCausalLM(LLM):
    def __init__(self, config: LLMConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.name = config.model
        self.revision = config.revision or "main"
        self.device = _resolve_device(config.device)
        dtype = _resolve_dtype(config.dtype)

        load_kwargs: Dict[str, Any] = {"revision": self.revision}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
        self.model.eval()
        self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Generation needs left padding so that every sequence's last position is
        # the true final token.
        self.tokenizer.padding_side = "left"
        self.use_chat_template = bool(config.chat_template) and getattr(
            self.tokenizer, "chat_template", None
        )
        self._resolved_revision = getattr(self.model.config, "_commit_hash", None) or self.revision

    # -- prompting ---------------------------------------------------------
    def _wrap(self, prompt: str) -> str:
        """Render through the model's chat template when it has one.

        Instruct-tuned backbones (both of the paper's open-weight models) expect
        their own template; a raw completion prompt degrades them badly.
        """
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    # -- generation --------------------------------------------------------
    def generate(self, prompts: Sequence[str], **kwargs: Any) -> List[str]:
        import torch

        if not prompts:
            return []
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.config.temperature))
        batch_size = int(kwargs.get("batch_size", self.config.batch_size)) or 1

        outputs: List[str] = []
        for start in range(0, len(prompts), batch_size):
            chunk = [self._wrap(p) for p in prompts[start : start + batch_size]]
            encoded = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=bool(self.config.max_input_tokens),
                max_length=self.config.max_input_tokens or None,
                add_special_tokens=not self.use_chat_template,
            ).to(self.device)
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=self.config.top_p)
            else:
                # Paper appendix A.3: greedy decoding, temperature 0.
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                generated = self.model.generate(**encoded, **gen_kwargs)
            prompt_len = encoded["input_ids"].shape[1]
            for row in generated:
                outputs.append(
                    self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                )
        return outputs

    # -- scoring (Equation 4) ---------------------------------------------
    def score(self, prompt: str, continuation: str) -> ScoredSequence:
        """Sum of log P(continuation_i | prompt, continuation_<i).

        Prompt tokens are masked out, so the returned scores -- and therefore the
        length normalisation in Equation 4 -- cover the continuation only.
        """
        import torch

        prompt_ids = self.tokenizer(
            self._wrap(prompt), add_special_tokens=not self.use_chat_template
        )["input_ids"]
        full_ids = prompt_ids + self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            return ScoredSequence(token_logprobs=[], num_tokens=0)

        max_len = self.config.max_input_tokens or getattr(self.model.config, "max_position_embeddings", 0)
        if max_len and len(full_ids) > max_len:
            # Truncate from the left so the continuation, whose tokens are being
            # scored, always survives intact.
            cut = len(full_ids) - max_len
            full_ids = full_ids[cut:]
            prompt_ids = prompt_ids[cut:] if cut < len(prompt_ids) else []

        input_ids = torch.tensor([full_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits.float()
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = input_ids[0, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(len(prompt_ids) - 1, 0)
        selected = gathered[start:]
        values = [float(v) for v in selected.tolist()]
        return ScoredSequence(token_logprobs=values, num_tokens=len(values))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "revision": str(self._resolved_revision),
            "class": "HFCausalLM",
            "dtype": self.config.dtype,
            "chat_template": bool(self.use_chat_template),
        }


@register_llm("huggingface")
def _build_hf(config: LLMConfig) -> LLM:
    return HFCausalLM(config)


@register_llm("hf")
def _build_hf_alias(config: LLMConfig) -> LLM:
    return HFCausalLM(config)
