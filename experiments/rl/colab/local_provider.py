"""Local open-weight LLM provider for Colab RL experiments.

Serves a Qwen instruct model as a ``harnessx.Provider`` so the tau3 dialogue
harness (``DialogueRunner``) can roll out against a fine-tune-able model. Two
additional capabilities exist purely for the GRPO trainer:

- ``tokenize`` / ``apply_chat_template`` — reconstruct training inputs.
- ``compute_logprobs`` — per-token log-probs under the current LoRA policy and
  under the frozen base (reference) policy, so the trainer can compute
  importance ratios and the KL anchor without a second rollout.

GPU code (unsloth/transformers) is imported lazily inside methods so this module
imports cleanly on machines without torch (CI / unit tests). Tool calling is
prompt-instructed: the assistant replies with a JSON tool call or a plain
answer, parsed by :func:`parse_assistant`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from harnessx.events import Message, MessageRole
from harnessx.providers.base import Provider, ProviderResponse

_TOOL_CALL_SYSTEM = (
    "You control a set of tools. When a tool is needed, respond with ONLY a "
    "JSON object of the form: "
    '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}} '
    "Otherwise respond with a normal plain-text message. Never invent tools."
)


def _render_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    return "Available tools (JSON schema):\n" + json.dumps(tools, ensure_ascii=False)


def parse_assistant(
    text: str, tools: list[dict[str, Any]] | None = None
) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse a raw assistant response into (content, tool_calls).

    A JSON block with ``tool_call`` yields a single tool call; otherwise the
    text is returned as content. Returns ``(None, [])`` on empty output.
    """
    text = (text or "").strip()
    if not text:
        return None, []
    tool_names = {t.get("function", {}).get("name") for t in (tools or []) if t.get("function")}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            call = data.get("tool_call") or data.get("tool_calls")
            if isinstance(call, dict):
                name = call.get("name")
                if name and (not tool_names or name in tool_names):
                    return None, [{"name": name, "arguments": call.get("arguments") or {}, "id": f"call_{abs(hash(name)):x}"}]
            if isinstance(call, list):
                parsed = []
                for c in call:
                    if isinstance(c, dict) and c.get("name"):
                        parsed.append({"name": c["name"], "arguments": c.get("arguments") or {}, "id": f"call_{abs(hash(c['name'])):x}"})
                if parsed:
                    return None, parsed
    return text, []


class LocalQwenProvider(Provider):
    """Roll out a Qwen instruct model locally with optional LoRA adapter."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        device: str = "cuda",
        max_seq_length: int = 4096,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__(model_name)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self._model = None
        self._tokenizer = None
        self._base = None
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load model + tokenizer via unsloth (lazy; GPU only)."""
        if self._loaded:
            return
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:  # pragma: no cover - Colab only path
            raise RuntimeError("unsloth not installed; run on Colab with GPU") from exc

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=0,
            use_rslora=False,
            loftq_config=None,
        )
        self._device = self._model.device
        self._loaded = True

    def _ensure_ref(self) -> None:
        if self._base is None:
            from unsloth import FastLanguageModel

            base, _tok = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            base.eval()
            self._base = base

    # -- provider interface --------------------------------------------------

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.load()
        prompt = self._render(messages, tools)
        text, token_ids = self._generate(prompt)
        content, tool_calls = parse_assistant(text, tools)
        raw = {"text": text, "token_ids": token_ids}
        if tool_calls:
            raw["tool_calls"] = tool_calls
        return ProviderResponse(
            content=content or text,
            tool_calls=tool_calls,
            stop_reason="end_turn",
            raw=raw,
        )

    # -- GRPO helpers ---------------------------------------------------------

    def tokenize(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> list[int]:
        """Return prompt token ids for a message list (chat template applied)."""
        self.load()
        out = self._tokenizer.apply_chat_template(
            self._render(messages, tools),
            add_generation_prompt=True,
            tokenize=True,
        )
        return out[0]

    def compute_logprobs(
        self, prompt_tokens: list[int], completion_tokens: list[int]
    ) -> dict[str, list[float]]:
        """Per-token log-probs of ``completion_tokens`` (current + reference)."""
        import torch

        self.load()
        self._ensure_ref()
        self._model.eval()
        self._base.eval()

        input_ids = prompt_tokens + completion_tokens
        seq = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        position_ids = torch.arange(len(input_ids), device=self._device).unsqueeze(0)
        attention_mask = torch.ones_like(seq)

        with torch.no_grad():
            cur = self._model(input_ids=seq, attention_mask=attention_mask, position_ids=position_ids).logits
            ref = self._base(input_ids=seq, attention_mask=attention_mask, position_ids=position_ids).logits

        start = len(prompt_tokens)
        cur_lp = torch.log_softmax(cur[0, start - 1 : -1], dim=-1).gather(
            1, seq[0, start:].unsqueeze(-1)
        ).squeeze(-1).tolist()
        ref_lp = torch.log_softmax(ref[0, start - 1 : -1], dim=-1).gather(
            1, seq[0, start:].unsqueeze(-1)
        ).squeeze(-1).tolist()
        return {"logprobs": cur_lp, "ref_logprobs": ref_lp}

    def save_lora(self, path: str) -> None:

        self.load()
        self._model.save_pretrained_merged(path, tokenizer=self._tokenizer, save_method="lora")

    def load_lora(self, path: str) -> None:
        from unsloth import FastLanguageModel

        self.load()
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=path, max_seq_length=self.max_seq_length, dtype=None, load_in_4bit=True
        )

    # -- training helpers -------------------------------------------------------

    def training_logprobs(
        self, examples: list[tuple[list[int], list[int]]], micro_batch: int = 8
    ) -> tuple[list[Any], list[int], list[int]]:
        """Batched, gradient-enabled per-token log-probs of completions.

        Returns ``(logp_per_example, starts, lengths)`` where each element of
        ``logp_per_example`` is a 1-D tensor aligned to the completion tokens.
        """
        import torch

        self.load()
        self._model.train()
        outs: list[Any] = []
        starts: list[int] = []
        lengths: list[int] = []
        for i in range(0, len(examples), micro_batch):
            chunk = examples[i : i + micro_batch]
            input_ids, attention_mask, position_ids, chunk_starts, chunk_lengths = self._pad_examples(chunk)
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids).logits
            logits = logits.to(torch.bfloat16)
            logp = (
                logits[:, :-1, :].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                - logits[:, :-1, :].logsumexp(dim=-1)
            )
            for k in range(len(chunk)):
                s, L = chunk_starts[k], chunk_lengths[k]
                outs.append(logp[k, s - 1 : s - 1 + L])
                starts.append(s)
                lengths.append(L)
        return outs, starts, lengths

    def _pad_examples(
        self, examples: list[tuple[list[int], list[int]]]
    ) -> tuple[Any, Any, Any, list[int], list[int]]:
        import torch

        self.load()
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        max_len = max((len(p) + len(c)) for p, c in examples)
        batch = len(examples)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=self._device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=self._device)
        starts: list[int] = []
        lengths: list[int] = []
        for k, (prompt, completion) in enumerate(examples):
            seq = prompt + completion
            pad = max_len - len(seq)
            input_ids[k, pad:] = torch.tensor(seq, dtype=torch.long, device=self._device)
            attention_mask[k, pad:] = 1
            starts.append(pad + len(prompt))
            lengths.append(len(completion))
        position_ids = attention_mask.cumsum(dim=-1) - 1
        return input_ids, attention_mask, position_ids, starts, lengths

    # -- internals ------------------------------------------------------------

    def _render(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        system_parts: list[str] = []
        if tools:
            system_parts.append(_TOOL_CALL_SYSTEM + "\n" + _render_tools(tools))
        for m in messages:
            if m.role == MessageRole.SYSTEM and m.content:
                system_parts.append(m.content)
        conv: list[dict[str, Any]] = []
        if system_parts:
            conv.append({"role": "system", "content": "\n\n".join(system_parts)})
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                continue
            conv.append({"role": m.role.value, "content": m.content})
        return conv

    def _generate_text(self, conv: list[dict[str, Any]]) -> str:
        text, _ = self._generate(conv)
        return text

    def _generate(self, conv: list[dict[str, Any]]) -> tuple[str, list[int]]:
        from unsloth import FastLanguageModel

        inputs = self._tokenizer.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        outputs = FastLanguageModel.generate(
            self._model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        prompt_len = input_ids.shape[1]
        new_ids = outputs[0][prompt_len:].tolist()
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, new_ids

    def __repr__(self) -> str:
        return f"LocalQwenProvider(model={self.model_name!r}, device={self.device!r})"