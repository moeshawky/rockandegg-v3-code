"""Inference surface shaped like ``/v1/systemone``.

One request carries a state plus typed questions. The state is encoded exactly
once; every option row is produced from that single KV cache. No text is
generated. Confidence is the maximum softmax probability, optionally rescaled
by a held-out-fitted temperature; the ECE contract is measured, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from .config import MARKER_ID, QTYPE_CHOICE, QTYPE_NOUL, QTYPE_SCORE

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .model import RockAndEggV3SystemOne

_QTYPE_BY_NAME = {
    "choice": QTYPE_CHOICE,
    "score": QTYPE_SCORE,
    "noul": QTYPE_NOUL,
}


@dataclass(frozen=True)
class Question:
    """One typed question with its candidate options."""

    type: str
    options: list[str]
    instruction: str = ""


@dataclass(frozen=True)
class SystemOneRequest:
    """A single /v1/systemone request: one state, many typed questions."""

    state: str
    questions: list[Question]


@dataclass(frozen=True)
class QuestionDecision:
    """Per-question decision output."""

    type: str
    probabilities: list[float]
    chosen_key: str
    confidence: float


@dataclass(frozen=True)
class SystemOneResponse:
    """Per-request decision output: one decision per input question."""

    decisions: list[QuestionDecision]


class SystemOneInference:
    """Stateless adapter from text requests to SystemOne logits.

    The adapter builds token ids for the single state and for every option row,
    calls the model once, and formats the result. It holds no generation path.

    Attributes:
        model: the decision model.
        tokenizer: tokenizer whose vocabulary contains ``<|im_end|>``.
        device: device of the model parameters.
        confidence_temperature: optional held-out-fitted temperature for the
            reported confidence. Defaults to 1.0 (raw softmax).
    """

    def __init__(
        self,
        model: RockAndEggV3SystemOne,
        tokenizer: PreTrainedTokenizerBase,
        max_state_len: int = 256,
        max_option_len: int = 64,
        confidence_temperature: float = 1.0,
    ) -> None:
        """Initialize the inference adapter."""
        self.model = model
        self.tokenizer = tokenizer
        self.max_state_len = max_state_len
        self.max_option_len = max_option_len
        self.confidence_temperature = confidence_temperature
        self.device = next(model.parameters()).device
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    def _encode_state(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and right-pad the state to ``S``; return ids and a mask."""
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_state_len,
        )
        ids = cast("torch.Tensor", enc["input_ids"])
        mask = cast("torch.Tensor", enc["attention_mask"])
        if ids.shape[1] < self.max_state_len:
            pad_n = self.max_state_len - ids.shape[1]
            ids = torch.cat(
                [ids, torch.full((1, pad_n), self.pad_id, dtype=ids.dtype)], dim=1
            )
            mask = torch.cat([mask, torch.zeros((1, pad_n), dtype=mask.dtype)], dim=1)
        return ids.to(self.device), mask.to(self.device)

    def _encode_options(
        self, questions: list[Question]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Build padded option rows with the marker as the readout token."""
        rows: list[list[int]] = []
        counts: list[int] = []
        for question in questions:
            counts.append(len(question.options))
            for option in question.options:
                text = f"{question.instruction} [SEP] {option} [SEP]"
                ids = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, self.max_option_len - 1),
                )["input_ids"]
                rows.append([*ids, MARKER_ID])
        width = max(len(row) for row in rows)
        suffix_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        suffix_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            suffix_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            suffix_mask[i, : len(row)] = 1
        return suffix_ids.to(self.device), suffix_mask.to(self.device), counts

    def __call__(self, request: SystemOneRequest) -> SystemOneResponse:
        """Run one request through a single state encoding."""
        qtype_ids = []
        for question in request.questions:
            if question.type not in _QTYPE_BY_NAME:
                msg = f"unknown question type: {question.type!r}"
                raise ValueError(msg)
            if not question.options:
                msg = "a question must have at least one option"
                raise ValueError(msg)
            qtype_ids.append(_QTYPE_BY_NAME[question.type])

        state_ids, state_mask = self._encode_state(request.state)
        suffix_ids, suffix_mask, counts = self._encode_options(request.questions)
        with torch.inference_mode():
            logits_list = self.model.forward_case(
                state_ids, suffix_ids, suffix_mask, counts, qtype_ids, state_mask
            )

        decisions: list[QuestionDecision] = []
        for question, logits in zip(request.questions, logits_list, strict=True):
            scaled = logits.float() / self.confidence_temperature
            probs = F.softmax(scaled, dim=-1)
            idx = int(probs.argmax().item())
            decisions.append(
                QuestionDecision(
                    type=question.type,
                    probabilities=probs.tolist(),
                    chosen_key=question.options[idx],
                    confidence=float(probs.max().item()),
                )
            )
        return SystemOneResponse(decisions=decisions)
