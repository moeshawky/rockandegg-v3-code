"""Text-level adapter for the RockAndEgg-313M-v3 SystemOne decision model.

One request carries a state plus typed questions. The state is encoded exactly
once; every option row is produced from that single KV cache. **No text is
generated.** Confidence is the maximum softmax probability, optionally rescaled
by a held-out-fitted temperature; the ECE contract is measured, never assumed.

``Question.answer_keys`` is an optional, backward-compatible stable-identity
facility. When a caller renders *scoring text* that differs from the answer
identity it wants observed (for example a state-grounded Noul option), the
adapter returns the matching ``answer_keys`` entry as ``chosen_key`` instead of
the scoring string. Omitting it preserves the original behaviour exactly:
``chosen_key`` is the selected ``options`` string. Probabilities, ordering and
confidence are untouched by this facility.

Example::

    from transformers import AutoModel, AutoTokenizer
    from systemone_inference import Question, SystemOneInference, SystemOneRequest

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    adapter = SystemOneInference(model, tokenizer, max_state_len=256)

    request = SystemOneRequest(
        state="The deployment failed at 03:12 UTC.",
        questions=[
            Question(
                type="choice",
                instruction="What should the operator do next?",
                options=["Roll back the release", "Ignore it"],
            )
        ],
    )
    response = adapter(request)
    print(response.decisions[0].chosen_key, response.decisions[0].confidence)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# <|im_end|> is a single token (id 151645) in the shipped Qwen3 tokenizer and
# is the option marker. <|opt|> does not exist in the vocabulary.
MARKER_ID = 151645

QTYPE_CHOICE = 0
QTYPE_SCORE = 1
QTYPE_NOUL = 2

_QTYPE_BY_NAME = {
    "choice": QTYPE_CHOICE,
    "score": QTYPE_SCORE,
    "noul": QTYPE_NOUL,
}


@dataclass(frozen=True)
class Question:
    """One typed question with its candidate options.

    Attributes:
        type: ``choice``, ``score`` or ``noul``.
        options: scoring strings, in the order the head scores them.
        instruction: instruction prepended to every option row.
        answer_keys: optional stable identities, one per option. When set,
            ``chosen_key`` reports ``answer_keys[argmax]`` rather than the
            scoring string. Default ``None`` keeps the legacy observable key.
    """

    type: str
    options: list[str]
    instruction: str = ""
    answer_keys: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate the optional stable-key mapping at construction time."""
        if self.answer_keys is not None:
            if len(self.answer_keys) != len(self.options):
                msg = "answer_keys must have exactly one entry per option"
                raise ValueError(msg)
            if not all(isinstance(k, str) and k for k in self.answer_keys):
                msg = "answer_keys must be non-empty strings"
                raise ValueError(msg)

    def observable_key(self, index: int) -> str:
        """Return the caller-visible identity for a winning option index."""
        if self.answer_keys is not None:
            return self.answer_keys[index]
        return self.options[index]


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
        last_option_truncation: ``(truncated_rows, option_rows)`` recorded by
            the most recent ``_encode_options`` call. The HTTP layer reads it
            instead of tokenizing every option a second time.
    """

    def __init__(
        self,
        model: torch.nn.Module,
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
        self.last_option_truncation: tuple[int, int] = (0, 0)
        # State-length reporting for the most recent ``_encode_state`` call.
        # ``last_state_tokens`` is the exact token count of the input state
        # before any capacity limit is applied; ``last_state_truncated`` says
        # whether that count exceeded ``max_state_len``; ``last_physical_state_len``
        # is the padded execution length actually handed to the backbone. The
        # HTTP layer reads these instead of tokenizing the state a second time.
        self.last_state_tokens: int = 0
        self.last_state_truncated: bool = False
        self.last_physical_state_len: int = max_state_len

    def _encode_state(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize once, truncate to capacity, right-pad to the physical length.

        The tokenization is performed *without* truncation so the exact input
        length is known from the same pass that produces the ids; slicing to
        ``max_state_len`` is head truncation and yields the same ids the former
        ``truncation=True, max_length=...`` call produced. Capacity (how much
        state is admitted) and physical execution length (how long the cached
        sequence is) are both derived from ``max_state_len`` here and are
        reported separately via ``last_state_tokens`` /
        ``last_physical_state_len`` so callers are not forced to conflate them.
        """
        ids_list = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        n_real = len(ids_list)
        ids_list = ids_list[: self.max_state_len]
        n_kept = len(ids_list)
        pad_n = self.max_state_len - n_kept
        ids = torch.tensor(
            [ids_list + [self.pad_id] * pad_n], dtype=torch.long, device=self.device
        )
        mask = torch.tensor(
            [[1] * n_kept + [0] * pad_n], dtype=torch.long, device=self.device
        )
        self.last_state_tokens = n_real
        self.last_state_truncated = n_real > self.max_state_len
        self.last_physical_state_len = self.max_state_len
        return ids, mask

    def _encode_options(
        self, questions: list[Question]
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Build padded option rows with the marker as the readout token.

        Every option row is tokenized in ONE batched call. Each text is cut to
        ``limit + 1`` tokens: the ids are exactly the ones the former per-row
        ``max_length=limit`` call produced (head truncation), and the extra slot
        reveals whether anything was cut. The per-row truncation verdict is
        recorded in ``last_option_truncation`` so the HTTP layer can report it
        without tokenizing every option a second time.
        """
        limit = max(1, self.max_option_len - 1)
        counts: list[int] = []
        texts: list[str] = []
        for question in questions:
            counts.append(len(question.options))
            for option in question.options:
                texts.append(f"{question.instruction} [SEP] {option} [SEP]")

        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=limit + 1,
        )["input_ids"]

        rows: list[list[int]] = []
        truncated = 0
        for ids in encoded:
            if len(ids) > limit:
                truncated += 1
            rows.append([*ids[:limit], MARKER_ID])
        self.last_option_truncation = (truncated, len(rows))

        row_lens = [len(row) for row in rows]
        width = max(row_lens)
        # One tensor build for every row: flatten once, then copy slices into
        # the padded buffer (no per-row ``torch.tensor(row)`` construction).
        flat_ids = torch.tensor(
            [token for row in rows for token in row], dtype=torch.long
        )
        suffix_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        suffix_mask = torch.zeros((len(rows), width), dtype=torch.long)
        offset = 0
        for i, n in enumerate(row_lens):
            suffix_ids[i, :n] = flat_ids[offset : offset + n]
            suffix_mask[i, :n] = 1
            offset += n
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
                    chosen_key=question.observable_key(idx),
                    confidence=float(probs.max().item()),
                )
            )
        return SystemOneResponse(decisions=decisions)
