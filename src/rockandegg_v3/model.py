"""RockAndEgg-313M-v3 native SystemOne decision model.

Physical contract (AGENTS.md Iron Law):

* no LM head and no token generation -- pure typed decisions;
* encode the state once into a ``DynamicCache``, never re-encode per question
  or option;
* isolate options in batch rows and mix options only in the head;
* no custom 4D attention mask and no RoPE reset -- stock ``DynamicCache`` plus
  stock ``Qwen3Model`` forward paths only.

Optional components (late cross-attention, option mixer, cosine prior) are
gated by :class:`~rockandegg_v3.config.HeadConfig` and are OFF by default per
the operator's evidence-based decision.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .config import MODEL_PATH, NUM_QTYPES, HeadConfig

if TYPE_CHECKING:
    from transformers import DynamicCache


class RockAndEggV3SystemOne(nn.Module):
    """Headless Qwen3 backbone plus a SystemOne decision head.

    The backbone emits hidden states for a single state prefill; its KV cache
    is forked across all option rows so the state is encoded exactly once.
    Each ``(question, option)`` pair is a separate batch row. The head reduces
    those rows to one logit vector per question. No generation path exists.

    Attributes:
        config: resolved head configuration.
        backbone: frozen ``Qwen3Model`` (``architectures == ['Qwen3Model']``).
    """

    def __init__(
        self, config: HeadConfig | None = None, model_path: str = MODEL_PATH
    ) -> None:
        """Build the model from a head config and the local checkpoint."""
        super().__init__()
        self.config = config if config is not None else HeadConfig()
        dtype = self.config.torch_dtype
        self.backbone = AutoModel.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa"
        )
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        d = self.config.d_model
        self.qtype_emb = nn.Embedding(NUM_QTYPES, d)
        self.state_norm = nn.RMSNorm(d)
        self.opt_norm = nn.RMSNorm(d)
        self.final_norm = nn.RMSNorm(d)

        if self.config.use_cross_attn:
            self.state_cross_attn = nn.MultiheadAttention(
                d, num_heads=self.config.cross_attn_heads, batch_first=True
            )
        if self.config.use_option_mixer:
            layer = nn.TransformerEncoderLayer(
                d,
                nhead=self.config.cross_attn_heads,
                dim_feedforward=self.config.mixer_ffn,
                dropout=self.config.mixer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.option_mixer = nn.TransformerEncoder(layer, self.config.mixer_layers)

        self.scorer = nn.Sequential(
            nn.Linear(d, self.config.scorer_hidden),
            nn.SiLU(),
            nn.Linear(self.config.scorer_hidden, 1),
        )
        # Cosine prior is off by default: scale starts at zero and the flag
        # gates its computation. When enabled by the operator, train this up.
        self.prior_scale = nn.Parameter(torch.zeros(()))
        self.log_temp = nn.Parameter(torch.zeros(NUM_QTYPES))

        # from_pretrained only casts the backbone. Cast the head too, otherwise
        # fp16 inputs meet fp32 weights and normalization/logit paths diverge.
        self.to(dtype)

    @staticmethod
    def _last_non_pad(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the last non-padding hidden state for each batch row."""
        idx = mask.long().cumsum(dim=1).argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    def encode_state(
        self, state_ids: torch.Tensor, state_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Encode the state once and return normalized hidden states + cache.

        This is the single state pass. The resulting cache is the progenitor
        that every option row inherits via ``batch_repeat_interleave``. When a
        mask is supplied (1 = real token, 0 = padding), it is forwarded to the
        backbone so padding never enters attention and the readout vector is
        taken from the last real token. Omitting it preserves the legacy
        unmasked path used by the numeric gate harness.

        Args:
            state_ids: ``[1, S]`` state token ids.
            state_mask: optional ``[1, S]`` mask; ``1`` marks a real token.

        Returns:
            Normalized ``[1, S, d]`` hidden states and the progenitor cache.
        """
        out = self.backbone(
            input_ids=state_ids, attention_mask=state_mask, use_cache=True
        )
        state_hidden = self.state_norm(out.last_hidden_state)
        return state_hidden, out.past_key_values

    def _read_options(
        self,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
        cache: DynamicCache,
        state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run all option rows through a forked cache; read their final states.

        The cached-state portion of the full mask mirrors ``state_mask`` when
        present, so an option row cannot attend to padded state positions.
        """
        m = suffix_ids.shape[0]
        fork = copy.deepcopy(cache)
        fork.batch_repeat_interleave(m)  # in-place; returns None
        if state_mask is not None:
            state_part = state_mask.expand(m, -1)
        else:
            state_part = torch.ones(
                (m, cache.get_seq_length()),
                dtype=suffix_mask.dtype,
                device=suffix_mask.device,
            )
        full_mask = torch.cat([state_part, suffix_mask], dim=1)
        out = self.backbone(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=fork,
            use_cache=False,
        )
        return self._last_non_pad(out.last_hidden_state, suffix_mask)

    def forward_case(
        self,
        state_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
        question_option_counts: list[int],
        qtype_ids: list[int],
        state_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Return one scaled logit vector per question from a single state pass.

        Args:
            state_ids: ``[1, S]`` state token ids.
            suffix_ids: ``[M, L]`` option-row token ids.
            suffix_mask: ``[M, L]`` option-row padding mask.
            question_option_counts: number of option rows per question.
            qtype_ids: question type id (0 choice, 1 score, 2 noul) per question.
            state_mask: optional ``[1, S]`` state padding mask.

        Returns:
            One ``[n_k]`` logit tensor per question, in input order.
        """
        if len(question_option_counts) != len(qtype_ids):
            msg = "question_option_counts and qtype_ids must have equal length"
            raise ValueError(msg)

        state_hidden, cache = self.encode_state(state_ids, state_mask)
        opt = self._read_options(suffix_ids, suffix_mask, cache, state_mask)
        return self._head_forward(
            opt, state_hidden, question_option_counts, qtype_ids, state_mask
        )

    def forward_batch(
        self,
        state_ids: torch.Tensor,
        state_mask: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
        rows_per_case: int,
        real_counts: list[int],
        qtype_ids: list[int],
    ) -> torch.Tensor:
        """Score ``B`` cases in one backbone pass, returning ``[B, M]`` logits.

        This is the batched analogue of :meth:`forward_case`. Every case gets
        one state row, and all option rows are padded to the same width
        ``rows_per_case = M``. The single state cache is forked once with an
        integer repeat (every case claims exactly M rows), so the state is
        still encoded only once per case and options remain isolated in their
        own rows. Padded option positions receive a large negative logit so a
        zero padded target contributes no probability mass.

        Args:
            state_ids: ``[B, S]`` state token ids.
            state_mask: ``[B, S]`` state padding mask.
            suffix_ids: ``[B*M, L]`` option-row token ids, case-major order.
            suffix_mask: ``[B*M, L]`` option-row padding mask.
            rows_per_case: padded option width ``M`` for every case.
            real_counts: actual option count per case (``<= M``).
            qtype_ids: question type id per case.

        Returns:
            ``[B, M]`` logits; columns ``>= real_counts[b]`` are suppressed.
        """
        b = int(state_ids.shape[0])
        if len(real_counts) != b or len(qtype_ids) != b:
            msg = "real_counts and qtype_ids must have one entry per case"
            raise ValueError(msg)

        state_hidden, cache = self.encode_state(state_ids, state_mask)
        fork = copy.deepcopy(cache)
        fork.batch_repeat_interleave(rows_per_case)
        state_part = state_mask.repeat_interleave(rows_per_case, dim=0)
        full_mask = torch.cat([state_part, suffix_mask], dim=1)
        out = self.backbone(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=fork,
            use_cache=False,
        )
        opt = self._last_non_pad(out.last_hidden_state, suffix_mask).view(
            b, rows_per_case, -1
        )

        state_vec = F.normalize(self._last_non_pad(state_hidden, state_mask), dim=-1)
        logits = torch.full(
            (b, rows_per_case), -1e4, device=opt.device, dtype=opt.dtype
        )
        for i in range(b):
            n = int(real_counts[i])
            h = opt[i : i + 1, :n]
            prior = (
                (F.normalize(h.squeeze(0), dim=-1) @ state_vec[i : i + 1].T).squeeze(-1)
                if self.config.use_cosine_prior
                else torch.zeros(n, device=h.device, dtype=h.dtype)
            )
            q_emb = self.qtype_emb(
                torch.tensor([qtype_ids[i]], device=h.device)
            ).unsqueeze(1)
            h = h + q_emb
            if self.config.use_cross_attn:
                key_padding = ~state_mask[i : i + 1].bool()
                attended = self.state_cross_attn(
                    self.opt_norm(h),
                    state_hidden[i : i + 1],
                    state_hidden[i : i + 1],
                    key_padding_mask=key_padding,
                    need_weights=False,
                )[0]
                h = h + attended
            if self.config.use_option_mixer:
                h = self.final_norm(self.option_mixer(h))
            score = self.scorer(h).squeeze(0).squeeze(-1)
            temp = self.log_temp[qtype_ids[i]].exp().clamp(0.5, 5.0)
            logits[i, :n] = (score + self.prior_scale * prior) / temp
        return logits

    def _head_forward(
        self,
        opt: torch.Tensor,
        state_hidden: torch.Tensor,
        question_option_counts: list[int],
        qtype_ids: list[int],
        state_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Reduce option-row hidden states to per-question logits.

        Shared by the single-device and multi-device forward paths so both
        produce identical logits from identical option vectors. The readout of
        the state vector uses the last real token when a mask is supplied.
        """
        if len(question_option_counts) != len(qtype_ids):
            msg = "question_option_counts and qtype_ids must have equal length"
            raise ValueError(msg)

        if state_mask is not None:
            state_vec = F.normalize(
                self._last_non_pad(state_hidden, state_mask), dim=-1
            )
        else:
            state_vec = F.normalize(state_hidden[:, -1, :], dim=-1)
        logits: list[torch.Tensor] = []
        offset = 0
        for n_k, qtype in zip(question_option_counts, qtype_ids, strict=True):
            h = opt[offset : offset + n_k].unsqueeze(0)
            offset += n_k

            if self.config.use_cosine_prior:
                prior = (F.normalize(h.squeeze(0), dim=-1) @ state_vec.T).squeeze(-1)
            else:
                prior = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)

            q_emb = self.qtype_emb(torch.tensor([qtype], device=h.device))
            h = h + q_emb.unsqueeze(1)

            if self.config.use_cross_attn:
                key_padding = ~state_mask.bool() if state_mask is not None else None
                attended = self.state_cross_attn(
                    self.opt_norm(h),
                    state_hidden,
                    state_hidden,
                    key_padding_mask=key_padding,
                    need_weights=False,
                )[0]
                h = h + attended
            if self.config.use_option_mixer:
                h = self.final_norm(self.option_mixer(h))

            score = self.scorer(h).squeeze(0).squeeze(-1)
            temp = self.log_temp[qtype].exp().clamp(0.5, 5.0)
            logits.append((score + self.prior_scale * prior) / temp)
        return logits
