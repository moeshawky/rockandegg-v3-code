"""RockAndEgg-313M-v3 SystemOne decision model.

A headless Qwen3 backbone plus a native typed-decision head. There is **no
language-modeling head and no token-generation path**: the model emits a
probability distribution and a confidence value per question.

Physical contract:

* encode the state exactly once into a ``DynamicCache`` and never re-encode it
  per question or option;
* isolate every option in its own batch row and mix options only inside the
  head;
* use stock ``DynamicCache`` and the stock ``Qwen3Model`` forward path — no
  custom 4D attention mask and no RoPE reset.

The optional components (late state cross-attention, option mixer, cosine
prior) are OFF by default and are configured through the ``systemone_head``
block in ``config.json``.

Loading requires ``trust_remote_code=True``::

    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

See ``systemone_inference.py`` in this repository for the text-level
``/v1/systemone``-shaped adapter.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from .configuration_rockandegg_v3 import RockAndEggV3Config


def _apply_backbone_freeze(module: object, *args: object) -> None:
    """Re-apply the declared freeze after weights are assigned.

    ``from_pretrained`` builds the model under ``init_empty_weights`` and then
    replaces meta parameters with loaded tensors. A freeze applied only in
    ``__init__`` is therefore lost; this load hook restores it.
    """
    head = getattr(module, "head_config", {}) or {}
    if not head.get("backbone_frozen", True):
        return
    backbone = getattr(module, "backbone", None)
    if backbone is not None:
        for param in backbone.parameters():
            param.requires_grad_(False)


class RockAndEggV3SystemOneModel(PreTrainedModel):
    """Headless Qwen3 backbone plus a SystemOne decision head.

    The backbone emits hidden states for a single state prefill; its KV cache
    is forked across all option rows so the state is encoded exactly once. Each
    ``(question, option)`` pair is a separate batch row. The head reduces those
    rows to one scaled logit vector per question.

    Attributes:
        config: resolved :class:`RockAndEggV3Config`.
        backbone: frozen ``Qwen3Model`` (``architectures == ["Qwen3Model"]``).
    """

    config_class = RockAndEggV3Config
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = False
    _no_split_modules: ClassVar[list[str]] = ["Qwen3DecoderLayer"]

    def __init__(self, config: RockAndEggV3Config) -> None:
        """Build the backbone and the decision head from the configuration."""
        super().__init__(config)
        head = config.systemone_head
        self.head_config = dict(head)

        self.backbone = Qwen3Model(config.backbone_config)
        # config.json declares backbone_frozen; honour it so the published
        # artifact does not advertise a freeze it never performed.
        _apply_backbone_freeze(self)
        self.register_load_state_dict_post_hook(_apply_backbone_freeze)

        d = config.hidden_size
        self.qtype_emb = nn.Embedding(head["qtype_embedding"], d)
        self.state_norm = nn.RMSNorm(d)
        self.opt_norm = nn.RMSNorm(d)
        self.final_norm = nn.RMSNorm(d)

        if head["use_cross_attn"]:
            self.state_cross_attn = nn.MultiheadAttention(
                d, num_heads=head["cross_attn_heads"], batch_first=True
            )
        if head["use_option_mixer"]:
            layer = nn.TransformerEncoderLayer(
                d,
                nhead=head["cross_attn_heads"],
                dim_feedforward=head["mixer_ffn"],
                dropout=head["mixer_dropout"],
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.option_mixer = nn.TransformerEncoder(layer, head["mixer_layers"])

        self.scorer = nn.Sequential(
            nn.Linear(d, head["scorer_hidden"]),
            nn.SiLU(),
            nn.Linear(head["scorer_hidden"], 1),
        )
        # Cosine prior is off by default: the scale starts at zero and the flag
        # gates its computation.
        self.prior_scale = nn.Parameter(torch.zeros(()))
        self.log_temp = nn.Parameter(torch.zeros(head["qtype_embedding"]))

        self.post_init()

    def to(self, *args: Any, **kwargs: Any) -> RockAndEggV3SystemOneModel:
        """Re-apply the declared freeze after dtype/device moves."""
        result = super().to(*args, **kwargs)
        _apply_backbone_freeze(result)
        return result

    def eval(self, *args: Any, **kwargs: Any) -> RockAndEggV3SystemOneModel:
        """Re-apply the declared freeze after from_pretrained replaces weights.

        Transformers 5 assigns fresh nn.Parameter objects during loading, so a
        freeze applied only in __init__ or via a load-state hook is lost.
        from_pretrained always calls eval() afterwards, making this the reliable
        point to honor ``backbone_frozen``. Inference logits are unchanged.
        """
        result = super().eval(*args, **kwargs)
        _apply_backbone_freeze(result)
        return result

    @staticmethod
    def _last_non_pad(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the last non-padding hidden state for each batch row."""
        idx = mask.long().cumsum(dim=1).argmax(dim=1)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]

    def encode_state(
        self, state_ids: torch.Tensor, state_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, Any]:
        """Encode the state once and return normalized hidden states + cache.

        This is the single state pass. The resulting cache is the progenitor
        that every option row inherits via ``batch_repeat_interleave``. When a
        mask is supplied (1 = real token, 0 = padding) it is forwarded to the
        backbone, so padding never enters attention and the readout vector is
        taken from the last real token.

        Args:
            state_ids: ``[1, S]`` state token ids.
            state_mask: optional ``[1, S]`` padding mask; ``1`` marks a real
                token.

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
        cache: Any,
        state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run all option rows through a forked cache; read their final states."""
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

        Every case gets one state row and all option rows are padded to the
        same width ``rows_per_case = M``. The single state cache is forked once
        with an integer repeat, so the state is still encoded only once per
        case and options remain isolated in their own rows. Padded option
        positions receive a large negative logit.

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
                if self.head_config["use_cosine_prior"]
                else torch.zeros(n, device=h.device, dtype=h.dtype)
            )
            q_emb = self.qtype_emb(
                torch.tensor([qtype_ids[i]], device=h.device)
            ).unsqueeze(1)
            h = h + q_emb
            if self.head_config["use_cross_attn"]:
                key_padding = ~state_mask[i : i + 1].bool()
                attended = self.state_cross_attn(
                    self.opt_norm(h),
                    state_hidden[i : i + 1],
                    state_hidden[i : i + 1],
                    key_padding_mask=key_padding,
                    need_weights=False,
                )[0]
                h = h + attended
            if self.head_config["use_option_mixer"]:
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
        """Reduce option-row hidden states to per-question logits."""
        if len(question_option_counts) != len(qtype_ids):
            msg = "question_option_counts and qtype_ids must have equal length"
            raise ValueError(msg)

        if state_mask is not None:
            state_vec = F.normalize(
                self._last_non_pad(state_hidden, state_mask), dim=-1
            )
        else:
            state_vec = F.normalize(state_hidden[:, -1, :], dim=-1)

        outputs: list[torch.Tensor] = []
        offset = 0
        for n_k, qtype in zip(question_option_counts, qtype_ids, strict=True):
            h = opt[offset : offset + n_k].unsqueeze(0)
            offset += n_k

            if self.head_config["use_cosine_prior"]:
                prior = (F.normalize(h.squeeze(0), dim=-1) @ state_vec.T).squeeze(-1)
            else:
                prior = torch.zeros(h.size(1), device=h.device, dtype=h.dtype)

            q_emb = self.qtype_emb(torch.tensor([qtype], device=h.device))
            h = h + q_emb.unsqueeze(1)
            if self.head_config["use_cross_attn"]:
                key_padding = ~state_mask.bool() if state_mask is not None else None
                attended = self.state_cross_attn(
                    self.opt_norm(h),
                    state_hidden,
                    state_hidden,
                    key_padding_mask=key_padding,
                    need_weights=False,
                )[0]
                h = h + attended
            if self.head_config["use_option_mixer"]:
                h = self.final_norm(self.option_mixer(h))
            score = self.scorer(h).squeeze(0).squeeze(-1)
            temp = self.log_temp[qtype].exp().clamp(0.5, 5.0)
            outputs.append((score + self.prior_scale * prior) / temp)
        return outputs
