"""Configuration for the RockAndEgg-313M-v3 SystemOne decision model.

The released artifact is a headless ``Qwen3Model`` backbone plus a native
SystemOne decision head. The backbone keeps the stock Qwen3 configuration
fields at the top level of ``config.json``; the head is described by the
``systemone_head`` block.

Loading the model requires ``trust_remote_code=True`` because the decision head
is not part of any stock Transformers architecture::

    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

The model has **no language-modeling head and no generation path**. It returns
typed decisions, not text.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

# Head defaults. Every value here is mirrored by the ``systemone_head`` block
# in config.json; the block always wins when present.
# NOTE: max_state_len / max_option_len are documentation metadata only.
# The qualified truncation limits (256/64) are enforced by the serving
# adapter and server constants; editing these keys does not alter runtime
# tokenization. backbone_frozen IS consumed by the modeling module.
HEAD_DEFAULTS: dict[str, Any] = {
    "qtype_embedding": 3,
    "scorer_hidden": 256,
    "cross_attn_heads": 16,
    "mixer_layers": 2,
    "mixer_ffn": 2048,
    "mixer_dropout": 0.1,
    "use_cross_attn": False,
    "use_option_mixer": False,
    "use_cosine_prior": False,
    "max_state_len": 256,
    "max_option_len": 64,
    "backbone_frozen": True,
    "marker_token": "<|im_end|>",
    "marker_id": 151645,
}

# Keys in config.json that describe the assembled v3 model rather than the
# Qwen3 backbone, and must not be forwarded to Qwen3Config.
_BACKBONE_EXCLUDE = {
    "architectures",
    "auto_map",
    "model_type",
    "systemone_head",
    "transformers_version",
    "dtype",
    "torch_dtype",
}


class RockAndEggV3Config(PretrainedConfig):
    """Configuration of the assembled backbone + decision head.

    Attributes:
        systemone_head: resolved head parameters (see :data:`HEAD_DEFAULTS`).
        backbone_config: the stock Qwen3 configuration of the frozen backbone.
    """

    model_type = "rockandegg_v3_systemone"

    def __init__(
        self, systemone_head: dict[str, Any] | None = None, **kwargs: Any
    ) -> None:
        """Resolve head parameters and derive the backbone configuration."""
        head = dict(HEAD_DEFAULTS)
        if systemone_head:
            for key, value in systemone_head.items():
                if value is not None:
                    head[key] = value
        self.systemone_head = head
        kwargs.pop("systemone_head", None)
        super().__init__(**kwargs)

    @property
    def backbone_config(self) -> Qwen3Config:
        """Build the stock Qwen3 configuration for the frozen backbone."""
        kwargs = {
            key: value
            for key, value in self.to_dict().items()
            if key not in _BACKBONE_EXCLUDE
        }
        kwargs["model_type"] = "qwen3"
        kwargs["architectures"] = ["Qwen3Model"]
        return Qwen3Config(**kwargs)
