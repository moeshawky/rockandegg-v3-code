"""Configuration and verified constants for the SystemOne decision head.

Shape constants come from the released config.json. The defaults encode the
release decisions: optional components off, frozen backbone, ECE bound 0.05.
External model paths are configurable through ``RAE_BACKBONE`` (environment)
or the corresponding CLI flags on the training/verification scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Frozen v2 backbone. Prefer a local directory for offline runs; the public
# Hub identifier is the portable fallback. Released v2 weights are F32 while
# the config declares bfloat16, so dtype is always passed explicitly (T4 has
# no bf16 -> fp16).
MODEL_PATH = os.environ.get("RAE_BACKBONE", "moeshawky/RockAndEgg-313M-v2")

# Shipped tokenizer: <|im_end|> is a single token (id 151645). <|opt|> is absent
# from the vocabulary and must not be used as an option marker.
MARKER_TOKEN = "<|im_end|>"  # noqa: S105 - tokenizer marker, not a credential
MARKER_ID = 151645

# Question type ids used by the qtype embedding and temperature scaling.
QTYPE_CHOICE = 0
QTYPE_SCORE = 1
QTYPE_NOUL = 2
NUM_QTYPES = 3

# Verified backbone shape (config.json; VALIDATION C1-C3).
D_MODEL = 1024
NUM_LAYERS = 10
KV_HEADS = 8
HEAD_DIM = 128

# Export-gate thresholds (prompt_v3_spec.md § Export gates).
ECE_BOUND = 0.05
ORDER_TOL = 1e-6


@dataclass(frozen=True)
class HeadConfig:
    """Operator-resolved configuration for the decision head.

    Attributes:
        d_model: hidden size of the backbone and head.
        use_cross_attn: enable late state cross-attention (default off).
        use_option_mixer: enable the permutation-equivariant option mixer
            (default off; source paper measured it neutral).
        use_cosine_prior: enable the cosine retrieval prior / PFR (default
            off; C outperforms PFR under every objective in the source work).
        cross_attn_heads: attention heads for cross-attention and mixer.
        mixer_layers: number of TransformerEncoder layers.
        mixer_ffn: feed-forward width of each mixer layer.
        mixer_dropout: dropout inside mixer layers (inactive in eval mode).
        scorer_hidden: width of the nonlinear scorer bottleneck.
        option_dropout: probability of dropping a non-gold distractor.
        emd_weight: weight of squared EMD on score questions.
        ece_bound: maximum allowed expected calibration error.
        order_tol: maximum allowed permutation max-abs-difference.
        freeze_backbone: keep Qwen3 weights frozen (operator decision).
        dtype_name: ``"fp16"`` on Turing/T4 (no bf16) or ``"fp32"``.
    """

    d_model: int = D_MODEL
    use_cross_attn: bool = False
    use_option_mixer: bool = False
    use_cosine_prior: bool = False
    cross_attn_heads: int = 16
    mixer_layers: int = 2
    mixer_ffn: int = 2048
    mixer_dropout: float = 0.1
    scorer_hidden: int = 256
    option_dropout: float = 0.3
    emd_weight: float = 0.5
    ece_bound: float = ECE_BOUND
    order_tol: float = ORDER_TOL
    freeze_backbone: bool = True
    dtype_name: str = "fp16"

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype selected by ``dtype_name``."""
        return torch.float16 if self.dtype_name == "fp16" else torch.float32
