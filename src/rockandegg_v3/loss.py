"""Training objective for typed SystemOne decisions.

Soft-label cross-entropy on every question; squared Wasserstein (EMD) on score
questions only; per-question-type temperature scaling lives in the model.
Option dropout removes non-gold distractors with probability ``p`` while always
retaining the gold option. No RLCD and no REINFORCE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import QTYPE_SCORE


def option_dropout(
    target_probs: torch.Tensor,
    gold_idx: int,
    drop_prob: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop non-gold options from a soft-label target and renormalize.

    The gold option is always kept. Each non-gold option is dropped
    independently with probability ``drop_prob``. The remaining mass is
    renormalized to sum to one. Draws are taken on CPU so a CPU ``generator``
    stays reproducible even when ``target_probs`` lives on a GPU.

    Args:
        target_probs: ``[n]`` soft-label probability vector.
        gold_idx: index of the gold option, always retained.
        drop_prob: probability of dropping each non-gold option.
        generator: optional torch generator for reproducible draws.

    Returns:
        A ``[n]`` target vector with dropped entries zeroed and mass restored.
    """
    if drop_prob <= 0.0:
        return target_probs
    n = target_probs.shape[-1]
    draws = torch.rand(n, generator=generator)
    keep = draws >= drop_prob
    keep[gold_idx] = True
    dropped = target_probs * keep.to(
        device=target_probs.device, dtype=target_probs.dtype
    )
    total = dropped.sum(dim=-1, keepdim=True)
    return dropped / total.clamp_min(1e-12)


def verdict_pact_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    qtype: int,
    emd_weight: float = 0.5,
) -> torch.Tensor:
    """Soft-label cross-entropy plus squared EMD on score questions.

    The EMD term compares cumulative distributions of the predicted and target
    probabilities and is applied only when ``qtype == QTYPE_SCORE``. Choice and
    noul questions receive plain soft-label CE.

    Args:
        logits: ``[n]`` unscaled logits over options.
        target_probs: ``[n]`` target probability distribution.
        qtype: question type id (1 selects the EMD term).
        emd_weight: coefficient of the squared-EMD term.

    Returns:
        Scalar loss tensor.
    """
    log_p = F.log_softmax(logits, dim=-1)
    ce = -(target_probs * log_p).sum()
    if qtype != QTYPE_SCORE:
        return ce
    probs = log_p.exp()
    cdf_diff = torch.cumsum(probs, dim=-1) - torch.cumsum(target_probs, dim=-1)
    return ce + emd_weight * (cdf_diff**2).mean()
