"""Per-layer update accounting for the full-backbone campaign.

Provides a *measured* refutation of the null hypothesis "only the 268,805-param
head changed": snapshot every named parameter before a training segment, then
compute actual weight deltas and update-to-weight ratios grouped by the ten
backbone blocks, the final backbone norm, the embedding, and the decision head.

A frozen-backbone or head-only run cannot produce nonzero deltas in all ten
blocks — this module makes that property explicit and recorded, not assumed.
"""

from __future__ import annotations

import re
from typing import Any

import torch


def snapshot_params(model: Any, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Detached clones of every named parameter, moved off-GPU by default."""
    return {
        name: p.detach().to(device).clone()
        for name, p in model.named_parameters()
    }


def _group_of(name: str) -> str:
    """Map a parameter name to a reporting group."""
    m = re.search(r"backbone\.layers\.(\d+)\.", name)
    if m:
        return f"block_{int(m.group(1))}"
    if name == "backbone.norm.weight":
        return "backbone_final_norm"
    if name.startswith("backbone."):
        return "backbone_embedding"
    return "decision_head"


def compute_update_report(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
    step: int,
) -> dict[str, Any]:
    """Actual weight deltas + update/weight ratios per group.

    ratio = ||delta||_2 / max(||w_before||_2, eps). A nonzero ratio in every
    block_{0..9} and backbone_final_norm is the executed proof that the whole
    backbone updated — not a config label.
    """
    groups: dict[str, dict[str, Any]] = {}
    for name, w0 in before.items():
        w1 = after.get(name)
        if w1 is None:
            continue
        delta = (w1.float() - w0.float())
        dn = torch.linalg.vector_norm(delta).item()
        wn = torch.linalg.vector_norm(w0.float()).item()
        g = _group_of(name)
        rec = groups.setdefault(
            g,
            {"n_tensors": 0, "delta_l2": 0.0, "weight_l2": 0.0, "delta_sq": 0.0, "weight_sq": 0.0},
        )
        rec["n_tensors"] += 1
        rec["delta_l2"] += dn
        rec["weight_l2"] += wn
        rec["delta_sq"] += dn * dn
        rec["weight_sq"] += wn * wn

    out: dict[str, Any] = {}
    for g, rec in groups.items():
        ratio = (rec["delta_sq"] ** 0.5) / max(rec["weight_sq"] ** 0.5, 1e-12)
        out[g] = {
            "n_tensors": rec["n_tensors"],
            "delta_l2_sum": round(rec["delta_l2"], 10),
            "weight_l2_sum": round(rec["weight_l2"], 6),
            "update_to_weight_ratio": round(ratio, 10),
        }
    blocks = [f"block_{i}" for i in range(10)]
    present = [b for b in blocks if b in out and out[b]["delta_l2_sum"] > 0]
    norm_ok = out.get("backbone_final_norm", {}).get("delta_l2_sum", 0) > 0
    head_ok = out.get("decision_head", {}).get("delta_l2_sum", 0) > 0
    return {
        "at_step": step,
        "groups": out,
        "blocks_with_nonzero_delta": len(present),
        "all_ten_blocks_updated": len(present) == 10,
        "final_norm_updated": bool(norm_ok),
        "decision_head_updated": bool(head_ok),
        "full_backbone_proven": bool(len(present) == 10 and norm_ok and head_ok),
    }


def optimizer_membership(model: Any, opt: Any) -> dict[str, Any]:
    """Verify optimizer parameter groups match the intended trainable sets."""
    groups = opt.param_groups
    bb_param = sum(p.numel() for p in model.backbone.parameters())
    head_param = sum(
        p.numel() for n, p in model.named_parameters() if not n.startswith("backbone.")
    )
    group_sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    group_lrs = [g["lr"] for g in groups]
    all_bb_trainable = all(p.requires_grad for p in model.backbone.parameters())
    all_head_trainable = all(
        p.requires_grad for n, p in model.named_parameters() if not n.startswith("backbone.")
    )
    return {
        "backbone_params": bb_param,
        "head_params": head_param,
        "total_trainable": bb_param + head_param,
        "optimizer_group_sizes": group_sizes,
        "optimizer_group_lrs": group_lrs,
        "all_backbone_requires_grad": bool(all_bb_trainable),
        "all_head_requires_grad": bool(all_head_trainable),
        "optimizer_covers_all_trainable": sorted(group_sizes) == sorted([bb_param, head_param]),
    }
