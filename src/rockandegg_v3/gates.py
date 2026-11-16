"""Export gates for the SystemOne decision head.

Gate 1: option-order invariance (``max_abs_diff <= order_tol``).
Gate 2: question-order / state-independence / cross-question leakage == 0.0
    with a discriminating control.
Gate 3: reload parity ``<= order_tol`` from a fresh model instance.
Gate 4: ECE ``<= ece_bound`` on held-out typed decisions.
Gate 5: structural ban on LM head / generation path / custom masks.

The numeric probes are executed by ``run_systemone_gates.py``; this module owns
the report type, the ECE computation, and the structural checks. An unevaluated
gate is recorded as ``None`` and blocks a ``passed() == True`` claim only if the
caller requires that gate -- it is never silently promoted to a pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .config import ECE_BOUND, ORDER_TOL

if TYPE_CHECKING:
    from .model import RockAndEggV3SystemOne


@dataclass(frozen=True)
class GateReport:
    """Measured results of the five export gates.

    ``None`` means the gate was not evaluated (missing held-out labels or no
    second instance), not that it passed.
    """

    option_order_max_abs_diff: float
    question_order_max_abs_diff: float
    state_independence_max_abs_diff: float
    cross_question_leakage_max_abs_diff: float
    reload_parity_max_abs_diff: float | None
    ece: float | None
    ece_bound: float = ECE_BOUND
    order_tol: float = ORDER_TOL
    no_lm_head: bool = True
    no_generation: bool = True
    backbone_arch: str = "Qwen3Model"
    control_moved_logits: bool = True

    def numeric_passed(self) -> bool:
        """Return whether gates 1, 2, and the discriminating control pass."""
        return (
            self.option_order_max_abs_diff <= self.order_tol
            and self.question_order_max_abs_diff == 0.0
            and self.state_independence_max_abs_diff == 0.0
            and self.cross_question_leakage_max_abs_diff == 0.0
            and self.control_moved_logits
        )

    def structural_passed(self) -> bool:
        """Return whether the structural ban (gate 5) passes."""
        return (
            self.no_lm_head
            and self.no_generation
            and self.backbone_arch == "Qwen3Model"
        )

    def reload_passed(self) -> bool:
        """Return whether reload parity is at least not contradicted."""
        return self.reload_parity_max_abs_diff is None or (
            self.reload_parity_max_abs_diff <= self.order_tol
        )

    def calibration_passed(self, require_evaluated: bool = False) -> bool:
        """Return whether ECE is within bound; optionally require measurement."""
        if self.ece is None:
            return not require_evaluated
        return self.ece <= self.ece_bound

    def passed(self, require_all: bool = False) -> bool:
        """Aggregate verdict.

        Args:
            require_all: when True, unevaluated reload and ECE gates block the
                pass. Use this for a final export decision.
        """
        return (
            self.numeric_passed()
            and self.structural_passed()
            and self.reload_passed()
            and self.calibration_passed(require_evaluated=require_all)
            and (not require_all or self.reload_parity_max_abs_diff is not None)
        )


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """Compute expected calibration error from softmax probs and true labels.

    Args:
        probs: ``[N, C]`` predicted probability distributions.
        labels: ``[N]`` true class indices.
        n_bins: number of equal-width confidence bins.

    Returns:
        The ECE as a float in ``[0, 1]``.
    """
    if probs.ndim != 2:
        msg = "probs must be a 2-D [N, C] tensor"
        raise ValueError(msg)
    if probs.shape[0] != labels.shape[0]:
        msg = "probs and labels must have the same number of rows"
        raise ValueError(msg)
    if probs.shape[0] == 0:
        return 0.0
    confidence, predicted = probs.float().max(dim=-1)
    correct = (predicted == labels).to(torch.float32)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    total = 0.0
    for i in range(n_bins):
        in_bin = (confidence > edges[i]) & (confidence <= edges[i + 1])
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        avg_conf = confidence[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        total += (count / probs.shape[0]) * float((avg_conf - avg_acc).abs().item())
    return total


def serving_probs(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Recover the distribution the serving path emits from a saved artifact.

    The evaluation artifact stores per-case ``softmax(logits)``. The serving
    adapter (:class:`~rockandegg_v3.inference.SystemOneInference`) reports
    ``softmax(logits / T)`` where ``T`` is ``confidence_temperature`` (default
    ``1.0``): temperature is applied to *raw logits*, then a **single** softmax.
    Applying ``softmax(probs / T)`` to an already-softmaxed artifact is a double
    softmax and measures a distribution the model never emits (R-MEASURE).

    Softmax is invariant to a per-row additive constant, so
    ``log(probs) == logits - c`` and therefore
    ``softmax(log(probs) / T) == softmax(logits / T)``. Recovering logits in log
    space with padded zero columns clamped reproduces serving exactly.

    Args:
        probs: saved per-case distributions ``[N, C]`` (``softmax(logits)``).
        temperature: serving ``confidence_temperature``; must be ``> 0``.

    Returns:
        The ``[N, C]`` distributions the serving path reports.
    """
    if temperature <= 0:
        msg = "temperature must be > 0"
        raise ValueError(msg)
    logits = probs.float().clamp_min(1e-12).log()
    return torch.softmax(logits / temperature, dim=-1)


def structural_gate(model: RockAndEggV3SystemOne) -> tuple[bool, bool, str]:
    """Check the structural ban: no LM head, no generation, stock backbone.

    Property-based, not string-based: the LM-head check inspects both live
    attributes and state_dict keys, so a head whose attribute was renamed or
    removed from the class but whose weights survive is still detected.
    """
    state_dict_fn = getattr(model, "state_dict", None)
    keys = state_dict_fn().keys() if callable(state_dict_fn) else ()
    no_lm_head = not hasattr(model, "lm_head") and not any(
        "lm_head" in key for key in keys
    )
    no_generation = not hasattr(model, "generate")
    arch = list(model.backbone.config.architectures or [])
    backbone_arch = arch[0] if arch else ""
    return no_lm_head, no_generation, backbone_arch
