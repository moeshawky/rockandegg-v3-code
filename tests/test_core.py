"""Unit tests for config, loss, calibration, and gate-report logic.

These tests are CPU-only and do not load the checkpoint; the GPU gate run is
performed by run_systemone_gates.py against the real model.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rockandegg_v3 import (
    ECE_BOUND,
    ORDER_TOL,
    QTYPE_CHOICE,
    QTYPE_SCORE,
    GateReport,
    HeadConfig,
    compute_ece,
    option_dropout,
    serving_probs,
    structural_gate,
    verdict_pact_loss,
)


def test_head_config_defaults_encode_operator_decisions() -> None:
    """Optional components are off, backbone frozen, thresholds declared."""
    cfg = HeadConfig()
    assert cfg.use_cross_attn is False
    assert cfg.use_option_mixer is False
    assert cfg.use_cosine_prior is False
    assert cfg.freeze_backbone is True
    assert cfg.ece_bound == ECE_BOUND == 0.05
    assert cfg.order_tol == ORDER_TOL == 1e-6
    assert cfg.option_dropout == 0.3


def test_option_dropout_keeps_gold_and_renormalizes() -> None:
    """Gold mass is retained and the returned target sums to one."""
    target = torch.tensor([0.1, 0.7, 0.2])
    gen = torch.Generator().manual_seed(0)
    out = option_dropout(target, gold_idx=1, drop_prob=0.5, generator=gen)
    assert out[1] > 0
    assert torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-6)


def test_option_dropout_zero_prob_is_identity() -> None:
    """With drop_prob 0 the target is returned unchanged."""
    target = torch.tensor([0.25, 0.25, 0.5])
    out = option_dropout(target, gold_idx=0, drop_prob=0.0)
    assert torch.equal(out, target)


def test_choice_loss_has_no_emd_term() -> None:
    """Choice questions use plain soft-label CE."""
    logits = torch.tensor([1.0, 2.0, 0.5])
    target = torch.tensor([0.2, 0.5, 0.3])
    loss = verdict_pact_loss(logits, target, QTYPE_CHOICE)
    ce = -(target * torch.log_softmax(logits, dim=-1)).sum()
    assert torch.isclose(loss, ce)


def test_score_loss_adds_squared_emd() -> None:
    """Score questions add a non-negative squared-EMD term."""
    logits = torch.tensor([1.0, 2.0, 0.5])
    target = torch.tensor([0.2, 0.5, 0.3])
    ce = verdict_pact_loss(logits, target, QTYPE_CHOICE)
    score = verdict_pact_loss(logits, target, QTYPE_SCORE)
    assert score >= ce


def test_compute_ece_perfect_calibration() -> None:
    """Perfect calibration means confidence equals accuracy (one-hot correct)."""
    probs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    labels = torch.tensor([0, 0])
    assert compute_ece(probs, labels) == 0.0


def test_compute_ece_detects_miscalibration() -> None:
    """Confident wrong predictions produce positive ECE."""
    probs = torch.tensor([[0.99, 0.01], [0.98, 0.02]])
    labels = torch.tensor([1, 1])
    assert compute_ece(probs, labels) > 0.0


def test_serving_probs_at_default_temperature_is_identity() -> None:
    """At T=1.0 the served distribution equals the saved softmax exactly.

    This pins the R-MEASURE fix: a double softmax ``softmax(probs / T)`` would
    NOT be the identity even at T=1.0, and must never be used for the ECE gate.
    """
    probs = torch.tensor([[0.65, 0.30, 0.05], [0.10, 0.20, 0.70]])
    emitted = serving_probs(probs, 1.0)
    assert torch.allclose(emitted, probs, atol=1e-6)
    # Negative control: the old double softmax differs from the served probs.
    double = torch.softmax(probs / 1.0, dim=-1)
    assert not torch.allclose(double, probs, atol=1e-6)


def test_serving_probs_matches_raw_logit_temperature_scaling() -> None:
    """Recovered log-space scaling equals applying T to the original logits."""
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, 3.0, 0.0]])
    saved = torch.softmax(logits, dim=-1)
    for temp in (0.5, 1.0, 0.84, 2.0):
        expected = torch.softmax(logits / temp, dim=-1)
        assert torch.allclose(serving_probs(saved, temp), expected, atol=1e-6)


def test_serving_probs_rejects_nonpositive_temperature() -> None:
    """A non-positive serving temperature is a caller error, not a silent pass."""
    import pytest

    with pytest.raises(ValueError, match="temperature must be > 0"):
        serving_probs(torch.tensor([[0.5, 0.5]]), 0.0)


def test_gate_report_pass_requires_control_and_thresholds() -> None:
    """Numeric pass needs zero leakage, tolerance, and a moving control."""
    base = {
        "option_order_max_abs_diff": 3.6e-7,
        "question_order_max_abs_diff": 0.0,
        "state_independence_max_abs_diff": 0.0,
        "cross_question_leakage_max_abs_diff": 0.0,
        "reload_parity_max_abs_diff": 1e-7,
        "ece": 0.03,
    }
    ok = GateReport(**base, control_moved_logits=True)
    assert ok.numeric_passed()
    assert ok.passed(require_all=True)
    bad = GateReport(**base, control_moved_logits=False)
    assert not bad.numeric_passed()


def test_gate_report_unmeasured_ce_is_not_a_fail_but_blocks_full_export() -> None:
    """An unmeasured ECE is not silently promoted to a pass."""
    base = {
        "option_order_max_abs_diff": 0.0,
        "question_order_max_abs_diff": 0.0,
        "state_independence_max_abs_diff": 0.0,
        "cross_question_leakage_max_abs_diff": 0.0,
        "reload_parity_max_abs_diff": 0.0,
        "ece": None,
    }
    r = GateReport(**base)
    assert r.calibration_passed() is True
    assert r.calibration_passed(require_evaluated=True) is False
    assert r.passed(require_all=False) is True
    assert r.passed(require_all=True) is False


def test_structural_gate_detects_lm_head_and_generation() -> None:
    """Structural gate rejects an LM head, a generate method, or wrong arch."""
    backbone = SimpleNamespace(config=SimpleNamespace(architectures=["Qwen3Model"]))
    clean = SimpleNamespace(backbone=backbone)
    assert structural_gate(clean) == (True, True, "Qwen3Model")  # type: ignore[arg-type]

    with_head = SimpleNamespace(backbone=backbone, lm_head=object())
    assert structural_gate(with_head)[0] is False  # type: ignore[arg-type]

    with_gen = SimpleNamespace(backbone=backbone, generate=lambda: None)
    assert structural_gate(with_gen)[1] is False  # type: ignore[arg-type]

    bad_arch = SimpleNamespace(
        backbone=SimpleNamespace(
            config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        )
    )
    assert structural_gate(bad_arch)[2] == "Qwen3ForCausalLM"  # type: ignore[arg-type]

    # Property-based: a hidden lm_head weight is detected even with no attribute.
    hidden_head = SimpleNamespace(
        backbone=backbone,
        state_dict=lambda: {"decoder.lm_head.weight": object()},
    )
    assert structural_gate(hidden_head)[0] is False  # type: ignore[arg-type]
