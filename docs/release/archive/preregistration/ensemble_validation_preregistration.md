# Pre-registration — Ensemble Independent Validation (Phase III)

Frozen BEFORE any inference on the clean panel. Any deviation invalidates the decision gate.

- Date: 2026-11-18 (UTC)
- Clean panel: `artifacts/clean_panel_manifest.json`
- Panel: `LocalLLaMA/typed-decisions`, config `all`, revision `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6`, split `test`, 400 dossiers / 2000 decisions.
- Frozen sorted-dossier-id SHA-256: `441ba1b97b99f2e409e82e7186a1458751696f2e3fb7de5a573fe65185b00852`
- Why clean for THESE three checkpoints: dossier-disjoint from the 1,200 train dossiers from which the three seeds' 1,020 training dossiers were drawn (split_seed=0); never used to select these three checkpoints (selection was DEV-only per `full_backbone_training_audit.md` §10–11 and `eval_full_backbone.py`); not DEV, not RESERVED, not SSF-1.
- Excluded from this decision: DEV (90/450, repeatedly investigated), RESERVED (90/450, used in `final_eval.json`), SSF-1 (calibration/sealed). No temperature scaling; raw T=1.0 probabilities only.

## Hypotheses

- H1: the three full-backbone checkpoints carry complementary decision knowledge; the probability ensemble generalizes better than seed 0 on clean data.
- H0: the DEV-observed ensemble gain (356/450 vs 347/450) is sampling variation / evaluation-specific behavior.

## Inference controls (identical across arms)

- fp32 weights, `dtype=fp32`, device cuda:0; `max_state_len=256`, `max_option_len=64`; production `SystemOneInference` adapter; same `render_options`/`serialize_state`; options in canonical order (no permutation at inference).
- Models: seed0 `fullbb_bf16_e2style/best/model` (sha256 `1b7fee92…`), seed1 `…_s1/best/model` (`209ae601…`), seed2 `…_s2/best/model` (`39af46c6…`).
- Ensemble = arithmetic mean of the three per-decision probability vectors, aligned by option key; argmax from the mean vector.

## Pre-registered metrics

Overall accuracy; per-type accuracy (choice/score/noul); paired per-decision error overlap, corrections and regressions; NLL (gold under predicted distribution); Brier (choice/noul); ECE (15 equal-width bins, same implementation as `typed_eval.ece_score`); confidence distribution per arm; paired dossier-clustered bootstrap (5,000 resamples, seed 20261118, dossiers as clusters).

## Decision rule (GO / HOLD / NO-GO)

GO requires ALL FOUR:

1. `accuracy(ensemble) > accuracy(seed0)` on the clean panel.
2. Dossier-clustered paired 95% CI for `Δ = accuracy(ensemble) − accuracy(seed0)` has **lower bound > 0**.
3. No per-type accuracy regression worse than **3.0 percentage points** absolute vs seed0 for any of choice/score/noul.
4. Calibration non-regression: `ECE(ensemble) ≤ ECE(seed0) + 0.02` AND `NLL(ensemble) ≤ NLL(seed0) + 0.02`.

HOLD if Δ > 0 but the CI includes 0, or if lineage/clean-panel integrity is unresolved, or if the panel has fewer than 90 independent dossiers.

NO-GO if Δ ≤ 0, or any per-type regression exceeds 3.0 pts, or calibration regresses beyond the pre-registered bounds.

No fourth outcome. No additional unregistered comparison may be substituted if the gate fails. Raw predictions for every arm are preserved.
