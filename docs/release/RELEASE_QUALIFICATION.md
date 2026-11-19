# Release Qualification — RockAndEgg-313M-v3 Full-Backbone FP16

Date: 2026-11-19. Final status: **RELEASED — RESEARCH/PREVIEW** (production ECE gate
outstanding). This document records exactly what was verified and what was not.

## 1. Architecture verification (fresh process)

| Check | Result |
|---|---|
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) ✅ |
| Tensor count / dtype | 122 tensors, all `torch.float16`, all finite ✅ |
| `lm_head` | absent (attr + config) ✅ |
| `generate()` | absent (attr + config) ✅ |
| Decision types | `choice`, `score`, `noul` all emit normalized distributions ✅ |
| Confidence | equals max probability; raw T=1.0 ✅ |
| Option-order handling | permutation moves probabilities with options; chosen key stable ✅ |
| Determinism (GPU) | identical repeated outputs ✅ |
| Backbone frozen at serve | yes (`requires_grad=False` at load) ✅ |
| Config dtype | float16; all tensors float16 ✅ |

Evidence: `evidence/server_smoke.json` (fresh-process load), `evidence/quality_metrics.json`.

## 2. Full-backbone training evidence

All 10 blocks + final norm + head trainable; 112/112 backbone tensors changed with
finite grads; optimizer covered 312,892,928 + 268,805 params (`layer_update_report.json`,
`gradient_verification.json`, `artifacts/smoke_fullbb/result.json`). The distillation
backbone-ambiguity was resolved: `requires_grad_(True)` and optimizer coverage prove it
was trainable there too; `backbone_frozen=True` was only an export metadata flip.

## 3. Precision cast & FP32→FP16 parity

Fresh-process FP32→FP16 parity: FP16 artifact cast directly from BF16-valued weights
(`to(float16)`, no FP32 intermediate). Storage identity vs BF16 artifact under identical
compute: max abs prob diff **0.0**, 0 argmax flips (exact). Native BF16 on T4 is strictly
worse than FP16 on every axis (agreement 0.984, 32 flips, accuracy 72.35% vs 72.85%,
~11× slower). FP16 is the correct deployment precision for Turing.

## 4. Precision-qualification gates (pre-registered, `evidence/precision_parity.json`)

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| N1 argmax agreement vs FP32-compute ref | ≥ 0.99 | 0.9975 | PASS |
| N2 mean abs prob diff vs FP32-compute ref | ≤ 1e-3 | 6.54e-4 | PASS |
| N3 max abs prob diff vs FP32-compute ref | ≤ 5.0e-3 | **5.038e-3** | **FAIL** |
| N4 abs accuracy delta vs FP32-compute ref | ≤ 0.005 | +0.05 pp | PASS |
| N5 per-type delta vs FP32-compute ref | ≤ 0.01 | all ≤ 0.01 | PASS |

**N3 is reported as a failure.** It is a 0.76% overshoot on a single decision out of
2,000. It does not demonstrate a material accuracy regression (+0.05 pp accuracy delta;
5 of 2,000 argmax flips). It blocks the stricter qualification tier; the release is
published as research/preview.

**The FP32-compute reference** is FP32 compute over the BF16-valued weights — the
FP32 master checkpoint is unrecoverable. It is not a master-weight reference.

## 5. Evaluation evidence

| Metric | this release (FP16) | Source |
|---|---|---|
| Accuracy | **73.25% [71.05, 75.35]** | `evidence/quality_metrics.json` |
| NLL | 0.6945 | `evidence/quality_metrics.json` |
| Brier | 0.0777 | `evidence/quality_metrics.json` |
| ECE (M=15) | **0.1323** | `evidence/quality_metrics.json` |
| Score MAE | 0.290 | `evidence/quality_metrics.json` |
| Per-type A: choice / noul / score | 70.50% / 82.17% / 68.63% | `evidence/quality_metrics.json` |

Comparison arms: head-only FP32 34.25% [32.25, 36.30]; Laya 76.65% [74.60, 78.65].
Paired (dossier-clustered bootstrap, 5000, seed 20261116): A−B_fp32 +39.00 pp
[+35.95, +42.10]; A−C −3.40 pp [−5.50, −1.45].

## 6. Calibration status (gate NOT met)

Production ECE ≤ 0.05 gate **fails**: FP16 test panel ECE **0.1323** (BF16-era panels
0.1599 RESERVED, 0.2755 SSF-1 sealed). C2 rejected (sealed NLL 1.0510 vs raw 1.0510→
1.4563). No eligible unused representative calibration partition exists; no parameters
were fit against exposed evaluation data. Probabilities are uncalibrated T=1.0; no
`calibration.json` is shipped or loaded. This is the sole reason the release is
RESEARCH/PREVIEW and not production-qualified.

## 7. Remote verification

HF tag `v3-fullbackbone-fp16-1` → commit `48e42035a66299b803d1a545963913f90dbc3a2e`
downloaded into a fresh directory: 46/46 files re-hashed byte-identical against
`RELEASE_MANIFEST.json` before tagging; parameter count 313,161,733; 3 decision types;
remote↔local decision parity verified on publish. All historical tags verified unchanged.

## 8. Outstanding limitations

- Uncalibrated probabilities; do not use `confidence` as a calibrated guarantee.
- N3 max-probability bound vs FP32 compute FAIL (5.038e-3 > 5.0e-3).
- Does not beat Laya on the matched panel (−3.40 pp; CI excludes zero; not a universal ranking).
- No superiority claim on speed; concurrency unmeasured.
- Historical config metadata defects (stale `training_dataset`/`backbone_frozen`) are
  corrected in the release config and disclosed in `TRAINING_PROVENANCE.md`.
- Operator-stated dev figures differ from artifacts; the discrepancy is disclosed, not
  blended into the old figures.
