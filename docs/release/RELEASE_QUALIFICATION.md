# Release Qualification — RockAndEgg-313M-v3 Full-Backbone BF16

Date: 2026-11-19. Final status: **RELEASED — RESEARCH/PREVIEW** (production ECE gate
outstanding). This document records exactly what was verified and what was not.

## 1. Architecture verification (fresh process)

| Check | Result |
|---|---|
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) ✅ |
| Tensor count / shapes vs source | 122 tensors, all shapes match ✅ |
| `lm_head` | absent (attr + config) ✅ |
| `generate()` | absent (attr + config) ✅ |
| Decision types | `choice`, `score`, `noul` all emit normalized distributions ✅ |
| Confidence | equals max probability; raw T=1.0 ✅ |
| Option-order handling | permutation moves probabilities with options; chosen key stable ✅ |
| Determinism (GPU) | identical repeated outputs ✅ |
| Backbone frozen at serve | yes (`requires_grad=False` at load) ✅ |
| Config dtype | bfloat16; all tensors bf16 ✅ |

Evidence: `artifacts/release_integrity.json` (16/16 checks pass).

## 2. Full-backbone training evidence

All 10 blocks + final norm + head trainable; 112/112 backbone tensors changed with
finite grads; optimizer covered 312,892,928 + 268,805 params (`layer_update_report.json`,
`gradient_verification.json`, `artifacts/smoke_fullbb/result.json`). The distillation
backbone-ambiguity was resolved: `requires_grad_(True)` and optimizer coverage prove it
was trainable there too; `backbone_frozen=True` was only an export metadata flip.

## 3. Clean-load / FP32→BF16 parity

Fresh-process FP32→BF16 decision parity: 0 argmax flips over 50 real decisions, max
probability difference 0.0241. CPU and RTX PRO 6000 loads both work; CPU↔GPU max
difference 0.0191. Conversion stats: max abs weight diff 0.02125, mean 2.04e-5
(`bf16_conversion_report.json`, `artifacts/release_integrity.json`).

## 4. Inference parity and performance

RTX PRO 6000, label-blind inputs, `artifacts/release_performance.json`:

| fan-out | backbone calls (naive) | p50 ms | p95 ms | req/s | peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|
| F1 | 2 (2) | 14.35 | 14.67 | 69.7 | 654 |
| F2 | 2 (3) | 14.13 | 14.21 | 70.8 | 668 |
| F4 | 2 (5) | 14.31 | 14.41 | 69.9 | 694 |
| F8 | 2 (9) | 14.31 | 14.45 | 69.9 | 694 |

Encode-once proven at tensor level: one `[1,256]` state prefill + one `[4,21]` option-row
pass at every fan-out; naive re-encoding would need up to 9 calls. No speed superiority
over Laya is claimed (no matched serving comparison was authorized).

## 5. Evaluation evidence

- DEV selection: seed 0 best 0.7800 @ step 320.
- Clean panel: seed 0 0.7335; ensemble 0.7545 (research context only, +2.1 pts CI
  `[+0.0085, +0.0340]`).
- Distillation: FAILURE (treatment 0.7180 vs control 0.7310).
- RESERVED: student 0.7644, Laya OOF 0.7844; student−Laya Δ −0.0200 CI
  `[−0.0667, +0.0223]` includes zero — no claim of beating Laya.

## 6. Calibration status (gate NOT met)

Production ECE ≤ 0.05 gate **fails**: raw ECE 0.1599 (RESERVED), 0.2755 (SSF-1 sealed).
C2 rejected (sealed NLL 1.4563 vs raw 1.0510). No eligible unused representative
calibration partition exists; no parameters were fit against exposed evaluation data.
Probabilities are uncalibrated T=1.0; no `calibration.json` is shipped or loaded.
This is the sole reason the release is RESEARCH/PREVIEW and not production-qualified.

## 7. Remote verification

HF preview revision `0b60f73eda4293b4891f27c3106a977cef536a1f` downloaded into a fresh
directory: all 17 file hashes match, 313,161,733 params, remote↔local decision parity
exact (0.0 max diff, identical chosen keys), main preserved at `87c7549…`.
Evidence: `artifacts/remote_verification.json` (5/5 gates pass).

## 8. Outstanding limitations

- Uncalibrated probabilities; do not use `confidence` as a calibrated guarantee.
- Does not beat Laya on the matched held-out benchmark.
- Ensemble and distilled variants are documented research results, not deployed.
- Historical config metadata defects (stale `training_dataset`/`backbone_frozen`) are
  corrected in the release config and disclosed in `TRAINING_PROVENANCE.md`.
- Operator-stated dev figures differ from artifacts; selection is robust under both and
  the discrepancy is disclosed in `MODEL_SELECTION_REPORT.md`.
