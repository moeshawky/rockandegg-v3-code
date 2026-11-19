# Pre-registration — Phase IV/V matched distillation experiment

Frozen BEFORE any student training or final-panel inference. Any deviation invalidates the release recommendation.

- Date: 2026-11-18 (UTC)
- Trigger: Phase III verdict **GO** (artifacts/ensemble_validation_report.json) — ensemble beats seed0 on clean panel, +2.1 pts, CI [0.0085, 0.034].
- Final evaluation panel: `artifacts/final_eval_panel_v1_manifest.json` — 200 dossiers / 1000 decisions, deterministic hash-ranked half of the frozen clean panel, sorted-id SHA-256 `369fc01ed21b1cdf65e0973e0ca466305e2328a24b2e50e628dd14c999148130`. Never used for training, calibration, selection, or teacher generation.
- Teacher-data restriction handling: the three teacher seeds all trained on the same 1,020 dossiers (split_seed=0), so in-sample teacher predictions on those dossiers are NOT valid supervision. Transfer set = the 90 DEV dossiers / 450 decisions, which were held out from all three teachers' training AND from seed0's training. Teacher targets on DEV are genuinely out-of-sample.

## Student initialization

Seed 0 `fullbb_bf16_e2style/best/model` (sha256 `1b7fee92…d87ac8`). Rationale: the operator's stated initial candidate, the checkpoint selected by the original campaign on DEV, and the checkpoint reconciled with the C2 calibration record. Seed 1 scored 1.45 pts higher on the clean panel but that difference is not a licensed reason to switch: selecting the init from the validation panel would contaminate the final comparison. Seed 0 is retained. Original checkpoint is never modified; student is saved to a new run dir.

## Matched control vs treatment (ONE intended difference)

Both arms:
- identical initial weights (seed 0), identical 450 DEV decisions, identical sample order, identical optimizer (AdamW, backbone lr 2.5e-5 / head lr 1e-4, weight_decay 0.01), identical 3% warmup + cosine schedule, identical 3 epochs, identical effective batch 16, identical seed 20261118, identical precision (bf16 autocast, fp32 master weights), identical scoring path.
- Control target: 100% gold soft labels for choice/score/noul (score keeps gold soft-CE + squared-EMD term unchanged).
- Treatment target: choice/noul = `0.75 × gold + 0.25 × ensemble_probabilities` (uncalibrated T=1.0, arithmetic mean of the three seeds on DEV); score = 100% gold (unchanged). The 75/25 ratio is a hypothesis under test, NOT optimized against any evaluation panel.

No architecture change, no context change, no weight averaging, no serving-contract change, no teacher weight blending.

## Pre-registered metrics & gates (final panel)

Accuracy overall and per type; paired dossier-clustered bootstrap (5000 resamples) for treatment−control, treatment−seed0, control−seed0; NLL, Brier, ECE(15), confidence distributions; per-type regression analysis; parameter count; checkpoint sha256; latency p50 and peak VRAM; inference parity on a fixed probe.

**SUCCESS** requires ALL:
1. `accuracy(treatment) > accuracy(control)` with paired 95% CI lower bound > 0.
2. `accuracy(treatment) > accuracy(seed0)` with paired 95% CI lower bound > 0.
3. No per-type accuracy regression > 3.0 pts vs control for choice/score/noul.
4. Checkpoint integrity: no LM head, no generate(), parameter count == 313,161,733, sha256 recorded, inference parity on fixed probe == 0.0.

**FAILURE** if any gate fails. **INCONCLUSIVE** if the required clean evidence is unavailable. No fourth outcome; no unregistered re-tuning against the final panel.

Calibration is evaluated separately and reported, never used to improve argmax accuracy. The existing export gate is ECE ≤ 0.05; the raw (uncalibrated) full-backbone checkpoints have ECE ~0.13 on clean data, so a calibrated release still requires a valid calibration partition. C2's sealed NLL regression (1.4563 vs 1.0510) is reconciled separately in FINAL_QUALIFICATION_REPORT.md and is NOT silently discarded.
