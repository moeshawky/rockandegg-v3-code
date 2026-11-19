# Distillation Experiment — matched control vs ensemble-distilled student (Phase IV/V)

Date: 2026-11-18. Configuration pre-registered BEFORE training in `experiments/ensemble_validation_20261118/distillation_preregistration.md`. Trigger: Phase III GO (see `ENSEMBLE_VALIDATION_REPORT.md`). All figures are executed measurements.

## 1. Teacher-data restriction — how it was satisfied

All three teacher seeds trained on the same 1,020 dossiers (`split_seed=0`), so in-sample teacher predictions on those dossiers would be memorized and invalid as supervision. **Transfer set = the 90 DEV dossiers / 450 decisions**, which `split_by_dossier` + `split_dev_reserve` held out from the training of all three teachers AND from seed0. Teacher vectors were generated fresh by running all three seeds on DEV through the production adapter and averaging their per-decision probabilities (`artifacts/distillation_teacher_probs_dev.json`, 450 decisions). No in-sample teacher target was used.

## 2. Matched design (one intended difference)

| Property | Control | Treatment |
|---|---|---|
| Initialization | seed0 `fullbb_bf16_e2style/best/model` (sha256 `1b7fee92…`) | identical |
| Training data | 450 DEV decisions | identical |
| Sample order / seed | `make_batches(..., 16, 20261118+epoch)` | identical |
| Optimizer | AdamW, backbone 2.5e-5 / head 1e-4, wd 0.01 | identical |
| Schedule | 3% warmup + cosine, 87 steps, 3 epochs | identical |
| Precision | bf16 autocast, fp32 master weights | identical |
| Score target | 100% gold (soft CE; squared-EMD weight 0.0 as in selected config) | identical |
| choice/noul target | 100% gold | **0.75 gold + 0.25 ensemble (T=1.0)** |

270 of 450 decisions (the non-score ones) received the ensemble blend. The 75/25 ratio was fixed a priori and was NOT tuned on any evaluation panel. Both arms ran 87 updates (~12 s each); loss fell 1.146→0.964 (control) and 1.151→0.983 (treatment). Full config: `experiments/ensemble_validation_20261118/distill_student.py`.

## 3. Final-panel evaluation (frozen 200 dossiers / 1000 decisions)

Panel: `artifacts/final_eval_panel_v1_manifest.json`, id SHA-256 `369fc01ed21b1cdf65e0973e0ca466305e2328a24b2e50e628dd14c999148130`, a deterministic hash-ranked half of the clean panel. Never used for training, calibration, selection or teacher generation. **Caveat, stated honestly:** the ensemble-validation run scored all 400 clean-panel dossiers, so this half's aggregate was seen there; no selection or tuning was performed on it, and it was fully unseen by both students.

| Arm | Accuracy | choice | score | noul | ECE(15) | Brier | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed0 baseline | 0.7310 | 0.7095 | 0.6875 | 0.8092 | 0.1276 | 0.0755 | 0.7971 |
| gold-only control | 0.7310 | 0.7365 | 0.6825 | 0.7895 | 0.0962 | 0.0882 | 0.8139 |
| ensemble-distilled | **0.7180** | 0.6858 | 0.6750 | 0.8059 | 0.1056 | 0.0819 | 0.8084 |

Paired dossier-clustered bootstrap (5,000 resamples, n=200 dossiers):

| Contrast | Δ | 95% CI | excludes 0 |
|---|---:|---:|---|
| treatment − control | **−0.0130** | [−0.0290, +0.0020] | no (directional negative) |
| treatment − seed0 | −0.0130 | [−0.0400, +0.0120] | no |
| control − seed0 | ≈0.0000 | [−0.0290, +0.0270] | no |

Per-type treatment − control (dossier-clustered): **choice −0.0425 [−0.0675, −0.0200] — CI excludes zero, a significant regression**; score −0.0075 [−0.0325, +0.0175]; noul +0.0100 [−0.0151, +0.0350].

## 4. Pre-registered gates / verdict

| Gate | Result |
|---|---|
| 1. treatment > control with CI lower > 0 | ❌ (−1.3 pts, CI includes 0) |
| 2. treatment > seed0 with CI lower > 0 | ❌ |
| 3. no per-type regression > 3 pts | ❌ (choice −4.25 pts, CI excludes 0) |
| 4. structural integrity (no LM head, no generate(), 313,161,733 params, backbone_frozen=True) | ✅ both students |

**Verdict: FAILURE.** The ensemble's clean-panel advantage did NOT transfer into a single student. The 25% ensemble blend on choice/noul *reduced* choice accuracy significantly while the gold-only control preserved seed0 accuracy (and improved calibration, ECE 0.096 vs 0.128). Plausible mechanism (diagnostic, not further tested): on choice the student was already strong and gold is a 100%-authoritative target; mixing in a ~79%-accurate ensemble distribution injects label noise that dominates the 25% weight. No fourth outcome was manufactured: the 75/25 ratio was not re-tuned and no multi-seed campaign was launched, per the pre-registration.

## 5. Artifacts

- Control student: `experiments/ensemble_validation_20261118/students/control` (sha256 `25788c75…5d0f`)
- Treatment student: `experiments/ensemble_validation_20261118/students/treatment` (sha256 `ebc30af2…3486`) — experimental only, **NOT qualified for release**
- Machine-readable report: `artifacts/distillation_experiment_report.json`
- Raw predictions: `artifacts/distillation_final_predictions.json`
- Teacher targets: `artifacts/distillation_teacher_probs_dev.json`
- Harness/log: `experiments/ensemble_validation_20261118/distill_student.py`, `distill.log`
