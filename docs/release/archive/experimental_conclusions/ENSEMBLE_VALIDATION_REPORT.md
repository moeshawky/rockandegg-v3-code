# Ensemble Validation Report — RockAndEgg-313M-v3 (Phase III)

Date: 2026-11-18. All figures are executed measurements. Decision rule was pre-registered BEFORE inference in `artifacts/ensemble_validation_preregistration.md`.

## 1. Clean panel (frozen before inference)

| Property | Value |
|---|---|
| Source | `LocalLLaMA/typed-decisions`, config `all`, revision `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6`, split `test` |
| Manifest | `artifacts/clean_panel_manifest.json` |
| Size | 400 dossiers / 2000 decisions |
| Sorted-dossier-id SHA-256 | `441ba1b97b99f2e409e82e7186a1458751696f2e3fb7de5a573fe65185b00852` |
| Workflow coverage | agent_trace_observability, customer_service, invoice_processing, security_incidents |

Cleanliness argument (see `DATA_LINEAGE_AUDIT.md`): the three seeds were trained on 1,020 dossiers drawn from the 1,200-dossier TRAIN split with `split_seed=0`; TEST dossiers are disjoint from all 1,200. These three checkpoints were selected on the 90-dossier DEV set only (`eval_full_backbone.py`, `full_backbone_training_audit.md` §10–11); TEST was never used to select them. DEV (repeatedly investigated), RESERVED (used in `final_eval.json`) and SSF-1 (calibration/sealed) were excluded from this decision. No temperature scaling; raw T=1.0 probabilities.

## 2. Checkpoint identity (verified by hash and fresh-process inference)

| Arm | Checkpoint | model.safetensors SHA-256 |
|---|---|---|
| seed0 | `artifacts/runs/fullbb_bf16_e2style/best/model` | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` |
| seed1 | `.../fullbb_bf16_e2style_s1/best/model` | `209ae6012ecd0d3c2b974954a7b0e66d99a1b84605a5a11eb24d0edffea52e3d` |
| seed2 | `.../fullbb_bf16_e2style_s2/best/model` | `39af46c6663a2dd3b6d47eb56d3fa45d9200cd440b686f0928e94ae39d57bcd1` |

Seed2's export dir lacks tokenizer files; a new dir (`experiments/ensemble_validation_20261118/seed2_infer_dir`) symlinks its weights/config to the original and borrows the identical seed0 tokenizer. No checkpoint was modified. All arms: fp32, `SystemOneInference` adapter, `max_state_len=256`, `max_option_len=64`, canonical option order.

## 3. Results on the clean panel (2000 decisions)

| Arm | Accuracy | choice | score | noul | ECE(15) | Brier | NLL | mean conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| seed0 | 0.7335 | 0.7083 | 0.6850 | 0.8233 | 0.1313 | 0.0772 | 0.7985 | 0.602 |
| seed1 | 0.7480 | 0.7283 | 0.6987 | 0.8333 | 0.1253 | 0.0724 | 0.7857 | 0.623 |
| seed2 | 0.7435 | 0.7150 | 0.6913 | 0.8417 | 0.1276 | 0.0682 | 0.7778 | 0.616 |
| **ensemble** | **0.7545** | 0.7250 | **0.7163** | 0.8350 | 0.1474 | 0.0657 | 0.7761 | 0.607 |

Ensemble = arithmetic mean of the three per-decision probability vectors, aligned by option key; argmax from the mean.

Paired dossier-clustered bootstrap (5,000 resamples, dossiers as clusters, seed 20261118, n=400 independent dossiers):

| Contrast | Δ accuracy | 95% CI | excludes 0 |
|---|---:|---:|---|
| ensemble − seed0 | **+0.0210** | [+0.0085, +0.0340] | **yes** |
| seed1 − seed0 | +0.0145 | [−0.0015, +0.0300] | no |
| seed2 − seed0 | +0.0100 | [−0.0070, +0.0275] | no |

Per-type paired deltas (dossier-clustered): choice +0.0113 [−0.0100, +0.0313]; score +0.0313 [+0.0100, +0.0525]; noul +0.0200 [−0.0063, +0.0450]. Per-decision deltas: choice +0.0167, score +0.0313, noul +0.0117 (equal-decision weighting; both weightings reported for transparency — they differ because dossiers have different type mixes).

Paired error overlap vs seed0: **106 corrections**, **64 regressions**, 1,403 both-right, 427 both-wrong. Net +42 correct decisions = +2.1 pts. The ensemble is better than *every* individual seed, while no individual seed beats seed0 with a CI excluding zero.

## 4. Decision gate (pre-registered)

| Gate | Criterion | Result |
|---|---|---|
| 1 | ensemble acc > seed0 acc | 0.7545 > 0.7335 ✅ |
| 2 | paired 95% CI lower bound > 0 | +0.0085 > 0 ✅ |
| 3 | no per-type regression > 3 pts | all per-type deltas positive ✅ |
| 4 | ECE ≤ seed0 + 0.02 and NLL ≤ seed0 + 0.02 | ECE +0.0164, NLL −0.0224 ✅ |

**Verdict: GO.** H1 (complementary decision knowledge) is supported on clean data; H0 (sampling variation) is rejected for this panel. Honest caveat: the ensemble is slightly *worse calibrated* than seed0 (ECE 0.1474 vs 0.1313, within the pre-registered +0.02 tolerance) while being more accurate and lower-NLL. The earlier DEV-only finding (356/450 vs 347/450, CI [−0.22, +4.22] pts) was inconclusive; this clean-panel result is the resolving evidence, not a re-labeling of DEV.

## 5. Traceability

- Raw predictions: `artifacts/ensemble_validation_predictions.json` (every decision, all four arms)
- Machine-readable report: `artifacts/ensemble_validation_report.json`
- Harness: `experiments/ensemble_validation_20261118/ensemble_validate.py`, log `run.log`
- Pre-registration: `artifacts/ensemble_validation_preregistration.md`
- Panel pin: `artifacts/clean_panel_manifest.json`
