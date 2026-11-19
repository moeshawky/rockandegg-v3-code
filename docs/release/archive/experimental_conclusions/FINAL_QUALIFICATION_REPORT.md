# Final Qualification Report — RockAndEgg-313M-v3 ensemble distillation

Date: 2026-11-18. Evidence chain: `DATA_LINEAGE_AUDIT.md` → `ENSEMBLE_VALIDATION_REPORT.md` → `DISTILLATION_EXPERIMENT.md` → this file. Every metric traces to a manifest, checkpoint hash and evaluation run recorded in the reports.

## 1. Executive decision

**FAILURE — the ensemble advantage does not survive single-student distillation.** No student checkpoint is qualified for release; the published checkpoint was not touched and no push/publish was performed.

What was actually established:

1. The three full-backbone seeds share one training set (1,020 dossiers, `split_seed=0`) but differ in initialization; on a frozen, dossier-disjoint 400-dossier clean panel the **probability ensemble beats seed0 by +2.1 pts (95% CI [+0.85, +3.40], 400 independent dossiers)** and beats every individual seed. This resolves the earlier inconclusive DEV-only result.
2. That advantage **did not transfer** into one student. A matched, one-change control/treatment experiment on an out-of-sample transfer set produced a distilled student at 0.7180 vs gold-only control 0.7310 and seed0 0.7310 on a 200-dossier final panel; the choice-type regression was −4.25 pts with a CI excluding zero. All four pre-registered success gates failed except structural integrity.
3. Independently of distillation, the release contract remains **not satisfied** because of calibration (§3).

## 2. Qualification metrics (final panel, 200 dossiers / 1000 decisions)

| Arm | Accuracy | choice | score | noul | NLL | Brier | ECE(15) | Params | SHA-256 (model.safetensors) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed0 (best single model) | 0.7310 | 0.7095 | 0.6875 | 0.8092 | 0.7971 | 0.0755 | 0.1276 | 313,161,733 | `1b7fee92…d87ac8` |
| gold-only control student | 0.7310 | 0.7365 | 0.6825 | 0.7895 | 0.8139 | 0.0882 | 0.0962 | 313,161,733 | `25788c75…5d0f` |
| distilled student | 0.7180 | 0.6858 | 0.6750 | 0.8059 | 0.8084 | 0.0819 | 0.1056 | 313,161,733 | `ebc30af2…3486` |

Paired dossier bootstrap (n=200): treatment−control −0.0130 [−0.0290, +0.0020]; treatment−seed0 −0.0130 [−0.0400, +0.0120]; choice regression −0.0425 [−0.0675, −0.0200] (significant). Full tables in `DISTILLATION_EXPERIMENT.md` §3.

## 3. Calibration kept separate from accuracy (required)

- The governing export gate is **ECE ≤ 0.05**. No arm meets it on the clean/final panels: seed0 ECE 0.1276, control 0.0962, distilled 0.1056 (ensemble 0.1474 on the 400-dossier panel). These are uncalibrated T=1.0 numbers.
- The only prior attempt to satisfy the gate was post-hoc calibration C2 on SSF-1. **C2 is rejected and not deployed.** Its sealed-set NLL was **1.4563 versus the raw reference 1.0510** (paired CI [+0.308, +0.504], excludes zero) — a significant regression on the pre-registered primary metric. DEV noul accuracy was 1.000, so the fitted noul temperature hit the search lower bound (T=exp(−4)=0.0183), a degenerate perfect-separation fit; DEV contained only `threshold_breach` while sealed contained five other families. Post-mortem: C1 (single T=0.212) would have given sealed NLL 0.8423, but reselecting on sealed data would repeat the protocol violation. C2 is reconciled here, not silently discarded; the one-step rollback is to leave `calibration.json` unapplied in production (no production file loads it, per `audit_correction.py`).
- No new temperature was fitted on the clean/final panels: temperature scaling cannot change argmax accuracy and the designated calibration partition was consumed as the distillation transfer set. **Release remains blocked on calibration regardless of the distillation result.**
- Convention for all Phase III/V numbers: `V3Arm` never loads `calibration.json`, so every metric in this report is **uncalibrated T=1.0** (this resolves the audit's open question about calibration application during TEST measurement; `calibration.json` was not applied in these runs).

## 4. Inference cost and integrity

- Architecture/deployment unchanged: headless 10-layer Qwen3 backbone (312,892,928 params) + 268,805-param SystemOne head; no `lm_head`, no `generate()` (verified on both students and the base); `backbone_frozen=True` in exported configs; 313,161,733 total params.
- Inference is non-generative. Measured SSF-1-protocol latency for this checkpoint family (RTX PRO 6000, fan-out 1): p50 ≈ 13.1 ms; fan-out 8: ≈ 21.6 ms; calibration overhead +0.06 to +0.12 ms (`calibration_report.md` §4). Peak training memory 41.90 GiB at effective batch 64 (`training_report.md` §2).
- Both student checkpoints pass structural integrity (gate 4). Neither is qualified on performance.
- Lineage audit (`DATA_LINEAGE_AUDIT.md` §7) confirmed defects that do not alter these results but are recorded: the 3 teacher configs carry stale `backbone_frozen:true` / `training_dataset:SargeDev/...` metadata despite genuinely unfrozen typed-decisions training; seed2's checkpoint lacks tokenizer files (handled by a symlink inference dir); and the older Jev-distill pipeline has confirmed question-level dossier leakage (616/1200 dossiers span panels) that is **separate** from these 3 seeds and was not transferred to them.

## 5. Artifact integrity and reproducibility

- Environment actually used: `/opt/pytorch/bin/python` (Py3.13), torch 2.13.0+cu130, transformers 5.23.0, 1× RTX PRO 6000 Blackwell (97 GB, sm_120). The workspace `AGENTS.md` hardware (2×T4, `.venv`, torch 2.11) is stale for this machine; all decisions followed direct measurement.
- Pre-registrations frozen before evidence: `artifacts/ensemble_validation_preregistration.md`, `experiments/ensemble_validation_20261118/distillation_preregistration.md`.
- Panel pins: `artifacts/clean_panel_manifest.json` (400 dossiers), `artifacts/final_eval_panel_v1_manifest.json` (200 dossiers).
- Raw predictions preserved: `artifacts/ensemble_validation_predictions.json`, `artifacts/distillation_final_predictions.json`, `artifacts/distillation_teacher_probs_dev.json`.
- Machine-readable reports: `artifacts/ensemble_validation_report.json`, `artifacts/distillation_experiment_report.json`.
- No existing checkpoint, dataset or manifest was overwritten; no publish/push; new run dirs only.

## 6. Recommendation and next step

Do **not** deploy a distilled student and do **not** promote the 3-model ensemble as a release (tripling inference cost for +2.1 pts does not by itself justify the deployment complexity). The defensible single-model candidate remains seed0 (or, on the 400-dossier panel, seed1 at 0.7480 — but seed1−seed0 CI includes zero, so there is no significant single-seed winner).

If further work is authorized, the exact missing evidence is: a teacher-transfer design that does not inject label noise into authoritative gold targets for choice (e.g., OOF teacher predictions or a teacher-weight schedule validated on a *new* untouched panel), plus a valid calibration partition with a degeneracy gate to satisfy ECE ≤ 0.05. These are new, separately pre-registered experiments — not a re-run of this gate.
