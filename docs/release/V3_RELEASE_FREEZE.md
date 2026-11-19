# V3 Release Freeze — RockAndEgg-313M-v3 Full-Backbone FP16

Release date: 2026-11-19. Status: **FROZEN — RESEARCH/PREVIEW**.
The release identity, weights, artifacts and remote references agree; no gate was
weakened or hidden. The BF16 preview artifact remains published on HF branch
`preview/fullbackbone-bf16-seed0` @ `0b60f73…` and is not modified.

## 1. Exact model identity

| Item | Exact value |
|---|---|
| Model | `moeshawky/RockAndEgg-313M-v3` |
| Variant | Full-backbone FP16, seed 0 |
| HF tag | `v3-fullbackbone-fp16-1` |
| HF tag commit | `48e42035a66299b803d1a545963913f90dbc3a2e` |
| Released FP16 model SHA-256 | `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28` |
| Released model bytes | 626,337,154 |
| `config.json` SHA-256 | `06d9d4bad660e7c40163ae16b7f870bbdca5ce77e7956b89c9e9420017111c03` |
| Precision provenance | cast from published BF16 artifact `c8b3aafb…` (direct `to(float16)`, no FP32 intermediate); labelled BF16-derived, never as an FP32 export |
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) |
| Tensors | 122, all `torch.float16`, all finite |

## 2. Exact source identity

| Item | Exact value |
|---|---|
| GitHub repository | `moeshawky/rockandegg-v3-code` (public, sanitized history) |
| Source commit | `aaa274566ab749cd7b993dcfb3c728dedc8c7fdf` |
| Source tag | none (release engineering committed in alpha repo `moe/rockandegg-v3-code` @ `aaa2745`) |
| Canonical main tag (BF16) | `v3-fullbackbone-main-1` → `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8` |
| Historical main tag | `v3-head-only-main` → `87c754951f9d4bd04c90e9ec78fc54295075e86c` |
| Historical head-only checkpoint SHA-256 | `3b208b86d6b5b87306091cbaec1b71007454eb4864b47efc636b4f633c875ac8` |
| Preserved preview branch | `preview/fullbackbone-bf16-seed0` → `0b60f73eda4293b4891f27c3106a977cef536a1f` |

No tag was rewritten; no force-push; no historical file deleted.

## 3. Qualification status

**RELEASED — RESEARCH/PREVIEW; NOT production qualified.**

- Production ECE bound **0.05** unchanged and **NOT met**: FP16 test panel ECE 0.1323
  (BF16-era panels 0.1599 RESERVED, 0.2755 SSF-1 sealed).
- C2 calibration rejected: sealed NLL regressed 1.0510 → 1.4563 (degenerate noul fit).
- No eligible unused representative calibration population exists.
- N3 pre-registered max-probability bound vs FP32 compute **FAIL** (5.038e-3 > 5.0e-3,
  0.76% overshoot on one decision of 2,000); reported as a failure, not relaxed.
- Probabilities are uncalibrated raw softmax at **T=1.0**; `confidence` is not a
  calibrated guarantee. No new temperature was fitted; no exposed partition reused.
- No superiority over Laya is claimed (student−Laya −3.40 pp, CI [−5.50, −1.45]).

## 4. Frozen architectural contract

- 10× Qwen3-derived decoder blocks + native SystemOne decision head; 313,161,733 params.
- Decision types: `choice`, `score`, `noul`. No LM head, no `generate()`, no token output.
- Native `DynamicCache` preserved; one state prefill + one option-row pass.

## 5. Frozen behavioral contract

- Shared-state inference: exactly **2** backbone forwards at every fan-out for
  `rae_shared` (verified hook-verified); `rae_independent` uses `2F`; `laya_native` 1.
- T4 (cuda:0, fp16, serial single-request): shared 22.49 req/s / 43.5 ms p50 vs
  independent 8.71 req/s / 111.9 ms p50 vs Laya 15.20 req/s / 65.1 ms p50.
  These do **not** generalize to other hardware or to concurrent serving.
- Fresh-process integrity: 122 tensors, all float16, no `lm_head`, no `generate`;
  FP32→BF16 storage identity 0.0 (FP16 artifact vs BF16 artifact under identical compute);
  remote↔local inference parity verified byte-identical on publish.

## 6. Research cycle closed

Seed 0 selected; seeds 1/2, the three-model ensemble, and the 75/25 distilled treatment
are research artifacts only. C2 failed qualification. No further training, distillation,
checkpoint averaging, architectural changes, or evaluation-driven selection for this release.

## 7. Conditions for reopening development

Development may reopen ONLY for a separately preregistered milestone that:
1. Supplies fresh, representative, dossier-grouped typed-decision calibration data,
   disjoint from all training/DEV/RESERVED/SSF-1/evaluation partitions; and
2. Uses a degeneracy-aware calibration procedure (reject fits where per-type accuracy
   reaches 1.000), without reusing exposed partitions; and
3. Achieves ECE ≤ 0.05 with paired CIs on the eligible held-out population.

Any new model variant requires a new versioned branch/tag; this frozen identity is immutable.

## 8. Non-blocking documentation limitation

The copy of release documentation inside this source repository (`docs/release/`) was
converted to describe the FP16 release at tag `v3-fullbackbone-fp16-1`. The BF16 preview
remains documented in its frozen HF revision `0b60f73…` (not modified) and in the alpha
repo `moe/rockandegg-v3-code`. No artifact integrity or release-identity problem exists.

## 9. Evidence archive

`evidence/` holds machine-readable results: `quality_metrics.json` (16/16 arm metrics),
`arm_accuracy_ci.json` (paired dossier bootstrap), `reconciliation_accuracy.json`
(three-convention comparison + divergent decision IDs), `precision_parity.json`
(N-gates), `fp16_invariance.json` (export gates), `throughput_results.json`,
`latency_results.json`, `memory_results.json`, `shared_state_instrumentation.json`,
`server_smoke.json`, `precision_cast_report.json`, `environment.json`. All payload
hashes verified against `RELEASE_MANIFEST.json` / `SHA256SUMS` (44 files).
