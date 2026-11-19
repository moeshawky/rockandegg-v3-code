# Model Card: RockAndEgg-313M-v3 — Full-Backbone BF16 (RESEARCH / PREVIEW)

**Model ID / Version:** `moeshawky/RockAndEgg-313M-v3`, branch `preview/fullbackbone-bf16-seed0` @ revision `0b60f73eda4293b4891f27c3106a977cef536a1f`
**Architecture:** 10-layer Qwen3-derived decoder backbone + native SystemOne typed-decision head; no LM head, no `generate()` path
**Source (authoritative identity):** `V3_RELEASE_FREEZE.md:1-30`; source code `moeshawky/rockandegg-v3-code` commit `215cc710a9da62a8d56a8a8e676a41293dc5fc1b`, tag `v3-fullbackbone-preview-1`
**Last Verified:** 2026-11-19 01:05 UTC (all claims below re-measured this session; see §7)
**Verification Command:** `release/smoke_documented_interface.py` and `release/verify_release.py` (both executed this session, outputs in §7)

> **RESEARCH / PREVIEW — NOT PRODUCTION QUALIFIED.** Probabilities are raw softmax at
> **T=1.0** and are **uncalibrated**. The production ECE ≤ 0.05 gate is **NOT met**
> (0.1599 RESERVED, 0.2755 SSF-1 sealed). `confidence` is not a calibrated probability
> guarantee. Source: `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/CALIBRATION_LIMITATIONS.md`, `artifacts/release_integrity.json`.

## 1. Intended Use

**Designed for:** research on compact typed-decision models and shared-state
(encode-once) inference. One state + typed questions (`choice`, `score`, `noul`) → one
probability distribution + `confidence` per question. Verified by `artifacts/smoke_interface.json`
(3/3 decision types, 10/10 checks).
**Out of scope:** production decisions requiring calibrated confidence; any risk-threshold
gating; token generation (no generation path exists — `artifacts/release_integrity.json:no_generate=true`).
**Primary users:** researchers and integrators evaluating shared-state inference on
BF16-capable GPUs (Ampere+/Blackwell) or CPU.

## 2. Identity & Artifacts (exact values, not adjectives)

| Item | Value | Source |
|---|---|---|
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) | `artifacts/release_integrity.json:param_count` |
| Tensors | 122; all shapes match source | `artifacts/release_integrity.json:tensor_keys/shapes` |
| Precision | bfloat16 (all tensors/params) | `artifacts/release_integrity.json:all_bf16` |
| Released model SHA-256 | `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5` | `RELEASE_MANIFEST.json` |
| Selected FP32 source SHA-256 | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` | `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/TRAINING_PROVENANCE.md` |
| Historical head-only checkpoint SHA-256 | `3b208b86d6b5b87306091cbaec1b71007454eb4864b47efc636b4f633c875ac8` | `V3_RELEASE_FREEZE.md` |
| Historical main revision (preserved) | `87c754951f9d4bd04c90e9ec78fc54295075e86c`, tag `v3-head-only-main` | HF remote, verified this session |
| License | Apache-2.0; Qwen3 attribution in `ATTRIBUTION.md` | `LICENSE`, `ATTRIBUTION.md` |

## 3. Performance (Measured — Sekel ratios, not adjectives)

Conditions: 1× NVIDIA RTX PRO 6000 Blackwell (driver 595.112.07), `cuda:0`, bfloat16,
torch 2.13.0+cu130 / transformers 5.23.0, 65 label-blind typed-decision inputs,
5 warmup + 40 timed iterations per fan-out, `torch.cuda.synchronize()`.
Source: `artifacts/release_performance.json` (captured output this session).

| Fan-out | Backbone calls (naive) | p50 ms | p95 ms | req/s | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|
| F1 | 2 (2) | 14.347 | 14.670 | 69.7 | 654.0 |
| F2 | 2 (3) | 14.132 | 14.209 | 70.8 | 668.1 |
| F4 | 2 (5) | 14.312 | 14.408 | 69.9 | 693.8 |
| F8 | 2 (9) | 14.310 | 14.446 | 69.9 | 693.8 |

Encode-once verified: exactly **2** backbone forwards at every fan-out (one `[1,256]`
state prefill + one `[4,21]` option-row pass); naive re-encoding needs up to 9 calls at
F8. These numbers are hardware-specific and must not be generalized to other devices.
No controlled comparison against Laya was run; no speed superiority is claimed.

**FP32→BF16 conversion (measured):** 313,161,733 values; max abs diff 0.02125, mean
abs diff 2.04e-5; 0 argmax flips over 50 real decisions, max probability diff 0.0241.
Source: `bf16_conversion_report.json`, `artifacts/release_integrity.json`.
**CPU↔GPU:** max probability diff 0.0191 (`artifacts/release_integrity.json`).

## 4. Training Data (evidence-first sourcing)

| Source | Description | Provenance | Size |
|---|---|---|---|
| `LocalLLaMA/typed-decisions`, config `all` | Typed decisions (choice/score/noul) over dossiers | revision `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6` | 1,200 train dossiers / 6,000 decisions |

Full-backbone fine-tuning (no frozen layers): AdamW backbone lr 2.5e-5 / head lr 1e-4,
wd 0.01, 3% warmup + cosine, 8 epochs, micro-batch 64, bf16 autocast / fp32 master
weights, soft-label CE, option-permutation augmentation. Dossier-grouped split: 1,020
train / 180 val → 90 DEV / 90 RESERVED (`split_seed=0`). Selected seed 0 by DEV best
0.7800 @ step 320; three-seed DEV best 0.7800/0.7533/0.7711. Sources:
`release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/TRAINING_PROVENANCE.md`, public `FULL_BACKBONE.md`, `artifacts/lineage_audit.json`.

## 5. Evaluation (measured; reproduce commands)

| Benchmark / panel | Split | Metric | Result | Source |
|---|---|---|---|---|
| DEV selection | 90 dossiers | best accuracy | 0.7800 (seed 0) | `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/TRAINING_PROVENANCE.md` |
| Clean panel | 400 dossiers / 2,000 dec | seed0 acc | 0.7335 | `artifacts/ensemble_validation_report.json` |
| Clean panel (ensemble, research only) | 400 dossiers | acc; Δ vs seed0 | 0.7545; +0.0210 CI[+0.0085,+0.0340] | `artifacts/ensemble_validation_report.json` |
| Matched distillation (failed) | 200 dossiers | treatment/control acc | 0.7180 / 0.7310 | `artifacts/distillation_experiment_report.json` |
| RESERVED | 90 dossiers | student / Laya OOF | 0.7644 / 0.7844 (Δ CI includes 0) | `FINAL_QUALIFICATION_REPORT.md` |
| Integrity | fresh process | checks passed | 16/16 | `artifacts/release_integrity.json` |
| Clean-process smoke (documented interface) | remote artifact | checks passed | 10/10 | `artifacts/smoke_interface.json` |
| Remote reload verification | fresh download | gates passed | 5/5; remote↔local parity 0.0 | `artifacts/remote_verification.json` |

Reproduce: `release/smoke_documented_interface.py` (loads the published artifact via
`revision` + `subfolder` and re-runs all three decision types + fan-out);
`release/verify_remote.py` (fresh remote download + hash + parity). No accuracy
evaluation is re-run here (panels are exposed and frozen).

## 6. Limitations & Failure Modes (G-ERR)

| Symptom | Likely cause | Trigger condition | Mitigation |
|---|---|---|---|
| `confidence` misread as calibrated | uncalibrated raw T=1.0 softmax | any use as risk/gate signal | do not use for gating; see `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/CALIBRATION_LIMITATIONS.md` |
| ECE gate failure | no eligible unused calibration population | production qualification attempt | new preregistered dossier-grouped calibration milestone |
| No Laya superiority | Δ CI includes zero | claiming benchmark win | make no such claim (0.7644 vs 0.7844) |
| Token generation attempted | no LM head / `generate()` | calling generation APIs | use typed-decision interface only |
| BF16 on T4 (no bf16 hardware) | Turing lacks bf16 | loading bf16 weights on T4 | use CPU float32 or a non-bf16 variant |
| Published-card placeholders | frozen revision not edited | copying snippet from HF card verbatim | use exact `revision`/`subfolder` from §1 and `V3_OPERATIONAL_HANDOFF.md` |

**Known weaknesses:** uncalibrated probabilities; ensemble (+2.1 pts) and distilled
treatment are research-only, not deployed; historical config metadata defects disclosed
in `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/TRAINING_PROVENANCE.md`; operator-stated dev figures differ from artifacts (selection
robust under both). **Edge cases:** empty questions/options rejected by the canonical
server; option order permutes probabilities with options (chosen key stable — integrity check).

## 7. Verification (living documentation)

Executed this session (2026-11-19 01:05 UTC):
```bash
# clean-process smoke of exact documented interface against the published artifact
HF_HUB_DISABLE_PROGRESS_BARS=1 USE_TF=0 PYTHONPATH=/scratch/rockandegg-v3-code:src \
  /opt/pytorch/bin/python release/smoke_documented_interface.py
# Expected: all_pass=true, 10 checks (model/tokenizer/adapter/3 types/normalization/confidence/encode_once)

# fresh-process artifact integrity
PYTHONPATH=src:/scratch/RockAndEgg-313M-v3 /opt/pytorch/bin/python release/verify_release.py
# Expected: all_pass=true, 16 checks, param_count=313161733, bf16 parity 0 flips/50

# remote consistency (fresh download + hash + parity)
/opt/pytorch/bin/python release/verify_remote.py
# Expected: all_pass=true, 5 gates, remote revision 0b60f73..., parity 0.0
```
**Expected output:** all three scripts exit 0 with `all_pass: true`.
**Re-verification frequency:** before any downstream integration and on any model/hardware change.

## 8. Source Citations

- Model weights/config: `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/model.safetensors`, `config.json`
- Canonical inference API: `systemone_inference.py:91` (`SystemOneInference`), `:105` (`__init__`, `confidence_temperature=1.0` at `:111`), `:167` (`__call__`), `:188` (temperature scaling)
- Training code: `scripts/train_full_backbone.py` (public source repo); `release/RockAndEgg-313M-v3-fullbackbone-bf16-preview/TRAINING_PROVENANCE.md`
- Evaluation code: `release/verify_release.py`, `release/perf_release.py`, `release/smoke_documented_interface.py`
- Measured results: `artifacts/release_integrity.json`, `artifacts/release_performance.json`, `artifacts/smoke_interface.json`, `artifacts/remote_verification.json`
- Freeze/identity: `V3_RELEASE_FREEZE.md`, `RELEASE_MANIFEST.json`, `PUBLICATION_REPORT.md`

> **Published-card note:** the copy of this card inside the frozen preview revision
> (`fullbackbone-bf16-preview/README.md` @ `0b60f73…`) still uses `<preview-branch>` /
> `<preview-subdir>` placeholders in its load snippet. It is intentionally not modified
> to preserve revision `0b60f73…` byte-for-byte. Exact identifiers are in §1 and in
> `V3_OPERATIONAL_HANDOFF.md`.
