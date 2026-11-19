# FP16 Qualification — Pre-registered Acceptance Criteria

**Written before the FP16 qualification run. Timestamp: 2026-11-19 (pre-run).**

These thresholds are fixed in advance. No threshold in this file may be edited
after a measurement is read. If a criterion fails, the failure is reported and
the artifact is not promoted; the criterion is not relaxed.

## Scope of the comparison

| Role | Artifact | Identity |
|---|---|---|
| Candidate | FP16 standalone export | derived from the published BF16 release |
| Reference | BF16 published release | `model.safetensors` SHA-256 `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5` |

**FP32 master status is established separately.** The recorded FP32 master
(`1b7fee92…`) is not present in this environment as of pre-registration. If it is
recovered before the run, it becomes the primary reference and BF16 becomes the
secondary reference; if it is not, the reference is BF16 and every downstream
statement must say "derived from BF16", never "native FP32 export".

## Structural gates (must all pass)

| ID | Criterion | Threshold |
|---|---|---|
| S1 | Tensor key set identical to reference | exact set equality, 122/122 |
| S2 | Every tensor shape identical to reference | exact |
| S3 | Total parameters | exactly 313,161,733 |
| S4 | Backbone parameters | exactly 312,892,928 |
| S5 | SystemOne head parameters | exactly 268,805 |
| S6 | LM-head tensors present | exactly 0 |
| S7 | `generate` attribute present on loaded model | exactly False |
| S8 | All candidate tensors finite (no NaN/Inf) | 313,161,733 / 313,161,733 |
| S9 | Every candidate tensor dtype | `torch.float16` |

## Conversion-fidelity gates

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| C1 | Overflow count during BF16→FP16 cast (|v| > 65504) | 0 | FP16 max finite is 65504; any overflow is a hard fail |
| C2 | Values that become non-finite after cast | 0 | hard fail |
| C3 | Global max absolute weight difference | ≤ 0.05 | BF16→FP16 is exact in mantissa for in-range values; only subnormal/underflow can move values. 0.05 bounds any subnormal rounding |
| C4 | Values changing at all | reported, not gated | informative: the exact-cast property predicts ≈0 outside the subnormal range |

## Runtime gates on T4

| ID | Criterion | Threshold |
|---|---|---|
| R1 | Fresh-process load from a clean cwd | exit code 0 |
| R2 | Loaded parameter dtype | all `torch.float16` |
| R3 | Peak allocated VRAM for one multi-question request | ≤ 4 GiB |
| R4 | State length 256 tokens executes | no error |
| R5 | Option length 64 tokens executes | no error |
| R6 | All three decision types (`choice`, `score`, `noul`) return decisions | 3/3 |
| R7 | Probability normalization | every decision \|Σp − 1\| ≤ 1e-5 |
| R8 | Option-order invariance (permute options, chosen key preserved) | max probability deviation ≤ 1e-6 |

## Numerical-parity gates (FP16 vs BF16 reference, clean panel)

| ID | Criterion | Threshold | Basis |
|---|---|---|---|
| N1 | Argmax (chosen-key) agreement over 2,000 clean-panel decisions | ≥ 99.0% | independent prior measurement on a different panel reported 477/29,955 changed options (98.4% agreement); 99.0% is a stricter published bar, not a fitted one |
| N2 | Mean absolute probability difference | ≤ 1.0e-3 | independent prior measurement reported 7.17e-4 mean delta |
| N3 | Max absolute probability difference | ≤ 5.0e-3 | prior per-question maxima were 1.1e-3 – 1.6e-3 on examples; 5e-3 allows tail growth |
| N4 | Absolute overall-accuracy difference vs reference | ≤ 0.5 percentage points | accuracy is a coarse statistic; a larger move signals a broken cast |
| N5 | Per-type accuracy difference vs reference | ≤ 1.0 percentage point each | per-type n is smaller, so a wider band is allowed than N4 |

## Decision rule

- **All of S1–S9, C1–C3, R1–R8 pass and N1–N5 pass** → FP16 artifact is
  QUALIFIED for publication as `v3-fullbackbone-fp16-1`.
- **Any structural or runtime gate fails** → NOT QUALIFIED. Do not publish.
- **Only a numerical-parity gate fails** → NOT QUALIFIED for promotion; publish
  the evidence and the failure, keep BF16 canonical.
- **FP32 master recovered mid-run** → rerun against FP32 as primary reference and
  report both; do not silently swap the reference.

## Explicit non-claims

- This does not qualify production calibration. The production ECE gate (≤ 0.05)
  remains a separate concern and is currently NOT MET (recorded raw ECE 0.1599
  RESERVED / 0.2755 sealed).
- A pass here does not assert that FP16 is lossless in general; it asserts the
  measured deltas above on the stated panel and artifacts.
