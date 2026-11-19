# FP16 Qualification Verdict — gate-by-gate

Run: 2026-11-19. Artifact: `/tmp/opencode/work/A_provenance/fp16`
(sha256 `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28`,
626,337,154 B). Gate definitions: `PRECISION_ACCEPTANCE_CRITERIA.md`
(written before any result existed). Evidence: `precision_parity.json`,
`qualify_fp16.json`, `A_provenance/precision_cast_report.json`.

## Structural and runtime gates — ALL PASS

S1–S9 and R1–R8 all pass. Highlights:

| Gate | Result |
|---|---|
| S3/S4/S5 params | 313,161,733 = 312,892,928 + 268,805 — exact |
| S6/S7 no LM head, no generate | 0 lm_head tensors; `hasattr(model,'generate') is False` |
| S9 dtype | all 122 tensors `torch.float16`, all finite |
| R8 option-order invariance | **0.0** (exact; historical head-only was 3.58e-7) |
| R3 peak VRAM | 1,339.2 MiB (bound 4,096) |
| R4/R5 max lengths | state 256, option 64 — both execute |

## Cast-fidelity gates — ALL PASS

| Gate | Threshold | Measured |
|---|---:|---:|
| C1 overflow to ±inf | 0 | **0** |
| C2 non-finite after cast | 0 | **0** |
| C3 global max abs weight diff | ≤ 0.05 | **2.98e-8** |
| max abs source value | — | 96.5 (well under fp16 max 65504) |

The cast is *provably exact in the fp16 normal range*: 312,572,427 / 312,572,427
values bit-identical. The only movement is 588,743 values that land in the fp16
subnormal window (rounding) and 236 that underflow to zero. Total changed:
47,755 of 313,161,733 (0.015%).

## Artifact-fidelity gate — PASS, EXACTLY

Comparison **A**: same weights, same runtime (fp16), two files.

| Metric | Result |
|---|---|
| decisions compared | 2,000 |
| argmax agreement | **1.0000** (0 flips) |
| mean / p99 / max abs probability difference | **0.0 / 0.0 / 0.0** |
| accuracy | 0.7285 vs 0.7285 |

**This is the artifact question, and it is answered exactly.** The FP16 file is
not merely "close to" the BF16 file — under identical compute it is the same
function on this panel.

## Numerical-parity gates — ONE FAILURE

Comparison **B**: FP16 artifact at fp16 compute vs the BF16 artifact's weights at
**fp32 compute** (highest-precision available reference; the FP32 master is
unrecoverable).

| Gate | Threshold | Measured | Verdict |
|---|---:|---:|---|
| N1 argmax agreement | ≥ 0.99 | 0.9975 (5 flips / 2,000) | **PASS** |
| N2 mean abs prob diff | ≤ 1.0e-3 | 6.54e-4 | **PASS** |
| N3 max abs prob diff | ≤ 5.0e-3 | **5.038e-3** | **FAIL** |
| N4 accuracy delta | ≤ 0.5 pp | 0.05 pp | **PASS** |
| N5 per-type delta | ≤ 1.0 pp | choice 0.33, noul 0.0, score 0.125 pp | **PASS** |

N3 misses by 3.8e-4 — 0.75% beyond the bound, on the single worst of 4,600
probability values (p99 is 3.22e-3). **The threshold is not relaxed and the
failure is not reclassified.** Two facts bound what it means:

1. Comparison A is exactly 0.0, and the cast moves no weight by more than
   2.98e-8. A 3e-8 weight perturbation cannot cause a 5e-3 probability move.
   **Therefore the entire B delta is fp16-vs-fp32 compute precision, not the
   FP16 file.**
2. Comparison **C** against *native BF16 compute* — the arm the old docs treated
   as the reference — is far worse on every axis: 0.984 agreement, 32 flips,
   4.78e-3 mean, 4.32e-2 max, and accuracy **0.7235 < 0.7285**. On this hardware
   BF16 is the lower-precision and lower-accuracy runtime (E measured it 11×
   slower than FP16; T4 is SM 7.5 and has no native BF16 tensor cores).

## Ruling

- **Cast fidelity: PASS. Artifact fidelity vs published BF16: PASS exactly.**
- **Pre-registered numerical-parity gate: FAIL on N3 only.** Reported as a
  failure everywhere the artifact is described.
- **Consequence:** the artifact is a faithful, storage-exact FP16 export and is
  the only viable T4 runtime (BF16 is emulated and 11× slower). It is **not**
  described as "parity-qualified against BF16" and it is **not**
  production-qualified — the separate ECE gate (≤ 0.05) remains NOT MET at
  0.1599 RESERVED / 0.2755 sealed.

No criterion in `PRECISION_ACCEPTANCE_CRITERIA.md` was edited after measurement.
