# V3 Release Freeze — RockAndEgg-313M-v3 Full-Backbone BF16

Release date: 2026-11-19. Status: **FROZEN WITH DOCUMENTATION LIMITATIONS**.
The release identity, weights, artifacts and remote references agree; one non-blocking
documentation discrepancy is recorded in §8. No gate was weakened or hidden.

## 1. Exact model identity

| Item | Exact value |
|---|---|
| Model | `moeshawky/RockAndEgg-313M-v3` |
| Variant | Full-backbone BF16, seed 0 |
| HF preview branch | `preview/fullbackbone-bf16-seed0` |
| HF preview revision | `0b60f73eda4293b4891f27c3106a977cef536a1f` |
| Artifact subfolder | `fullbackbone-bf16-preview` |
| Released BF16 model SHA-256 | `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5` |
| Selected source FP32 checkpoint SHA-256 | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` |
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) |

## 2. Exact source identity

| Item | Exact value |
|---|---|
| GitHub repository | `moeshawky/rockandegg-v3-code` |
| Release commit (authoritative) | `215cc710a9da62a8d56a8a8e676a41293dc5fc1b` |
| Release tag | `v3-fullbackbone-preview-1` (annotated `e3c6436…`, resolves to `215cc71`) |
| `main` HEAD after doc-only correction | `8912fb3fe03d7eb1f9eb2c7bf6a7ef48461ae98e` |
| Historical main tag | `v3-head-only-main` → `87c754951f9d4bd04c90e9ec78fc54295075e86c` |
| Historical head-only checkpoint SHA-256 | `3b208b86d6b5b87306091cbaec1b71007454eb4864b47efc636b4f633c875ac8` |

The `main` advance `215cc71 → 8912fb3` is a documentation-only commit that added exact
preview identifiers to `README.md`; it changed no weights, code or checkpoints. The
authoritative release identity remains the tag `v3-fullbackbone-preview-1` → `215cc71`.

## 3. Qualification status

**RELEASED — RESEARCH/PREVIEW; NOT production qualified.**

- Production ECE bound **0.05** unchanged and **NOT met**: raw ECE 0.1599 (RESERVED),
  0.2755 (SSF-1 sealed).
- C2 calibration rejected: sealed NLL regressed 1.0510 → 1.4563 (degenerate noul fit).
- No eligible unused representative calibration population exists.
- Probabilities are uncalibrated raw softmax at **T=1.0**; `confidence` is not a
  calibrated guarantee. No new temperature was fitted; no exposed partition reused.
- No superiority over Laya is claimed (student−Laya CI includes zero).

## 4. Frozen architectural contract

- 10× Qwen3-derived decoder blocks + native SystemOne decision head; 313,161,733 params.
- Decision types: `choice`, `score`, `noul`. No LM head, no `generate()`, no token output.
- Native `DynamicCache` preserved; one state prefill + one option-row pass.

## 5. Frozen behavioral contract

- Shared-state inference: exactly **2** backbone forwards at F1/F2/F4/F8 (verified).
- RTX PRO 6000 (cuda:0, bfloat16, label-blind inputs): p50 14.13–14.35 ms, p95 14.21–14.67 ms,
  throughput ~69.7–70.8 req/s, peak VRAM 654–694 MiB. These do **not** generalize to other hardware.
- Fresh-process integrity: 16/16 checks; FP32→BF16 parity 0 argmax flips / 50 decisions;
  remote↔local inference parity exact (max diff 0.0).
- Clean-process smoke test of the documented interface: 10/10 checks (load model,
  tokenizer, `SystemOneInference`, 3 decision types, normalization, confidence, encode-once).

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

The copy of the model card inside the frozen preview revision
(`fullbackbone-bf16-preview/README.md` at `0b60f73…`) uses generic
`<preview-branch>`/`<preview-subdir>` placeholders in its load snippet and does not spell
out the source commit/revision. Correcting it would require a new HF commit that would move
the preview branch off `0b60f73…`, violating §1/§3 of the freeze mandate and the
"do not replace the published model / preserve identity" constraints. The exact identifiers
are therefore published in the GitHub source README (doc-only commit `8912fb3`), in the
local canonical `MODEL_CARD.md` and `RELEASE_MANIFEST.json`, and in
`V3_OPERATIONAL_HANDOFF.md`. This is a non-blocking documentation discrepancy; no artifact
integrity or release-identity problem exists.

## 9. Evidence archive

`V3_RELEASE_ARCHIVE/` holds all 5 release reports, `V3_RELEASE_FREEZE.md`,
`V3_OPERATIONAL_HANDOFF.md`, integrity/remote/smoke/performance results, both
preregistrations, and experimental conclusions. `ARCHIVE_HASHES.json` records SHA-256 for
**20** archived files and explicitly excludes itself (**21** files on disk); zero hash
mismatches. The published artifact manifest records **17** payload files + the manifest
itself = **18** published files (independently confirmed against the remote); all payload
hashes verified. This payload-vs-manifest distinction is intentional and consistent across
both manifests.
