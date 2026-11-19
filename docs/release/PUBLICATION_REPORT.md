# Publication Report — RockAndEgg-313M-v3 Full-Backbone BF16

Date: 2026-11-19. Final status: **RELEASED — RESEARCH/PREVIEW**.

## 1. Git commit and tag (source repository)

- Repository: `https://github.com/moeshawky/rockandegg-v3-code` (public, sanitized history).
- Base: `74658b1c7f3d2572d02c863626a8b4954e4221cd`.
- Release commit: `215cc710a9da62a8d56a8a8e676a41293dc5fc1b` — "Document selected
  full-backbone training path".
- Tag: `v3-fullbackbone-preview-1` (annotated, object `e3c64366ea29ea9de1e76e89c5ae11241bf7ffc7`).
- Pushed by fast-forward; original public history preserved; no force-push; the
  historical head-only implementation retained unchanged.
- Added: `scripts/train_full_backbone.py`, `scripts/full_backbone_utils.py`,
  `scripts/full_backbone_config.json`, `FULL_BACKBONE.md`; updated `README.md`.
- Not committed: checkpoints, datasets, private manifests, credentials, transcripts, logs.
- Tests before commit: public suite 25 passed / 2 skipped (skips are the explicitly
  private Laya harness). Local workspace suite: 29 passed / 2 failed-at-import / 1 skipped
  (missing optional `rl_agent_api`/`rl_common`), reported honestly.

## 2. Hugging Face repository and revision

- Repository: `https://huggingface.co/moeshawky/RockAndEgg-313M-v3`.
- Historical main revision preserved: `87c754951f9d4bd04c90e9ec78fc54295075e86c`.
- Immutable tag created: `v3-head-only-main` → `87c7549…` (verified).
- Preview branch: `preview/fullbackbone-bf16-seed0`.
- Preview revision: `0b60f73eda4293b4891f27c3106a977cef536a1f`.
- Artifact subfolder: `fullbackbone-bf16-preview/` (18 files, including manifest).
- No force-push; main untouched; no experimental checkpoints or ensembles uploaded.

## 3. Remote file manifest (published)

`ATTRIBUTION.md`, `CALIBRATION_LIMITATIONS.md`, `EVALUATION_REPORT.md`, `LICENSE`,
`README.md`, `RELEASE_MANIFEST.json`, `TRAINING_PROVENANCE.md`,
`bf16_conversion_report.json`, `config.json`, `configuration_rockandegg_v3.py`,
`merges.txt`, `model.safetensors` (SHA-256
`c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5`),
`modeling_rockandegg_v3.py`, `requirements.txt`, `systemone_inference.py`,
`tokenizer.json`, `tokenizer_config.json`, `vocab.json`.
Authoritative per-file SHA-256 in the published `RELEASE_MANIFEST.json`.

## 4. Remote reload result

`artifacts/remote_verification.json` — all 5 gates pass:

| Gate | Result |
|---|---|
| Main preserved at `87c7549…` | ✅ |
| File set matches manifest (+ self) | ✅ |
| All 17 file hashes match after fresh download | ✅ |
| Parameter count 313,161,733 | ✅ |
| Remote↔local inference parity (max diff 0.0, same chosen keys) | ✅ |

The exact remote revision was downloaded into a fresh directory, reloaded through the
canonical `AutoModel` + `SystemOneInference` path, and executed on `choice`, `score` and
`noul` requests. Output matched the local release candidate exactly.

## 5. Remaining qualification status

- **Production qualification: NOT achieved.** ECE ≤ 0.05 gate fails (0.1599 RESERVED,
  0.2755 SSF-1 sealed); C2 rejected; no eligible unused calibration partition.
- Research/preview publication is authorized and completed under an explicit
  uncalibrated label; no calibrated-confidence claim is made.
- No claim of beating Laya; ensemble and distilled variants are research-only.
- Next legitimate step (out of scope here): obtain a fresh, representative, unused typed
  population and pass a degeneracy-gated ECE fit to reach production qualification.

## 4. Promotion to canonical main (2026-11-19)

The full-backbone BF16 seed-0 checkpoint was promoted to the default `main`
artifact of `moeshawky/RockAndEgg-313M-v3`. The historical head-only release is
preserved and independently retrievable.

- New main commit: `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8`.
- New immutable tag: `v3-fullbackbone-main-1` → `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8`
  (verified resolves to the new main commit).
- Release model SHA-256: `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5`
  (verified on fresh download).
- Old main commit preserved: `87c754951f9d4bd04c90e9ec78fc54295075e86c`.
- Historical tag preserved: `v3-head-only-main` → `87c7549…` (verified unchanged).
- Preview branch preserved: `preview/fullbackbone-bf16-seed0` → `0b60f73…`.
- Published root files: 24 (model at root, no subfolder; obsolete `fp16/` variant
  and stale head-only artifacts removed in the same commit).
- Verification: `verify_main.py` passed 24/24 files + param count 313,161,733 +
  3 decision types + probability normalization + option-order stability + raw
  T=1.0 + no calibration file. Exact README loading snippet executed against the
  published revision tag successfully.

Final status: **RELEASED — CANONICAL MAIN / RESEARCH-PREVIEW**.
Production calibration remains outstanding (ECE ≤ 0.05 NOT met).
