# V3 Release Dossier (documentation + evidence)

This directory preserves the closure record for the **RockAndEgg-313M-v3 full-backbone
BF16 research/preview release**. Status: **FROZEN WITH DOCUMENTATION LIMITATIONS**.
The model is **RESEARCH / PREVIEW, NOT production qualified** — probabilities are
uncalibrated (raw T=1.0) and the ECE ≤ 0.05 gate is NOT met.

## Canonical identifiers

- Model: `moeshawky/RockAndEgg-313M-v3`, branch `preview/fullbackbone-bf16-seed0`,
  revision `0b60f73eda4293b4891f27c3106a977cef536a1f`
- Released model SHA-256: `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5`
- Source release commit: `215cc710a9da62a8d56a8a8e676a41293dc5fc1b`, tag `v3-fullbackbone-preview-1`
- Historical head-only `main`: `87c754951f9d4bd04c90e9ec78fc54295075e86c`, tag `v3-head-only-main`

## Documents

| File | Purpose |
|---|---|
| `MODEL_CARD.md` | Living, source-cited model card (measure-first, last verified 2026-11-19) |
| `V3_RELEASE_FREEZE.md` | Exact identity, frozen contract, conditions for reopening development |
| `V3_OPERATIONAL_HANDOFF.md` | First-time integrator guide; measured RTX PRO 6000 conditions |
| `MODEL_SELECTION_REPORT.md` | Candidate identities, frozen selection criterion, rejected candidates |
| `RELEASE_QUALIFICATION.md` | Architecture/integrity/parity/performance verification; limitations |
| `PUBLICATION_REPORT.md` | Exact Git commit/tag, HF revision, remote reload result |
| `RELEASE_MANIFEST.json` | Model identity, source/model revisions, file hashes cross-reference |

## Evidence (`evidence/`) and full archive (`archive/`)

`evidence/` holds the machine-readable results: `release_integrity.json` (16/16),
`release_performance.json` (encode-once, F1–F8), `smoke_interface.json` (10/10),
`remote_verification.json` (5/5), `archive_hashes.json`. `archive/` holds the durable
evidence package including both preregistrations and the experimental conclusions
(data lineage, ensemble validation, distillation, final qualification).

## Known documentation limitation

The model card copy inside the frozen HF preview revision uses generic
`<preview-branch>` / `<preview-subdir>` placeholders in its load snippet; exact
identifiers are in `MODEL_CARD.md` and `V3_OPERATIONAL_HANDOFF.md`. The published
revision is intentionally not modified to preserve `0b60f73…` byte-for-byte.
