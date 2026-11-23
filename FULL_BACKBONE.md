# Full-Backbone Training Path (selected release procedure)

> **Current-release notice (2026-11-23):** this document preserves the historical
> full-backbone BF16/FP16 procedure. The canonical default weights are now
> **R_final3** (SHA-256 `bd30d80e…`, promotion commit `837d3d5…`). The current
> source-only addition is the opt-in semantic rendering helper documented in
> [`docs/release/SEMANTIC_RENDERING_RELEASE.md`](docs/release/SEMANTIC_RENDERING_RELEASE.md).
> No training procedure or weight changed.

This repository preserves **two** training procedures. They must not be confused.

## 1. Historical head-only procedure (kept for provenance)

`scripts/train_jev_distill.py` freezes the Qwen3 backbone and trains only the
268,805-parameter SystemOne decision head on `SargeDev/jev-distill-corpus-v3`
(Apache-2.0, revision `75111a13…`). This is the procedure that produced the
**published** `moeshawky/RockAndEgg-313M-v3` checkpoint on `main` (frozen backbone,
312,892,928 backbone tensors bit-identical to v2). It is retained unchanged and its
history is not rewritten. It is **not** the procedure behind the full-backbone canonical release.

## 2. Selected full-backbone procedure (this release)

`scripts/train_full_backbone.py` trains all 10 decoder blocks + final norm + head
(312,892,928 + 268,805 trainable parameters) end-to-end on the matched
`LocalLLaMA/typed-decisions` corpus (config `all`, revision
`e6aa632c4db73b59c7713bb21066ad1a8e1bfce6`). It keeps the architecture contract:
one state encode per forward, option rows isolated in batch, no LM head, no generation.

Reproduce the selected seed-0 configuration:

```bash
python3 scripts/train_full_backbone.py --config scripts/full_backbone_config.json
```

The config supplies the model id (the **F32** v3 root weights — not the frozen-head
release — or a local copy), the dataset revision, the dossier-grouped split
(`split_seed=0`, 1,020 train / 180 val; val → 90 DEV / 90 RESERVED), the optimizer
(AdamW, backbone lr 2.5e-5 / head lr 1e-4, wd 0.01, 3% warmup + cosine), 8 epochs,
micro-batch 64, and bf16 autocast with FP32 master weights. Checkpoint selection uses
the frozen DEV criterion (`selection_set=DEV`, `reserved_used_for_selection=false`).

Measured provenance of the selected checkpoint (workspace investigation, 2026-11):
DEV best 0.7800 at step 320; all 112 backbone tensors changed with finite grads;
optimizer covered all 312,892,928 backbone + 268,805 head parameters. Three
independent seeds: 0.7800 / 0.7533 / 0.7711 DEV best.

## Release status

The seed-0 full-backbone checkpoint cast to BF16 was first published on the versioned
preview branch `preview/fullbackbone-bf16-seed0` and was subsequently promoted to the
canonical `main` artifact of `moeshawky/RockAndEgg-313M-v3` under the immutable tag
`v3-fullbackbone-main-1` (commit `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8`). The
preview branch and its revision `0b60f73eda4293b4891f27c3106a977cef536a1f` remain
preserved. It is a **research/preview** release, not production qualified: the
ECE ≤ 0.05 gate is NOT met (raw ECE 0.1599 RESERVED / 0.2755 SSF-1 sealed), C2
calibration is rejected, and probabilities are uncalibrated T=1.0. See the model
card's `CALIBRATION_LIMITATIONS.md`. The three-model ensemble (research context only,
+2.1 pts on a clean panel) and the failed 75/25 distillation are documented in
`EVALUATION_REPORT.md`; neither is the deployed model.

## Test note

The public `tests/` suite covers the core model, gates, parallel path, ECE definition
and typed-decision encoding. Two typed-decision tests import `benchmark_vs_laya`,
which requires the optional private `rl_agent_api` / `rl_common` Laya-runtime modules
not shipped in a public candidate; in the 2026-11 workspace run those two tests failed
at import and one ECE test skipped for the same reason, while 29 others passed. Those
are explicit environment limitations, not logic regressions, and are reported as such
rather than relabelled as passes.
