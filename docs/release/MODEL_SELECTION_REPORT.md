# Model Selection Report — RockAndEgg-313M-v3 finalization

Date: 2026-11-19. All identities verified by SHA-256 and fresh-process load. Selection
criterion was frozen before any release evaluation (see §2).

## 1. Candidate identities

| Seed | Checkpoint | model.safetensors SHA-256 | DEV best | Step | Tokenizer |
|---|---|---|---:|---:|---|
| 0 | `artifacts/runs/fullbb_bf16_e2style/best/model` | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` | **0.7800** | 320 | present |
| 1 | `.../fullbb_bf16_e2style_s1/best/model` | `209ae6012ecd0d3c2b974954a7b0e66d99a1b84605a5a11eb24d0edffea52e3d` | 0.7533 | 400 | present |
| 2 | `.../fullbb_bf16_e2style_s2/best/model` | `39af46c6663a2dd3b6d47eb56d3fa45d9200cd440b686f0928e94ae39d57bcd1` | 0.7711 | 400 | **missing** |

All three are genuine full-backbone checkpoints (312,892,928 backbone + 268,805 head
trainable; all 112 backbone tensors changed; optimizer covered all parameters).

## 2. Selection criterion (frozen before evaluation)

The original campaign's frozen selection rule applies: each seed's best checkpoint was
chosen by validation accuracy on the 90-dossier DEV set (`selection_set=DEV`,
`reserved_used_for_selection=false` in every `summary.json`). The operator additionally
directed: use the original frozen criterion; do not pick a seed merely for the highest
number on an already-inspected panel; if the reserved student was pre-selected, consider
it the preferred candidate; otherwise use the original validation procedure.

## 3. Selected checkpoint

**Seed 0** — `fullbb_bf16_e2style/best/model`, SHA-256 `1b7fee92…d87ac8`.

Justification:
1. Highest original DEV best (0.7800 @ step 320) under the frozen criterion.
2. Designated `student_checkpoint` in `artifacts/final_eval.json` — the pre-selected
   reserved student.
3. Pinned as the student initialization in the November-18 distillation pre-registration
   specifically to avoid validation-driven selection.
4. Loads standalone with its own tokenizer; seeds 1/2 do not improve defensibly.
5. Robust to the operator's stated dev figures (77.11/76.22/77.56): seed 0 also leads
   under those numbers. The artifact discrepancy (0.7800/0.7533/0.7711) is flagged, not
   hidden; the selection is unchanged under both.

The released FP16 artifact is a precision cast of this checkpoint:
`model.safetensors` SHA-256 `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28`
(direct `to(float16)` from the published BF16 artifact `c8b3aafb…`; no FP32 intermediate;
labelled BF16-derived, never as an FP32 export).

## 4. Rejected candidates and reasons

- **Seed 1** — +1.45 pts on the 400-dossier clean panel, but its paired CI vs seed 0
  includes zero (`[−0.0015, +0.0300]`) and selecting on an inspected panel contaminates
  the comparison. Not the originally-selected seed. Rejected.
- **Seed 2** — lower DEV best (0.7711) and its checkpoint lacks tokenizer files (cannot
  load standalone). Rejected.
- **Three-model ensemble** — 0.7545 vs seed 0 0.7335 on clean data (+2.1 pts, CI
  `[+0.0085, +0.0340]`), a real research finding, but the product objective is one
  compact model and the operator forbade ensemble serving. Not selected.
- **75/25 distilled treatment** — matched experiment FAILURE (treatment 0.7180 vs
  control 0.7310; choice regression −4.25 pts, CI excludes zero). Not selected.
- **Historical head-only `main` checkpoint** (`3b208b86…`) — lower accuracy, frozen
  backbone; preserved on main, not the full-backbone release.

## 5. Release selection requirements — verified

- Genuine full-backbone provenance: yes (§1, `layer_update_report.json`).
- Exact SHA-256: recorded above and in `RELEASE_MANIFEST.json`.
- Reproducible training config: `scripts/full_backbone_config.json` (public source).
- Loads through canonical model implementation: yes (`AutoModel`, `trust_remote_code`).
- Native SystemOne interface preserved: yes (3 decision types, no LM head/generate).
- Expected architecture / parameter count: 313,161,733 — verified.
- Fresh-process loading and inference: yes (16/16 integrity checks; remote parity exact).
