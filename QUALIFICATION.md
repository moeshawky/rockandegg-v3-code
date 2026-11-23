# Qualification status — RockAndEgg-313M-v3

> **Current-release notice (2026-11-23):** this ledger records the historical
> qualification. For the current R_final3 source-only release see
> [`docs/release/SEMANTIC_RENDERING_RELEASE.md`](docs/release/SEMANTIC_RENDERING_RELEASE.md)
> and `evidence/semantic_rendering_validation.json`. Probability confidence
> remains **uncalibrated**; the semantic renderer changes option text and stable
> keys only, not probability semantics.

This file is an evidence ledger, not a claim catalogue. A claim is promoted
only when an executed operation in **this workspace** measures the property it
asserts. "The script exists" is never treated as "the measurement ran".

Verdicts: **VERIFIED** (independently re-derived here) · **SUPPORTED** (derived
from a present artifact but not an executed measurement) · **NOT VERIFIED**
(no supporting measurement artifact in this workspace) · **CONTRADICTED** (a
present artifact refutes the claim as written).

## Verified here

| Claim | Evidence |
| --- | --- |
| 122 tensors, 313,161,733 parameters | Independent parse of `model.safetensors` header in the published snapshot. |
| Backbone/head split 312,892,928 + 268,805 | Same header parse (tensor-name prefix). |
| Zero `lm_head` tensors; no `generate()`/`CausalLM` call site | Header key scan + source grep across published and candidate code. |
| Qualified limits 256 state / 64 option tokens | `config.json` keys `max_state_len`/`max_option_len`; server and adapter constants. |
| KV cache at S=512 is 20 MiB fp16 / 40 MiB fp32 | Arithmetic from present `head_dim=128`, `num_key_value_heads=8`, `num_hidden_layers=10`: `2*8*128*2*10*512 = 20,971,520 B`. The historical "measured" result JSON is absent. |
| Server endpoints, error shape and exit classes 3/4 | Source inspection + 16 executed CPU tests. |
| Compute semantics of source and published decision path are equivalent | Function-by-function comparison (`encode_state`, `_read_options`, `forward_case`, `forward_batch`, `_head_forward`); only naming/config-access differences. |
| Frozen backbone, trainable head, soft-label CE + squared EMD, temperature scaling, option dropout 0.3 | Code inspection of `src/rockandegg_v3/{model,loss}.py` and `scripts/train_jev_distill.py`. |

## Reproduced in this workspace (public inputs, executed)

The missing measurement artifacts were regenerated rather than merely
disclosed. External inputs came from public Hugging Face repositories:
`moeshawky/RockAndEgg-313M-v2` (frozen backbone),
`SargeDev/jev-distill-corpus-v3` revision `75111a13a8455032f0435050bfc7e0c9026ecfe8`,
`LocalLLaMA/typed-decisions`, `PolyAI/banking77`. The head checkpoint used for
the gates was extracted **unchanged** from the published `model.safetensors`.
Runtime: torch 2.11.0+cu128, transformers 5.16.1, 2× Tesla T4.

| Claim | Development value | Reproduced here | Status |
| --- | --- | --- | --- |
| Held-out accuracy | 0.5263 | **0.5263228** (n=29,955) | REPRODUCED |
| ECE fp32 / fp16 | 0.03656 / 0.03677 | **0.036654 / 0.036764** (T=1.0, serving path) | REPRODUCED |
| Option-permutation invariance | 3.6e-7 fp16 / 2.4e-7 fp32 | **3.58e-7 fp16 / 1.19e-7 fp32** | REPRODUCED |
| Question order / state independence / leakage | 0.0 | **0.0 / 0.0 / 0.0** | REPRODUCED |
| Moving control | 3.390625 | **3.390625** | REPRODUCED |
| Fresh-process reload parity | ≤1e-6 | **0.0**, child exit **0** | REPRODUCED |
| Held-out case count | 29,955 | **29,955** | REPRODUCED |
| FP16 probability mean delta / changed options | 7.23e-4 / 490 | **7.17e-4 / 477** | REPRODUCED (small env delta) |
| FP16 tensor identity | 122/122, max 0.0 | **122/122, max 0.0** | REPRODUCED |
| KV cache at S=512 | 20 / 40 MiB | **20.0 / 40.0 MiB** | REPRODUCED |
| Backbone forwards per request | 2 vs 1+n | **2** at 2/4/6/9 options | REPRODUCED |
| Full-export verdict | approved | **true** with ECE supplied | REPRODUCED |
| v3_release benchmark arm | 0.3425 | **0.3395** (400 cases) | REPRODUCED |
| v3_matched_typed benchmark arm | 0.5230 | **0.47** (fresh matched head) | REPRODUCED AT DIFFERENT VALUE |
| BANKING77 linear probe | 0.8344 | **0.8666** | FALSIFIED CLAIM CONFIRMED |
| Dual-GPU split parity | ~2.9e-3 fp16, tol 1e-2 | **4.88e-3** | REPRODUCED (within 1e-2, not 1e-6) |

Additional executed checks: one-request adapter latency mean 41.45 ms
(S=256, 3 questions, fp16; **not** a Laya/Verdict comparison); canonical HF
freeze fix verified at 112/112 backbone parameters frozen with byte-identical
decisions.

## Not reproduced (with reason)

* **Laya benchmark arm (0.7665).** The public Laya checkpoint ships no
  inference wrapper; the harness requires a private agent API and a private ECE
  metric helper. No substitute was invented — that would measure a
  different system. The third-party figure remains third-party published.
* **Backbone bit-identity to the F32 v2 source.** The public v2 Hub file is the
  FP16 rendering of that source; equality against the F32 v3 backbone is not
  expected and one export-metadata probe correctly reports FAILED for this
  reason. The F32 source is not public.
* **The original training run record.** A fresh 80,000-case reproduction was
  executed on 2×T4: 5,002 steps in 2,956.5 s, held-out accuracy **0.52499** on
  29,955 cases (documented 0.5263). It is a new run, not the historical log.
  Small numeric differences (477 vs 490 flips, 0.47 vs 0.523 matched arm) are
  reported as measured under this runtime and are not blended into old values.
  The training report's held-out ECE (0.0668) differs from the gate ECE
  (0.03665) because the report applies the advisory fitted temperature while
  serving deliberately uses T=1.0; no temperature is wired into the model.

## Reconciled contradictions

* **Invariance.** Structural equivariance does not imply finite-precision
  equality. Correct wording: "≤ 1e-6 measured; option permutation 3.6e-7 fp16
  and 2.4e-7 fp32; exactly 0.0 only in exact arithmetic."
* **Exit code 5.** Documented historically but no branch existed. Removed from
  the public server docstring and documentation; only classes 3 and 4 remain.
* **"No separate package install".** Corrected: no *model-specific* package is
  needed, but torch, transformers, huggingface-hub (<2.0), fastapi, uvicorn and
  pydantic must be installed (`requirements-server.txt`).
* **State length.** An earlier validation context used 128 tokens; the release
  qualifies 256. Keep 256 as the single external bound and treat 512 as
  unqualified.
* **Config metadata.** `systemone_head.max_state_len` and `max_option_len` are
  documentation metadata only; runtime limits are enforced by the adapter and
  server. `backbone_frozen` is now honoured by the published modeling code in
  this candidate.
* **`reload_passed` ambiguity.** A `None` reload result must not be reported as
  a pass. Treat a missing parity measurement as NOT VERIFIED.
