# RockAndEgg-313M-v3 — source repository (public release candidate)

A compact, **non-generative** typed-decision model. It does not produce text.
It returns one typed decision per question (`choice`, `score`, `noul`) with a
probability distribution and a confidence value.

This repository contains the training implementations, the architecture, the
qualification scripts and the tests. The published weights, tokenizer and the
**canonical HTTP server** live in
[`moeshawky/RockAndEgg-313M-v3`](https://huggingface.co/moeshawky/RockAndEgg-313M-v3).
Do not create a second server implementation here: `serving/systemone_server.py`
is a reviewed reference copy of that canonical file, kept in sync for auditing.

> **Two training procedures, do not confuse them.** `scripts/train_jev_distill.py`
> is the **historical head-only** (frozen-backbone) procedure behind the previous
> `main` weights (now preserved at tag `v3-head-only-main`). `scripts/train_full_backbone.py`
> is the **authoritative full-backbone** procedure behind the new canonical `main`
> release (`v3-fullbackbone-main-1`). See `FULL_BACKBONE.md`. The backbone was
> fully fine-tuned during training and is fixed only during inference.
> The full-backbone artifact is an explicitly **uncalibrated research/preview**
> release (ECE ≤ 0.05 gate NOT met); no calibrated-confidence claim is made for it.

## Canonical main release (exact identifiers)

The full-backbone BF16 seed-0 checkpoint is now the **default `main` artifact** of
the model repository. It ships at the repository root and loads without a subfolder.
The historical head-only release remains available via its tag.

| Item | Exact value |
|---|---|
| Model | `moeshawky/RockAndEgg-313M-v3` |
| Canonical release tag | `v3-fullbackbone-main-1` |
| FP16 precision tag | `v3-fullbackbone-fp16-1` |
| Release commit | `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8` |
| Release model SHA-256 | `c8b3aafb5ac4b5aad5808e49cd13a184205146711d0fc9e4ac123172c4141fd5` |
| FP16 model SHA-256 | `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28` |
| Source checkpoint (FP32) SHA-256 | `1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92` |
| Parameters | 313,161,733 |
| Source commit (this repo) | `215cc710a9da62a8d56a8a8e676a41293dc5fc1b` |
| Source tag (preview) | `v3-fullbackbone-preview-1` — immutable identifier for the original preview publication; **not** rewritten |
| Source tag (canonical main) | `v3-fullbackbone-main-1` — distinct immutable identifier for the canonical-main promotion |
| Historical head-only tag | `v3-head-only-main` → `87c754951f9d4bd04c90e9ec78fc54295075e86c` |
| Preserved preview branch | `preview/fullbackbone-bf16-seed0` → `0b60f73eda4293b4891f27c3106a977cef536a1f` |

Load the canonical main release (no subfolder required):

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "moeshawky/RockAndEgg-313M-v3"
revision = "v3-fullbackbone-main-1"
model = AutoModel.from_pretrained(
    model_id, revision=revision, trust_remote_code=True, dtype=torch.bfloat16
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision, trust_remote_code=True
)
assert sum(p.numel() for p in model.parameters()) == 313_161_733
```

Load the historical head-only release (preserved, independently retrievable):

```python
model = AutoModel.from_pretrained(
    "moeshawky/RockAndEgg-313M-v3", revision="v3-head-only-main",
    trust_remote_code=True, dtype=torch.bfloat16,
)
```

## What is actually novel

The intervention is **shared-state inference**, not a new classifier:

1. A frozen, headless 10-layer Qwen3 backbone encodes the *state once* into a
   native `DynamicCache`.
2. That cache is deep-copied and expanded across independently batched
   candidate option rows (`batch_repeat_interleave`). The state is **never**
   re-encoded per question or option.
3. Each option row carries its own causal suffix terminated by the marker token
   `<|im_end|>` (id `151645`). Rows are isolated; the decision head mixes them.
4. Stock causal attention and position IDs are reused — no custom attention
   mask, no rotary-embedding surgery.
5. The head emits a distribution over options. There is no LM head, no
   `generate()` method and no token generation anywhere in the artifact.

Measured structural facts, independently re-derived from the published
`model.safetensors` header and `config.json`:

| Fact | Value |
| --- | --- |
| Tensors | 122 |
| Parameters | 313,161,733 (backbone 312,892,928 + head 268,805) |
| `lm_head` tensors | 0 |
| Backbone layers / hidden / KV heads / head dim | 10 / 1024 / 8 / 128 |
| State / option token limit (max) | 256 / 64 tokens |
| Generation flag in `config.json` | `false` |

Lineage: Laya (decision architecture), Verdict-MQ (shared-state hypothesis
evaluation and invariance), RockAndEgg (native Qwen3 `DynamicCache` reuse and
batched candidate evaluation). The backbone is Qwen3-0.6B-Base; see
`ATTRIBUTION.md`.

## Install

Python ≥ 3.10. The model-specific code ships in the model repository, but the
generic runtime libraries are **required** and are not bundled.

```bash
python -m pip install -r serving/requirements-server.txt   # serving only
python -m pip install -e '.[dev]'                            # training + tests
```

Note: `transformers 5.x` declares `huggingface-hub>=1.5,<2.0`. A newer hub
(for example 2.4.0) breaks `import transformers`; keep the pin below 2.0.
T4 is Turing and has **no bf16** — inference uses fp16 on CUDA and fp32 on CPU.

## Load and run inference

```bash
hf download moeshawky/RockAndEgg-313M-v3 --local-dir ./rae-v3
cd rae-v3
python systemone_server.py          # binds 127.0.0.1:8012
curl -s localhost:8012/ready
curl -s localhost:8012/v1/systemone -H 'content-type: application/json' -d '{
  "state": "The deployment failed at 03:12 UTC.",
  "questions": [{"type": "choice",
                 "instruction": "What should the operator do next?",
                 "options": ["Roll back the release", "Ignore it"]}]
}'
```

Endpoints: `POST /v1/systemone`, `GET /healthz`, `GET /ready`.
Readiness reflects the loaded model and the selected device; `gpu_available`
is informational, so a CPU deployment reports `ready: true`.
One HTTP request returns one decision per submitted question. Empty questions
and empty options are rejected with the canonical four-field error block
(`error / Cause / Action / Exit`); exit classes are `3` (bad request) and `4`
(inference failed). There is no exit class 5 — startup fails fast, so a
serving process is always ready.

Server configuration (all optional, safe defaults):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAE_HOST` / `RAE_PORT` | `127.0.0.1` / `8012` | Bind address. Non-loopback requires `RAE_API_TOKEN` and enforces Bearer auth on every route. |
| `RAE_MODEL_ID` / `RAE_MODEL_REVISION` | local dir | A Hub-sourced model **must** be pinned to an exact commit SHA. |
| `RAE_CODE_DIGEST` | unset | If set, the local adapter/modeling code digest must match before serving. |
| `RAE_BACKBONE` | `moeshawky/RockAndEgg-313M-v2` | Frozen v2 backbone used by training and verification scripts. |

The default bind is localhost only. An unauthenticated inference endpoint must
not be exposed to a non-loopback interface; use an authenticating reverse proxy
and TLS, or set the token gate described above.

## Layout

```
src/rockandegg_v3/    model, loss, gates, inference, parallel, config
scripts/              training (head-only + full-backbone), export, verification, gate runners
FULL_BACKBONE.md      selected full-backbone procedure vs historical head-only procedure
tests/                CPU unit tests
serving/              reviewed reference copy of the canonical HF server,
                      adapter, custom modeling code and CPU server tests
LICENSE               Apache-2.0
REPRODUCTION.md       how to reproduce training and qualification
QUALIFICATION.md      claim-to-evidence status (what is and is not verified)
ATTRIBUTION.md        licenses and third-party lineage
PUBLIC_MANIFEST.md    public-facing file manifest
```

## Tests

```bash
python -m pytest tests/ -q            # 23 passed, 2 skipped in the audit run
python -m pytest serving/test_systemone_server_cpu.py -q   # 16 passed
ruff check .
```

The two skips are explicit: they compare against a private Laya benchmark
harness that is not shipped in this public candidate. Skipped tests are not
reported as passes. See `QUALIFICATION.md` for the status of all numeric claims.

## Honest limitations

* No training corpus, model weights or checkpoints are bundled. The missing
  measurement artifacts were **regenerated from public inputs** (frozen v2
  backbone, the Apache-2.0 corpus at revision `75111a13…`, typed-decisions,
  BANKING77) on two T4 GPUs and are included in `evidence/`. `QUALIFICATION.md`
  records each reproduced value alongside its historical value. Small
  differences (e.g. 477 vs 490 fp16 choice changes) are reported as measured,
  not blended into the old figures.
* The Laya benchmark arm cannot be independently reproduced: the public Laya
  checkpoint ships no inference wrapper. Its third-party accuracy stays
  third-party published; only the v3 arms were reproduced.
* The shared-state speed claim is structural (two backbone forwards at every
  measured fan-out, F1–F8), not a claim of wall-clock superiority. A controlled
  matched single-request benchmark on one RTX PRO 6000 (64 states, 24 warmups,
  BF16) measured RockAndEgg at 14.325/14.767/15.498/16.691 ms p50 (F1–F8)
  versus Laya at 12.403/13.285/13.740/14.697 ms; Laya was faster on that
  workload. See `BENCHMARK.md` in the model repository.
* A newer matched in-process benchmark on one T4 (fp16) compared the optimized
  (A+D) serving path shipped here with the original published implementation and
  with native Laya over identical text. The optimized path is 1.52 ms faster than
  the original (95% CI [1.42, 1.62], ~4.3%, exact decision parity) and faster than
  Laya in all 11 populated configurations (1.05–2.00× depending on fan-out;
  stratified ~1.33×). These are in-process latency and memory results only — no
  accuracy claim is derived from them, and the older ~1.48× T4 figure is not
  reproduced as a single settled number. See
  [`docs/INFERENCE_BENCHMARKS.md`](docs/INFERENCE_BENCHMARKS.md).
* The intended differentiator is computation sharing, not superior accuracy.
  On the matched benchmark the alternative model scored higher accuracy; this
  release does not present that comparison as an accuracy victory.
* The ECE gate is artifact-dependent. Without the held-out prediction file it
  is reported NOT EVALUATED, never as a pass.
* The public candidate has fresh Git history. Original repository history is
  deliberately not reproduced.

## Release dossier

The closure record for the full-backbone BF16 research/preview release — model card,
freeze, operational handoff, qualification, selection/publication reports and the full
evidence archive — is in [`docs/release/`](docs/release/README.md). The exact
release-engineering scripts are in [`scripts/release/`](scripts/release/README.md).

## License

Apache-2.0 — see `LICENSE` and `ATTRIBUTION.md`.
