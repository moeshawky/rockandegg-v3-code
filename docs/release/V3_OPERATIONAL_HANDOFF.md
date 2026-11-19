# V3 Operational Handoff — RockAndEgg-313M-v3 Full-Backbone FP16 (RESEARCH / PREVIEW)

This document is written for an engineer encountering RockAndEgg v3 for the first time.
It needs no access to the private research history — just the identifiers and steps below.

## 1. What RockAndEgg v3 is

A compact (313,161,733-parameter), **non-generative** typed-decision model derived from a
10-layer Qwen3 backbone. It does not produce text. For one input state plus one or more
typed questions, it returns one decision per question of type `choice`, `score`, or
`noul`, each with a probability distribution over options and a `confidence` value.
There is no LM head and no `generate()` path. It is licensed Apache-2.0.

## 2. What it is designed to do

Serve typed decisions efficiently over a shared state. The novel mechanism is
**shared-state inference**: the state is encoded once into a native `DynamicCache`, and
all candidate option rows fan out from that single cache inside one batch. No option is
re-encoded against the state. This is a computation-sharing design, not a claim of
superior accuracy.

## 3. Typed-decision interface

A request carries one `state` (text) and a list of `Question(type, options, instruction)`:
- `choice` — distribution over the named options; `chosen_key` is the arg-max option.
- `score` — distribution over ordinal levels; the decision reports the arg-max level.
- `noul` — two-way `false`/`true` distribution; `chosen_key` is the arg-max label.

A response is one `QuestionDecision` per question with `probabilities` (sum ≈ 1),
`chosen_key`, and `confidence` (= max probability). Empty questions/options are rejected
by the canonical server with the four-field error block (`error / Cause / Action / Exit`).

## 4. Shared-state advantage (encode-once)

Verified on the published artifact: a request produces exactly **two** backbone
forwards regardless of fan-out — one `[1, 256]` state prefill plus one `[4, 64]`
option-row pass — at F1, F2, F4 and F5. Naive per-option re-encoding would need up to 10
calls at F5. The native `DynamicCache` is preserved; no custom attention mask or RoPE
surgery. No alternative server or abstraction layer should be introduced.

## 5. Requirements

- Python ≥ 3.10; torch ≥ 2.11; transformers ≥ 5.16,<6; huggingface-hub ≥ 1.5,<2.0;
  safetensors ≥ 0.4. Install: `python -m pip install -r requirements.txt` (from the
  artifact folder) or the source repo's `serving/requirements-server.txt` for the server.
- `transformers 5.x` requires `huggingface-hub < 2.0` (a newer hub breaks import).
- FP16 weights load on any CUDA device (T4/Turing supported natively) or CPU
  (`dtype="float32"`). No bf16 required.

## 6. Load and run (executable example)

Exact identifiers (also pinned in `V3_RELEASE_FREEZE.md`):

```python
from transformers import AutoModel, AutoTokenizer
from systemone_inference import Question, SystemOneInference, SystemOneRequest

REPO = "moeshawky/RockAndEgg-313M-v3"
REV  = "v3-fullbackbone-fp16-1"   # FP16 research/preview tag

model = AutoModel.from_pretrained(
    REPO, revision=REV, trust_remote_code=True, dtype="float16",
).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(REPO, revision=REV)

adapter = SystemOneInference(model, tok)   # confidence_temperature defaults to 1.0
req = SystemOneRequest(
    state="The payment service returned HTTP 503 after the deploy at 03:12 UTC.",
    questions=[
        Question(type="choice", options=["Roll back", "Ignore", "Add logging"], instruction="next action?"),
        Question(type="score", options=["low", "medium", "high", "critical"]),
        Question(type="noul", options=["false", "true"]),
    ],
)
for d in adapter(req).decisions:
    print(d.type, d.chosen_key, d.probabilities, d.confidence)  # probabilities sum ~= 1
```

CPU: replace `.to("cuda")` with `.to("cpu")` and use `dtype="float32"`. This exact
interface was smoke-tested clean-process: model/tokenizer load, adapter init, all three
decision types, normalization, confidence=max, and encode-once all passed (10/10).

For an HTTP server, use the canonical `systemone_server.py` in the model repository
(`POST /v1/systemone`, `GET /healthz`, `GET /ready`), not a reimplementation.

## 7. Measured performance (exact conditions — do not generalize)

Measured on **1× NVIDIA Tesla T4** (driver 580.178.04), `cuda:0`, **fp16**,
transformers 5.16.1 / torch 2.11.0+cu128, serial single-request workload,
3 rounds per cell, 100 timed requests per cell, wall-clock with `torch.cuda.synchronize()`:

| Arm | Backbone calls (naive) | mean p50 ms | req/s | decisions/s | peak VRAM MiB |
|---|---:|---:|---:|---:|---:|
| `rae_shared` (this release) | 2 | 43.5 | 22.49 | 67.46 | 2,575 |
| `rae_independent` (naive) | 2F | 111.9 | 8.71 | 26.13 | 2,297 |
| `laya_native` | 1 | 65.1 | 15.20 | 45.60 | 3,001 |

These numbers are specific to this hardware/runtime/build and must not be quoted for
other devices. They are not serving throughput (single-request, no concurrency) and are
not a controlled comparison against Laya or any other system. The shared path trades
+278 MiB peak memory for less compute (2 vs 2F forwards).

## 8. Known limitations and status

- **RESEARCH / PREVIEW — NOT production qualified.** Probabilities are uncalibrated
  raw softmax at T=1.0. Production ECE bound 0.05 is NOT met (0.1323 on the FP16 test
  panel; BF16-era panels 0.1599 / 0.2755). C2 calibration rejected (sealed NLL 1.0510
  → 1.4563). `confidence` is **not** a calibrated guarantee — do not use it for risk
  thresholds or gating.
- No accuracy superiority over Laya is claimed. On this panel Laya measured higher
  (−3.40 pp, CI [−5.50, −1.45], excludes zero); this is not a universal ranking.
- No superiority claim on speed; concurrency unmeasured and could invert the shared-path
  result. Encoded-once is structural, not a wall-clock superiority claim.
- N3 pre-registered max-probability bound vs FP32 compute is NOT met (5.038e-3 > 5.0e-3).

## 9. Reporting failures or regressions

When reporting an issue, include:
1. Exact model revision (`v3-fullbackbone-fp16-1`), `dtype`, and source commit.
2. Full input (state + questions) and the decision output (probabilities, chosen key, confidence).
3. Hardware/GPU, driver, torch/transformers/hub versions, OS, and device (CPU/CUDA).
4. Expected vs observed behavior and whether probabilities still sum to ~1.
5. For latency regressions: fan-out, warmup/iteration counts, and whether measurement
   conditions match §7 (do not compare across different hardware).

Log issues against `moeshawky/rockandegg-v3-code`. Do not respond to a regression by
reopening training or recalibration — this release is frozen; new work requires a new
versioned milestone under the conditions in `V3_RELEASE_FREEZE.md` §7.
