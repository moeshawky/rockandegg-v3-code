# V3 Operational Handoff — RockAndEgg-313M-v3 Full-Backbone BF16 (RESEARCH / PREVIEW)

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
forwards regardless of fan-out — one `[1, 256]` state prefill plus one `[4, 21]`
option-row pass — at F1, F2, F4 and F8. Naive per-option re-encoding would need up to 9
calls at F8. The native `DynamicCache` is preserved; no custom attention mask or RoPE
surgery. No alternative server or abstraction layer should be introduced.

## 5. Requirements

- Python ≥ 3.10; torch ≥ 2.11; transformers ≥ 5.16,<6; huggingface-hub ≥ 1.5,<2.0;
  safetensors ≥ 0.4. Install: `python -m pip install -r requirements.txt` (from the
  artifact folder) or the source repo's `serving/requirements-server.txt` for the server.
- `transformers 5.x` requires `huggingface-hub < 2.0` (a newer hub breaks import).
- BF16 weights require an BF16-capable GPU (e.g. RTX PRO 6000 / Ampere+ / Blackwell).
  T4 (Turing) has no bf16: for T4 use an fp16/fp32 variant or CPU with `dtype="float32"`.

## 6. Load and run (executable example)

Exact identifiers (also pinned in `V3_RELEASE_FREEZE.md`):

```python
from transformers import AutoModel, AutoTokenizer
from systemone_inference import Question, SystemOneInference, SystemOneRequest

REPO = "moeshawky/RockAndEgg-313M-v3"
REV  = "0b60f73eda4293b4891f27c3106a977cef536a1f"   # preview revision
SUB  = "fullbackbone-bf16-preview"

model = AutoModel.from_pretrained(
    REPO, revision=REV, subfolder=SUB, trust_remote_code=True, dtype="bfloat16",
).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(REPO, revision=REV, subfolder=SUB)

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

Measured on **1× NVIDIA RTX PRO 6000 Blackwell Server Edition** (driver 595.112.07),
`cuda:0`, `bfloat16`, transformers 5.23.0 / torch 2.13.0+cu130, 65 label-blind
typed-decision inputs drawn from the exposed clean panel (used as input payloads only),
40 warmup+timed iterations per fan-out, wall-clock with `torch.cuda.synchronize()`:

| Fan-out | backbone calls (naive) | p50 ms | p95 ms | req/s | peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|
| F1 | 2 (2) | 14.35 | 14.67 | 69.7 | 654 |
| F2 | 2 (3) | 14.13 | 14.21 | 70.8 | 668 |
| F4 | 2 (5) | 14.31 | 14.41 | 69.9 | 694 |
| F8 | 2 (9) | 14.31 | 14.45 | 69.9 | 694 |

These numbers are specific to this hardware/runtime/build and must not be quoted for
other devices. They are not a controlled comparison against Laya or any other system.

## 8. Known limitations and status

- **RESEARCH / PREVIEW — NOT production qualified.** Probabilities are uncalibrated
  raw softmax at T=1.0. Production ECE bound 0.05 is NOT met (0.1599 RESERVED, 0.2755
  SSF-1 sealed). C2 calibration rejected (sealed NLL 1.0510 → 1.4563). `confidence` is
  **not** a calibrated guarantee — do not use it for risk thresholds or gating.
- No accuracy superiority over Laya is claimed. The three-model ensemble (+2.1 pts on a
  clean panel) and the failed 75/25 distillation are research context only, not deployed.
- The published card at the frozen revision uses generic branch/subfolder placeholders;
  the exact identifiers are in this handoff, the freeze document, and the source README.

## 9. Reporting failures or regressions

When reporting an issue, include:
1. Exact model revision (`0b60f73…`), subfolder, and `dtype` used.
2. Full input (state + questions) and the decision output (probabilities, chosen key, confidence).
3. Hardware/GPU, driver, torch/transformers/hub versions, OS, and device (CPU/CUDA).
4. Expected vs observed behavior and whether probabilities still sum to ~1.
5. For latency regressions: fan-out (F1/F2/F4/F8), warmup/iteration counts, and whether
   the measurement conditions match §7 (do not compare across different hardware).

Log issues against `moeshawky/rockandegg-v3-code`. Do not respond to a regression by
reopening training or recalibration — this release is frozen; new work requires a new
versioned milestone under the conditions in `V3_RELEASE_FREEZE.md` §7.
