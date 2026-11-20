# Inference Performance

This document reports measured inference performance of RockAndEgg-313M-v3 and
compares it with a native Laya implementation. RockAndEgg is a non-generative
typed-decision model: given one state and one or more questions, it returns a
`choice`, `score`, or `noul` decision per question with a probability
distribution and a confidence value.

The architecture and the serving improvements described here are **portable**;
they are not specific to any one GPU. The measurements below were taken on one
NVIDIA T4 under the environment in §2, and the exact latency figures are not
assumed to transfer unchanged to other hardware, CUDA versions, attention
implementations, batch shapes, or serving environments.

## 1. Model identities

| Item | Optimized RockAndEgg | Native Laya |
|---|---|---|
| Implementation | This model repository, A+D serving path | `Laya` typed-decisions, native `RLAgent` path |
| Checkpoint SHA-256 | `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28` | `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e` |
| Parameters | 313,161,733 | 421,293,827 |
| Encoder | 10-layer Qwen3 backbone (headless) + SystemOne head | ModernBERT-large derived encoder + decision head |
| Precision on this hardware | fp16 | fp16 (config declares bf16; runtime downshifts to fp16 autocast on T4) |

The comparison is **native implementation versus native implementation** under
each model's configured runtime path. Parameter-storage dtype was not matched
between the two implementations, so this is not a controlled precision-equal
comparison.

## 2. Environment

| Item | Value |
|---|---|
| GPU | One NVIDIA T4, 16 GB (Turing, SM 7.5, no native bf16) |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.18.0 |
| Precision | fp16 throughout |
| Attention | SDPA (no FlashAttention install) |
| Optimizations | no `torch.compile`, no CUDA graphs |
| Boundary | in-process inference (not HTTP-serving latency) |

## 3. Workload definitions

* **F1** — one question sharing a state. **F2** — two distinct questions sharing
  one state. (F4 and F8 were requested but see §8.)
* **K2 / K5** — number of candidate options (2 or 5), **not** fan-out.
* **S64 / S128 / S256** — state-length buckets defined by the RockAndEgg
  tokenizer. The Laya tokenizer produces different token counts; buckets are
  not identical across models.
* Text is drawn from the `LocalLLaMA/typed-decisions` test split (400 cases,
  2,000 typed decisions). Every arm receives identical raw text.

## 4. Methodology

* Fresh process per arm per round; arm order alternated across rounds.
* 20 warmup iterations and 100 timed iterations per configuration per round;
  3 rounds. Inputs drawn with a fixed seed from each configuration's case pool
  (never a single repeated case).
* CUDA synchronization before and after timed operations, CUDA events for GPU
  stages, and `time.perf_counter()` for end-to-end latency.
* Dossier-clustered bootstrap (10,000 resamples, fixed seed) for paired
  confidence intervals; multiple questions from one dossier are treated as one
  cluster, not independent samples.

## 5. Inference Performance results

### A+D serving optimization vs the original published implementation

| Measurement | Optimized (A+D) | Original published |
|---|---:|---:|
| Mean paired in-process latency difference | reference | +1.52 ms slower |
| 95% CI of difference | — | [1.42, 1.62] ms |
| Relative latency | 0.957× (−4.3%) | 1.000× |
| Configurations improved | 11 / 11 | — |
| Decision parity | max abs Δp = 0, zero choice flips | — |

The A+D change lowers in-process latency by approximately 4.3% with exact
decision parity. (An earlier, separate HTTP-path measurement found a larger
per-request gain; HTTP latency is not compared here.)

### Optimized RockAndEgg vs native Laya

| Measurement | Optimized RockAndEgg | Native Laya |
|---|---:|---:|
| F1 p50 relative result | reference | 1.05–1.22× slower |
| F2 p50 relative result | reference | 1.26–2.00× slower |
| Stratified paired ratio | reference | ~1.33× slower |
| Mean paired latency difference | reference | +13.17 ms slower |
| 95% CI of difference | — | [12.82, 13.52] ms |
| Configurations where Laya is slower | 11 / 11 | — |
| Peak reserved VRAM (heaviest config) | ~0.85 GB | ~2.68 GB |

RockAndEgg is faster in every populated configuration. The difference is
modest at F1 and substantially larger at F2, where two questions share one
state. RockAndEgg encodes the state once and reuses its KV representation; Laya
re-encodes the state for each question row.

The peak-VRAM difference is real but must not be attributed solely to KV
reuse: the models differ in parameter count (313M vs 421M) and in
parameter-storage characteristics.

## 6. Reproduce the performance check

The published artifact can be loaded and served directly:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "moeshawky/RockAndEgg-313M-v3"
model = AutoModel.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.float16
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
assert sum(p.numel() for p in model.parameters()) == 313_161_733
```

The HTTP server in this repository (`serving/systemone_server.py`) exposes
`GET /healthz`, `GET /ready`, and `POST /v1/systemone`. Run the CPU smoke tests
with `python -m pytest serving/test_systemone_server_cpu.py -q`.

## 7. Relationship to earlier benchmarks

A previously reported ~1.48× T4 advantage is **not reproduced as a single
settled number** here. The current stratified ratio is ~1.33×, with a
configuration-dependent range of roughly 1.05–2.00× measured under the
conditions in §2. A separate matched workload on an RTX PRO 6000 favored Laya
on p50 latency. These results differ by hardware and workload and should not be
combined or cherry-picked.

## 8. Limitations

1. Measured on one NVIDIA T4 under the environment in §2; numbers do not
   transfer unchanged to other hardware or serving environments.
2. In-process comparison, not HTTP-serving latency.
3. Of 48 requested configurations, 11 were populated. F4, F8, and Xlong were
   not available in the source corpus and were not fabricated.
4. F1 and F2 case pools differ in workflow composition and state-length
   distribution; their relative scaling is descriptive, not a clean causal
   isolation of fan-out.
5. The ~1.33× figure does not establish a universal advantage on every GPU or
   workload.
6. The memory difference reflects model size and parameter-storage
   characteristics as well as runtime architecture.
7. No new accuracy or calibration evaluation was performed; this is a latency
   and memory comparison only.
8. Stage-level timing breakdowns are not reported: host-to-device time for the
   RockAndEgg path was not measurable, some Laya stage times are derived, and
   some execution counters are configured rather than instrumented.
