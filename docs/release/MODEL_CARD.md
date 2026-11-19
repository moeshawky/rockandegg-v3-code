> **Update (2026-11-19):** this card describes the preview artifact (BF16 seed-0). The
> full-backbone **FP16** seed-0 research/preview release is now published at tag
> `v3-fullbackbone-fp16-1` on `moeshawky/RockAndEgg-313M-v3` (commit `48e42035…`).
> See the model repository's root `README.md` for the canonical card and loading snippet.

# Model Card: RockAndEgg-313M-v3 — Full-Backbone FP16 (RESEARCH / PREVIEW)

**Model ID / Version:** `moeshawky/RockAndEgg-313M-v3`, tag `v3-fullbackbone-fp16-1`
**Architecture:** 10-layer Qwen3-derived decoder backbone + native SystemOne typed-decision head; no LM head, no `generate()` path
**Source (authoritative identity):** `V3_RELEASE_FREEZE.md:1-30`; source code `moeshawky/rockandegg-v3-code` commit `aaa274566ab749cd7b993dcfb3c728dedc8c7fdf`
**Last Verified:** 2026-11-19 (all claims below re-measured this session; see §7)
**Verification Command:** fresh-process load of `model.safetensors`; `evidence/server_smoke.json` (3/3 decision types, 10/10 checks)

> **RESEARCH / PREVIEW — NOT PRODUCTION QUALIFIED.** Probabilities are raw softmax at
> **T=1.0** and are **uncalibrated**. The production ECE ≤ 0.05 gate is **NOT met**
> (0.1323 on the 400-dossier FP16 test panel; BF16-era panels 0.1599 RESERVED / 0.2755 SSF-1
> sealed). `confidence` is not a calibrated probability guarantee. Source:
> `evidence/CALIBRATION_LIMITATIONS.md`, `evidence/quality_metrics.json`.

## 1. Intended Use

**Designed for:** research on compact typed-decision models and shared-state
(encode-once) inference. One state + typed questions (`choice`, `score`, `noul`) → one
probability distribution + `confidence` per question. Verified by `evidence/server_smoke.json`
(3/3 decision types, 10/10 checks).
**Out of scope:** production decisions requiring calibrated confidence; any risk-threshold
gating; token generation (no generation path exists — `systemone_inference.py` holds no `generate`).
**Primary users:** researchers and integrators evaluating shared-state inference on
Turing-class GPUs (T4) via fp16, or CPU via fp32.

## 2. Identity & Artifacts (exact values, not adjectives)

| Item | Value | Source |
|---|---|---|
| Parameters | 313,161,733 (312,892,928 backbone + 268,805 head) | `evidence/quality_metrics.json`, `evidence/server_smoke.json` |
| Tensors | 122; all shapes match source | `evidence/server_smoke.json` |
| Precision | float16 (all tensors/params) | `config.json` dtype `float16`, 122/122 tensors `torch.float16` |
| Released model SHA-256 | `17a54706be165ce246e963488ca7d4ebfd39394926c71325fa8e664c64c83e28` | `RELEASE_MANIFEST.json`, `SHA256SUMS` |
| Released model bytes | 626,337,154 | `RELEASE_MANIFEST.json` |
| `config.json` SHA-256 | `06d9d4bad660e7c40163ae16b7f870bbdca5ce77e7956b89c9e9420017111c03` | `SHA256SUMS` |
| Precision provenance | cast from published BF16 artifact `c8b3aafb…` (direct `to(float16)`, no FP32 intermediate) | `evidence/precision_cast_report.json`, `TRAINING_PROVENANCE.md` |
| HF tag | `v3-fullbackbone-fp16-1` → commit `48e42035a66299b803d1a545963913f90dbc3a2e` | HF remote, verified this session |
| Historical head-only tag | `v3-head-only-main` → `87c754951f9d4bd04c90e9ec78fc54295075e86c` | HF remote, preserved |
| Canonical main tag | `v3-fullbackbone-main-1` → `c8d6f8c1e3cc32456ddb80640a7339789ca7dff8` | HF remote, preserved |
| Preview branch | `preview/fullbackbone-bf16-seed0` → `0b60f73eda4293b4891f27c3106a977cef536a1f` | HF remote, preserved |
| License | Apache-2.0; Qwen3 attribution in `ATTRIBUTION.md` | `LICENSE`, `ATTRIBUTION.md` |

## 3. Performance (Measured — Sekel ratios, not adjectives)

Conditions: 1× NVIDIA Tesla T4 (driver 580.178.04), `cuda:0`, fp16,
torch 2.11.0+cu128 / transformers 5.16.1, serial single-request workload,
3 rounds per cell, 100 timed requests per cell.
Source: `evidence/throughput_results.json`, `evidence/latency_results.json`.

| Arm | Backbone calls (naive) | mean p50 ms | req/s | decisions/s |
|---|---:|---:|---:|---:|
| `rae_shared` (this release) | 2 | 43.5 | 22.49 | 67.46 |
| `rae_independent` (naive) | 2F | 111.9 | 8.71 | 26.13 |
| `laya_native` | 1 | 65.1 | 15.20 | 45.60 |

Encode-once verified: exactly **2** backbone forwards at every fan-out for
`rae_shared` (one state prefill + one option-row pass); `rae_independent` needs `2F`;
`laya_native` re-encodes the state once per question row (1 forward containing its
batched comparison workload). Hook-verified in `evidence/shared_state_instrumentation.json`.
These numbers are hardware-specific (single-request, one GPU, no concurrency) and must
not be generalized to other devices or to serving throughput. No controlled comparison
against Laya was run; no speed superiority is claimed.

## 4. Training Data (evidence-first sourcing)

| Source | Description | Provenance | Size |
|---|---|---|---|
| `LocalLLaMA/typed-decisions`, config `all` | Typed decisions (choice/score/noul) over dossiers | revision `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6` | 400 test dossiers / 2,000 decisions |

Canonical scorer: decision = `argmax` of model probabilities; target = dataset `gold["label"]`
for **every** type (`choice`, `score`, `noul`). Panel identity sha256
`441ba1b97b99f2e409e82e7186a1458751696f2e3fb7de5a573fe65185b00852`; 0 train overlap.

## 5. Evaluation (measured; reproduce commands)

| Benchmark / panel | Split | Metric | Result | Source |
|---|---|---|---|---|
| Clean panel (this release, FP16) | 400 dossiers / 2,000 dec | accuracy [95% CI] | **73.25% [71.05, 75.35]** | `evidence/quality_metrics.json` |
| Clean panel (head-only FP32) | 400 dossiers / 2,000 dec | accuracy [95% CI] | 34.25% [32.25, 36.30] | `evidence/quality_metrics.json` |
| Clean panel (head-only FP16) | 400 dossiers / 2,000 dec | accuracy [95% CI] | 33.95% [31.90, 36.05] | `evidence/quality_metrics.json` |
| Clean panel (Laya) | 400 dossiers / 2,000 dec | accuracy [95% CI] | **76.65% [74.60, 78.65]** | `evidence/quality_metrics.json` |
| Paired A − B_fp32 | — | Δ [95% CI] | **+39.00 pp [+35.95, +42.20]** | `evidence/arm_accuracy_ci.json` |
| Paired A − Laya | — | Δ [95% CI] | **−3.40 pp [−5.50, −1.45]** | `evidence/arm_accuracy_ci.json` |
| NLL / Brier / ECE(M=15) | this release | metrics | 0.6945 / 0.0777 / 0.1323 | `evidence/quality_metrics.json` |

Per decision type (arm A): `choice` 70.50%, `noul` 82.17%, `score` 68.63%.
Reproduce: recompute from `evidence/predictions.jsonl` + the pinned dataset
(revision above). No accuracy evaluation is re-run here (panel is exposed and frozen).
**Non-determinism:** all arm figures are single-run point estimates on one 2,000-decision
panel. The dossier-clustered bootstrap quantifies evaluation-sample uncertainty; it does
not measure run-to-run inference variability.

## 6. Precision qualification (N-gates, pre-registered)

Three comparisons (`evidence/precision_parity.json`, 400 dossiers / 2,000 decisions):

| Comparison | Agreement | Argmax flips | Mean abs prob diff | Max abs prob diff |
|---|---|---|---|---|
| A: FP16 artifact vs BF16 artifact, identical compute | **1.0000** | **0** | **0.0** | **0.0** |
| B: FP16 artifact vs FP32-compute reference | 0.9975 | 5 / 2,000 | 6.54e-4 | **5.038e-3 (N3 FAIL)** |
| C: FP16 artifact vs native BF16 compute on T4 | 0.9840 | 32 / 2,000 | 4.78e-3 | 4.32e-2 |

**The pre-registered N3 threshold (≤ 5.0e-3) is NOT met and is reported as a failure.**
Acceptance criteria were fixed before the run. Measured: 5.038e-3, a 0.76% overshoot
on a single decision out of 2,000. Not relaxed, not reclassified, not rounded down.

**What the N3 failure does and does not mean.** Three questions are separate:
(1) *Did the pre-registered threshold pass?* **No** — 5.038e-3 > 5.0e-3, reported as a failure.
(2) *Is a material accuracy regression demonstrated?* **No** — arm B's accuracy differs by
+0.05 pp (72.85% vs 72.90%) and 5 of 2,000 argmax calls flip; the maximum-probability
overshoot is a single-decision numerical bound, not evidence of a large accuracy loss.
(3) *Does this block a research/preview release?* **No** — it blocks the stricter
qualification tier; the release is published as research/preview.
The **FP32-compute reference** is FP32 compute over the BF16-valued weights, not the
unrecovered FP32 master. Arm C shows native BF16 on Turing is strictly worse than FP16
on every axis (lower agreement, more flips, lower accuracy, ~11× slower). FP16 is the
correct deployment precision for Turing.

## 7. Verification (living documentation)

Executed this session (2026-11-19):
```bash
# fresh-process load of the published artifact
HF_HUB_DISABLE_PROGRESS_BARS=1 USE_TF=0 PYTHONPATH=. \
  /kaggle/working/rockandegg-v3-code/.venv/bin/python -c "
from transformers import AutoModel, AutoTokenizer
from systemone_inference import SystemOneInference
m = AutoModel.from_pretrained('.', trust_remote_code=True, dtype='float16').cuda().eval()
assert not hasattr(m, 'lm_head') and not hasattr(m, 'generate')
t = AutoTokenizer.from_pretrained('.', trust_remote_code=True)
a = SystemOneInference(m, t)
r = a(__import__('systemone_inference').SystemOneRequest(
    state='s', questions=[__import__('systemone_inference').Question('choice','i',['a','b'])])
print(r.decisions[0].chosen_key, sum(r.decisions[0].probabilities))
"
# Expected: exit 0; no lm_head; no generate; normalized probabilities.
# Quality recompute (no inference):
#   .venv/bin/python work/lead/canonical_metrics.py   # recomputes quality_metrics.json from predictions.jsonl
```
**Expected output:** fresh-process load exit 0 with 122 tensors / all float16 / no LM head /
no generation; the recompute reproduces `evidence/quality_metrics.json` exactly.
**Re-verification frequency:** before any downstream integration and on any model/hardware change.

## 8. Source Citations

- Model weights/config: `model.safetensors` (sha256 `17a54706…`), `config.json` (sha256 `06d9d4ba…`)
- Canonical inference API: `systemone_inference.py:91` (`SystemOneInference`), `:167` (`__call__`)
- Training code: `scripts/train_full_backbone.py` (public source repo)
- Evaluation code: `work/lead/canonical_metrics.py` (recompute), `work/B_quality/bench_quality.py` (harness)
- Measured results: `evidence/quality_metrics.json`, `evidence/arm_accuracy_ci.json`,
  `evidence/reconciliation_accuracy.json`, `evidence/precision_parity.json`,
  `evidence/throughput_results.json`
- Provenance/limitations: `evidence/TRAINING_PROVENANCE.md`, `evidence/CALIBRATION_LIMITATIONS.md`
- Release manifest: `RELEASE_MANIFEST.json`, `SHA256SUMS`

> **Historical note:** the BF16 preview artifact (`c8b3aafb…`) remains published on HF
> branch `preview/fullbackbone-bf16-seed0` @ `0b60f73…` and is not modified. The FP16
> artifact is a precision cast of it; it is labelled BF16-derived, never as an FP32 export.
