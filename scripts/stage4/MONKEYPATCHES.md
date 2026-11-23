# Runtime / training-time patches (Stage 4 reproduction)

These are the code paths that were **modified, monkeypatched, or exec-rewritten** during the
Stage 3/4 work. Anyone reproducing the checkpoints or debugging serving parity must apply them.
Provenance for the archived copies: `docs/release/archive/ARCHIVE_HASHES.json` and the Stage 4
source snapshot (`moeshawky/rockandegg-stage4-source`, private).

---

## 1. `gradsafe()` — source rewrite of `forward_batch` (`stage4_train.py`)

```python
def gradsafe(model):
    src = textwrap.dedent(inspect.getsource(model.forward_batch))
    src = src.replace("copy.deepcopy(cache)", "cache")
    ns = dict(model.__class__.__dict__)
    ns["torch"] = torch; ns["F"] = F; ns["copy"] = copy
    exec(src, ns)
    model.forward_batch = types.MethodType(ns["forward_batch"], model)
```

- **What it does:** removes the per-option `copy.deepcopy(cache)` inside `forward_batch`, so all
  option-suffix continuations reuse a single cache object.
- **Why:** the deepcopy is a memory/time hazard under gradient accumulation across many option
  rows; sharing the cache is numerically equivalent for a *read-only* cache but much cheaper.
- **Blast radius:** the shipped `serving/modeling_rockandegg_v3.py` (and every checkpoint export)
  still contains the original `copy.deepcopy(cache)`. Therefore:
  - **Training** ran with the patched `forward_batch` (shared cache).
  - **Serving / evaluation** (`stage4_eval.py`, `SystemOneInference`) uses the **unpatched** code.
- **Measured result on `R_final3` weights (fp16, eager, T4, 2026-11-23):** patched vs unpatched
  `forward_batch` agree to **0.0 logit delta** across 12 cases (3 oracle statuses × 4 questions),
  all argmaxes identical. Both differ from `forward_case` by at most **0.015625 logit** (fp16
  accumulation from different row padding/order), with **zero decision flips**. The serving code is
  therefore safe as shipped; the gradsafe patch is a no-op numerically on these weights.

## 2. `unfreeze()` — parameter-scope selection (`stage4_train.py`)

```python
if scope == "final3":  # layers 7, 8, 9 + head
elif scope == "layer9":  # layer 9 + head
# head: qtype_emb, scorer, log_temp
```

Scope is a **training-time** decision only; it leaves no trace in the exported weights beyond
which tensors differ. Verified post-hoc by int16 view: final3 changed exactly
`backbone.layers.{7,8,9}` (11 tensors each) + head (6) = 39 tensors, 0 unauthorized.

## 3. Modified modeling file (head-only experiment)

`MODIFIED_headonly_modeling_rockandegg_v3.py` in the archived `v3/checkpoint_code/` is a **locally
edited** copy of the v3 modeling file that differs from the published artifact. It belongs to the
earlier head-only repair track, **not** to Stage 4 final3. Preserved verbatim in the source snapshot
for provenance; do not ship it with the Stage 4 checkpoint.

## 4. `_read_options` cache footgun (in the shipped code, not a patch)

`_read_options` calls `cache.batch_repeat_interleave(m)` **in place**; a `DynamicCache` is
single-use. Any probe or training loop that forks must call `encode_state` afresh (or clone).
This is a property of the shipped code and a known source of corrupted measurements.

## 5. Environment deltas (not code, but version-sensitive)

- Artifact metadata says `transformers 5.23.0`; the runtime that produced Stage 4 had **5.16.1**.
  Treated as an environment delta, not a defect.
- Remote-code cache id `ae31427eaa448a03` is not a git revision.
- `attn_implementation="eager"` and FP16 were used throughout. Do not switch to SDPA/Flash for
  parity comparisons.
- Reproduction runtime for this PR: torch 2.14.1+cu130, transformers 5.22.1, Python 3.13.13,
  Tesla T4 (sm_75). See `requirements.txt` in this directory.