# Reproduction — RockAndEgg-313M-v3

Nothing in this repository downloads large data automatically, and no training
run is required to load or serve the released model. Reproduction has two
independent parts.

## 1. Serving / inference reproduction (no external data)

```bash
python -m pip install -r serving/requirements-server.txt
hf download moeshawky/RockAndEgg-313M-v3 --local-dir ./rae-v3
cd rae-v3 && python systemone_server.py
```

Then run the CPU contract tests from this source repository:

```bash
python -m pytest serving/test_systemone_server_cpu.py -q
```

These tests load no weights: they replace the lifespan with a deterministic
fake adapter and exercise CPU readiness, GPU readiness, bounded inference,
health responsiveness, option-truncation reporting, input validation, error
disclosure, binding configuration and the exit-code taxonomy.

## 2. Training reproduction (external inputs required)

The training corpus and the frozen v2 backbone are **not bundled**. Supply
both explicitly; the script fails with a clear error if the dataset directory
is missing. It never silently substitutes a different backbone or corpus.

```bash
python -m pip install -e '.[dev]'

python scripts/train_jev_distill.py \
  --backbone /path/to/RockAndEgg-313M-v2-f32 \
  --data-dir /path/to/jev_distill_v3 \
  --out-dir ./training \
  --cases 80000 --epochs 1 --seed 20261116
```

`--backbone` accepts a local directory or a supported Hub identifier
(default `moeshawky/RockAndEgg-313M-v2`). `--data-dir` must contain the
`train.jsonl`, `calibration.jsonl` and `test_set_30k.jsonl` splits of
`SargeDev/jev-distill-corpus-v3` (Apache-2.0). `RAE_BACKBONE` is the
environment-variable equivalent used by the verification scripts.

**Preserved training configuration** (do not change when reproducing):

| Setting | Value |
| --- | --- |
| Backbone | Frozen Qwen3 10-layer headless v2, fp16 on T4 / fp32 on CPU |
| Trainable parameters | SystemOne head only |
| Objective | Soft-label cross-entropy + squared Earth-Mover's Distance (score questions) |
| Sampling | Temperature 1.0; per-type temperature scaling fitted on `calibration` |
| Regularisation | Option dropout 0.3; no RLCD, no REINFORCE |
| State / option limits | 256 / 64 tokens |
| Distributed strategy | Two-device manual data parallelism; head gradients averaged; primary optimizer steps |
| Seed | 20261116 |

## 3. Qualification reproduction

```bash
PYTHONPATH=src:scripts python scripts/run_systemone_gates.py --help
PYTHONPATH=src:scripts python scripts/gpu_verify.py --help   # CUDA device required
PYTHONPATH=src:scripts python scripts/verify_export_metadata.py
```

Gate runners require the published checkpoint, the held-out prediction files
and (for transfer probes) the external corpus. Those artifacts are absent from
this source-only candidate. Without them the numeric gates are **NOT RUN**,
not passed. See `QUALIFICATION.md`.

Export procedure (requires the trained head checkpoint and the v2 backbone):
`scripts/export_v3_checkpoint.py` assembles the headless backbone plus head;
`scripts/export_fp16_variant.py` derives the fp16 snapshot. The fp16 artifact
must be qualified separately — a weight-cast report is not a parity gate.
