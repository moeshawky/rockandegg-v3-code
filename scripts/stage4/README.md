# Stage 4 reproduction (RockAndEgg-313M-v3)

This directory contains the Stage 4 training and evaluation code that produced the
`R_final3` candidate checkpoint, replayed here for independent reproduction.

- `stage4_train.py` — trainer (scope `final3` unfreezes backbone layers 7/8/9 + head;
  `gradsafe()` source-rewrite of `forward_batch`).
- `stage4_eval.py` — evaluator on typed-decisions DEV, v1/v2 test + composition,
  and the frozen independent panel.
- `curriculum_v1.py` / `curriculum_v1_eval.py` — v1 renderer, split logic, and
  symbolic oracle (DAG solver).
- `curriculum_v2.py` — v2 scenario generator (three-valued oracle).
- `eval_panel.py` — frozen independent synthetic panel generator (810 scenarios).
- `corrected_data.py` — typed-decisions dataset loader and row evaluator.
- `corrected_metrics_v2.py` — metric helpers.
- `v1_graph.py` / `v2_env.py` — symbolic oracles.
- `MONKEYPATCHES.md` — runtime/training patches (incl. `gradsafe`).
- `STAGE4_FILE_HASHES.json` — archived file hashes for provenance.

## Inputs (not bundled)

- Model checkpoint: `moeshawky/RockAndEgg-313M-v3` (FP16 `main`, or the Stage 3
  backbone at tag `v3-fullbackbone-main-1`). Set `RAE_MODEL_DIR` to a local clone.
- Typed-decisions dataset: `LocalLLaMA/typed-decisions` (Apache-2.0), revision
  `e6aa632c4db73b59c7713bb21066ad1a8e1bfce6`; place the two parquet files under
  `$RAE_PROBE_DIR/real_data/`.
- Stage 3 main checkpoint: set `RAE_STAGE3_MAIN` to a local clone (used by the trainer).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `RAE_PROBE_DIR` | `./probe_data` | Where curriculum/panel JSON and outputs live. |
| `RAE_MODEL_DIR` | `./rae-v3` | Local RockAndEgg v3 checkpoint (with `systemone_inference.py`). |
| `RAE_STAGE3_MAIN` | `./stage3-main` | Stage 3 main checkpoint used for distillation/retention. |
| `RAE_OUTPUT_DIR` | `./rockandegg_stage4` | Where trained checkpoints are written. |

## Run

```bash
pip install -r requirements.txt
export RAE_MODEL_DIR=/path/to/rae-v3
export RAE_PROBE_DIR=/path/to/probe_data
python curriculum_v1.py          # writes probe_data/curriculum_v1.json
python curriculum_v2.py          # writes probe_data/curriculum_v2.json
python eval_panel.py             # writes probe_data/eval_panel.json
python stage4_eval.py /path/to/ckpt   # evaluates one checkpoint
```

## License

Apache-2.0. The typed-decisions dataset is Apache-2.0; Qwen3-0.6B-Base (backbone donor)
is Apache-2.0. See the repository `ATTRIBUTION.md`.