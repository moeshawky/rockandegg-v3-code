"""Export the RockAndEgg-313M-v3 SystemOne model as an HF-format checkpoint.

The full v3 model is NOT headless: it is a headless (no-LM-head) Qwen3
backbone PLUS a native SystemOne decision head. Only the causal LM head and
the generation path are absent; the decision head (qtype embedding, norms,
optional cross-attention/mixer, 256-unit scorer, temperature/prior params) is
exported in model.safetensors and declared in config.json under
``systemone_head``.

The exported directory is consumed by `sia model-headroom --model-path`, which
reads config.json for architecture detection and samples safetensors weight
matrices for R-03.

Run: PYTHONPATH=src .venv/bin/python export_v3_checkpoint.py

Output: export/rockandegg-313m-v3/{config.json,model.safetensors,manifest.json}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from rockandegg_v3 import MODEL_PATH, HeadConfig, RockAndEggV3SystemOne

EXPORT_DIR = Path(__file__).resolve().parent / "export" / "rockandegg-313m-v3"
SEED = 20261116


def main() -> None:
    """Export a deterministic FP32 checkpoint of the full v3 system."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-checkpoint",
        type=str,
        default="",
        help="Optional trained head checkpoint (torch.save payload with a 'head' key).",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    cfg = HeadConfig(dtype_name="fp32")
    model = RockAndEggV3SystemOne(cfg, MODEL_PATH).eval()
    if args.head_checkpoint:
        payload = torch.load(
            args.head_checkpoint, map_location="cpu", weights_only=True
        )
        head = payload["head"]
        state = model.state_dict()
        unknown = [key for key in head if key not in state]
        if unknown:
            msg = f"unknown head keys in checkpoint: {unknown}"
            raise ValueError(msg)
        state.update(head)
        model.load_state_dict(state, strict=True)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_config = Path(MODEL_PATH) / "config.json"
    exported_config = json.loads(source_config.read_text())
    # The source config declares bfloat16 while its on-disk weights are F32.
    # This export writes F32 weights (cfg dtype_name="fp32"), so correct the
    # authoritative dtype field to match the artifact; a naive from_pretrained
    # would otherwise cast to an unsupported dtype on T4. No consumer here
    # relies on the inherited string (all loaders pass dtype explicitly).
    exported_config["dtype"] = "float32"
    # The stock config describes only the Qwen3 backbone. Declare the decision
    # head explicitly so the artifact cannot be mistaken for the bare v2 model.
    exported_config["systemone_head"] = {
        "type": "native_typed_decision_head",
        "lm_head": False,
        "generation": False,
        "qtype_embedding": 3,
        "scorer": "1024->256->1",
        "use_cross_attn": cfg.use_cross_attn,
        "use_option_mixer": cfg.use_option_mixer,
        "use_cosine_prior": cfg.use_cosine_prior,
        "max_state_len": 256,
        "trained": bool(args.head_checkpoint),
        "training_dataset": (
            "SargeDev/jev-distill-corpus-v3" if args.head_checkpoint else None
        ),
    }
    (EXPORT_DIR / "config.json").write_text(json.dumps(exported_config, indent=2))

    weights = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    weights_path = EXPORT_DIR / "model.safetensors"
    save_file(weights, str(weights_path), metadata={"format": "pt"})

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest = {
        "model_name": "RockAndEgg-313M-v3-SystemOne",
        "source_backbone": MODEL_PATH,
        "head_config": {
            "dtype": "fp32",
            "freeze_backbone": cfg.freeze_backbone,
            "use_cross_attn": cfg.use_cross_attn,
            "use_option_mixer": cfg.use_option_mixer,
            "use_cosine_prior": cfg.use_cosine_prior,
        },
        "seed": SEED,
        "tensor_count": len(weights),
        "sha256": digest,
        "head_checkpoint": args.head_checkpoint or None,
        "note": (
            "Frozen Qwen3 backbone plus a SystemOne head trained on "
            "SargeDev/jev-distill-corpus-v3."
            if args.head_checkpoint
            else "Frozen Qwen3 backbone plus a randomly initialized, untrained "
            "SystemOne head. Not a trained decision model."
        ),
    }
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"export_dir": str(EXPORT_DIR), **manifest}, indent=2))


if __name__ == "__main__":
    main()
