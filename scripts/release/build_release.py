"""Build the RESEARCH/PREVIEW BF16 release artifact from selected seed 0.

Selected checkpoint: artifacts/runs/fullbb_bf16_e2style/best/model
(model.safetensors sha256 1b7fee92...), the original DEV-selected full-backbone
checkpoint. This script only READS that checkpoint; it never modifies it.

The source master weights are FP32 (trained under bf16 autocast). The release
casts them to BF16 (required release dtype) and MEASURES the numerical effect —
per-tensor and decision-level parity on a fixed, label-blind probe — rather than
asserting "cast is harmless".

Run:
  USE_TF=0 PYTHONPATH=src:/scratch/RockAndEgg-313M-v3 \
    /opt/pytorch/bin/python release/build_release.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path("/scratch/rockandegg-v3-code")
SRC = ROOT / "artifacts/runs/fullbb_bf16_e2style/best/model"
TOK_SRC = Path("/scratch/RockAndEgg-313M-v3")
OUT = ROOT / "release/RockAndEgg-313M-v3-fullbackbone-bf16-preview"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. cast FP32 master weights to BF16, measuring every tensor ----
    tensors: dict[str, torch.Tensor] = {}
    stats = {
        "source": str(SRC / "model.safetensors"),
        "source_sha256": sha256(SRC / "model.safetensors"),
        "source_dtype": "float32",
        "target_dtype": "bfloat16",
        "tensors": 0,
        "values": 0,
        "changed_values": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "per_tensor": [],
    }
    total_diff = 0.0
    with safe_open(str(SRC / "model.safetensors"), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            src = handle.get_tensor(key)
            dst = src.to(torch.bfloat16)
            tensors[key] = dst
            sf = src.float()
            df = dst.float()
            diff = (sf - df).abs()
            mx = float(diff.max())
            total_diff += float(diff.sum())
            stats["tensors"] += 1
            stats["values"] += src.numel()
            stats["changed_values"] += int((diff > 0).sum())
            stats["max_abs_diff"] = max(stats["max_abs_diff"], mx)
            stats["per_tensor"].append({"name": key, "shape": list(src.shape),
                                         "max_abs_diff": mx})
    stats["mean_abs_diff"] = total_diff / stats["values"]
    save_file(tensors, str(OUT / "model.safetensors"), metadata=metadata)
    stats["output_sha256"] = sha256(OUT / "model.safetensors")
    stats["output_bytes"] = (OUT / "model.safetensors").stat().st_size
    stats["all_bf16"] = all(t.dtype == torch.bfloat16 for t in tensors.values())
    (OUT / "bf16_conversion_report.json").write_text(json.dumps(stats, indent=2))

    # ---- 2. config: declare bfloat16; correct stale provenance metadata ----
    cfg = json.loads((SRC / "config.json").read_text())
    cfg["dtype"] = "bfloat16"
    head = cfg["systemone_head"]
    head["training_dataset"] = "LocalLLaMA/typed-decisions (config all, full-backbone fine-tune)"
    head["training_dataset_revision"] = "e6aa632c4db73b59c7713bb21066ad1a8e1bfce6"
    head["training_split"] = "train (1020 dossiers, split_seed=0; DEV/RESERVED held out)"
    head["backbone_frozen"] = True  # accurate for the SERVED artifact (inference only)
    head["full_backbone_trained"] = True
    head["release_status"] = "research-preview (production ECE gate NOT met)"
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # ---- 3. custom code, inference, tokenizer, license ----
    copy_from_src = ["configuration_rockandegg_v3.py", "modeling_rockandegg_v3.py"]
    for name in copy_from_src:
        shutil.copy2(SRC / name, OUT / name)
    shutil.copy2(Path("/scratch/RockAndEgg-313M-v3") / "systemone_inference.py",
                 OUT / "systemone_inference.py")
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = TOK_SRC / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    shutil.copy2(TOK_SRC / "LICENSE", OUT / "LICENSE")

    print(json.dumps({
        "output": str(OUT),
        "model_sha256": stats["output_sha256"],
        "source_sha256": stats["source_sha256"],
        "all_bf16": stats["all_bf16"],
        "max_abs_diff": stats["max_abs_diff"],
        "mean_abs_diff": stats["mean_abs_diff"],
        "changed_values": stats["changed_values"],
        "values": stats["values"],
    }, indent=2))


if __name__ == "__main__":
    main()
