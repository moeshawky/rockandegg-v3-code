"""Export the FP16 variant of the RockAndEgg-313M-v3 release artifact.

Reads the shipped FP32 ``model.safetensors``, casts every tensor to FP16, and
writes a self-contained ``fp16/`` model directory (weights + config + custom
code + tokenizer) that loads with ``subfolder="fp16"``.

The export is *measurement-first*: it records what FP16 actually does to these
weights — how many values round, how many underflow to zero, how many become
subnormal — because "cast it to fp16" is not a claim that fp16 is harmless.

Run: ``PYTHONPATH=src .venv/bin/python export_fp16_variant.py``
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "hf_repo" / "RockAndEgg-313M-v3"
SRC = REPO / "model.safetensors"
DST_DIR = REPO / "fp16"
DST = DST_DIR / "model.safetensors"

# Files the fp16 subdirectory needs to be a self-contained model directory.
COPY_FILES = [
    "configuration_rockandegg_v3.py",
    "modeling_rockandegg_v3.py",
    "systemone_inference.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14  # 6.1035e-05
FP16_MIN_SUBNORMAL = 2.0**-24  # 5.9605e-08


def sha256(path: Path) -> str:
    """Stream a file through SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Cast the release weights to FP16 and report the measured effects."""
    DST_DIR.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    stats = {
        "tensors": 0,
        "values": 0,
        "changed": 0,
        "unchanged": 0,
        "underflowed_to_zero": 0,
        "became_subnormal": 0,
        "overflowed_to_inf": 0,
        "max_abs_original": 0.0,
        "max_abs_fp16": 0.0,
        "max_abs_diff": 0.0,
        "sum_abs_original": 0.0,
        "sum_abs_fp16": 0.0,
    }

    with safe_open(SRC, framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle:
            src = handle.get_tensor(key)
            dst = src.to(torch.float16)
            tensors[key] = dst

            src_f = src.float()
            dst_f = dst.float()
            diff = (src_f - dst_f).abs()

            stats["tensors"] += 1
            stats["values"] += src.numel()
            stats["changed"] += int((diff > 0).sum().item())
            stats["unchanged"] += int((diff == 0).sum().item())
            stats["underflowed_to_zero"] += int(
                ((src_f != 0) & (dst_f == 0)).sum().item()
            )
            stats["became_subnormal"] += int(
                ((dst_f != 0) & (dst_f.abs() < FP16_MIN_NORMAL)).sum().item()
            )
            stats["overflowed_to_inf"] += int((src_f.abs() > FP16_MAX).sum().item())
            stats["max_abs_original"] = max(
                stats["max_abs_original"], float(src_f.abs().max())
            )
            stats["max_abs_fp16"] = max(stats["max_abs_fp16"], float(dst_f.abs().max()))
            stats["max_abs_diff"] = max(stats["max_abs_diff"], float(diff.max()))
            stats["sum_abs_original"] += float(src_f.abs().sum())
            stats["sum_abs_fp16"] += float(dst_f.abs().sum())

    save_file(tensors, str(DST), metadata=metadata)
    stats["dtype"] = "float16"
    stats["all_f16"] = all(t.dtype == torch.float16 for t in tensors.values())

    # ---- config: same model, dtype declared as float16 ----
    cfg = json.loads((REPO / "config.json").read_text())
    cfg["dtype"] = "float16"
    (DST_DIR / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for name in COPY_FILES:
        src = REPO / name
        if src.exists():
            shutil.copy2(src, DST_DIR / name)

    report = {
        "artifact": str(DST),
        "sha256": sha256(DST),
        "bytes": DST.stat().st_size,
        "source_sha256": sha256(SRC),
        "source_bytes": SRC.stat().st_size,
        "measured": stats,
    }
    (DST_DIR / "fp16_export_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print("\nfp16/ contents:")
    for item in sorted(DST_DIR.iterdir()):
        print(f"  {item.name:38s} {item.stat().st_size:>14,} B")


if __name__ == "__main__":
    main()
