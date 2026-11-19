"""Generate RELEASE_MANIFEST.json with SHA-256 for every file in the artifact."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REL = Path("/scratch/rockandegg-v3-code/release/RockAndEgg-313M-v3-fullbackbone-bf16-preview")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    files = {}
    for p in sorted(REL.iterdir()):
        if p.is_file() and p.name != "RELEASE_MANIFEST.json":
            files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    manifest = {
        "model_id": "moeshawky/RockAndEgg-313M-v3",
        "release_name": "fullbackbone-bf16-preview",
        "release_state": "RESEARCH/PREVIEW (NOT production qualified)",
        "selected_source_checkpoint_sha256": "1b7fee92a9f8825ac001100232ad87e740bd7e0808ee49abc46386cc89c53f92",
        "release_model_sha256": files["model.safetensors"]["sha256"],
        "param_count": 313161733,
        "architecture": "10x Qwen3-decoder backbone + native SystemOne head; no lm_head, no generate",
        "precision": "bfloat16",
        "dtype_conversion": {
            "from": "float32 (master weights)",
            "to": "bfloat16",
            "report": "bf16_conversion_report.json",
            "fp32_to_bf16_argmax_flips_of_50": 0,
        },
        "training": {
            "objective": "soft-label CE (+ squared-EMD for score in other configs; EMD weight 0 here)",
            "full_backbone": True,
            "optimizer": "AdamW lr_backbone 2.5e-5 / lr_head 1e-4, wd 0.01, 3% warmup + cosine",
            "epochs": 8, "micro_batch": 64, "precision": "bf16 autocast / fp32 master",
            "dataset": "LocalLLaMA/typed-decisions", "config": "all",
            "dataset_revision": "e6aa632c4db73b59c7713bb21066ad1a8e1bfce6",
            "selected_best_step": 320, "dev_best_accuracy": 0.7800,
            "provenance_doc": "TRAINING_PROVENANCE.md",
        },
        "evaluation": {
            "clean_panel_accuracy_seed0": 0.7335,
            "ensemble_accuracy_research_context_only": 0.7545,
            "integrity_report": "../artifacts/release_integrity.json",
            "performance_report": "../artifacts/release_performance.json",
            "evaluation_doc": "EVALUATION_REPORT.md",
        },
        "calibration": {
            "production_ece_gate": "NOT MET",
            "ece_bound": 0.05,
            "raw_ece_reserved": 0.1599,
            "raw_ece_ssf1_sealed": 0.2755,
            "c2_status": "rejected (sealed NLL 1.4563 vs raw 1.0510)",
            "limitations_doc": "CALIBRATION_LIMITATIONS.md",
            "temperature": 1.0,
        },
        "dependencies": ["torch>=2.11", "transformers>=5.16,<6", "huggingface-hub>=1.5,<2.0", "safetensors>=0.4"],
        "license": "Apache-2.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (REL / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"n_files": len(files), "model_sha256": manifest["release_model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
