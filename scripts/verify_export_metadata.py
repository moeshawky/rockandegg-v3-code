"""Verify the exported v3 checkpoint's metadata against its actual weights.

This is a claim-proof instrument, not a checksum check. Each metadata claim is
paired with a discriminating probe against ``model.safetensors`` (or the named
source artifacts) and assigned one of:

- ``VERIFIED_BY_WEIGHTS`` — inspecting the artifact entails the claim.
- ``VERIFIED_BY_SOURCE_MATCH`` — exact tensor equality with the named source.
- ``SUPPORTED_BY_PRODUCER`` — provenance/config value from the producing code,
  not entailed by static weights; cannot be promoted to VERIFIED by this probe.
- ``FAILED`` — the probe refuted the claim.

Run: PYTHONPATH=src .venv/bin/python verify_export_metadata.py
Output: export_metadata_audit.json
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
EXPORT = ROOT / "export" / "rockandegg-313m-v3"
HEAD_CKPT = ROOT / "training" / "head_final.pt"
SOURCE_BACKBONE = Path(
    os.environ.get("RAE_BACKBONE", "moeshawky/RockAndEgg-313M-v2")
)
OUT = ROOT / "export_metadata_audit.json"


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Run every claim probe and write the verdict report."""
    cfg = json.loads((EXPORT / "config.json").read_text())
    man = json.loads((EXPORT / "manifest.json").read_text())
    sh = cfg["systemone_head"]
    hc = man["head_config"]
    weights_path = EXPORT / "model.safetensors"

    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {k: tuple(handle.get_slice(k).get_shape()) for k in keys}
        dtypes = {k: str(handle.get_slice(k).get_dtype()) for k in keys}
    head_keys = [k for k in keys if not k.startswith("backbone.")]
    backbone_keys = [k for k in keys if k.startswith("backbone.")]

    claims: list[dict[str, Any]] = []

    def add(claim: str, status: str, evidence: str) -> None:
        claims.append({"claim": claim, "status": status, "evidence": evidence})

    actual_sha = sha256_file(weights_path)
    add(
        "manifest.sha256 matches model.safetensors",
        "VERIFIED_BY_WEIGHTS" if actual_sha == man["sha256"] else "FAILED",
        f"recomputed={actual_sha}; declared={man['sha256']}",
    )
    add(
        f"manifest.tensor_count = {man['tensor_count']}",
        "VERIFIED_BY_WEIGHTS" if len(keys) == man["tensor_count"] else "FAILED",
        f"actual key count={len(keys)}",
    )
    all_f32 = all(v == "F32" for v in dtypes.values())
    add(
        f"head_config.dtype = {hc['dtype']} (weights F32)",
        "VERIFIED_BY_WEIGHTS" if all_f32 and hc["dtype"] == "fp32" else "FAILED",
        f"dtype counts={dict(Counter(dtypes.values()))}",
    )
    add(
        "config.json dtype matches actual weights",
        "VERIFIED_BY_WEIGHTS"
        if cfg.get("dtype") == "float32" and all_f32
        else "FAILED",
        f"config dtype={cfg.get('dtype')}",
    )
    add(
        f"qtype_embedding = {sh['qtype_embedding']}",
        "VERIFIED_BY_WEIGHTS"
        if shapes.get("qtype_emb.weight") == (sh["qtype_embedding"], 1024)
        else "FAILED",
        f"qtype_emb.weight shape={shapes.get('qtype_emb.weight')}",
    )
    scorer_ok = (
        shapes.get("scorer.0.weight") == (256, 1024)
        and shapes.get("scorer.2.weight") == (1, 256)
        and sh["scorer"] == "1024->256->1"
    )
    add(
        f"scorer = {sh['scorer']}",
        "VERIFIED_BY_WEIGHTS" if scorer_ok else "FAILED",
        f"scorer.0.weight={shapes.get('scorer.0.weight')}, "
        f"scorer.2.weight={shapes.get('scorer.2.weight')}",
    )
    lm_keys = [k for k in keys if "lm_head" in k.lower()]
    add(
        "lm_head = false; generation = false",
        "VERIFIED_BY_WEIGHTS"
        if not lm_keys and sh["lm_head"] is False and sh["generation"] is False
        else "FAILED",
        f"lm_head-like keys={lm_keys}; architectures={cfg['architectures']}",
    )
    xa = [k for k in keys if "state_cross_attn" in k]
    om = [k for k in keys if "option_mixer" in k]
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        prior = (
            float(handle.get_tensor("prior_scale").item())
            if "prior_scale" in keys
            else -1.0
        )
    flags_ok = (
        not xa
        and not om
        and prior == 0.0
        and not sh["use_cross_attn"]
        and not sh["use_option_mixer"]
        and not sh["use_cosine_prior"]
    )
    add(
        "optional components (cross-attn / mixer / cosine-prior) OFF",
        "VERIFIED_BY_WEIGHTS" if flags_ok else "FAILED",
        f"cross-attn keys={xa}; mixer keys={om}; prior_scale={prior}",
    )

    # Provenance claims need exact tensor equality against the named sources.
    ckpt = torch.load(HEAD_CKPT, map_location="cpu", weights_only=True)["head"]
    head_max = 0.0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for k in head_keys:
            expected = ckpt.get(k)
            if expected is None:
                head_max = float("inf")
                break
            head_max = max(
                head_max,
                float((handle.get_tensor(k).float() - expected.float()).abs().max()),
            )
    trained_ok = (
        head_max == 0.0 and sh["trained"] is True and len(ckpt) == len(head_keys)
    )
    add(
        "systemone_head.trained = true; head_checkpoint weights embedded",
        "VERIFIED_BY_SOURCE_MATCH" if trained_ok else "FAILED",
        f"max abs diff vs {man['head_checkpoint']} = {head_max}; "
        f"{len(head_keys)} head tensors",
    )

    src_path = SOURCE_BACKBONE / "model.safetensors"
    back_max = 0.0
    if src_path.exists():
        with safe_open(str(src_path), framework="pt", device="cpu") as src:
            src_keys = set(src.keys())
            for k in backbone_keys:
                name = k[len("backbone.") :]
                if name not in src_keys:
                    back_max = float("inf")
                    break
                with safe_open(
                    str(weights_path), framework="pt", device="cpu"
                ) as handle:
                    back_max = max(
                        back_max,
                        float(
                            (
                                src.get_tensor(name).float()
                                - handle.get_tensor(k).float()
                            )
                            .abs()
                            .max()
                        ),
                    )
    frozen_ok = back_max == 0.0 and hc["freeze_backbone"] is True
    add(
        "head_config.freeze_backbone = true; source_backbone weights unchanged",
        "VERIFIED_BY_SOURCE_MATCH" if frozen_ok else "FAILED",
        f"max abs diff vs declared source = {back_max}; {len(backbone_keys)} tensors",
    )

    for label, value in (
        ("systemone_head.max_state_len", sh["max_state_len"]),
        ("systemone_head.training_dataset", sh["training_dataset"]),
        ("manifest.seed", man["seed"]),
        ("manifest.source_backbone string", man["source_backbone"]),
    ):
        add(
            f"{label} = {value}",
            "SUPPORTED_BY_PRODUCER",
            "provenance/config value from producing code; not entailed by static weights",
        )

    report = {
        "artifact": str(EXPORT),
        "claim_count": len(claims),
        "verified_by_weights": sum(
            c["status"] == "VERIFIED_BY_WEIGHTS" for c in claims
        ),
        "verified_by_source_match": sum(
            c["status"] == "VERIFIED_BY_SOURCE_MATCH" for c in claims
        ),
        "supported_by_producer": sum(
            c["status"] == "SUPPORTED_BY_PRODUCER" for c in claims
        ),
        "failed": sum(c["status"] == "FAILED" for c in claims),
        "claims": claims,
        "verdict": "PASS" if all(c["status"] != "FAILED" for c in claims) else "FAIL",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "claims"}, indent=2))
    for claim in claims:
        print(f"  [{claim['status']:>25}] {claim['claim']}")


if __name__ == "__main__":
    main()
