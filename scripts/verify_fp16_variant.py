"""Verify the FP16 variant of the RockAndEgg-313M-v3 release artifact.

Four independent checks, each answering a different question:

1. **Weight identity** — does the FP16 file hold exactly the weights the
   qualified FP16 compute path uses? The export gates ran with
   ``HeadConfig(dtype_name="fp16")`` (``gate_results_trained.json`` records
   ``config.dtype == "fp16"``), so the gated model is an FP16 model. If the
   FP16 file equals that model's state dict tensor for tensor, the FP16
   artifact *is* the gated configuration.
2. **Loadability** — does the documented Hub path work? ``subfolder="fp16"``
   with ``trust_remote_code=True`` must produce a working model.
3. **Consumer parity** — do the emitted decisions match the FP32 artifact's?
4. **Divergence** — how far do the probability vectors move? A cast is not
   free; this measures the cost.

Run: ``PYTHONPATH=src .venv/bin/python verify_fp16_variant.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "hf_repo" / "RockAndEgg-313M-v3"
FP16_DIR = REPO / "fp16"

# The same two-question request the model card documents.
STATE = (
    "Deployment RAE-313M failed at 03:12 UTC. The on-call engineer "
    "rolled back the release and opened an incident."
)
QUESTIONS = [
    {
        "type": "choice",
        "instruction": "What happened?",
        "options": ["The deployment failed", "The deployment succeeded"],
    },
    {
        "type": "score",
        "instruction": "How severe was it?",
        "options": ["none", "low", "medium", "high"],
    },
]


def run_example(model_dir: str) -> list[dict[str, Any]]:
    """Load a model directory and emit the card's request as plain dicts."""
    from transformers import AutoModel, AutoTokenizer

    sys.path.insert(0, str(Path(model_dir).resolve()))
    from systemone_inference import Question, SystemOneInference, SystemOneRequest

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval()
    adapter = SystemOneInference(model, tokenizer, max_state_len=256, max_option_len=64)
    request = SystemOneRequest(
        state=STATE,
        questions=[Question(**q) for q in QUESTIONS],
    )
    response = adapter(request)
    return [
        {
            "type": d.type,
            "chosen_key": d.chosen_key,
            "confidence": d.confidence,
            "probabilities": list(d.probabilities),
        }
        for d in response.decisions
    ]


def check_weight_identity() -> dict[str, Any]:
    """Compare the FP16 file to the FP16 model the gates measured.

    The gates (``run_systemone_gates.py``) build ``HeadConfig()`` — whose
    ``dtype_name`` defaults to ``"fp16"`` — and then load the trained head from
    ``--head-checkpoint``. Reproducing that configuration requires loading the
    same checkpoint; comparing against a randomly initialised head would
    measure nothing but the seed.

    The trained head is a **weight**, not code, and is deliberately not part of
    this repository. When it is absent this check reports ``SKIPPED`` with the
    reason and the exact path to supply, rather than crashing or — worse —
    silently passing against a random initialisation. The remaining checks
    (loadability, consumer parity) still run.
    """
    from rockandegg_v3 import HeadConfig, RockAndEggV3SystemOne

    fp16_file = load_file(str(FP16_DIR / "model.safetensors"))

    head_path = ROOT / "training" / "head_final.pt"
    if not head_path.is_file():
        return {
            "status": "SKIPPED",
            "reason": "trained head checkpoint not present",
            "head_path": str(head_path),
            "how_to_supply": (
                "Place the trained head checkpoint at training/head_final.pt, "
                "or build the model from the released artifact "
                "(moeshawky/RockAndEgg-313M-v3, subfolder='fp16') instead."
            ),
            "what_this_means": (
                "Weight identity against the gated fp16 configuration was NOT "
                "checked here. It is not inferred and not assumed."
            ),
            "tensors_in_file": len(fp16_file),
            "dtypes": sorted({str(t.dtype) for t in fp16_file.values()}),
        }

    model = RockAndEggV3SystemOne(HeadConfig(dtype_name="fp16")).to("cuda:0").eval()
    payload = torch.load(head_path, map_location="cuda:0", weights_only=True)
    state = model.state_dict()
    state.update(payload["head"])
    model.load_state_dict(state, strict=True)
    live = model.state_dict()

    keys_match = sorted(fp16_file) == sorted(live)
    diffs = {}
    worst = 0.0
    for key in sorted(fp16_file):
        a = fp16_file[key].float()
        b = live[key].float().cpu()
        d = float((a - b).abs().max()) if a.shape == b.shape else float("nan")
        diffs[key] = d
        if d == d:  # not NaN
            worst = max(worst, d)

    dtypes = {str(t.dtype) for t in fp16_file.values()}
    return {
        "status": "CHECKED",
        "compared_against": "HeadConfig(dtype_name='fp16') + training/head_final.pt",
        "tensors_in_file": len(fp16_file),
        "tensors_in_live_fp16_model": len(live),
        "keys_match": keys_match,
        "dtypes": sorted(dtypes),
        "max_abs_diff_vs_gated_fp16_model": worst,
        "n_tensors_differing": sum(1 for d in diffs.values() if d > 0),
    }


def check_loadability() -> dict[str, Any]:
    """Load through the documented Hub path: subfolder + trust_remote_code."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    kwargs = {"subfolder": "fp16", "trust_remote_code": True}
    cfg = AutoConfig.from_pretrained(str(REPO), **kwargs)
    tok = AutoTokenizer.from_pretrained(str(REPO), **kwargs)
    model = AutoModel.from_pretrained(str(REPO), **kwargs).eval()
    return {
        "config_class": type(cfg).__name__,
        "model_class": type(model).__name__,
        "config_dtype": str(getattr(cfg, "dtype", None)),
        "tokenizer_class": type(tok).__name__,
        "backbone_arch": model.config.architectures,
        "has_lm_head": hasattr(model, "lm_head"),
        "has_generate": hasattr(model, "generate"),
        "first_weight_dtype": str(next(model.parameters()).dtype),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }


def main() -> int:
    """Run every check and print a JSON report."""
    # The assembled model directories (hf_repo/) hold weights, not code, so a
    # source checkout of this repository does not carry them. Say so and point
    # at the Hub instead of crashing on a missing file.
    if not (REPO / "model.safetensors").is_file():
        print(
            "assembled model directory not found: "
            f"{REPO / 'model.safetensors'}\n"
            "This repository carries the code, not the weights. Download the\n"
            "release artifacts first, e.g.:\n"
            "  hf download moeshawky/RockAndEgg-313M-v3 --local-dir "
            f"{REPO}\n"
            "then re-run this script.",
            file=sys.stderr,
        )
        return 2

    report: dict[str, Any] = {
        "fp16_artifact": str(FP16_DIR / "model.safetensors"),
        "fp16_sha256_source": json.loads(
            (FP16_DIR / "fp16_export_report.json").read_text()
        )["sha256"],
    }

    report["weight_identity"] = check_weight_identity()
    report["loadability_subfolder"] = check_loadability()

    fp32_out = run_example(str(REPO))
    fp16_out = run_example(str(FP16_DIR))
    report["fp32_output"] = fp32_out
    report["fp16_output"] = fp16_out

    same_choice = [
        a["chosen_key"] == b["chosen_key"]
        for a, b in zip(fp32_out, fp16_out, strict=True)
    ]
    deltas = []
    for a, b in zip(fp32_out, fp16_out, strict=True):
        pa = np.asarray(a["probabilities"], dtype=float)
        pb = np.asarray(b["probabilities"], dtype=float)
        deltas.append(
            {
                "type": a["type"],
                "chosen_key_matches": a["chosen_key"] == b["chosen_key"],
                "max_abs_prob_delta": float(np.abs(pa - pb).max()),
                "mean_abs_prob_delta": float(np.abs(pa - pb).mean()),
                "confidence_delta": abs(a["confidence"] - b["confidence"]),
            }
        )
    report["divergence"] = {
        "per_question": deltas,
        "all_chosen_keys_match": all(same_choice),
        "max_abs_prob_delta": max(d["max_abs_prob_delta"] for d in deltas),
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
