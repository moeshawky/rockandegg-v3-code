"""Clean-process smoke test of the exact interface documented in the operational handoff.

Runs a NEW interpreter against the published preview artifact (no local release dir),
using the documented load path: revision + subfolder + trust_remote_code + dtype.
Verifies install path, model load, tokenizer load, SystemOneInference init, all three
decision types, shared-state fan-out at F1/F2/F4/F8, output normalization, and
encode-once (2 backbone forwards). No accuracy evaluation is performed.

Run:
  HF_HUB_DISABLE_PROGRESS_BARS=1 USE_TF=0 PYTHONPATH=/scratch/rockandegg-v3-code:src \
    /opt/pytorch/bin/python release/smoke_documented_interface.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = "moeshawky/RockAndEgg-313M-v3"
REV = "preview/fullbackbone-bf16-seed0"
SUB = "fullbackbone-bf16-preview"
ROOT = Path("/scratch/rockandegg-v3-code")


def main() -> None:
    rep: dict = {"repo": REPO, "revision": REV, "subfolder": SUB}
    model = AutoModel.from_pretrained(
        REPO, revision=REV, subfolder=SUB, trust_remote_code=True, dtype="bfloat16",
    ).to("cuda:0").eval()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV, subfolder=SUB)
    sys.path.insert(0, str(ROOT / "release" / "RockAndEgg-313M-v3-fullbackbone-bf16-preview"))
    from systemone_inference import Question, SystemOneInference, SystemOneRequest  # noqa: PLC0415
    adapter = SystemOneInference(model, tok)  # defaults to T=1.0
    rep["model_load"] = True
    rep["tokenizer_load"] = True
    rep["marker_id_ok"] = tok.convert_tokens_to_ids("<|im_end|>") == 151645
    rep["param_count"] = sum(p.numel() for p in model.parameters())
    rep["raw_t1"] = adapter.confidence_temperature == 1.0

    state = "The payment service returned HTTP 503 after the deploy at 03:12 UTC."
    req = SystemOneRequest(state=state, questions=[
        Question(type="choice", options=["Roll back", "Ignore", "Add logging"], instruction="next action?"),
        Question(type="score", options=["low", "medium", "high", "critical"]),
        Question(type="noul", options=["false", "true"]),
    ])
    out = adapter(req)
    rep["n_decisions"] = len(out.decisions)
    rep["decision_types"] = [d.type for d in out.decisions]
    rep["all_normalized"] = all(abs(sum(d.probabilities) - 1.0) < 1e-3 for d in out.decisions)
    rep["confidence_is_max"] = all(
        abs(d.confidence - max(d.probabilities)) < 1e-4 for d in out.decisions)
    rep["sample"] = [{"type": d.type, "chosen": d.chosen_key,
                      "probs": [round(p, 4) for p in d.probabilities],
                      "confidence": round(d.confidence, 4)} for d in out.decisions]

    # shared-state fan-out: count backbone forwards at F1/F2/F4/F8
    from typing import Any
    calls: list[tuple[int, ...]] = []
    orig = model.backbone.forward

    def counting(*args: Any, **kwargs: Any) -> Any:
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is not None:
            calls.append(tuple(ids.shape))
        return orig(*args, **kwargs)

    model.backbone.forward = counting
    fanout = {}
    for n in (1, 2, 4, 8):
        calls.clear()
        with torch.inference_mode():
            adapter(SystemOneRequest(state=state, questions=[
                Question(type="choice", options=["opt%d" % i for i in range(n)]*2, instruction="x")]))
        fanout[f"F{n}"] = {"backbone_calls": len(calls), "shapes": calls}
    model.backbone.forward = orig
    rep["fanout"] = fanout
    rep["encode_once"] = all(v["backbone_calls"] == 2 for v in fanout.values())

    checks = {
        "model_load": rep["model_load"], "tokenizer_load": rep["tokenizer_load"],
        "marker_id": rep["marker_id_ok"], "param_count": rep["param_count"] == 313161733,
        "raw_t1": rep["raw_t1"], "three_decisions": rep["n_decisions"] == 3,
        "decision_types": rep["decision_types"] == ["choice", "score", "noul"],
        "normalized": rep["all_normalized"], "confidence": rep["confidence_is_max"],
        "encode_once": rep["encode_once"],
    }
    rep["checks"] = checks
    rep["all_pass"] = all(checks.values())
    (ROOT / "artifacts/smoke_interface.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"all_pass": rep["all_pass"], "checks": checks,
                      "sample": rep["sample"],
                      "fanout_calls": {k: v["backbone_calls"] for k, v in fanout.items()}}, indent=2))


if __name__ == "__main__":
    main()
