"""Canonical shared-state inference + performance for the BF16 release artifact.

Uses the shipped SystemOneInference adapter (no competing implementation).
Measures the F1/F2/F4/F8 fan-out protocol in calibration_report.md §4 and the
'encode state once' claim. Inputs are label-blind: real states/options from the
exposed 400-dossier panel are used only as performance inputs; no labels are read
and no selection/calibration decision is made from them.

Run:
  USE_TF=0 PYTHONPATH=/scratch/rockandegg-v3-code:src:/scratch/RockAndEgg-313M-v3 \
    /opt/pytorch/bin/python release/perf_release.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path("/scratch/rockandegg-v3-code")
REL = ROOT / "release/RockAndEgg-313M-v3-fullbackbone-bf16-preview"


def main() -> None:
    report: dict = {"artifact": str(REL), "device": "cuda:0", "precision": "bfloat16"}
    model = AutoModel.from_pretrained(str(REL), trust_remote_code=True,
                                      dtype=torch.bfloat16).to("cuda:0").eval()
    tok = AutoTokenizer.from_pretrained(str(REL))
    sys.path.insert(0, str(REL))
    from systemone_inference import Question, SystemOneInference, SystemOneRequest  # noqa: PLC0415
    adapter = SystemOneInference(model, tok, max_state_len=256, max_option_len=64)

    # label-blind inputs from the clean test panel (states + option text only)
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test",
                      revision="e6aa632c4db73b59c7713bb21066ad1a8e1bfce6")
    cases = []
    for r in ds:
        c = json.loads(r["state"])
        # extract a representative state string
        st = json.dumps(c)[:400]
        qs = json.loads(r["questions"])
        for q in qs.values():
            opts = q.get("criteria")
            if isinstance(opts, dict):
                opt_list = [f"{k}: {v}" for k, v in list(opts.items())[:8]]
            else:
                opt_list = [f"level {i}" for i in range(min(8, len(q.get("criteria", []))))]
            if len(opt_list) >= 1:
                cases.append((st, opt_list))
        if len(cases) >= 64:
            break
    report["n_input_cases"] = len(cases)

    # ---- backbone-call counting (encode-once claim) on the SERVED model ----
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
    state, opts = cases[0]
    for fan in (1, 2, 4, 8):
        calls.clear()
        req = SystemOneRequest(state=state, questions=[
            Question(type="choice", options=opts[:fan], instruction="perf")])
        with torch.inference_mode():
            _ = adapter(req)
        report.setdefault("fanout", {})[f"F{fan}"] = {
            "backbone_calls": len(calls), "shapes": calls,
            "naive_reencode_calls": 1 + fan,
        }
    model.backbone.forward = orig

    # ---- latency: warmup then timed runs at each fan-out ----
    def bench(fan: int, n_iter: int = 40) -> dict:
        state, opts = cases[0]
        req = SystemOneRequest(state=state, questions=[
            Question(type="choice", options=opts[:fan], instruction="perf")])
        for _ in range(5):  # warmup
            with torch.inference_mode():
                _ = adapter(req)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = adapter(req)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        return {
            "fan_out": fan,
            "warmup_iters": 5, "timed_iters": n_iter,
            "p50_ms": round(statistics.median(lat), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "min_ms": round(lat[0], 3), "max_ms": round(lat[-1], 3),
            "throughput_req_per_s": round(1000.0 / statistics.median(lat), 1),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        }

    report["latency"] = {f"F{fan}": bench(fan) for fan in (1, 2, 4, 8)}
    rep_flag = all(v["backbone_calls"] == 2 for v in report["fanout"].values())
    report["encode_once_verified"] = rep_flag
    (ROOT / "artifacts/release_performance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"encode_once_verified": rep_flag,
                      "fanout": {k: v["backbone_calls"] for k, v in report["fanout"].items()},
                      "latency": report["latency"]}, indent=2))


if __name__ == "__main__":
    main()
