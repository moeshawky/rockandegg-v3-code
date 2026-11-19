"""Fresh-process integrity verification of the BF16 release artifact.

Run in a NEW interpreter (no state carried from the build):
  USE_TF=0 PYTHONPATH=src:/scratch/RockAndEgg-313M-v3 \
    /opt/pytorch/bin/python release/verify_release.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

ROOT = Path("/scratch/rockandegg-v3-code")
REL = ROOT / "release/RockAndEgg-313M-v3-fullbackbone-bf16-preview"
SRC = ROOT / "artifacts/runs/fullbb_bf16_e2style/best/model"
DEV_PATH = ROOT / "experiments/ensemble_validation_20261118/seed2_infer_dir"


def main() -> None:
    rep: dict = {"artifact": str(REL)}
    cfg = json.loads((REL / "config.json").read_text())
    rep["config_dtype"] = cfg.get("dtype")
    rep["architectures"] = cfg.get("architectures")
    rep["no_lm_head_declared"] = not cfg.get("systemone_head", {}).get("lm_head", True)
    rep["no_generation_declared"] = not cfg.get("systemone_head", {}).get("generation", True)

    # tensor shapes + dtype parity against source
    with safe_open(str(REL / "model.safetensors"), framework="pt") as rh, \
         safe_open(str(SRC / "model.safetensors"), framework="pt") as sh:
        rkeys, skeys = set(rh.keys()), set(sh.keys())
    rep["tensor_key_parity"] = rkeys == skeys
    rep["n_tensors"] = len(rkeys)
    shape_bad = []
    with safe_open(str(REL / "model.safetensors"), framework="pt") as rh, \
         safe_open(str(SRC / "model.safetensors"), framework="pt") as sh:
        for k in rkeys:
            if list(rh.get_slice(k).get_shape()) != list(sh.get_slice(k).get_shape()):
                shape_bad.append(k)
    rep["all_tensor_shapes_match_source"] = (len(shape_bad) == 0)
    rep["shape_mismatches"] = shape_bad[:10]

    # ---- GPU load via canonical path ----
    model = AutoModel.from_pretrained(str(REL), trust_remote_code=True,
                                      dtype=torch.bfloat16).to("cuda:0").eval()
    n_params = sum(p.numel() for p in model.parameters())
    rep["param_count"] = n_params
    rep["param_count_ok"] = (n_params == 313_161_733)
    rep["has_lm_head_attr"] = hasattr(model, "lm_head")
    rep["has_generate_attr"] = hasattr(model, "generate")
    rep["all_weights_bf16"] = all(p.dtype == torch.bfloat16 for p in model.parameters())
    rep["backbone_frozen_at_serve"] = all(
        not p.requires_grad for p in model.backbone.parameters()
    )

    tok = AutoTokenizer.from_pretrained(str(REL))
    rep["tokenizer_loads"] = tok is not None
    rep["marker_id_ok"] = (tok.convert_tokens_to_ids("<|im_end|>") == 151645)

    import sys
    sys.path.insert(0, str(REL))
    from systemone_inference import (  # noqa: PLC0415
        Question, SystemOneInference, SystemOneRequest,
    )
    adapter = SystemOneInference(model, tok, max_state_len=256, max_option_len=64)
    rep["raw_temperature_is_1"] = (adapter.confidence_temperature == 1.0)
    rep["calibration_json_absent_from_release"] = not (REL / "calibration.json").exists()

    def decide(state: str, opts: list[str], qtype: str = "choice") -> dict:
        req = SystemOneRequest(state=state, questions=[
            Question(type=qtype, options=opts, instruction="probe")])
        d = adapter(req).decisions[0]
        p = np.asarray(d.probabilities, dtype=float)
        return {"probs": p.tolist(), "sum": float(p.sum()),
                "argmax": int(p.argmax()), "conf": float(d.confidence),
                "chosen": d.chosen_key}

    state = "The payment service returned HTTP 503 for three of five requests after the deploy at 03:12 UTC."
    opts = ["Roll back the release", "Ignore the alert", "Increase log verbosity"]
    d_choice = decide(state, opts, "choice")
    rep["probe_choice"] = d_choice
    rep["prob_normalized_1"] = abs(d_choice["sum"] - 1.0) < 1e-5
    rep["confidence_equals_maxprob"] = abs(d_choice["conf"] - max(d_choice["probs"])) < 1e-5

    # decision type behavior: all three types
    rep["probe_score"] = decide(state, ["low", "medium", "high", "critical"], "score")
    rep["probe_noul"] = decide(state, ["false", "true"], "noul")
    rep["score_normalized"] = abs(rep["probe_score"]["sum"] - 1.0) < 1e-5
    rep["noul_normalized"] = abs(rep["probe_noul"]["sum"] - 1.0) < 1e-5

    # option-order handling: permutation must permute probabilities identically
    perm = [opts[i] for i in [2, 0, 1]]
    d_perm = decide(state, perm, "choice")
    rep["option_order_permutation"] = {
        "orig_probs": d_choice["probs"], "perm_probs": d_perm["probs"],
        "perm_argmax_key": d_perm["chosen"],
    }
    # same canonical state + same option SET, different order -> chosen key stable
    rep["option_order_chosen_key_stable"] = (
        d_choice["chosen"] == d_perm["chosen"]
    )

    # determinism: repeat on GPU
    d_repeat = decide(state, opts, "choice")
    rep["deterministic_gpu"] = (d_choice["probs"] == d_repeat["probs"])

    del model, adapter
    torch.cuda.empty_cache()

    # ---- CPU load of bf16 artifact ----
    model_cpu = AutoModel.from_pretrained(str(REL), trust_remote_code=True,
                                          dtype=torch.bfloat16).to("cpu").eval()
    adapter_cpu = SystemOneInference(model_cpu, tok)
    d_cpu = adapter_cpu(SystemOneRequest(state=state, questions=[
        Question(type="choice", options=opts, instruction="probe")])).decisions[0]
    rep["cpu_load_and_infer"] = True
    rep["cpu_vs_gpu_prob_maxdiff"] = float(np.max(np.abs(
        np.asarray(d_cpu.probabilities, dtype=float) - np.asarray(d_choice["probs"]))))
    del model_cpu, adapter_cpu

    # ---- FP32 source vs BF16 release decision parity on 10 real dossiers ----
    # label-blind: inputs only, no labels used for selection
    from datasets import load_dataset
    from typed_decisions_data import render_options, serialize_state
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test",
                      revision="e6aa632c4db73b59c7713bb21066ad1a8e1bfce6")
    sample = list(ds)[:10]

    from typed_eval import V3Arm
    fp32 = V3Arm(str(SRC), dtype="fp32", device="cuda:0")
    bf16 = V3Arm(str(REL), dtype="bf16", device="cuda:0")
    n_dec = 0
    argmax_flips = 0
    max_prob_diff = 0.0
    for r in sample:
        st = json.loads(r["state"])
        qs = json.loads(r["questions"])
        a = fp32.predict({"id": r["id"]}, st, qs)
        b = bf16.predict({"id": r["id"]}, st, qs)
        for qid in qs:
            n_dec += 1
            if "probabilities" in a[qid]:
                pa = np.asarray(list(a[qid]["probabilities"].values()), dtype=float)
                pb = np.asarray(list(b[qid]["probabilities"].values()), dtype=float)
            else:  # noul: single scalar p(true)
                pa = np.asarray([1.0 - a[qid]["noul"], a[qid]["noul"]], dtype=float)
                pb = np.asarray([1.0 - b[qid]["noul"], b[qid]["noul"]], dtype=float)
            max_prob_diff = max(max_prob_diff, float(np.max(np.abs(pa - pb))))
            if int(pa.argmax()) != int(pb.argmax()):
                argmax_flips += 1
    rep["fp32_vs_bf16_parity"] = {
        "n_decisions": n_dec,
        "argmax_flips": argmax_flips,
        "max_prob_abs_diff": max_prob_diff,
    }

    # ---- verdict ----
    checks = {
        "config_dtype_bf16": rep["config_dtype"] == "bfloat16",
        "no_lm_head": rep["no_lm_head_declared"] and not rep["has_lm_head_attr"],
        "no_generate": rep["no_generation_declared"] and not rep["has_generate_attr"],
        "param_count": rep["param_count_ok"],
        "tensor_keys": rep["tensor_key_parity"],
        "tensor_shapes": rep["all_tensor_shapes_match_source"],
        "all_bf16": rep["all_weights_bf16"],
        "tokenizer": rep["tokenizer_loads"] and rep["marker_id_ok"],
        "prob_normalization": rep["prob_normalized_1"] and rep["score_normalized"] and rep["noul_normalized"],
        "confidence": rep["confidence_equals_maxprob"],
        "option_order": rep["option_order_chosen_key_stable"],
        "determinism": rep["deterministic_gpu"],
        "raw_t1_no_calibration": rep["raw_temperature_is_1"] and rep["calibration_json_absent_from_release"],
        "cpu_load": rep["cpu_load_and_infer"] and rep["cpu_vs_gpu_prob_maxdiff"] < 0.05,
        "bf16_parity": rep["fp32_vs_bf16_parity"]["argmax_flips"] == 0,
        "backbone_frozen_serve": rep["backbone_frozen_at_serve"],
    }
    rep["checks"] = checks
    rep["all_pass"] = all(checks.values())
    (ROOT / "artifacts/release_integrity.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({"all_pass": rep["all_pass"], "checks": checks,
                      "parity": rep["fp32_vs_bf16_parity"],
                      "param_count": rep["param_count"],
                      "cpu_vs_gpu_maxdiff": rep["cpu_vs_gpu_prob_maxdiff"]}, indent=2))


if __name__ == "__main__":
    main()
