"""Independent held-out evaluation of the trained v3 head.

Loads the final head checkpoint, evaluates the reserved `calibration` split and
`test_set_30k`, and writes a predictions artifact that
`run_systemone_gates.py --ece-predictions` recomputes ECE from, so no ECE number
is self-reported.

Calibration contract: the export artifact stores ``softmax(logits)`` and the
serving path reports ``softmax(logits / T)`` with ``T = confidence_temperature``
(default 1.0). No fitted temperature is stored in the export config or read by
any serving path, so the gate MUST evaluate at the serving default T=1.0. A
log-space temperature fit is recorded as *advisory only* — it is not consumed
by serving and never drives the gate (an unconsumed temperature would be an
R-PLACEBO).

Run: ``PYTHONPATH=src .venv/bin/python evaluate_trained.py``

Use ``--dtype fp16`` to measure the FP16 variant of the same checkpoint and
``--out`` to keep that artifact separate from the FP32 one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from train_jev_distill import (
    BACKBONE,
    OUT,
    evaluate,
    read_cases,
    stratified_sample,
)
from transformers import AutoTokenizer

from rockandegg_v3 import HeadConfig, RockAndEggV3SystemOne, compute_ece, serving_probs

SERVING_TEMPERATURE = 1.0  # SystemOneInference default; no override is wired.


def fit_advisory_temperature(
    probs: list[list[float]], labels: list[int]
) -> tuple[float, float]:
    """Fit a temperature on the SERVING path for documentation only.

    Uses ``softmax(log(probs) / T)`` — a single softmax on recovered logits —
    which is what serving would emit if an integrator wired a temperature. It is
    NOT consumed by the gate or by serving today.
    """
    best_temp, best_ece = 1.0, float("inf")
    probs_t = torch.tensor(probs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    for i in range(59):
        temp = round(0.1 + 0.05 * i, 3)
        ece = compute_ece(serving_probs(probs_t, temp), labels_t)
        if ece < best_ece:
            best_temp, best_ece = temp, float(ece)
    return best_temp, best_ece


def main() -> None:
    """Write raw held-out predictions plus the serving-temperature provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help=(
            "compute dtype for the evaluation. fp32 reproduces the release "
            "artifact's measurements; fp16 measures the FP16 variant."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="artifact path (default: training/ece_predictions.json for fp32).",
    )
    args = parser.parse_args()

    report = json.loads((OUT / "training_report.json").read_text())
    checkpoint = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
    model = RockAndEggV3SystemOne(HeadConfig(dtype_name=args.dtype)).to("cuda:0").eval()
    state = model.state_dict()
    state.update(checkpoint["head"])
    model.load_state_dict(state, strict=True)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    eval_args = type(
        "Args", (), {"state_len": 256, "option_len": 64, "eval_batch": 8}
    )()
    calib_rows = stratified_sample(read_cases("calibration"), 3000, 20261116)
    calib = evaluate(model, calib_rows, tokenizer, "cuda:0", eval_args)  # type: ignore[arg-type]
    print(f"calibration evaluated: {len(calib['_labels'])} cases", flush=True)
    advisory_temp, advisory_ece = fit_advisory_temperature(
        calib["_probs"], calib["_labels"]
    )

    test = read_cases("test_set_30k")
    metrics = evaluate(model, test, tokenizer, "cuda:0", eval_args)  # type: ignore[arg-type]
    print(f"held-out evaluated: {len(test)} cases", flush=True)
    payload: dict[str, Any] = {
        "probs": metrics["_probs"],
        "labels": metrics["_labels"],
        "temperature": SERVING_TEMPERATURE,
        "temperature_source": "serving default (SystemOneInference.confidence_temperature=1.0)",
        "dtype": args.dtype,
        "split": "test_set_30k",
        "dataset": "SargeDev/jev-distill-corpus-v3",
        "dataset_revision": report["dataset_revision"],
        "checkpoint": report["checkpoint"],
        "n": len(test),
        "advisory_confidence_temperature": advisory_temp,
        "advisory_calibration_ece": advisory_ece,
        "advisory_note": (
            "Log-space fit on calibration; NOT stored in export config and NOT "
            "consumed by serving. Recorded for an integrator who wires a "
            "confidence temperature; the gate evaluates at T=1.0."
        ),
        "calibration_n": len(calib["_labels"]),
        "accuracy_recomputed": metrics["accuracy"],
        "loss_recomputed": metrics["loss"],
    }
    out_path = (
        Path(args.out)
        if args.out
        else (
            OUT / "ece_predictions.json"
            if args.dtype == "fp32"
            else OUT / "ece_predictions_fp16.json"
        )
    )
    out_path.write_text(json.dumps(payload))
    summary = {k: v for k, v in payload.items() if k != "probs"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
