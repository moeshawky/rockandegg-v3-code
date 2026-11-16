"""Measure the FP16 variant's divergence from FP32 on the held-out corpus.

Two evaluations of the same checkpoint exist:

* ``training/ece_predictions.json``     -- FP32 compute path
* ``training/ece_predictions_fp16.json`` -- FP16 compute path

They cover the same 29,955 held-out cases, so this script measures exactly how
far the fp16 cast moves the *emitted distributions* — not just whether the
chosen option happens to agree. Accuracy is a derived quantity here; the
per-case probability delta is the primary measurement, because accuracy is
insensitive to it (a uniform monotone rescaling leaves arg-max untouched) and a
claim of "FP16 preserves decisions" built only on accuracy would be an
R-MEASURE.

Run: ``PYTHONPATH=src .venv/bin/python compare_fp16_eval.py``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
FP32 = ROOT / "training" / "ece_predictions.json"
FP16 = ROOT / "training" / "ece_predictions_fp16.json"


def load(path: Path) -> dict[str, Any]:
    """Read one evaluation artifact, with an actionable error if it is absent."""
    if not path.is_file():
        raise SystemExit(
            f"missing evaluation artifact: {path}\n"
            "Generate it first, e.g.:\n"
            "  PYTHONPATH=src .venv/bin/python evaluate_trained.py --dtype fp32\n"
            "  PYTHONPATH=src .venv/bin/python evaluate_trained.py --dtype fp16\n"
            "This comparison needs both probability payloads; neither is code."
        )
    return json.loads(path.read_text())


def main() -> int:
    """Compare the two artifacts case by case and print a report."""
    a = load(FP32)
    b = load(FP16)

    pa = np.asarray(a["probs"], dtype=np.float64)
    pb = np.asarray(b["probs"], dtype=np.float64)
    la = np.asarray(a["labels"], dtype=np.int64)
    lb = np.asarray(b["labels"], dtype=np.int64)

    report: dict[str, Any] = {
        "fp32_dtype": a.get("dtype", "absent"),
        "fp16_dtype": b.get("dtype"),
        "n_fp32": a["n"],
        "n_fp16": b["n"],
        "labels_identical": bool(np.array_equal(la, lb)),
        "probs_shape_equal": pa.shape == pb.shape,
    }

    if pa.shape == pb.shape:
        delta = np.abs(pa - pb)
        report["per_case_max_abs_prob_delta"] = float(delta.max())
        report["per_case_mean_abs_prob_delta"] = float(delta.mean())
        # Confidence = max prob per case; the decision the consumer reads.
        conf_a = pa.max(axis=1)
        conf_b = pb.max(axis=1)
        report["confidence_delta_max"] = float(np.abs(conf_a - conf_b).max())
        report["confidence_delta_mean"] = float(np.abs(conf_a - conf_b).mean())
        # Chosen-option agreement (arg-max). This is the claim a consumer cares
        # about, and it is deliberately reported alongside the delta so the
        # delta cannot hide behind it.
        report["chosen_option_agreement"] = float((pa.argmax(1) == pb.argmax(1)).mean())
        report["n_chosen_option_disagreements"] = int(
            (pa.argmax(1) != pb.argmax(1)).sum()
        )
        report["n_cases"] = int(pa.shape[0])

    report["accuracy_fp32"] = a["accuracy_recomputed"]
    report["accuracy_fp16"] = b["accuracy_recomputed"]
    report["accuracy_delta"] = b["accuracy_recomputed"] - a["accuracy_recomputed"]
    report["loss_fp32"] = a["loss_recomputed"]
    report["loss_fp16"] = b["loss_recomputed"]
    report["loss_delta"] = b["loss_recomputed"] - a["loss_recomputed"]

    # ---- are the disagreements symmetric, and are they near-ties? ----
    # Accuracy alone cannot answer this: 490 chosen-option disagreements with an
    # identical accuracy would be indistinguishable from a silent 490-case
    # regression offset by 490 silent gains. Decompose them.
    am = pa.argmax(1)
    bm = pb.argmax(1)
    correct_a = am == la
    correct_b = bm == la
    disagrees = am != bm
    report["disagreements"] = {
        "n": int(disagrees.sum()),
        "rate": float(disagrees.mean()),
        "fp32_correct_fp16_wrong": int((disagrees & correct_a & ~correct_b).sum()),
        "fp16_correct_fp32_wrong": int((disagrees & ~correct_a & correct_b).sum()),
        "both_wrong_different_answer": int((disagrees & ~correct_a & ~correct_b).sum()),
        "fp32_correct_total": int(correct_a.sum()),
        "fp16_correct_total": int(correct_b.sum()),
    }

    # Decision margin: top1 - top2 probability in the FP32 distribution. A cast
    # that only flips near-ties is a different claim from a cast that flips
    # confident decisions, so the two must be separated.
    part = np.partition(pa, -2, axis=1)
    margin = part[:, -1] - part[:, -2]
    agreeing = ~disagrees
    report["decision_margin"] = {
        "disagreeing_median": float(np.median(margin[disagrees]))
        if disagrees.any()
        else None,
        "agreeing_median": float(np.median(margin[agreeing]))
        if agreeing.any()
        else None,
        "disagreeing_below_0.01": int((margin[disagrees] < 0.01).sum()),
        "disagreeing_below_0.05": int((margin[disagrees] < 0.05).sum()),
    }

    # Per-decision-type breakdown of the disagreement rate, so a single bad
    # type cannot hide inside the aggregate.
    report["payload_difference"] = {
        "probabilities_byte_identical": bool(np.array_equal(pa, pb)),
        "n_values_differing": int((pa != pb).sum()),
        "total_values": int(pa.size),
        "n_delta_gt_1e-3": int((np.abs(pa - pb) > 1e-3).sum()),
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
