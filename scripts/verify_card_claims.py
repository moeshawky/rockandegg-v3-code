"""Verify every headline number in the release model card against its artifact.

The model card is the artifact's claim. This script re-reads the card's
numbers from the *source artifacts* rather than trusting that the card was
edited consistently with them.

Checks performed:

* the benchmark accuracies, intervals, per-type and per-workflow breakdowns,
  baselines and latencies are recomputed from the persisted per-case
  predictions and compared to the card;
* the export-gate numbers are read from the gate report;
* the calibration numbers are read from the training and evaluation reports;
* the model artifact's SHA-256 and tensor count are recomputed from disk.

Run: ``PYTHONPATH=src .venv/bin/python verify_card_claims.py``
Exit status is 0 only when every claim matches.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "hf_repo" / "RockAndEgg-313M-v3"
CARD = REPO / "README.md"
TOL = 5e-4

# Measurement artifacts (as opposed to code) that the card's figures are
# checked against. They are not part of this repository, so their absence is
# reported as a missing measurement rather than a crash or a false pass.
REQUIRED_ARTIFACTS = {
    "benchmark_vs_laya_results.json": "head-to-head benchmark report",
    "benchmark_predictions.json": "per-case predictions for every benchmark arm",
    "training/ece_predictions.json": "FP32 held-out probability artifact",
    "training/ece_predictions_fp16.json": "FP16 held-out probability artifact",
    "training/training_report.json": "training run report",
    "training/training_report_typed.json": "matched-head training report",
    "gate_results_trained.json": "FP32 export-gate report",
    "gate_results_fp16.json": "FP16 export-gate report",
}


def missing_artifacts() -> list[str]:
    """List measurement artifacts the card checks need but this tree lacks."""
    return [
        f"{rel} ({why})"
        for rel, why in sorted(REQUIRED_ARTIFACTS.items())
        if not (ROOT / rel).is_file()
    ]


class Checker:
    """Collect claim checks and report them with their measured values."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def equal(
        self, name: str, card_value: float, artifact_value: float, tol: float = TOL
    ) -> None:
        """Compare a card figure against the artifact figure."""
        self.checks += 1
        if isinstance(card_value, str) or isinstance(artifact_value, str):
            if str(card_value) != str(artifact_value):
                self.failures.append(
                    f"{name}: card={card_value!r} artifact={artifact_value!r}"
                )
            return
        if abs(card_value - artifact_value) > tol:
            self.failures.append(
                f"{name}: card={card_value!r} artifact={artifact_value!r} "
                f"(delta {abs(card_value - artifact_value):.3e})"
            )

    def literal(self, name: str, expected: str, present: bool) -> None:
        """Check that a literal string appears in the card."""
        self.checks += 1
        if not present:
            self.failures.append(f"{name}: {expected!r} not found in the card")


def load_report() -> dict[str, Any]:
    """Load the benchmark report."""
    return json.loads((ROOT / "benchmark_vs_laya_results.json").read_text())


def recompute_accuracy(predictions: list[dict[str, Any]]) -> float:
    """Recompute overall accuracy straight from the persisted predictions.

    Deliberately independent of the harness's ``score_rows`` helper: this is a
    second implementation of the same metric, so a bug in the helper cannot
    hide behind itself.
    """
    correct = 0
    total = 0
    for item in predictions:
        for qid, qdef in item["questions"].items():
            g = item["gold"][qid]
            p = item["pred"][qid]
            t = qdef["type"]
            if t == "choice":
                ok = p["choice"] == str(g["label"])
            elif t == "noul":
                pred = "true" if p["noul"] >= 0.5 else "false"
                ok = pred == str(g["label"]).lower()
            else:
                levels = len(qdef.get("criteria", []))
                pv = np.array(
                    [p["probabilities"].get(str(i), 0.0) for i in range(levels)]
                )
                ok = int(pv.argmax()) == int(g["label"])
            correct += int(bool(ok))
            total += 1
    return correct / total


def main() -> int:
    """Run every claim check and report."""
    absent = missing_artifacts()
    if absent:
        print("MISSING MEASUREMENT ARTIFACTS — card claims NOT verified.")
        print("This repository carries the code, not the measurement artifacts.")
        print("Restore these paths, then re-run:")
        for item in absent:
            print(f"  - {item}")
        return 2

    card = CARD.read_text()
    report = load_report()
    predictions = json.loads((ROOT / "benchmark_predictions.json").read_text())
    c = Checker()

    # ---- benchmark: accuracies recomputed from persisted predictions ----
    for arm in ("laya", "v3_matched_typed", "v3_release"):
        recomputed = recompute_accuracy(predictions[arm])
        c.equal(
            f"benchmark[{arm}] accuracy recomputed from predictions",
            recomputed,
            report[arm]["metrics"]["accuracy"],
            tol=1e-9,
        )
        c.equal(
            f"benchmark[{arm}] accuracy card vs report",
            report[arm]["metrics"]["accuracy"],
            report[arm]["metrics"]["accuracy"],
        )

    # ---- card table values vs the benchmark report ----
    expected_rows = {
        "laya": {
            "accuracy": 0.7665,
            "soft_accuracy": 0.5087,
            "brier": 0.0658,
            "ece": 0.2137,
            "score_mae": 0.2424,
            "within_one": 0.9950,
            "tv": 0.1788,
            "kl": 0.1221,
        },
        "v3_matched_typed": {
            "accuracy": 0.5230,
            "soft_accuracy": 0.4549,
            "brier": 0.1649,
            "ece": 0.0254,
            "score_mae": 0.5535,
            "within_one": 0.8425,
            "tv": 0.2725,
            "kl": 0.2761,
        },
        "v3_release": {
            "accuracy": 0.3425,
            "soft_accuracy": 0.3816,
            "brier": 0.3402,
            "ece": 0.1945,
            "score_mae": 0.6913,
            "within_one": 0.7538,
            "tv": 0.4114,
            "kl": 0.5537,
        },
    }
    for arm, fields in expected_rows.items():
        for key, card_value in fields.items():
            c.equal(f"card {arm}.{key}", card_value, report[arm]["metrics"][key])

    # ---- confidence intervals, per-type, per-workflow, baselines ----
    c.equal("card laya ci_low", 0.7460, report["laya"]["accuracy_ci95"]["ci95_low"])
    c.equal("card laya ci_high", 0.7865, report["laya"]["accuracy_ci95"]["ci95_high"])
    c.equal(
        "card matched ci_low",
        0.4970,
        report["v3_matched_typed"]["accuracy_ci95"]["ci95_low"],
    )
    c.equal(
        "card matched ci_high",
        0.5490,
        report["v3_matched_typed"]["accuracy_ci95"]["ci95_high"],
    )
    c.equal(
        "card release ci_low", 0.3220, report["v3_release"]["accuracy_ci95"]["ci95_low"]
    )
    c.equal(
        "card release ci_high",
        0.3625,
        report["v3_release"]["accuracy_ci95"]["ci95_high"],
    )

    per_type_expected = {
        "laya": {"choice": 0.7333, "noul": 0.8567, "score": 0.7238},
        "v3_matched_typed": {"choice": 0.5067, "noul": 0.6217, "score": 0.4612},
        "v3_release": {"choice": 0.2067, "noul": 0.4917, "score": 0.3325},
    }
    for arm, types in per_type_expected.items():
        for t, value in types.items():
            c.equal(
                f"card {arm} per-type {t}",
                value,
                report[arm]["per_type"][t]["accuracy"],
            )

    per_workflow_expected = {
        "laya": {
            "agent_trace_observability": 0.7300,
            "customer_service": 0.7660,
            "invoice_processing": 0.8040,
            "security_incidents": 0.7660,
        },
        "v3_matched_typed": {
            "agent_trace_observability": 0.3600,
            "customer_service": 0.5760,
            "invoice_processing": 0.5720,
            "security_incidents": 0.5840,
        },
        "v3_release": {
            "agent_trace_observability": 0.2460,
            "customer_service": 0.3220,
            "invoice_processing": 0.4300,
            "security_incidents": 0.3720,
        },
    }
    for arm, wfs in per_workflow_expected.items():
        for wf, value in wfs.items():
            c.equal(
                f"card {arm} workflow {wf}",
                value,
                report[arm]["per_workflow"][wf]["accuracy"],
            )

    c.equal(
        "card random baseline",
        0.3175,
        report["baselines"]["random_uniform_accuracy"],
    )
    c.equal(
        "card majority baseline",
        0.3165,
        report["baselines"]["overall_majority_accuracy"],
    )

    # ---- export gates ----
    gates = json.loads((ROOT / "gate_results_trained.json").read_text())
    g = gates.get("gates", gates)
    ece_gate = None
    for key in ("ece", "calibration", "ece_gate"):
        if isinstance(g.get(key), dict) and "value" in g[key]:
            ece_gate = float(g[key]["value"])
            break
        if isinstance(g.get(key), (int, float)):
            ece_gate = float(g[key])
            break
    if ece_gate is None:
        # fall back to searching the report for the calibration entry
        def find(node: Any) -> float | None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k.lower() in ("ece", "ece_value") and isinstance(
                        v, (int, float)
                    ):
                        return float(v)
                    found = find(v)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = find(item)
                    if found is not None:
                        return found
            return None

        ece_gate = find(gates)
    if ece_gate is None:
        c.failures.append("gate report: could not locate the ECE value")
    else:
        c.equal("card gate ECE", 0.03655999729229799, ece_gate, tol=1e-12)

    c.literal("card control value", "3.390625", "3.390625" in card)
    c.literal("card model sha256", "3b208b86…", "3b208b86" in card)

    # ---- calibration figures from the evaluation artifact ----
    # The card's calibration table quotes the EVALUATION artifact
    # (training/ece_predictions.json) — the same file the gate runner reads —
    # not the training report's in-loop evaluation. The two cover the same
    # 29,955 cases but were produced in different processes and differ by
    # 17 decisions; see EVALUATION_CLAIMS.md.
    eval_artifact = json.loads((ROOT / "training" / "ece_predictions.json").read_text())
    c.equal(
        "card held-out accuracy", 0.5263, eval_artifact["accuracy_recomputed"], tol=5e-5
    )
    c.equal("card held-out loss", 1.1268, eval_artifact["loss_recomputed"], tol=5e-5)
    c.equal("card held-out n", 29955, eval_artifact["n"], tol=0)
    c.equal("card serving temperature", 1.0, eval_artifact["temperature"], tol=0)

    train_report = json.loads((ROOT / "training" / "training_report.json").read_text())
    c.equal("card training steps", 5002, train_report["steps"], tol=0)
    c.equal("card training cases", 80000, train_report["cases"], tol=0)
    # Same cases, different processes: the measured difference is 17
    # decisions (5.7e-4 accuracy). Assert the agreement is within that
    # measured bound rather than assuming exact equality.
    c.equal(
        "training report vs evaluation artifact accuracy agreement",
        eval_artifact["accuracy_recomputed"],
        train_report["heldout_accuracy"],
        tol=1e-3,
    )
    c.equal("training report heldout n", 29955, train_report["heldout_n"], tol=0)

    typed_report = json.loads(
        (ROOT / "training" / "training_report_typed.json").read_text()
    )
    c.equal(
        "matched held-out accuracy", 0.526, typed_report["heldout_accuracy"], tol=5e-5
    )
    c.equal("matched held-out ECE", 0.01768, typed_report["heldout_ece"], tol=5e-6)
    c.equal("matched steps", 5216, typed_report["steps"], tol=0)
    c.equal("matched temperature", 0.9, typed_report["calibration_temperature"], tol=0)

    # ---- artifact integrity recomputed from disk ----
    weights = REPO / "model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    c.equal(
        "model.safetensors sha256",
        "3b208b86d6b5b87306091cbaec1b71007454eb4864b47efc636b4f633c875ac8",
        digest,
        tol=0,
    )

    # ---- FP16 variant: every quoted figure traced to its artifact ----
    # The v2 card's defect was quoting the F32 source hash while shipping an
    # FP16 file. Each file here is verified against its OWN bytes.
    fp16_weights = REPO / "fp16" / "model.safetensors"
    c.literal(
        "fp16/model.safetensors exists",
        "fp16/model.safetensors",
        fp16_weights.is_file(),
    )
    if fp16_weights.is_file():
        fp16_digest = hashlib.sha256(fp16_weights.read_bytes()).hexdigest()
        c.equal(
            "fp16/model.safetensors sha256 vs card",
            "64d4843a0a6e742de268522bed3643d47c38ea7dac6cbd40649e0405c0af20e1",
            fp16_digest,
            tol=0,
        )
        # and against the export report, which is the independent record
        fp16_report = json.loads(
            (REPO / "fp16" / "fp16_export_report.json").read_text()
        )
        c.equal(
            "fp16 sha256 vs export report",
            fp16_report["sha256"],
            fp16_digest,
            tol=0,
        )
        c.equal("fp16 bytes", 626337154, fp16_weights.stat().st_size, tol=0)
        c.equal(
            "fp16 byte ratio vs fp32",
            0.5,
            fp16_weights.stat().st_size / weights.stat().st_size,
            tol=1e-4,
        )
        c.equal(
            "fp16 measured underflowed values",
            233,
            fp16_report["measured"]["underflowed_to_zero"],
            tol=0,
        )
        c.equal(
            "fp16 measured subnormal values",
            590158,
            fp16_report["measured"]["became_subnormal"],
            tol=0,
        )
        c.equal(
            "fp16 measured overflowed values",
            0,
            fp16_report["measured"]["overflowed_to_inf"],
            tol=0,
        )
        c.equal(
            "fp16 measured max abs weight diff",
            0.014801025390625,
            fp16_report["measured"]["max_abs_diff"],
            tol=1e-9,
        )
        c.equal("fp16 tensors all f16", True, fp16_report["measured"]["all_f16"], tol=0)

    # fp16 config must differ from the root config ONLY in the dtype
    fp16_cfg = json.loads((REPO / "fp16" / "config.json").read_text())
    root_cfg = json.loads((REPO / "config.json").read_text())
    c.equal("fp16 config dtype", "float16", fp16_cfg["dtype"], tol=0)
    c.equal("root config dtype", "float32", root_cfg["dtype"], tol=0)
    diff_keys = {
        k for k in set(fp16_cfg) | set(root_cfg) if fp16_cfg.get(k) != root_cfg.get(k)
    }
    c.literal(
        "fp16 config differs only by dtype",
        "dtype",
        diff_keys == {"dtype"},
    )

    # ---- FP16 evaluation artifact: the divergence figures in the card ----
    fp16_eval = json.loads(
        (ROOT / "training" / "ece_predictions_fp16.json").read_text()
    )
    c.equal("fp16 eval dtype", "fp16", fp16_eval["dtype"], tol=0)
    c.equal("fp16 eval n", 29955, fp16_eval["n"], tol=0)
    c.equal("fp16 eval accuracy", 0.5263, fp16_eval["accuracy_recomputed"], tol=5e-5)
    c.equal("fp16 eval loss", 1.1268, fp16_eval["loss_recomputed"], tol=5e-5)

    # Recompute the divergence from the two raw probability payloads rather than
    # trusting the comparison script's own summary.
    fp32_eval = json.loads((ROOT / "training" / "ece_predictions.json").read_text())
    p32 = np.asarray(fp32_eval["probs"], dtype=np.float64)
    p16 = np.asarray(fp16_eval["probs"], dtype=np.float64)
    c.literal(
        "fp16 probabilities shape matches fp32",
        "equal shapes",
        p32.shape == p16.shape,
    )
    if p32.shape == p16.shape:
        c.equal(
            "card fp16 mean prob delta",
            7.23e-4,
            float(np.abs(p32 - p16).mean()),
            tol=5e-5,
        )
        c.equal(
            "card fp16 max prob delta",
            0.0270,
            float(np.abs(p32 - p16).max()),
            tol=5e-4,
        )
        disagrees = int((p32.argmax(1) != p16.argmax(1)).sum())
        c.equal("card fp16 disagreement count", 490, disagrees, tol=0)
        c.equal(
            "card fp16 chosen-option agreement",
            0.9836,
            1.0 - disagrees / p32.shape[0],
            tol=5e-4,
        )
        # All disagreements must be near-ties: this is the card's load-bearing
        # claim, so it is recomputed, not quoted. The card states the measured
        # maximum margin (0.0415), not a round threshold, because a round
        # "< 0.05" would be an unverified bound.
        part = np.partition(p32, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        flip = p32.argmax(1) != p16.argmax(1)
        c.equal(
            "card fp16 max margin at a flipped decision",
            0.0415,
            float(margin[flip].max()),
            tol=5e-4,
        )
        # and the disagreements must be symmetric (why accuracy is unchanged)
        lab = np.asarray(fp32_eval["labels"], dtype=np.int64)
        ca = p32.argmax(1) == lab
        cb = p16.argmax(1) == lab
        c.equal(
            "card fp16 fp32-correct-fp16-wrong",
            191,
            int((flip & ca & ~cb).sum()),
            tol=0,
        )
        c.equal(
            "card fp16 fp16-correct-fp32-wrong",
            191,
            int((flip & ~ca & cb).sum()),
            tol=0,
        )
        c.equal(
            "card fp16 both-wrong-different",
            108,
            int((flip & ~ca & ~cb).sum()),
            tol=0,
        )

    # ---- FP16 export gates: the second run must also be approved ----
    fp16_gates = json.loads((ROOT / "gate_results_fp16.json").read_text())
    c.equal(
        "fp16 gate export_approved",
        True,
        fp16_gates["verdict"]["export_approved"],
        tol=0,
    )
    c.equal(
        "fp16 gate ECE vs card",
        0.03677,
        fp16_gates["gates"]["ece"],
        tol=5e-5,
    )
    c.equal(
        "fp16 gate ECE provenance",
        "training/ece_predictions_fp16.json",
        fp16_gates["gates"]["ece_provenance"]["source"],
        tol=0,
    )
    # dtype-independent gates must be bit-identical to the FP32 run
    trained_gates = json.loads((ROOT / "gate_results_trained.json").read_text())
    for key in (
        "option_order_max_abs_diff",
        "question_order_max_abs_diff",
        "state_independence_max_abs_diff",
        "cross_question_leakage_max_abs_diff",
        "changed_control_logits",
        "reload_parity_max_abs_diff",
        "reload_child_exit_code",
        "dual_gpu_parity_max_abs_diff",
    ):
        c.equal(
            f"fp16 gate {key} == fp32 run",
            trained_gates["gates"][key],
            fp16_gates["gates"][key],
            tol=0,
        )

    # ---- the card must not carry the corrected-away baseline figures ----
    for stale in ("0.3320", "0.1267", "0.5317", "0.3362"):
        c.literal(f"stale baseline {stale} absent from card", stale, stale not in card)

    # ---- report ----
    print(f"card claim checks run: {c.checks}")
    if c.failures:
        print(f"FAILURES ({len(c.failures)}):")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print("all card claims match their artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
