"""Probe whether the held-out evaluation is bit-reproducible run to run.

Two evaluations of the SAME checkpoint on the SAME split produced accuracies
that differ on 17 of 29,955 decisions. This probe characterizes the
magnitude: it evaluates the same subset in two separate processes and
compares the per-decision outcomes.

Run: PYTHONPATH=src .venv/bin/python probe_eval_determinism.py --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from train_jev_distill import BACKBONE, evaluate, read_cases
from transformers import AutoTokenizer

from rockandegg_v3 import HeadConfig, RockAndEggV3SystemOne


def main() -> None:
    """Evaluate a fixed subset and dump per-decision correctness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", type=int, default=4000)
    parser.add_argument("--checkpoint", default="training/head_final.pt")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = RockAndEggV3SystemOne(HeadConfig(dtype_name="fp32")).to("cuda:0").eval()
    state = model.state_dict()
    state.update(checkpoint["head"])
    model.load_state_dict(state, strict=True)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    eval_args = type(
        "Args", (), {"state_len": 256, "option_len": 64, "eval_batch": 8}
    )()
    cases = read_cases("test_set_30k")[: args.cases]
    metrics = evaluate(model, cases, tokenizer, "cuda:0", eval_args)  # type: ignore[arg-type]

    probs = metrics["_probs"]
    labels = metrics["_labels"]
    correct = [
        int(int(torch.tensor(p).argmax()) == lab)
        for p, lab in zip(probs, labels, strict=True)
    ]
    Path(args.out).write_text(
        json.dumps(
            {
                "n": len(correct),
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"],
                "correct": correct,
                "probs_checksum": sum(round(sum(p), 9) for p in probs[:64]),
            }
        )
    )
    print(
        f"n={len(correct)} accuracy={metrics['accuracy']:.6f} "
        f"probs_checksum={sum(round(sum(p), 9) for p in probs[:64]):.9f}"
    )


if __name__ == "__main__":
    main()
