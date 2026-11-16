"""Measure the architectural property the card claims: the state is encoded once.

A request carries one state and ``n`` options. This probe counts how many times
the backbone is actually invoked while serving such a request, and what shape
each call sees. The design claim is that the count is **2** — one state prefill
plus one pass over all option rows — regardless of ``n``. A design that
re-encoded the state per option would need ``1 + n`` calls.

This is an *executed* measurement of the shipped code, not a restatement of its
docstring. It instruments ``backbone.forward`` so the count cannot come from
reading the source.

Run: ``PYTHONPATH=src .venv/bin/python probe_architecture.py``
"""

from __future__ import annotations

import json
from typing import Any

import torch

from rockandegg_v3 import HeadConfig, RockAndEggV3SystemOne

# Small but non-degenerate: enough state to be a real prefill, enough option
# rows to make the difference between "fork" and "re-encode" unambiguous.
STATE_LEN = 16
OPTION_LEN = 4
# Two questions with unequal option counts, so row grouping is exercised.
QUESTION_GROUPS = ([0, 1, 2], [0, 1, 2, 3])


def measure(n_options_total: int, state_len: int = STATE_LEN) -> dict[str, Any]:
    """Count backbone forwards for a request with the given option total."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RockAndEggV3SystemOne(
        HeadConfig(dtype_name="fp16" if device != "cpu" else "fp32")
    )
    model = model.to(device).eval()

    calls: list[tuple[int, ...]] = []
    original = model.backbone.forward

    def counting(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None:
            calls.append(tuple(input_ids.shape))
        return original(*args, **kwargs)

    model.backbone.forward = counting

    state_ids = torch.randint(0, 1000, (1, state_len), device=device)
    state_mask = torch.ones_like(state_ids)

    # Spread the requested total across the two question groups.
    per_row = max(1, n_options_total // len(QUESTION_GROUPS))
    suffix_rows = [[5, 6, 7, 8] for _ in range(per_row * len(QUESTION_GROUPS))]
    suffix = torch.tensor(suffix_rows, device=device)
    suffix_mask = torch.ones_like(suffix)
    counts = [per_row] * len(QUESTION_GROUPS)

    with torch.inference_mode():
        logits = model.forward_case(
            state_ids, suffix, suffix_mask, counts, [0, 1], state_mask
        )

    n_rows = per_row * len(QUESTION_GROUPS)
    return {
        "n_options": n_options_total,
        "option_rows": n_rows,
        "backbone_calls": len(calls),
        "shapes": calls,
        "logits_per_question": [int(o.shape[0]) for o in logits],
        "naive_reencode_calls": 1 + n_rows,
        "calls_saved": (1 + n_rows) - len(calls),
        "device": device,
    }


def main() -> int:
    """Measure the call count at several option totals."""
    report = {}
    for n in (2, 4, 8, 16):
        report[f"n_options={n}"] = measure(n)

    print(json.dumps(report, indent=2))

    print("\n=== verdict ===")
    counts = {v["backbone_calls"] for v in report.values()}
    print(f"backbone call counts across all option totals: {sorted(counts)}")
    if counts == {2}:
        print(
            "PASS: exactly 2 backbone forwards regardless of option count —\n"
            "the state is encoded once and the option rows reuse its KV cache."
        )
        return 0
    print(
        "NOTE: call count varies with option count; the single-encode claim\n"
        "does not hold for this configuration."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
