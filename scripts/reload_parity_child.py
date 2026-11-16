"""Fresh-process child for reload-parity gate 3.

Run by run_systemone_gates.py with a path to a saved state_dict. It builds a
brand-new model in a brand-new interpreter, loads the weights, regenerates the
same deterministic inputs, and prints logits as JSON on stdout. Keeping stdout
clean except for the final JSON line lets the parent preserve stream
provenance and the producer exit status.

Run: PYTHONPATH=src .venv/bin/python reload_parity_child.py <state_dict.pt>
"""

from __future__ import annotations

import json
import sys

import torch
from run_systemone_gates import DEV, MODEL, SEED, make_inputs, run_case
from transformers import AutoTokenizer

from rockandegg_v3 import HeadConfig, RockAndEggV3SystemOne


def main() -> None:
    """Load weights in a fresh process and emit logits for the fixed case."""
    if len(sys.argv) != 2:
        msg = "usage: reload_parity_child.py <state_dict.pt>"
        raise SystemExit(msg)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = RockAndEggV3SystemOne(HeadConfig()).to(DEV).eval()
    state = torch.load(sys.argv[1], map_location=DEV, weights_only=True)
    model.load_state_dict(state)

    state_ids, suffix_ids, suffix_mask, groups, state_mask = make_inputs(
        tokenizer, [3, 2, 4]
    )
    logits = run_case(
        model,
        state_ids,
        suffix_ids,
        suffix_mask,
        [len(g) for g in groups],
        state_mask,
    )
    print(json.dumps([tensor.float().cpu().tolist() for tensor in logits]))


if __name__ == "__main__":
    main()
