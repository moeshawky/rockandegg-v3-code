"""Run the export gates against the real checkpoint on GPU (fp16 on T4).

Run: PYTHONPATH=src .venv/bin/python run_systemone_gates.py > gate_results.json

This is the executable form of prompt_v3_spec.md section "Export gates". It
reuses the mechanism verified by gpu_verify.py but drives the packaged model.
ECE is reported as NOT EVALUATED until held-out typed labels are supplied; no
calibration gate is silently promoted to a pass.

Permutation probes physically reorder option rows. The model slices rows by
contiguous per-question counts, so relabeling group metadata alone would
permute the comparison, not the model input (a placebo probe).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from rockandegg_v3 import (
    GateReport,
    HeadConfig,
    RockAndEggV3SystemOne,
    available_devices,
    build_replicas,
    compute_ece,
    forward_case_multi,
    serving_probs,
    structural_gate,
)

MODEL = os.environ.get("RAE_BACKBONE", "moeshawky/RockAndEgg-313M-v2")
DEV = "cuda:0"
S = 512
SEED = 20261116


def make_inputs(
    tok: Any, options_per_q: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], torch.Tensor]:
    """Build a fixed-length state and isolated option rows."""
    state_text = "The customer asked about billing and refunds. " * 40
    enc_state = tok(
        state_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=S,
    )
    state_ids = enc_state["input_ids"].to(DEV)
    state_mask = enc_state["attention_mask"].to(DEV)
    if state_ids.shape[1] < S:
        pad = torch.full((1, S - state_ids.shape[1]), tok.eos_token_id, device=DEV)
        state_ids = torch.cat([state_ids, pad], dim=1)
        state_mask = torch.cat(
            [
                state_mask,
                torch.zeros(
                    (1, S - state_mask.shape[1]), dtype=state_mask.dtype, device=DEV
                ),
            ],
            dim=1,
        )

    suffix_texts: list[str] = []
    groups: list[list[int]] = []
    for q, n in enumerate(options_per_q):
        groups.append([])
        for j in range(n):
            suffix_texts.append(
                f"choice: Which team should handle this? [SEP] option_{j}: "
                f"billing refunds technical outages account access {q}. [SEP]"
            )
            groups[-1].append(len(suffix_texts) - 1)
    enc = tok(suffix_texts, add_special_tokens=False, padding=True, return_tensors="pt")
    return (
        state_ids,
        enc["input_ids"].to(DEV),
        enc["attention_mask"].to(DEV),
        groups,
        state_mask,
    )


def run_case(
    model: RockAndEggV3SystemOne,
    state_ids: torch.Tensor,
    suf_ids: torch.Tensor,
    suf_mask: torch.Tensor,
    counts: list[int],
    state_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """Run one full case through the packaged model."""
    with torch.inference_mode():
        return model.forward_case(
            state_ids, suf_ids, suf_mask, counts, [0] * len(counts), state_mask
        )


def reorder_rows(
    suf_ids: torch.Tensor,
    suf_mask: torch.Tensor,
    groups: list[list[int]],
    question_order: list[int],
    seed: int,
    permute_within: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]]]:
    """Physically reorder option rows into a new question/option order.

    Returns the new ids/mask, the position of each original question in the
    new order, and a per-question map from original row id to new row id.
    """
    generator = torch.Generator().manual_seed(seed)
    new_ids = torch.zeros_like(suf_ids)
    new_mask = torch.zeros_like(suf_mask)
    question_pos = [0] * len(groups)
    row_maps: list[dict[int, int]] = [{} for _ in groups]
    offset = 0
    for pos, qi in enumerate(question_order):
        grp = groups[qi]
        n = len(grp)
        order = (
            torch.randperm(n, generator=generator).tolist()
            if permute_within
            else list(range(n))
        )
        question_pos[qi] = pos
        for j, k in enumerate(order):
            original_row = grp[k]
            destination = offset + j
            new_ids[destination] = suf_ids[original_row]
            new_mask[destination] = suf_mask[original_row]
            row_maps[qi][original_row] = j  # local index within this question
        offset += n
    return new_ids, new_mask, question_pos, row_maps


def maxdiff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Maximum absolute difference between two per-question logit lists."""
    return max(
        float((x.float() - y.float()).abs().max()) for x, y in zip(a, b, strict=True)
    )


def main() -> None:
    """Execute all five gates and emit a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-checkpoint",
        type=str,
        default="",
        help="Optional trained head checkpoint to load before running the gates.",
    )
    parser.add_argument(
        "--ece-predictions",
        type=str,
        default="",
        help=(
            "JSON file with held-out 'probs' [[float,...]] (saved softmax(logits)), "
            "'labels' [int,...] and optional serving 'temperature'. ECE is "
            "recomputed on the distribution serving actually emits — "
            "softmax(logits/T), a SINGLE softmax — never a double softmax. A "
            "self-reported ECE number is never accepted."
        ),
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = HeadConfig()
    model = RockAndEggV3SystemOne(cfg).to(DEV).eval()
    if args.head_checkpoint:
        payload = torch.load(args.head_checkpoint, map_location=DEV, weights_only=True)
        state = model.state_dict()
        state.update(payload["head"])
        model.load_state_dict(state, strict=True)

    state_ids, suf_ids, suf_mask, groups, state_mask = make_inputs(tok, [3, 2, 4])
    counts = [len(g) for g in groups]

    # Gate 1: option-order invariance (rows physically permuted per question).
    base = run_case(model, state_ids, suf_ids, suf_mask, counts, state_mask)
    perm_ids, perm_mask, _, perm_maps = reorder_rows(
        suf_ids, suf_mask, groups, [0, 1, 2], SEED + 1, True
    )
    perm = run_case(model, state_ids, perm_ids, perm_mask, counts, state_mask)
    restored = [
        torch.stack([lg[perm_maps[qi][row]] for row in grp])
        for qi, (grp, lg) in enumerate(zip(groups, perm, strict=True))
    ]
    option_diff = maxdiff(base, restored)

    # Gate 2a: question-order invariance (whole contiguous blocks reordered).
    order = [2, 1, 0]
    qrev_ids, qrev_mask, qrev_pos, _ = reorder_rows(
        suf_ids, suf_mask, groups, order, SEED + 2, False
    )
    qrev = run_case(
        model,
        state_ids,
        qrev_ids,
        qrev_mask,
        [len(groups[qi]) for qi in order],
        state_mask,
    )
    qrestored = [qrev[qrev_pos[qi]] for qi in range(len(groups))]
    question_diff = maxdiff(base, qrestored)

    # Gate 2b: state independence, cross-question leakage, moving control.
    with torch.inference_mode():
        state_a = model.encode_state(state_ids, state_mask)[0]
    alt_ids = suf_ids.clone()
    for row in groups[0]:
        alt_ids[row] = suf_ids[row].flip(-1)
    alt = run_case(model, state_ids, alt_ids, suf_mask, counts, state_mask)
    with torch.inference_mode():
        state_b = model.encode_state(state_ids, state_mask)[0]
    state_diff = float((state_a - state_b).abs().max())
    leakage = maxdiff(base[1:], alt[1:])
    control = float((base[0] - alt[0]).abs().max())

    # Gate 3: fresh-process reload parity (same-process reload is insufficient).
    reload_diff: float | None = None
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "head.pt"
        torch.save(model.state_dict(), checkpoint)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "src")
        child = subprocess.run(  # noqa: S603 - fixed local interpreter + script
            [sys.executable, "reload_parity_child.py", str(checkpoint)],
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        child_rc = child.returncode
        child_stderr = child.stderr.strip()[-400:]
        if child_rc == 0 and child.stdout.strip():
            payload = json.loads(child.stdout.strip().splitlines()[-1])
            reloaded = [torch.tensor(x, device=DEV) for x in payload]
            reload_diff = maxdiff(base, reloaded)

    # Gate 4: ECE is recomputed from held-out probabilities and labels; a
    # self-reported ECE number is never accepted as evidence.
    ece: float | None = None
    ece_provenance: dict[str, Any] = {
        "status": "NOT EVALUATED: no held-out typed decision labels"
    }
    if args.ece_predictions:
        pred_path = Path(args.ece_predictions)
        pred = json.loads(pred_path.read_text())
        probs = torch.tensor(pred["probs"], dtype=torch.float32)
        labels = torch.tensor(pred["labels"], dtype=torch.long)
        temp = float(pred.get("temperature", 1.0))
        # Serving applies temperature to RAW LOGITS then one softmax
        # (SystemOneInference: softmax(logits / T)). The artifact stores
        # softmax(logits), so softmax(probs / T) is a double softmax that the
        # model never emits. serving_probs recovers the emitted distribution.
        emitted = serving_probs(probs, temp)
        ece = compute_ece(emitted, labels)
        ece_provenance = {
            "status": "MEASURED from held-out predictions on the serving path",
            "source": str(pred_path),
            "split": pred.get("split"),
            "temperature": temp,
            "temperature_source": pred.get("temperature_source", "artifact default"),
            "formula": "softmax(log(saved_probs)/T) == softmax(logits/T); single softmax",
            "n": int(labels.numel()),
        }

    # Optional dual-GPU parity: state encoded once, cache copied to peer.
    devices = available_devices()
    multi_parity: float | None = None
    if len(devices) > 1:
        replicas = build_replicas(cfg, model_path=MODEL, devices=devices)
        single = run_case(replicas[0], state_ids, suf_ids, suf_mask, counts, state_mask)
        with torch.inference_mode():
            multi = forward_case_multi(
                replicas,
                devices,
                state_ids,
                suf_ids,
                suf_mask,
                counts,
                [0] * len(counts),
                state_mask,
            )
        multi_parity = maxdiff(single, multi)
        del replicas
        torch.cuda.empty_cache()

    no_lm_head, no_generation, arch = structural_gate(model)
    report = GateReport(
        option_order_max_abs_diff=option_diff,
        question_order_max_abs_diff=question_diff,
        state_independence_max_abs_diff=state_diff,
        cross_question_leakage_max_abs_diff=leakage,
        reload_parity_max_abs_diff=reload_diff,
        ece=ece,
        ece_bound=cfg.ece_bound,
        order_tol=cfg.order_tol,
        no_lm_head=no_lm_head,
        no_generation=no_generation,
        backbone_arch=arch,
        control_moved_logits=control > 0.0,
    )
    payload = {
        "config": {
            "dtype": cfg.dtype_name,
            "freeze_backbone": cfg.freeze_backbone,
            "use_cross_attn": cfg.use_cross_attn,
            "use_option_mixer": cfg.use_option_mixer,
            "use_cosine_prior": cfg.use_cosine_prior,
            "head_checkpoint": args.head_checkpoint or None,
        },
        "gates": {
            "option_order_max_abs_diff": option_diff,
            "question_order_max_abs_diff": question_diff,
            "state_independence_max_abs_diff": state_diff,
            "cross_question_leakage_max_abs_diff": leakage,
            "changed_control_logits": control,
            "reload_parity_max_abs_diff": reload_diff,
            "reload_child_exit_code": child_rc,
            "reload_child_stderr": child_stderr,
            "dual_gpu_parity_max_abs_diff": multi_parity,
            "devices": devices,
            "ece": ece,
            "ece_provenance": ece_provenance,
        },
        "structural": {
            "no_lm_head": no_lm_head,
            "no_generation": no_generation,
            "backbone_arch": arch,
        },
        "verdict": {
            "numeric_passed": report.numeric_passed(),
            "structural_passed": report.structural_passed(),
            "reload_passed": report.reload_passed(),
            "calibration_passed": report.calibration_passed(),
            "passed_without_holdout": report.passed(require_all=False),
            "passed_full_export": report.passed(require_all=True),
            # Explicit and unambiguous: an unmeasured ECE means no export.
            # Added after dialectical challenge (a "passed_*" true field was
            # readable as approval even though full export was blocked).
            "export_approved": report.passed(require_all=True),
            # Separate diagnostic, NOT one of the five export gates. Measured
            # fp16 split path differs by kernel-shape nondeterminism (one-device
            # parity is exactly 0.0; fp32 split is ~2.7e-6). Do not fold this
            # into the 1e-6 invariance tolerance.
            "dual_gpu_within_order_tol": (
                multi_parity <= cfg.order_tol if multi_parity is not None else None
            ),
            "dual_gpu_measured_tolerance": 1e-2,
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
