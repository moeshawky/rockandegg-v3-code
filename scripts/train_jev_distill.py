"""Final SystemOne-head finetune on SargeDev/jev-distill-corpus-v3.

Trains only the v3 decision head on a frozen Qwen3 backbone using the release
objective: soft-label cross-entropy + squared EMD on score questions + per-type
temperature scaling + option dropout 0.3. No RLCD, no REINFORCE, no LM head,
no generation path. The state is encoded exactly once per case into a
DynamicCache and forked across option rows; options stay isolated in batch rows.

Two Tesla T4s are used with manual data parallelism: each replica processes a
batch of cases in one forward (option rows padded to a bucket width), head
gradients are averaged, the primary optimizer steps, and head weights are
re-synced to the peer. Backbone weights are frozen and identical on both
devices. Case batching raised measured throughput from ~18 to ~31 cases/s and
put both GPUs to work. Evaluation uses the same batched forward with a smaller
evaluation batch to stay within T4 memory.

Run:
    PYTHONPATH=src .venv/bin/python train_jev_distill.py --cases 80000

Evaluation uses the dataset's dedicated `calibration` split for temperature
fitting and its leakage-checked `test_set_30k` split for the single final ECE
read. Dataset credit: SargeDev/jev-distill-corpus-v3 (Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from rockandegg_v3 import (
    QTYPE_CHOICE,
    QTYPE_NOUL,
    QTYPE_SCORE,
    HeadConfig,
    RockAndEggV3SystemOne,
)
from rockandegg_v3.gates import compute_ece, serving_probs
from rockandegg_v3.loss import option_dropout, verdict_pact_loss

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "jev_distill_v3"
OUT = ROOT / "training"
BACKBONE = os.environ.get("RAE_BACKBONE", "moeshawky/RockAndEgg-313M-v2")
QTYPE_BY_KIND = {"choice": QTYPE_CHOICE, "score": QTYPE_SCORE, "noul": QTYPE_NOUL}


@dataclass(frozen=True)
class Case:
    """One typed-decision case in the shipped corpus schema."""

    case_id: str
    kind: str
    options: list[str]
    target: list[float]
    state: str
    question: str


@dataclass(frozen=True)
class EncodedCase:
    """Tokenized tensors for a single case, already on its target device."""

    state_ids: torch.Tensor
    state_mask: torch.Tensor
    suffix_ids: torch.Tensor
    suffix_mask: torch.Tensor
    counts: list[int]
    qtype_ids: list[int]
    target: torch.Tensor
    gold: int


@dataclass(frozen=True)
class EncodedBatch:
    """B cases padded to a common option width for one batched forward."""

    state_ids: torch.Tensor
    state_mask: torch.Tensor
    suffix_ids: torch.Tensor
    suffix_mask: torch.Tensor
    rows_per_case: int
    real_counts: list[int]
    qtype_ids: list[int]
    targets: torch.Tensor
    golds: list[int]


def option_buckets(
    order: list[int], cases: list[Case], bucket_size: int
) -> list[list[int]]:
    """Group indices by option count so padding stays minimal."""
    groups: dict[int, list[int]] = {}
    for idx in order:
        groups.setdefault(len(cases[idx].options), []).append(idx)
    batches: list[list[int]] = []
    for count in sorted(groups):
        pool = groups[count]
        for start in range(0, len(pool), bucket_size):
            batches.append(pool[start : start + bucket_size])
    return batches


def epoch_step_plan(
    epoch_batches: list[list[int]],
    n_devices: int,
    start_offset: int = 0,
) -> list[list[list[int]]]:
    """Split one epoch's batches into optimizer steps, one batch per replica.

    Each optimizer step consumes up to ``n_devices`` batches (one per
    data-parallel replica). The plan is deliberately a pure function of the
    epoch's batches so that the step arithmetic can be unit-tested without a
    GPU: a plan built from a *global* step counter instead of a per-epoch
    offset silently truncates every epoch after the first.

    Args:
        epoch_batches: the epoch's batches, already bucketed by option count.
        n_devices: number of data-parallel replicas.
        start_offset: batch index to resume from within this epoch (0 for a
            fresh epoch).

    Returns:
        The list of optimizer steps; each step is a list of per-device
        batches.
    """
    if n_devices < 1:
        msg = f"n_devices must be >= 1, got {n_devices}"
        raise ValueError(msg)
    return [
        epoch_batches[offset : offset + n_devices]
        for offset in range(start_offset, len(epoch_batches), n_devices)
    ]


def encode_batch(
    indices: list[int],
    cases: list[Case],
    tokenizer: Any,
    device: str,
    max_state_len: int,
    max_option_len: int,
) -> EncodedBatch:
    """Encode B cases into one padded state batch and one padded option batch."""
    items = [cases[i] for i in indices]
    width = max(len(c.options) for c in items)
    state_ids: list[torch.Tensor] = []
    state_masks: list[torch.Tensor] = []
    suffix_rows: list[list[int]] = []
    suffix_masks: list[list[int]] = []
    row_width = 0
    targets: list[list[float]] = []
    pad_id = tokenizer.eos_token_id
    marker_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for case in items:
        enc = tokenizer(
            case.state,
            add_special_tokens=False,
            truncation=True,
            max_length=max_state_len,
            return_tensors="pt",
        )
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] < max_state_len:
            n = max_state_len - ids.shape[1]
            ids = torch.cat([ids, torch.full((1, n), pad_id, dtype=ids.dtype)], dim=1)
            mask = torch.cat([mask, torch.zeros((1, n), dtype=mask.dtype)], dim=1)
        state_ids.append(ids[0])
        state_masks.append(mask[0])

        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for option in case.options:
            text = f"{case.question} [SEP] {option} [SEP]"
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_option_len - 1),
            )["input_ids"]
            row = [*tokens, marker_id]
            rows.append(row)
            masks.append([1] * len(row))
        row_width = max(row_width, *(len(r) for r in rows))
        suffix_rows.extend(rows)
        suffix_masks.extend(masks)
        for _ in range(width - len(rows)):
            suffix_rows.append([])
            suffix_masks.append([])
        padded_target = list(case.target) + [0.0] * (width - len(case.target))
        targets.append(padded_target)

    suffix_rows = [r + [pad_id] * (row_width - len(r)) for r in suffix_rows]
    suffix_masks = [m + [0] * (row_width - len(m)) for m in suffix_masks]
    return EncodedBatch(
        state_ids=torch.stack(state_ids).to(device),
        state_mask=torch.stack(state_masks).to(device),
        suffix_ids=torch.tensor(suffix_rows, dtype=torch.long, device=device),
        suffix_mask=torch.tensor(suffix_masks, dtype=torch.long, device=device),
        rows_per_case=width,
        real_counts=[len(c.options) for c in items],
        qtype_ids=[QTYPE_BY_KIND[c.kind] for c in items],
        targets=torch.tensor(targets, dtype=torch.float32, device=device),
        golds=[
            int(max(range(len(c.target)), key=lambda i: c.target[i])) for c in items
        ],
    )


def read_cases(split: str) -> list[Case]:
    """Read one jsonl split into typed cases."""
    path = DATA / f"{split}.jsonl"
    cases: list[Case] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            cases.append(
                Case(
                    case_id=str(row["id"]),
                    kind=str(row["kind"]),
                    options=[str(o) for o in row["options"]],
                    target=[float(p) for p in row["target"]],
                    state=str(row["state"]),
                    question=str(row["question"]),
                )
            )
    return cases


def stratified_sample(cases: list[Case], n: int, seed: int) -> list[Case]:
    """Sample n cases preserving the corpus kind distribution."""
    rng = random.Random(seed)  # noqa: S311 - deterministic ML sampling, not security
    by_kind: dict[str, list[Case]] = {}
    for case in cases:
        by_kind.setdefault(case.kind, []).append(case)
    out: list[Case] = []
    for _kind, pool in sorted(by_kind.items()):
        take = max(1, round(n * len(pool) / len(cases)))
        out.extend(rng.sample(pool, min(take, len(pool))))
    rng.shuffle(out)
    return out[:n]


def encode_case(
    case: Case,
    tokenizer: Any,
    device: str,
    max_state_len: int,
    max_option_len: int,
) -> EncodedCase:
    """Tokenize one case: a single state row and one row per option."""
    enc = tokenizer(
        case.state,
        add_special_tokens=False,
        truncation=True,
        max_length=max_state_len,
        return_tensors="pt",
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    if ids.shape[1] < max_state_len:
        pad_n = max_state_len - ids.shape[1]
        pad_id = tokenizer.eos_token_id
        ids = torch.cat([ids, torch.full((1, pad_n), pad_id, dtype=ids.dtype)], dim=1)
        mask = torch.cat([mask, torch.zeros((1, pad_n), dtype=mask.dtype)], dim=1)

    marker_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    rows: list[list[int]] = []
    for option in case.options:
        text = f"{case.question} [SEP] {option} [SEP]"
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_option_len - 1),
        )["input_ids"]
        rows.append([*tokens, marker_id])
    width = max(len(row) for row in rows)
    pad_id = tokenizer.eos_token_id
    suffix_ids = torch.tensor(
        [row + [pad_id] * (width - len(row)) for row in rows], dtype=torch.long
    )
    suffix_mask = torch.tensor(
        [[1] * len(row) + [0] * (width - len(row)) for row in rows],
        dtype=torch.long,
    )
    target = torch.tensor(case.target, dtype=torch.float32)
    return EncodedCase(
        state_ids=ids.to(device),
        state_mask=mask.to(device),
        suffix_ids=suffix_ids.to(device),
        suffix_mask=suffix_mask.to(device),
        counts=[len(case.options)],
        qtype_ids=[QTYPE_BY_KIND[case.kind]],
        target=target.to(device),
        gold=int(target.argmax().item()),
    )


def head_parameters(model: RockAndEggV3SystemOne) -> list[torch.nn.Parameter]:
    """Return only trainable parameters (the decision head)."""
    return [p for p in model.parameters() if p.requires_grad]


def sync_head_weights(
    source: RockAndEggV3SystemOne, targets: list[RockAndEggV3SystemOne]
) -> None:
    """Copy head weights from the primary replica to every peer."""
    src = dict(source.named_parameters())
    for target in targets:
        dst = dict(target.named_parameters())
        with torch.no_grad():
            for name, param in src.items():
                if param.requires_grad:
                    dst[name].copy_(param)


def average_head_gradients(
    models: list[RockAndEggV3SystemOne],
) -> None:
    """Average head gradients across replicas, in place on every replica."""
    primary = head_parameters(models[0])
    peers = [head_parameters(m) for m in models[1:]]
    n = len(models)
    for index, param in enumerate(primary):
        if param.grad is None:
            continue
        acc = param.grad.clone()
        for peer in peers:
            grad = peer[index].grad
            if grad is not None:
                acc += grad.to(param.device)
        acc /= n
        param.grad.copy_(acc)
        for peer in peers:
            if peer[index].grad is not None:
                peer[index].grad.copy_(acc.to(peer[index].device))


def cosine_lr(step: int, total: int, base_lr: float, warmup: float = 0.1) -> float:
    """Linear warmup then cosine decay to 10% of the base learning rate."""
    warm = max(1, int(total * warmup))
    if step < warm:
        return base_lr * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(
    model: RockAndEggV3SystemOne, path: Path, step: int, rng: dict[str, Any]
) -> None:
    """Atomically save head weights plus resume state."""
    payload = {
        "head": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if not k.startswith("backbone.")
        },
        "step": step,
        "rng": rng,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def evaluate(
    model: RockAndEggV3SystemOne,
    cases: list[Case],
    tokenizer: Any,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Measure soft loss and accuracy; collect raw probabilities for calibration.

    Evaluation is batched by option count with the same case-batched forward as
    training, so the held-out read takes seconds instead of tens of minutes.
    """
    model.eval()
    losses: list[float] = []
    correct = 0
    probs_all: list[list[float]] = []
    labels: list[int] = []
    batch_size = getattr(args, "eval_batch", 8)
    batches = option_buckets(list(range(len(cases))), cases, batch_size)
    for indices in batches:
        enc = encode_batch(
            indices, cases, tokenizer, device, args.state_len, args.option_len
        )
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=torch.float16, enabled=device != "cpu"),
        ):
            logits = model.forward_batch(
                enc.state_ids,
                enc.state_mask,
                enc.suffix_ids,
                enc.suffix_mask,
                enc.rows_per_case,
                enc.real_counts,
                enc.qtype_ids,
            )
        for i, _idx in enumerate(indices):
            n = enc.real_counts[i]
            case_logits = logits[i, :n].float()
            losses.append(
                float(
                    verdict_pact_loss(
                        case_logits, enc.targets[i, :n], enc.qtype_ids[i]
                    ).item()
                )
            )
            probs = F.softmax(case_logits, dim=-1).tolist()
            probs_all.append(probs)
            labels.append(enc.golds[i])
            correct += int(int(torch.tensor(probs).argmax()) == enc.golds[i])
    # Questions have different option counts. Pad with zeros before stacking:
    # zero-padding cannot change the argmax or the maximum probability, so the
    # ECE/accuracy read is unaffected while compute_ece gets a valid [N, C].
    width = max(len(p) for p in probs_all)
    padded = [p + [0.0] * (width - len(p)) for p in probs_all]
    return {
        "n": float(len(cases)),
        "loss": sum(losses) / len(losses),
        "accuracy": correct / len(cases),
        "_probs": padded,
        "_labels": labels,
    }


def fit_temperature(probs: list[list[float]], labels: list[int]) -> float:
    """Choose one scalar temperature minimizing ECE on the calibration split.

    The artifact stores ``softmax(logits)`` and the serving path reports
    ``softmax(logits / T)``: temperature is applied to raw logits, then ONE
    softmax. :func:`serving_probs` recovers that distribution from the saved
    probabilities. Applying ``softmax(probs / T)`` instead would be a double
    softmax over already-softmaxed values and would measure a distribution the
    model never emits.
    """
    best_temp, best_ece = 1.0, float("inf")
    probs_t = torch.tensor(probs, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    for temp in [0.5 + 0.05 * i for i in range(31)]:
        ece = compute_ece(serving_probs(probs_t, temp), labels_t)
        if ece < best_ece:
            best_temp, best_ece = round(temp, 3), ece
    return best_temp


def main() -> None:
    """Run the finetune, calibrate, evaluate once on the held-out split."""
    global DATA, OUT, BACKBONE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=80000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--micro", type=int, default=8, help="cases per GPU per step")
    parser.add_argument(
        "--resume", type=str, default="", help="head checkpoint to resume from"
    )
    parser.add_argument(
        "--resume-step",
        type=int,
        default=-1,
        help="override the saved step index when the batch size changes",
    )
    parser.add_argument("--state-len", type=int, default=256)
    parser.add_argument(
        "--dataset",
        type=str,
        default="jev",
        choices=["jev", "typed"],
        help=(
            "jev: SargeDev/jev-distill-corpus-v3 (the release corpus). "
            "typed: LocalLLaMA/typed-decisions train split, for the matched "
            "head-to-head comparison against Laya."
        ),
    )
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--option-len", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--val-cases", type=int, default=2000)
    parser.add_argument("--calib-cases", type=int, default=4000)
    parser.add_argument(
        "--heldout-cases", type=int, default=0, help="0 = full test_set_30k"
    )
    parser.add_argument(
        "--backbone", type=str, default=BACKBONE,
        help="local F32 v2 backbone directory or a public Hub identifier",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA),
        help="local directory with train/calibration/test_set_30k jsonl splits",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(OUT),
        help="directory for checkpoints and the training report",
    )
    args = parser.parse_args()

    # All external inputs are explicit. Defaults preserve the release
    # settings, but no other machine's private path is assumed.
    BACKBONE = args.backbone
    DATA = Path(args.data_dir)
    OUT = Path(args.out_dir)
    if not DATA.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {DATA}. Pass --data-dir with a "
            "local copy of SargeDev/jev-distill-corpus-v3 (jsonl splits)."
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    devices = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))] or ["cpu"]
    OUT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    cfg = HeadConfig(dtype_name="fp32")
    models = [RockAndEggV3SystemOne(cfg, model_path=BACKBONE).to(d) for d in devices]
    sync_head_weights(models[0], models[1:])
    params = [p for m in models for p in head_parameters(m)]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=devices[0] != "cpu")

    if args.dataset == "typed":
        from typed_decisions_data import corpus_stats, load_typed_decisions_cases

        # Matched-data run for the head-to-head benchmark. The corpus yields
        # 6,000 typed decisions over 1,200 cases. We hold out 800 (400 for
        # validation, 400 for calibration) so that nothing used to fit the
        # temperature ever reaches a training batch. Laya holds out 400 and
        # trains on 5,600; we train on 5,200, so our budget is the smaller one.
        pool = load_typed_decisions_cases("train")
        rng = random.Random(args.seed + 11)  # noqa: S311 - deterministic split
        shuffled = list(pool)
        rng.shuffle(shuffled)
        calib = shuffled[:400]
        val = shuffled[400:800]
        train = shuffled[800:]
        if args.cases and args.cases < len(train):
            train = stratified_sample(train, args.cases, args.seed)
        test = load_typed_decisions_cases("test")
        dataset_meta = {
            "dataset": "LocalLLaMA/typed-decisions",
            "dataset_revision": "all/train",
            "credit": "LocalLLaMA/typed-decisions",
            "split_note": (
                "matched-data run: 1,200 train cases -> 6,000 typed decisions; "
                "800 held out (400 val + 400 calibration); 5,200 train"
            ),
            "corpus_stats": corpus_stats(pool),
            "suffix": "_typed",
        }
    else:
        train_all = read_cases("train")
        train = stratified_sample(train_all, args.cases, args.seed)
        val = stratified_sample(read_cases("validation"), args.val_cases, args.seed + 1)
        calib = stratified_sample(
            read_cases("calibration"), args.calib_cases, args.seed + 2
        )
        test = read_cases("test_set_30k")
        if args.heldout_cases:
            test = stratified_sample(test, args.heldout_cases, args.seed + 3)
        dataset_meta = {
            "dataset": "SargeDev/jev-distill-corpus-v3",
            "dataset_revision": "75111a13a8455032f0435050bfc7e0c9026ecfe8",
            "credit": "SargeDev/jev-distill-corpus-v3 (Apache-2.0)",
            "split_note": "release corpus",
            "corpus_stats": {},
            "suffix": "",
        }
    print(
        json.dumps(
            {
                "devices": devices,
                "train_cases": len(train),
                "val_cases": len(val),
                "calibration_cases": len(calib),
                "heldout_cases": len(test),
                "micro_per_gpu": args.micro,
                "effective_batch": args.micro * len(devices),
                "lr": args.lr,
            }
        ),
        flush=True,
    )

    rng = torch.Generator().manual_seed(args.seed)
    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)  # noqa: S311 - shuffling ML batches
    batches = option_buckets(order, train, args.micro)
    # Steps per epoch are fixed by the bucket layout; every epoch replays the
    # same layout with a fresh shuffle, so the plan length per epoch is
    # constant. ``total_steps`` must therefore be steps-per-epoch * epochs —
    # this is the number the cosine LR schedule is stretched over.
    total_steps = math.ceil(len(batches) / len(devices)) * args.epochs
    step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=True)
        state = models[0].state_dict()
        state.update(resume["head"])
        models[0].load_state_dict(state, strict=True)
        sync_head_weights(models[0], models[1:])
        step = int(resume.get("step", 0))
        if args.resume_step >= 0:
            step = args.resume_step
    t0 = time.perf_counter()
    cases_done = 0
    best_val = float("inf")
    log: list[dict[str, Any]] = []

    for _epoch in range(args.epochs):
        # Each epoch reshuffles and re-buckets, so every case is seen once per
        # epoch. The previous code derived the start offset from the global
        # step counter, which made every epoch after the first an empty range:
        # only epoch 1 ever ran.
        epoch_order = list(range(len(train)))
        random.Random(args.seed + _epoch).shuffle(epoch_order)  # noqa: S311
        epoch_batches = option_buckets(epoch_order, train, args.micro)
        start_offset = 0
        if _epoch == 0 and args.resume:
            # Resume: skip the batches already consumed before the checkpoint.
            start_offset = step * len(devices)
        for step_batches in epoch_step_plan(epoch_batches, len(devices), start_offset):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            cases_done += sum(len(b) for b in step_batches)
            for replica, device in enumerate(devices):
                if replica >= len(step_batches):
                    continue
                torch.cuda.set_device(device)
                enc = encode_batch(
                    step_batches[replica],
                    train,
                    tokenizer,
                    device,
                    args.state_len,
                    args.option_len,
                )
                targets = torch.stack(
                    [
                        option_dropout(
                            enc.targets[i],
                            enc.golds[i],
                            cfg.option_dropout,
                            generator=rng,
                        )
                        for i in range(len(enc.golds))
                    ]
                )
                with torch.autocast(
                    "cuda", dtype=torch.float16, enabled=device != "cpu"
                ):
                    logits = models[replica].forward_batch(
                        enc.state_ids,
                        enc.state_mask,
                        enc.suffix_ids,
                        enc.suffix_mask,
                        enc.rows_per_case,
                        enc.real_counts,
                        enc.qtype_ids,
                    )
                losses = torch.stack(
                    [
                        verdict_pact_loss(
                            logits.float()[i],
                            targets[i],
                            enc.qtype_ids[i],
                            cfg.emd_weight,
                        )
                        for i in range(len(enc.golds))
                    ]
                )
                loss = losses.mean() / len(step_batches)
                scaler.scale(loss).backward()
                batch_loss += float(loss.detach())
            average_head_gradients(models)
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(step, total_steps, args.lr)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    json.dumps(
                        {
                            "step": step,
                            "total_steps": total_steps,
                            "loss": round(batch_loss, 5),
                            "lr": round(optimizer.param_groups[0]["lr"], 8),
                            "cases_per_sec": round(cases_done / elapsed, 2),
                            "elapsed_s": round(elapsed, 1),
                        }
                    ),
                    flush=True,
                )
            if step % args.checkpoint_every == 0:
                save_checkpoint(
                    models[0],
                    OUT / f"head_step_{step}{dataset_meta['suffix']}.pt",
                    step,
                    {"cpu": torch.random.get_rng_state()},
                )
            if step % args.validate_every == 0:
                metrics = evaluate(models[0], val, tokenizer, devices[0], args)
                mean_val = metrics["loss"]
                log.append(
                    {"step": step, "val": mean_val, "val_acc": metrics["accuracy"]}
                )
                print(
                    json.dumps(
                        {
                            "step": step,
                            "val_loss": round(mean_val, 5),
                            "val_acc": round(metrics["accuracy"], 4),
                        }
                    ),
                    flush=True,
                )
                if mean_val < best_val:
                    best_val = mean_val
                    save_checkpoint(
                        models[0],
                        OUT / f"head_best{dataset_meta['suffix']}.pt",
                        step,
                        {},
                    )

    final_path = OUT / f"head_final{dataset_meta['suffix']}.pt"
    save_checkpoint(models[0], final_path, step, {})

    calib_metrics = evaluate(models[0], calib, tokenizer, devices[0], args)
    temperature = fit_temperature(calib_metrics["_probs"], calib_metrics["_labels"])
    test_metrics = evaluate(models[0], test, tokenizer, devices[0], args)
    scaled = serving_probs(
        torch.tensor(test_metrics["_probs"], dtype=torch.float32), temperature
    )
    ece = compute_ece(scaled, torch.tensor(test_metrics["_labels"]))
    report = {
        "dataset": dataset_meta["dataset"],
        "dataset_revision": dataset_meta["dataset_revision"],
        "credit": dataset_meta["credit"],
        "split_note": dataset_meta["split_note"],
        "corpus_stats": dataset_meta["corpus_stats"],
        "objective": "soft-label CE + squared EMD(score) + option dropout 0.3; no RLCD",
        "devices": devices,
        "cases": len(train),
        "steps": step,
        "final_train_loss": batch_loss,
        # ``None`` (not Infinity) when validation never ran — a run shorter
        # than ``--validate-every`` steps has no best validation loss, and
        # ``Infinity`` is not valid strict JSON.
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "val_curve": log,
        "calibration_temperature": temperature,
        "calibration_ece": compute_ece(
            torch.tensor(calib_metrics["_probs"]),
            torch.tensor(calib_metrics["_labels"]),
        ),
        "heldout_split": (
            "typed-decisions/test" if args.dataset == "typed" else "test_set_30k"
        ),
        "heldout_accuracy": test_metrics["accuracy"],
        "heldout_ece": float(ece),
        "heldout_loss": test_metrics["loss"],
        "heldout_n": len(test),
        "checkpoint": str(final_path),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    report_path = OUT / f"training_report{dataset_meta['suffix']}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
