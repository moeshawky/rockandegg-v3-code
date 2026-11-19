"""Full-backbone SystemOne fine-tuning on typed-decisions.

Trains the production RockAndEgg-v3 model end-to-end (backbone UNFROZEN) on the
matched typed-decisions train split. Preserves the architecture contract: one
state encode per forward, options isolated in batch rows, no LM head, no
generation. Soft-label cross-entropy is the primary objective; squared EMD on
score questions is an optional controlled add-on.

Run (smoke):
    USE_TF=0 python3 scripts/train_full_backbone.py --config scripts/full_backbone_config.json

The JSON config is the single source of truth for every experiment; it is
copied verbatim into the run directory for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from typed_decisions_data import render_options, serialize_state

DATASET = "LocalLLaMA/typed-decisions"
CONFIG = "all"
DATASET_REVISION = "e6aa632c4db73b59c7713bb21066ad1a8e1bfce6"
MODEL = "moeshawky/RockAndEgg-313M-v3"  # public model id; override via config["model"]
MARKER_ID = 151645
QTYPE = {"choice": 0, "score": 1, "noul": 2}
ROOT = Path(__file__).resolve().parent
DEV_RESERVE_SEED = 20261119


@dataclass(frozen=True)
class Case:
    """One typed decision: a state, a question, options and a soft target."""

    case_id: str
    kind: str
    options: list[str]
    target: list[float]
    state: str
    question: str


@dataclass
class Batch:
    """Padded tensors for one optimizer micro-batch."""

    state_ids: torch.Tensor
    state_mask: torch.Tensor
    suffix_ids: torch.Tensor
    suffix_mask: torch.Tensor
    rows_per_case: int
    real_counts: list[int]
    qtype_ids: list[int]
    targets: torch.Tensor


def load_cases(split: str, max_cases: int | None = None) -> list[Case]:
    """Load one corpus split as per-question training cases."""
    ds = load_dataset(DATASET, CONFIG, split=split, revision=DATASET_REVISION)
    if max_cases is not None:
        ds = ds.select(range(min(max_cases, len(ds))))
    cases: list[Case] = []
    for row in ds:
        state = serialize_state(json.loads(row["state"]))
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, qdef in questions.items():
            g = gold.get(qid)
            if g is None:
                continue
            options, keys = render_options(qdef)
            if not options:
                continue
            probs = g["probabilities"]
            vals = [float(probs.get(k, 0.0)) for k in keys]
            total = sum(vals)
            target = [v / total for v in vals] if total > 0 else [1.0 / len(keys)] * len(keys)
            cases.append(
                Case(
                    case_id=f"{row['id']}:{qid}",
                    kind=str(qdef["type"]),
                    options=options,
                    target=target,
                    state=state,
                    question=str(qdef.get("instructions", "")),
                )
            )
    return cases


def apply_teacher_targets(
    cases: list[Case], teacher: dict[str, Any], alpha: float
) -> list[Case]:
    """Blend each gold soft target with a teacher distribution (train only)."""
    if alpha <= 0:
        return cases
    out: list[Case] = []
    for c in cases:
        dossier, _, qid = c.case_id.rpartition(":")
        vec = teacher.get("probabilities", {}).get(dossier, {}).get(qid)
        if vec is None or len(vec) != len(c.target):
            out.append(c)
            continue
        v = np.asarray(vec, dtype=float)
        total = v.sum()
        if total <= 0:
            out.append(c)
            continue
        v = v / total
        gold = np.asarray(c.target, dtype=float)
        mixed = alpha * v + (1.0 - alpha) * gold
        mixed = mixed / mixed.sum()
        out.append(replace(c, target=mixed.tolist()))
    return out


def dossier_id(case: Case) -> str:
    """Original dossier/case id (everything before the question suffix)."""
    return case.case_id.rsplit(":", 1)[0]


def split_by_dossier(cases: list[Case], val_frac: float, seed: int) -> tuple[list[Case], list[Case]]:
    """Hold out whole dossiers so no question from a case crosses the split."""
    rng = random.Random(seed)
    dossiers = sorted({dossier_id(c) for c in cases})
    rng.shuffle(dossiers)
    n_val = max(1, round(len(dossiers) * val_frac))
    val_ids = set(dossiers[:n_val])
    train = [c for c in cases if dossier_id(c) not in val_ids]
    val = [c for c in cases if dossier_id(c) in val_ids]
    return train, val


def split_dev_reserve(val_cases: list[Case]) -> tuple[list[Case], list[Case]]:
    """Deterministically split validation dossiers into DEV / RESERVED halves.

    Byte-identical logic to qualify_student.split_dev_reserve (kept local to
    avoid a circular import). RESERVED is NEVER used for selection here.
    """
    ids = sorted({dossier_id(c) for c in val_cases})
    rng = random.Random(DEV_RESERVE_SEED)
    rng.shuffle(ids)
    half = len(ids) // 2
    dev_ids, reserve_ids = set(ids[:half]), set(ids[half:])
    dev = [c for c in val_cases if dossier_id(c) in dev_ids]
    reserve = [c for c in val_cases if dossier_id(c) in reserve_ids]
    return dev, reserve


def permute_case(case: Case, rng: random.Random) -> Case:
    """Permute options and reorder the soft target to match (invariance aug).

    Score options are ordinal ("level 0", "level 1", ...): shuffling them would
    make squared-EMD / RPS compare non-adjacent cumulative masses and silently
    corrupt the ordinal objective. Permutation is therefore applied only to
    unordered choice and binary noul questions.
    """
    if case.kind == "score":
        return case
    order = list(range(len(case.options)))
    rng.shuffle(order)
    return Case(
        case_id=case.case_id,
        kind=case.kind,
        options=[case.options[i] for i in order],
        target=[case.target[i] for i in order],
        state=case.state,
        question=case.question,
    )


def encode_batch(
    indices: list[int],
    cases: list[Case],
    tokenizer: Any,
    device: str,
    max_state_len: int,
    max_option_len: int,
) -> Batch:
    """Encode B cases into padded state + option-row tensors."""
    items = [cases[i] for i in indices]
    width = max(len(c.options) for c in items)
    pad_id = tokenizer.eos_token_id
    state_ids, state_masks, suffix_rows, suffix_masks, targets = [], [], [], [], []
    row_width = 0
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
            ids = torch.cat([ids, torch.full((1, n), pad_id, dtype=ids.dtype)], 1)
            mask = torch.cat([mask, torch.zeros((1, n), dtype=mask.dtype)], 1)
        state_ids.append(ids[0])
        state_masks.append(mask[0])

        rows: list[list[int]] = []
        masks: list[list[int]] = []
        for option in case.options:
            text = f"{case.question} [SEP] {option} [SEP]"
            toks = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_option_len - 1),
            )["input_ids"]
            row = [*toks, MARKER_ID]
            rows.append(row)
            masks.append([1] * len(row))
        row_width = max(row_width, *(len(r) for r in rows))
        suffix_rows.extend(rows)
        suffix_masks.extend(masks)
        for _ in range(width - len(rows)):
            suffix_rows.append([])
            suffix_masks.append([])
        targets.append(list(case.target) + [0.0] * (width - len(case.target)))

    suffix_rows = [r + [pad_id] * (row_width - len(r)) for r in suffix_rows]
    suffix_masks = [m + [0] * (row_width - len(m)) for m in suffix_masks]
    return Batch(
        state_ids=torch.stack(state_ids).to(device),
        state_mask=torch.stack(state_masks).to(device),
        suffix_ids=torch.tensor(suffix_rows, dtype=torch.long, device=device),
        suffix_mask=torch.tensor(suffix_masks, dtype=torch.long, device=device),
        rows_per_case=width,
        real_counts=[len(c.options) for c in items],
        qtype_ids=[QTYPE[c.kind] for c in items],
        targets=torch.tensor(targets, dtype=torch.float32, device=device),
    )


def apply_option_dropout(targets: torch.Tensor, drop_prob: float, rng: Any) -> torch.Tensor:
    """Drop non-gold options from each soft target, keeping the argmax option."""
    if drop_prob <= 0:
        return targets
    out = targets.clone()
    for row in range(out.shape[0]):
        vals = out[row]
        gold = int(vals.argmax().item())
        for col in range(vals.shape[0]):
            if col != gold and vals[col].item() > 0 and rng.random() < drop_prob:
                out[row, col] = 0.0
        total = out[row].sum().clamp_min(1e-12)
        out[row] = out[row] / total
    return out


def amp_settings(cfg: dict[str, Any]) -> tuple[torch.dtype, bool]:
    """Resolve (autocast dtype, enabled) from the config's precision field.

    ``bf16`` is the historical workspace default; ``fp16`` reconstructs the
    operator's historical full-FT regime and is paired with a GradScaler in
    ``main``. ``fp32`` disables autocast entirely. This helper is the single
    source of truth so the train/eval/grad-probe paths cannot diverge.
    """
    precision = cfg.get("precision", "bf16" if cfg.get("bf16", True) else "fp32")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    return dtype, precision != "fp32"


def soft_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft-label cross-entropy over the real (non-padded) option columns.

    Targets sum to 1 over real columns and are zero on padded columns, so the
    dot product with log-softmax is the softmax cross-entropy; padded columns
    contribute exactly zero and their logits are -1e4 from the model.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(targets * logp).sum(dim=1).mean()


def emd_square(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Squared 1-D earth-mover distance over ordinal levels (batch)."""
    cp = torch.cumsum(p, dim=-1)
    cq = torch.cumsum(q, dim=-1)
    return ((cp - cq) ** 2).sum(dim=-1).mean()


def ordinal_rps(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Ranked probability score for ordinal levels, normalized to [0, 1]."""
    cp = torch.cumsum(p[:, :-1], dim=-1)
    cq = torch.cumsum(q[:, :-1], dim=-1)
    return ((cp - cq) ** 2).sum(dim=-1) / max(1, p.shape[-1] - 1)


def compute_loss(
    logits: torch.Tensor,
    batch: Batch,
    emd_weight: float,
    rps_weight: float = 0.0,
) -> torch.Tensor:
    """Soft CE plus controlled squared-EMD / ordinal-RPS on score questions."""
    ce = soft_ce_loss(logits, batch.targets)
    if emd_weight <= 0 and rps_weight <= 0:
        return ce
    idx = torch.tensor(batch.qtype_ids, device=logits.device)
    score_sel = idx == QTYPE["score"]
    if not score_sel.any():
        return ce
    probs = F.softmax(logits.float(), dim=-1)
    loss = ce
    if emd_weight > 0:
        loss = loss + emd_weight * emd_square(probs[score_sel], batch.targets[score_sel])
    if rps_weight > 0:
        loss = loss + rps_weight * ordinal_rps(probs[score_sel], batch.targets[score_sel])
    return loss


def make_batches(cases: list[Case], batch_size: int, seed: int) -> list[list[int]]:
    """Bucket by option count to minimize padding, then shuffle the buckets."""
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(cases):
        groups.setdefault(len(c.options), []).append(i)
    batches: list[list[int]] = []
    for width in sorted(groups):
        pool = groups[width]
        rng.shuffle(pool)
        for start in range(0, len(pool), batch_size):
            batches.append(pool[start : start + batch_size])
    rng.shuffle(batches)
    return batches


def _enable_grad_safe_cache_clone() -> None:
    """Make ``copy.deepcopy(DynamicCache)`` work while the backbone trains.

    The production ``forward_batch`` forks the single state cache with
    ``copy.deepcopy``. With an unfrozen backbone those cache tensors are
    non-leaf graph nodes, which torch refuses to deepcopy. A functional
    per-layer clone keeps the same fork-once semantics and preserves autograd.
    Inference (frozen backbone) is unaffected because its cache tensors are
    graph leaves.
    """
    from transformers.cache_utils import DynamicCache

    def _clone(self: Any, memo: Any) -> Any:
        out = copy.copy(self)
        out.layers = []
        for layer in self.layers:
            new_layer = copy.copy(layer)
            new_layer.keys = layer.keys.clone() if layer.keys is not None else None
            new_layer.values = layer.values.clone() if layer.values is not None else None
            out.layers.append(new_layer)
        return out

    if not getattr(DynamicCache, "_grad_safe_deepcopy", False):
        DynamicCache.__deepcopy__ = _clone
        DynamicCache._grad_safe_deepcopy = True


def load_model(cfg: dict[str, Any], device: str) -> Any:
    """Load the production model with the backbone unfrozen for training."""
    _enable_grad_safe_cache_clone()
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[cfg.get("weights_dtype", "fp32")]
    model = AutoModel.from_pretrained(cfg.get("model", MODEL), trust_remote_code=True, dtype=dtype)
    # The production class re-applies the freeze in __init__/eval()/to(). Flip
    # both the config dict and head_config so those hooks become no-ops.
    model.config.systemone_head["backbone_frozen"] = False
    model.head_config["backbone_frozen"] = False
    for p in model.backbone.parameters():
        p.requires_grad_(True)
    return model.to(device)


def build_optimizer(model: Any, cfg: dict[str, Any]) -> tuple[torch.optim.Optimizer, Any]:
    """AdamW with separate backbone/head learning rates and cosine schedule."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith("backbone.") and p.requires_grad
    ]
    lr_bb = cfg["lr_backbone"]
    lr_head = cfg["lr_head"]
    wd = cfg.get("weight_decay", 0.01)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_bb},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=wd,
    )
    return opt, (lr_bb, lr_head)


def lr_lambda(step: int, total: int, warmup_frac: float) -> float:
    """Linear warmup then cosine decay to zero."""
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate_val(model: Any, val_cases: list[Case], tokenizer: Any, cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Evaluate argmax accuracy on the internal validation split."""
    model.eval()
    amp_dtype, amp_enabled = amp_settings(cfg)
    batches = make_batches(val_cases, cfg.get("eval_batch_size", 32), cfg.get("seed", 0))
    correct = total = 0
    by_kind: dict[str, list[int]] = {}
    for idxs in batches:
        b = encode_batch(
            idxs, val_cases, tokenizer, device, cfg.get("max_state_len", 256), cfg.get("max_option_len", 64)
        )
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model.forward_batch(
                b.state_ids, b.state_mask, b.suffix_ids, b.suffix_mask,
                b.rows_per_case, b.real_counts, b.qtype_ids,
            )
        pred = logits.argmax(dim=-1).cpu().tolist()
        gold = [max(range(len(c.target)), key=lambda i: c.target[i]) for c in (val_cases[i] for i in idxs)]
        for j, (p, g) in enumerate(zip(pred, gold, strict=True)):
            kind = val_cases[idxs[j]].kind
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += int(p == g)
            by_kind[kind][1] += 1
            correct += int(p == g)
            total += 1
    model.train()
    out = {
        "accuracy": correct / max(1, total),
        "n": total,
        "by_kind": {k: {"accuracy": v[0] / max(1, v[1]), "n": v[1]} for k, v in by_kind.items()},
    }
    return out


def verify_grad_flow(model: Any, tokenizer: Any, cfg: dict[str, Any], device: str) -> dict[str, Any]:
    """Prove gradients reach both the state encoder and the option scorer."""
    probe = [
        Case("grad:0", "choice", ["a", "b", "c"], [0.2, 0.5, 0.3], "probe state", "which?"),
        Case("grad:1", "score", ["l0", "l1", "l2"], [0.1, 0.3, 0.6], "probe state", "level?"),
    ]
    b = encode_batch([0, 1], probe, tokenizer, device, cfg.get("max_state_len", 256), cfg.get("max_option_len", 64))
    model.train()
    model.zero_grad(set_to_none=True)
    amp_dtype, amp_enabled = amp_settings(cfg)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        logits = model.forward_batch(
            b.state_ids, b.state_mask, b.suffix_ids, b.suffix_mask,
            b.rows_per_case, b.real_counts, b.qtype_ids,
        )
        loss = compute_loss(logits, b, cfg.get("emd_weight", 0.0), cfg.get("rps_weight", 0.0))
    loss.backward()
    bb_grad = sum(
        p.grad.abs().sum().item() for p in model.backbone.parameters() if p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if not n.startswith("backbone.") and p.grad is not None
    )
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.item()),
        "backbone_grad_l1": float(bb_grad),
        "head_grad_l1": float(head_grad),
        "backbone_trainable": sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),
        "head_trainable": sum(
            p.numel() for n, p in model.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad
        ),
    }


def save_checkpoint(model: Any, opt: Any, sched: Any, out_dir: Path, meta: dict[str, Any], save_hf: bool) -> None:
    """Save a resumable training checkpoint and (optionally) an HF model dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "meta": meta,
        },
        out_dir / "resume.pt",
    )
    if save_hf:
        hf_dir = out_dir / "model"
        # Keep the declared freeze TRUE for the exported inference artifact;
        # training used an in-memory override only.
        model.config.systemone_head["backbone_frozen"] = True
        model.head_config["backbone_frozen"] = True
        model.save_pretrained(hf_dir, safe_serialization=True)
        (hf_dir / "generation_config.json").unlink(missing_ok=True)
        model.config.systemone_head["backbone_frozen"] = False
        model.head_config["backbone_frozen"] = False


def main() -> None:
    """Run one configured training experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment config")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["seed"] = int(cfg.get("seed", 0))

    run_dir = Path(cfg.get("run_dir", ROOT / "artifacts" / "runs" / cfg["name"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    device = cfg.get("device", "cuda:0")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    tokenizer = AutoTokenizer.from_pretrained(cfg.get("model", MODEL))
    all_cases = load_cases(cfg.get("split", "train"), cfg.get("max_cases"))
    # The data split is FIXED (split_seed) independent of the training seed, so
    # independent-seed replications share the same 1020/90/90 dossiers. Only
    # init/shuffle/augmentation randomness varies with cfg["seed"].
    split_seed = int(cfg.get("split_seed", 0))
    train_cases, val_cases = split_by_dossier(
        all_cases, cfg.get("val_fraction", 0.15), split_seed
    )
    # DEV-only selection: split val into DEV / RESERVED; RESERVED is never
    # evaluated or used for checkpoint choice during training.
    selection_cases = val_cases
    reserved_cases: list[Case] = []
    if cfg.get("dev_selection", True):
        selection_cases, reserved_cases = split_dev_reserve(val_cases)
        print(
            f"[train] selection=DEV dossiers={len({dossier_id(c) for c in selection_cases})} "
            f"RESERVED untouched dossiers={len({dossier_id(c) for c in reserved_cases})}",
            flush=True,
        )
    if cfg.get("teacher_probs_path") and cfg.get("teacher_alpha", 0.0) > 0:
        teacher = json.loads(Path(cfg["teacher_probs_path"]).read_text())
        train_cases = apply_teacher_targets(train_cases, teacher, cfg["teacher_alpha"])
        print(f"[train] teacher targets applied to train split from {cfg['teacher_probs_path']}", flush=True)
    print(
        f"[train] cases={len(all_cases)} train={len(train_cases)} val={len(val_cases)} "
        f"by_kind={ {k: sum(c.kind == k for c in train_cases) for k in QTYPE} }",
        flush=True,
    )

    model = load_model(cfg, device)
    grad_report = verify_grad_flow(model, tokenizer, cfg, device)
    (run_dir / "grad_flow.json").write_text(json.dumps(grad_report, indent=2))
    print(f"[train] grad_flow={json.dumps(grad_report)}", flush=True)
    if grad_report["backbone_grad_l1"] <= 0 or grad_report["head_grad_l1"] <= 0:
        raise RuntimeError("gradient flow check failed: backbone or head received no gradient")

    opt, lrs = build_optimizer(model, cfg)
    micro_bs = int(cfg.get("micro_batch_size", 32))
    accum = int(cfg.get("grad_accum", 2))
    epochs = int(cfg.get("epochs", 4))
    perm_aug = bool(cfg.get("option_permutation_aug", True))
    total_steps = math.ceil(len(train_cases) / (micro_bs * accum)) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, cfg.get("warmup_frac", 0.03))
    )
    amp_dtype, amp_enabled = amp_settings(cfg)
    precision = cfg.get("precision", "bf16" if cfg.get("bf16", True) else "fp32")
    # bf16/fp32 need no loss scaling; fp16 reconstructs the historical regime
    # and requires a GradScaler to avoid vanishing gradients. Disabled scaler
    # methods are no-ops, so one code path serves all precisions.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(precision == "fp16"),
        init_scale=float(cfg.get("grad_scaler_init_scale", 65536.0)),
        growth_interval=int(cfg.get("grad_scaler_growth_interval", 2000)),
    )
    print(f"[train] precision={precision} amp_dtype={amp_dtype} scaler_enabled={scaler.is_enabled()} init_scale={cfg.get('grad_scaler_init_scale', 65536.0)}", flush=True)

    eval_every = int(cfg.get("eval_every_steps", 0))
    step_reference = int(cfg.get("step_reference", 400))
    patience = int(cfg.get("early_stop_patience", 0))
    history: list[dict[str, Any]] = []
    step_evals: list[dict[str, Any]] = []
    step = 0
    t0 = time.perf_counter()
    best_acc = -1.0
    best_step = 0
    stale_evals = 0
    stop_training = False
    max_mem = 0

    def _evaluate_and_track(tag: str) -> dict[str, Any]:
        """Evaluate on the selection set (DEV), save best + step-400 ckpt."""
        nonlocal best_acc, best_step, stale_evals
        sel = evaluate_val(model, selection_cases, tokenizer, cfg, device)
        rec = {"tag": tag, "step": step, "selection_accuracy": sel["accuracy"], "selection": sel}
        step_evals.append(rec)
        print(f"[eval] {json.dumps(rec)}", flush=True)
        if sel["accuracy"] > best_acc:
            best_acc = sel["accuracy"]
            best_step = step
            stale_evals = 0
            save_checkpoint(model, opt, sched, run_dir / "best", rec, save_hf=True)
        else:
            stale_evals += 1
        if step_reference and step == step_reference:
            save_checkpoint(model, opt, sched, run_dir / f"step_{step_reference}", rec, save_hf=True)
        return rec

    for epoch in range(epochs):
        if stop_training:
            break
        model.train()
        rng = random.Random(cfg["seed"] + epoch)
        epoch_cases = list(train_cases)
        if perm_aug:
            epoch_cases = [
                permute_case(c, rng) if rng.random() < cfg.get("perm_prob", 0.5) else c
                for c in epoch_cases
            ]
        batches = make_batches(epoch_cases, micro_bs, cfg["seed"] + epoch)
        running = 0.0
        n_micro = 0
        opt.zero_grad(set_to_none=True)
        for bi, idxs in enumerate(batches):
            b = encode_batch(
                idxs, epoch_cases, tokenizer, device,
                cfg.get("max_state_len", 256), cfg.get("max_option_len", 64),
            )
            if cfg.get("option_dropout", 0.0) > 0:
                b.targets = apply_option_dropout(b.targets, cfg["option_dropout"], rng)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model.forward_batch(
                    b.state_ids, b.state_mask, b.suffix_ids, b.suffix_mask,
                    b.rows_per_case, b.real_counts, b.qtype_ids,
                )
                loss = compute_loss(logits, b, cfg.get("emd_weight", 0.0), cfg.get("rps_weight", 0.0))
            scaler.scale(loss / accum).backward()
            running += float(loss.item())
            n_micro += 1
            if (bi + 1) % accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)  # no-op when scaler disabled
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if eval_every and step % eval_every == 0:
                    _evaluate_and_track(f"step_{step}")
                    if patience and stale_evals >= patience:
                        print(f"[train] early stop: no DEV improvement for {stale_evals} evals", flush=True)
                        stop_training = True
                        break
            max_mem = max(max_mem, torch.cuda.max_memory_allocated(device))
            if stop_training:
                break
        train_loss = running / max(1, n_micro)
        epoch_eval = _evaluate_and_track(f"epoch_{epoch}_end")
        rec = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val": epoch_eval["selection"],
            "lr_backbone": sched.get_last_lr()[0],
            "lr_head": sched.get_last_lr()[1],
        }
        history.append(rec)
        print(f"[train] {json.dumps(rec)}", flush=True)
        save_checkpoint(
            model, opt, sched, run_dir / f"epoch_{epoch}", rec,
            save_hf=bool(cfg.get("save_hf_each_epoch", False)),
        )
        if patience and stale_evals >= patience:
            print(f"[train] early stop at epoch {epoch}", flush=True)
            stop_training = True

    elapsed = time.perf_counter() - t0
    summary = {
        "config": cfg,
        "precision": precision,
        "amp_dtype": str(amp_dtype),
        "scaler_enabled": scaler.is_enabled(),
        "grad_flow": grad_report,
        "history": history,
        "step_evals": step_evals,
        "best_val_accuracy": best_acc,
        "best_step": best_step,
        "selection_set": "DEV" if cfg.get("dev_selection", True) else "internal_val",
        "reserved_used_for_selection": False,
        "elapsed_s": elapsed,
        "peak_memory_gib": max_mem / 1024**3,
        "decisions_per_s": len(train_cases) * epochs / max(elapsed, 1e-6),
        "final_lrs": lrs,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] DONE best_val={best_acc:.4f} elapsed={elapsed:.1f}s peak={max_mem/1024**3:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
