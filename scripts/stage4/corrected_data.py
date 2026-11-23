"""CORRECTED canonical data pipeline for the RockAndEgg-v3 investigation.

Single source of truth for:
  * authoritative data load (LocalLLaMA/typed-decisions, config ``all``,
    revision e6aa632c4db73b59c7713bb21066ad1a8e1bfce6),
  * the EXACT historical split algorithm copied verbatim from
    scripts/train_full_backbone.py @ a4b528a4 (split_by_dossier / split_dev_reserve,
    split_seed=0, val_fraction=0.15, DEV_RESERVE_SEED=20261119),
  * the canonical option renderer copied from scripts/typed_decisions_data.py
    @ a4b528a4 (serialize_state / render_options / target_from_gold).

This module SUPERSEDES the exploratory ``real_data_common.py`` renderer and
split reconstruction. ``real_data_common.py`` is retained only for historical
reproduction of the old (invalid) artifacts and must not be imported by new
work.

No training, no parameter edits, read-only with respect to the checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
DATA_DIR = os.path.join(PROBE_DIR, "real_data")
sys.path.insert(0, MODEL_DIR)
sys.path.insert(0, PROBE_DIR)

from systemone_inference import Question  # noqa: E402

DATASET_NAME = "LocalLLaMA/typed-decisions"
CONFIG = "all"
DATASET_REVISION = "e6aa632c4db73b59c7713bb21066ad1a8e1bfce6"
MARKER_ID = 151645
QTYPE_ID = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAME = {0: "choice", 1: "score", 2: "noul"}

# Documented split parameters (scripts/full_backbone_config.json).
SPLIT_SEED = 0
VAL_FRACTION = 0.15
DEV_RESERVE_SEED = 20261119

# Canonical Noul fallback criteria (scripts/typed_decisions_data.py render_options).
NOUL_FALLBACK = {
    "false": "no, the statement does not hold",
    "true": "yes, the statement holds",
}


# ---------------------------------------------------------------------------
# Canonical renderer -- copied from typed_decisions_data.py @ a4b528a4
# ---------------------------------------------------------------------------
def serialize_state(state: Any) -> str:
    """Render a case state as the text the model actually sees (canonical)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(qdef: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Render one question's option texts and their keys (canonical).

    choice : ``f"{k}: {v}"`` when v else ``k``; keys in criteria order.
    score  : ``f"level {i}: {c}"``; keys ``["0","1",...]``.
    noul   : ``"false: " + (crit.get("false") or NOUL_FALLBACK["false"])`` and
             ``"true: " + (crit.get("true") or NOUL_FALLBACK["true"])``;
             keys ``["false","true"]``.
    """
    qtype = qdef["type"]
    crit = qdef.get("criteria")
    if qtype == "choice":
        crit = crit or {}
        options = [f"{k}: {v}" if v else k for k, v in crit.items()]
        keys = list(crit.keys())
    elif qtype == "score":
        crit = crit or []
        options = [f"level {i}: {c}" for i, c in enumerate(crit)]
        keys = [str(i) for i in range(len(options))]
    else:
        crit = crit or {}
        options = [
            "false: " + (crit.get("false") or NOUL_FALLBACK["false"]),
            "true: " + (crit.get("true") or NOUL_FALLBACK["true"]),
        ]
        keys = ["false", "true"]
    return options, keys


def target_from_gold(gold_q: dict[str, Any], keys: list[str]) -> list[float]:
    """Build the soft-label target for one question, aligned to ``keys``."""
    probs = gold_q["probabilities"]
    vals = [float(probs.get(k, 0.0)) for k in keys]
    total = sum(vals)
    if total <= 0.0:
        return [1.0 / len(keys)] * len(keys)
    return [v / total for v in vals]


# ---------------------------------------------------------------------------
# Authoritative data
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_authoritative() -> tuple[list[dict], list[dict]]:
    """Load the pinned train/test parquet rows in file order."""
    tr = pq.read_table(os.path.join(DATA_DIR, "train-all-all.parquet")).to_pylist()
    te = pq.read_table(os.path.join(DATA_DIR, "test-all-all.parquet")).to_pylist()
    return tr, te


def dossier_id(row_or_id) -> str:
    """Original dossier id (everything before the last ':' of a case id)."""
    cid = row_or_id if isinstance(row_or_id, str) else row_or_id["id"]
    return cid.rsplit(":", 1)[0]


# ---------------------------------------------------------------------------
# EXACT historical split -- copied from train_full_backbone.py @ a4b528a4
# ---------------------------------------------------------------------------
def split_by_dossier_ids(ids, val_frac: float = VAL_FRACTION, seed: int = SPLIT_SEED):
    """Verbatim port of train_full_backbone.split_by_dossier on a dossier-id list."""
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def split_dev_reserve(val_ids, seed: int = DEV_RESERVE_SEED):
    """Verbatim port of train_full_backbone.split_dev_reserve on a val-id set."""
    ids = sorted(set(val_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    dev_ids = set(ids[:half])
    reserved_ids = set(ids[half:])
    return dev_ids, reserved_ids


def exact_split(train_rows: list[dict]) -> dict[str, Any]:
    """Reproduce the original TRAIN/DEV/RESERVED dossier split exactly.

    Returns dossier-id sets plus per-split row indices and question counts.
    """
    dossier_ids = [dossier_id(r) for r in train_rows]
    train_ids, val_ids = split_by_dossier_ids(dossier_ids)
    dev_ids, reserved_ids = split_dev_reserve(val_ids)

    # Map dossier id -> row index (dossier ids are unique per row in this corpus).
    by_dossier: dict[str, int] = {}
    for i, d in enumerate(dossier_ids):
        by_dossier.setdefault(d, i)

    def indices(id_set):
        return sorted(by_dossier[d] for d in id_set)

    def qcount(rows_idx):
        return sum(len(json.loads(train_rows[i]["questions"])) for i in rows_idx)

    tr_idx, dv_idx, rs_idx = indices(train_ids), indices(dev_ids), indices(reserved_ids)
    return {
        "params": {
            "split_seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "dev_reserve_seed": DEV_RESERVE_SEED,
            "n_dossiers": len(dossier_ids),
            "expected_train": 1020,
            "expected_dev": 90,
            "expected_reserved": 90,
        },
        "train_ids": sorted(train_ids),
        "dev_ids": sorted(dev_ids),
        "reserved_ids": sorted(reserved_ids),
        "train_indices": tr_idx,
        "dev_indices": dv_idx,
        "reserved_indices": rs_idx,
        "counts": {
            "train_dossiers": len(train_ids),
            "dev_dossiers": len(dev_ids),
            "reserved_dossiers": len(reserved_ids),
            "train_questions": qcount(tr_idx),
            "dev_questions": qcount(dv_idx),
            "reserved_questions": qcount(rs_idx),
        },
    }


# ---------------------------------------------------------------------------
# Case building (one Case per question, as in the original load_cases)
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """One typed decision: a state, a question, options and a soft target."""

    case_id: str
    kind: str
    options: list[str]
    target: list[float]
    state: str
    question: str


def build_cases(rows: list[dict], row_indices=None) -> list[Case]:
    """Flatten selected dossiers into canonical per-question Cases."""
    if row_indices is not None:
        rows = [rows[i] for i in row_indices]
    cases: list[Case] = []
    for row in rows:
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
            target = target_from_gold(g, keys)
            cases.append(Case(
                case_id=f"{row['id']}:{qid}",
                kind=str(qdef["type"]),
                options=options,
                target=target,
                state=state,
                question=str(qdef.get("instructions", "")),
            ))
    return cases


# ---------------------------------------------------------------------------
# Evaluation through the frozen checkpoint (canonical renderer)
# ---------------------------------------------------------------------------
def evaluate_rows(model, adapter, rows: list[dict]) -> list[dict]:
    """Evaluate dossiers with the canonical renderer; one record per decision."""
    records = []
    with torch.inference_mode():
        for row in rows:
            state_text = serialize_state(json.loads(row["state"]))
            qdefs = json.loads(row["questions"])
            gold = json.loads(row["gold"])
            questions, keys_by_q, gold_by_q = [], {}, {}
            for qid, qdef in qdefs.items():
                opts, keys = render_options(qdef)
                questions.append(Question(type=qdef["type"],
                                          instruction=qdef["instructions"],
                                          options=opts))
                keys_by_q[qid] = keys
                gold_by_q[qid] = gold[qid]
            sids, smask = adapter._encode_state(state_text)
            sufids, sufmask, counts = adapter._encode_options(questions)
            qids = [QTYPE_ID[q.type] for q in questions]
            logits_list = model.forward_case(sids, sufids, sufmask, counts, qids, smask)
            for qi, (qid, qdef) in enumerate(qdefs.items()):
                lf = logits_list[qi].float()
                probs = F.softmax(lf, dim=-1)
                g = gold_by_q[qid]
                keys = keys_by_q[qid]
                gold_label = str(g["label"])
                gi = keys.index(gold_label) if gold_label in keys else -1
                pi = int(probs.argmax().item())
                pred_label = keys[pi]
                correct = int(pred_label == gold_label)
                if gi >= 0:
                    inc = [lf[j].item() for j in range(len(keys)) if j != gi]
                    margin = float(lf[gi].item() - (max(inc) if inc else 0.0))
                else:
                    margin = float("nan")
                soft = target_from_gold(g, keys)
                records.append({
                    "dossier_id": row["id"],
                    "workflow": row.get("workflow", ""),
                    "question_index": qi,
                    "question_id": qid,
                    "qtype": qdef["type"],
                    "gold_label": gold_label,
                    "predicted_label": pred_label,
                    "correct": correct,
                    "logits": [float(x) for x in lf],
                    "probabilities": [float(x) for x in probs],
                    "option_keys": keys,
                    "gold_soft": [float(x) for x in soft],
                    "margin_correct_minus_best_incorrect": margin,
                    "state_token_count": int(smask.sum().item()),
                    "option_token_count": int(sufmask.sum().item()),
                    "n_options": len(keys),
                    "state_truncated": int(smask.sum().item()) >= 256,
                })
    return records


if __name__ == "__main__":
    tr, te = load_authoritative()
    sp = exact_split(tr)
    print(json.dumps({"counts": sp["counts"], "params": sp["params"]}, indent=2))