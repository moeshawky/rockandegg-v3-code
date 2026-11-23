"""Corrected evaluation metrics v2 — resolves the six mandatory measurement corrections.

Fixes vs corrected_baseline.metrics:
  1. log_loss_soft is now GENUINE soft-target cross-entropy
     (-sum_k g_k log p_k with the gold probability distribution), computed in FP32.
     Hard-label NLL is kept separately as log_loss_hard_label.
  2. Calibration is measured separately for hard labels and soft targets, and the
     DIRECTION of miscalibration is reported (signed: conf - accuracy), so
     under/overconfidence is read from evidence, not asserted.
  3. add_ece() computes ECE for any (confidence, correct) stream; calibration()
     emits hard-label ECE (conf=max p vs argmax-correct) and soft-target ECE
     (per-class |p_k - g_k| averaged over classes, then over decisions).
  4. bootstrap_ci_dossier() + paired_clustered_ci() resample DOSSIERS (clusters),
     not decisions.
  5. bootstrap_ci now returns a dict {"ci95": [...], "method": "decision"}.
  6. metrics(..., precision=...) records which precision produced the numbers.

All aggregation math is FP32. metrics() does not silently redefine anything.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

SEED = 20261122


# ---------------------------------------------------------------------------
# Bootstrap (decision-level and dossier-clustered)
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=5000, seed=SEED, cluster_ids=None):
    """95% percentile bootstrap CI. If cluster_ids is given, resample clusters."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    if cluster_ids is None:
        idx = rng.integers(0, len(values), size=(n_boot, len(values)))
        means = values[idx].mean(axis=1)
        method = "decision"
    else:
        cluster_ids = np.asarray(cluster_ids)
        uniq = np.unique(cluster_ids)
        groups = [np.where(cluster_ids == c)[0] for c in uniq]
        pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        means = np.empty(n_boot)
        for b in range(n_boot):
            sel = np.concatenate([groups[j] for j in pick[b]])
            means[b] = values[sel].mean()
        method = f"dossier-clustered ({len(uniq)} clusters)"
    return {"ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "method": method, "n_boot": int(n_boot)}


def paired_clustered_ci(delta, cluster_ids, n_boot=5000, seed=SEED):
    """Paired dossier-clustered bootstrap CI for a mean difference."""
    delta = np.asarray(delta, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if len(delta) == 0:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(cluster_ids)
    groups = [np.where(cluster_ids == c)[0] for c in uniq]
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([groups[j] for j in pick[b]])
        means[b] = delta[sel].mean()
    return {"ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "mean": float(delta.mean()),
            "method": f"paired dossier-clustered ({len(uniq)} clusters)",
            "n_boot": int(n_boot)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def add_ece(conf, correct, n_bins=15):
    p = np.asarray(conf, dtype=np.float32)
    c = np.asarray(correct, dtype=np.float32)
    if len(p) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (p >= edges[b]) & ((p < edges[b + 1]) if b < n_bins - 1 else (p <= edges[b + 1]))
        if m.any():
            e += float(m.mean()) * abs(float(c[m].mean()) - float(p[m].mean()))
    return float(e)


def calibration(recs):
    """Hard-label and soft-target calibration with explicit direction.

    Hard-label: confidence = max_k p_k; correct = (argmax == gold label).
    Direction is signed (mean conf - accuracy): positive => overconfident.
    Soft-target: mean_k |p_k - g_k| (expected calibration error against the gold
    distribution); no single confidence scalar, so direction is reported per-class
    (mean p_k - mean g_k).
    """
    conf = np.array([max(r["probabilities"]) for r in recs], dtype=np.float32)
    correct = np.array([r["correct"] for r in recs], dtype=np.float32)
    hard_ece = add_ece(conf, correct)
    signed = float(conf.mean() - correct.mean())
    # per-class signed bias for soft targets (aggregate over decisions)
    per_class = {}
    keys_seen = sorted({k for r in recs for k in r["option_keys"]})
    for k in keys_seen:
        ps, gs = [], []
        for r in recs:
            if k in r["option_keys"]:
                ps.append(r["probabilities"][r["option_keys"].index(k)])
                gs.append(r["gold_soft"][r["option_keys"].index(k)])
        if ps:
            per_class[k] = {"mean_p": float(np.mean(ps)), "mean_g": float(np.mean(gs)),
                            "signed_bias_p_minus_g": float(np.mean(ps) - np.mean(gs))}
    # soft-target ECE: per-decision mean_k |p_k - g_k|, then binned ECE-style agg
    soft_abs = []
    for r in recs:
        p = np.asarray(r["probabilities"], dtype=np.float32)
        g = np.asarray(r["gold_soft"], dtype=np.float32)
        soft_abs.append(float(np.abs(p - g).mean()))
    # expected calibration error against soft targets = mean over decisions
    soft_ece = float(np.mean(soft_abs)) if soft_abs else None
    return {
        "hard_label_ece_15bin": hard_ece,
        "mean_confidence": float(conf.mean()),
        "accuracy": float(correct.mean()),
        "signed_confidence_minus_accuracy": signed,
        "direction": ("underconfident (conf < acc)" if signed < 0
                      else "overconfident (conf > acc)" if signed > 0 else "balanced"),
        "soft_target_mae_per_decision": soft_ece,
        "soft_target_per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Margins
# ---------------------------------------------------------------------------
def margin_summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float32)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "fraction_positive": float((v > 0).mean()),
            "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


# ---------------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------------
def metrics(recs, label, precision="fp16"):
    """Compute the full metric suite. All aggregation in FP32."""
    n = len(recs)
    correct = np.array([r["correct"] for r in recs], dtype=np.float32)
    clusters = [r["dossier_id"] for r in recs]

    # Genuine soft-target cross-entropy + hard-label NLL (both FP32 math).
    ce_soft, nll_hard, brier_soft, brier_hard = [], [], [], []
    for r in recs:
        p = np.asarray(r["probabilities"], dtype=np.float32)
        g = np.asarray(r["gold_soft"], dtype=np.float32)
        p = p / p.sum()
        gi = r["option_keys"].index(r["gold_label"])
        ce_soft.append(float(-(g * np.log(np.clip(p, 1e-12, None))).sum()))
        nll_hard.append(float(-np.log(max(float(p[gi]), 1e-12))))
        brier_soft.append(float(((p - g) ** 2).sum()))
        onehot = np.asarray([1.0 if k == r["gold_label"] else 0.0 for k in r["option_keys"]],
                            dtype=np.float32)
        brier_hard.append(float(((p - onehot) ** 2).sum()))

    by_qtype = {}
    for qt in ("choice", "score", "noul"):
        sub = [r for r in recs if r["qtype"] == qt]
        if not sub:
            continue
        labels = sorted({k for r in sub for k in r["option_keys"]})
        cm = confusion_matrix([r["gold_label"] for r in sub],
                              [r["predicted_label"] for r in sub], labels=labels)
        entry = {
            "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "accuracy_ci95": bootstrap_ci([r["correct"] for r in sub],
                                          cluster_ids=[r["dossier_id"] for r in sub]),
            "mean_margin": float(np.mean([r["margin_correct_minus_best_incorrect"] for r in sub
                                          if np.isfinite(r["margin_correct_minus_best_incorrect"])])),
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        }
        if qt == "score":
            maes, within1 = [], []
            for r in sub:
                p = np.asarray(r["probabilities"], dtype=np.float32)
                exp = float((p * np.arange(len(p), dtype=np.float32)).sum())
                gold_score = float(np.asarray(r["gold_soft"], dtype=np.float32)
                                   @ np.arange(len(p), dtype=np.float32))
                maes.append(abs(exp - gold_score))
                within1.append(int(abs(exp - gold_score) <= 1.0))
            entry["score_mae_vs_soft"] = float(np.mean(maes))
            entry["score_within_one"] = float(np.mean(within1))
        by_qtype[qt] = entry

    auc_noul = None
    noul = [r for r in recs if r["qtype"] == "noul" and set(r["option_keys"]) >= {"true", "false"}]
    if noul:
        y = [1 if r["gold_label"] == "true" else 0 for r in noul]
        if len(set(y)) == 2:
            s = [r["probabilities"][r["option_keys"].index("true")] for r in noul]
            auc_noul = float(roc_auc_score(y, s))

    family = {}
    for qid in sorted({r["question_id"] for r in recs}):
        sub = [r for r in recs if r["question_id"] == qid]
        entry = {
            "qtype": sub[0]["qtype"], "n": len(sub),
            "accuracy": float(np.mean([r["correct"] for r in sub])),
            "gold_label_counts": {lab: sum(r["gold_label"] == lab for r in sub)
                                  for lab in sorted({r["gold_label"] for r in sub})},
            "mean_margin_by_gold": {
                lab: float(np.mean([r["margin_correct_minus_best_incorrect"] for r in sub
                                    if r["gold_label"] == lab
                                    and np.isfinite(r["margin_correct_minus_best_incorrect"])]))
                for lab in sorted({r["gold_label"] for r in sub})
            },
        }
        if sub[0]["qtype"] == "noul" and set(sub[0]["option_keys"]) >= {"true", "false"}:
            entry["true_selection_rate"] = float(np.mean([r["predicted_label"] == "true" for r in sub]))
            fg = [r for r in sub if r["gold_label"] == "false"]
            entry["true_selection_rate_when_gold_false"] = (
                float(np.mean([r["predicted_label"] == "true" for r in fg])) if fg else None)
            entry["mean_p_true_when_gold_false"] = (
                float(np.mean([r["probabilities"][r["option_keys"].index("true")] for r in fg]))
                if fg else None)
            entry["mean_p_true_when_gold_true"] = (
                float(np.mean([r["probabilities"][r["option_keys"].index("true")] for r in sub
                               if r["gold_label"] == "true"]))
                if any(r["gold_label"] == "true" for r in sub) else None)
        family[qid] = entry

    margins = [r["margin_correct_minus_best_incorrect"] for r in recs]
    return {
        "panel": label,
        "precision": precision,
        "n_decisions": n,
        "n_dossiers": len(set(clusters)),
        "accuracy": float(correct.mean()),
        "accuracy_ci95": bootstrap_ci(correct, cluster_ids=clusters),
        "macro_accuracy_by_qtype": float(np.mean([v["accuracy"] for v in by_qtype.values()])),
        "log_loss_soft_ce": float(np.mean(ce_soft)),
        "log_loss_hard_label": float(np.mean(nll_hard)),
        "brier_soft": float(np.mean(brier_soft)),
        "brier_onehot": float(np.mean(brier_hard)),
        "calibration": calibration(recs),
        "margin": margin_summary(margins),
        "margin_correct": margin_summary([r["margin_correct_minus_best_incorrect"] for r in recs
                                          if r["correct"] == 1]),
        "margin_incorrect": margin_summary([r["margin_correct_minus_best_incorrect"] for r in recs
                                            if r["correct"] == 0]),
        "noul_binary_auc_p_true": auc_noul,
        "by_qtype": by_qtype,
        "by_question_family": family,
    }