"""Stage 4 evaluator — any checkpoint on v2 + independent panel + retention.

Evaluates a RockAndEgg checkpoint on:
  * typed-decisions DEV (retention),
  * v1 test + composition (Stage 3 gains retention),
  * v2 test (held-out families) + composition,
  * the fresh independent eval panel.

Reports: overall / by family / by depth / by oracle-status / by question-id
(choice, noul_reach, noul_impossible, noul_unknown), Noul balanced accuracy and
AUC per subtype, choice-on-unreachable, calibration, and scenario-clustered CIs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from corrected_data import load_authoritative, evaluate_rows  # noqa: E402
from corrected_metrics_v2 import metrics as td_metrics  # noqa: E402
from curriculum_v1_eval import evaluate_scenarios, clustered_ci, poisson_binomial_p  # noqa: E402


def _subset(recs, **kw):
    out = recs
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def _acc(recs):
    return float(np.mean([r["correct"] for r in recs])) if recs else None


def _noul_bacc(recs):
    y = [1 if r["gold"] == "true" else 0 for r in recs]
    if len(set(y)) < 2:
        return None
    pred = [1 if r["pred"] == "true" else 0 for r in recs]
    y = np.array(y); pred = np.array(pred)
    sens = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return float((sens + spec) / 2)


def _auc(recs):
    y = [1 if r["gold"] == "true" else 0 for r in recs]
    if len(set(y)) < 2:
        return None
    s = [r["probs"][r["keys"].index("true")] for r in recs]
    return float(roc_auc_score(y, s))


def summarize(recs, label):
    n = len(recs)
    clusters = [r["scenario_id"] for r in recs]
    p = [1.0 / r["n_options"] for r in recs]
    k = int(sum(r["correct"] for r in recs))
    pval, mu = poisson_binomial_p(k, p)

    by_family = {f: _acc(_subset(recs, family=f)) for f in sorted({r["family"] for r in recs})}
    by_depth = {str(d): _acc(_subset(recs, depth=d)) for d in sorted({r["depth"] for r in recs})}
    by_status = {s: _acc(_subset(recs, target_status=s)) for s in sorted({r.get("target_status") for r in recs if r.get("target_status")})}
    by_qid = {}
    for qid in sorted({r["qid"] for r in recs}):
        sub = _subset(recs, qid=qid)
        by_qid[qid] = {"n": len(sub), "accuracy": _acc(sub),
                       "uniform": float(np.mean([1.0 / r["n_options"] for r in sub]))}

    # noul per subtype
    noul_sub = {}
    for qid in [q for q in by_qid if q.startswith("noul")]:
        sub = _subset(recs, qid=qid)
        noul_sub[qid] = {"n": len(sub), "accuracy": _acc(sub), "balanced_accuracy": _noul_bacc(sub),
                         "auc": _auc(sub)}
    # choice by oracle status (unreachable recognition)
    ch = [r for r in recs if r["qtype"] == "choice"]
    choice_unreach = _acc([r for r in ch if r.get("target_status") == "UNREACHABLE"])
    choice_reach = _acc([r for r in ch if r.get("target_status") == "REACHABLE"])
    choice_unknown = _acc([r for r in ch if r.get("target_status") == "UNKNOWN"])
    # all noul balanced
    all_noul = [r for r in recs if r["qtype"] == "noul"]
    conf = np.array([max(r["probs"]) for r in recs], float)
    correct = np.array([r["correct"] for r in recs], float)
    ece = _ece(conf, correct)
    return {
        "panel": label, "n_decisions": n, "n_scenarios": len(set(clusters)),
        "accuracy": _acc(recs), "accuracy_ci95": clustered_ci(correct, clusters),
        "uniform_guess": float(np.mean(p)), "poisson_binomial_p_vs_uniform": pval,
        "by_family": by_family, "by_depth": by_depth, "by_status": by_status, "by_qid": by_qid,
        "noul_subtypes": noul_sub,
        "noul_all_balanced_accuracy": _noul_bacc(all_noul),
        "choice_on_reachable": choice_reach, "choice_on_unreachable": choice_unreach,
        "choice_on_unknown": choice_unknown,
        "ece": ece, "mean_conf": float(conf.mean()),
    }


def _ece(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for b in range(n_bins):
        m = (conf >= edges[b]) & ((conf < edges[b + 1]) if b < n_bins - 1 else (conf <= edges[b + 1]))
        if m.any():
            e += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(e)


def main():
    ckpt = sys.argv[1]
    out_name = sys.argv[2] if len(sys.argv) > 2 else "stage4_eval"
    model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.float16,
                                      attn_implementation="eager", local_files_only=True).to("cuda:0")
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    from systemone_inference import SystemOneInference
    ad = SystemOneInference(model, tok, 256, 64)

    c1 = json.load(open(os.path.join(PROBE_DIR, "curriculum_v1.json")))
    c2 = json.load(open(os.path.join(PROBE_DIR, "curriculum_v2.json")))
    panel = json.load(open(os.path.join(PROBE_DIR, "eval_panel.json")))["scenarios"]

    sets = {
        "v1_test": c1["test"], "v1_composition": c1["composition"],
        "v2_test": c2["test"], "v2_composition": c2["composition"],
        "panel": panel,
    }
    preds = {}
    for name, sc in sets.items():
        recs = evaluate_scenarios(model, ad, sc, ckpt, name)
        # enrich with target_status
        tmap = {s["scenario_id"]: s.get("target_status") for s in sc}
        for r in recs:
            r["target_status"] = tmap.get(r["scenario_id"])
        preds[name] = recs

    # typed-decisions DEV retention
    tr, _ = load_authoritative()
    man = json.load(open(os.path.join(PROBE_DIR, "corrected_split_manifest.json")))
    dev_rows = [tr[i] for i in man["split_ids"]["dev_row_indices"]]
    td_recs = evaluate_rows(model, ad, dev_rows)
    td_met = td_metrics(td_recs, "typed-decisions DEV FP16", precision="fp16")

    summary = {name: summarize(recs, name) for name, recs in preds.items()}
    summary["typed_decisions_dev"] = {"accuracy": td_met["accuracy"],
                                      "soft_ce": td_met["log_loss_soft_ce"],
                                      "noul_auc": td_met["noul_binary_auc_p_true"]}

    out = {"checkpoint": ckpt, "summary": summary, "predictions": preds}
    with open(os.path.join(PROBE_DIR, out_name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(PROBE_DIR, out_name + "_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump(preds, fh)

    print(f"checkpoint: {ckpt}")
    for name in ("typed_decisions_dev", "v1_test", "v1_composition", "v2_test", "v2_composition", "panel"):
        s = summary[name]
        if name == "typed_decisions_dev":
            print(f"  {name:16s} acc={s['accuracy']:.4f} softCE={s['soft_ce']:.4f} noul_auc={s['noul_auc']:.4f}")
        else:
            print(f"  {name:16s} acc={s['accuracy']:.4f} noul_bal={s['noul_all_balanced_accuracy']} "
                  f"choice_unreach={s['choice_on_unreachable']} noul_auc_reach={s['noul_subtypes'].get('noul_reach',{}).get('auc')}")


if __name__ == "__main__":
    main()