"""Curriculum v1 evaluator — frozen-baseline and post-training measurement.

Evaluates any RockAndEgg checkpoint on Curriculum v1 with:
  * accuracy by family x depth x qtype,
  * CORRECT baselines: per-question uniform-guess (1/n_options) with an exact
    Poisson-binomial p-value, a TRAIN-only majority baseline, and a label-prior,
  * Noul ROC-AUC where both labels occur,
  * soft cross-entropy (peak-0.85 target), Brier, hard-label NLL,
  * calibration SEPARATE from discrimination (hard-label ECE + signed margin),
  * scenario-clustered bootstrap CIs,
  * error-mechanism categorisation,
  * robustness: option-permutation, rename/irrelevant-fact invariance, and
    counterfactual consistency,
  * full prediction-level records with provenance (checkpoint id, split, state,
    option text, oracle evidence).

Usage:
  python3 curriculum_v1_eval.py baseline            # frozen published v3
  python3 curriculum_v1_eval.py eval <ckpt_dir> ...  # an experimental checkpoint
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from corrected_data import QTYPE_ID  # noqa: E402
from systemone_inference import Question, SystemOneInference  # noqa: E402

PEAK = 0.85
BOOT = 5000
SEED = 20261122


# ---------------------------------------------------------------------------
# soft target + helpers
# ---------------------------------------------------------------------------
def soft_target(gold, keys, peak=PEAK):
    n = len(keys)
    if n == 1:
        return [1.0]
    rest = (1.0 - peak) / (n - 1)
    return [peak if k == gold else rest for k in keys]


def poisson_binomial_pmf(p):
    dp = np.array([1.0])
    for pi in p:
        new = np.zeros(len(dp) + 1)
        new[:-1] += dp * (1.0 - pi)
        new[1:] += dp * pi
        dp = new
    return dp


def poisson_binomial_p(k_obs, p):
    pmf = poisson_binomial_pmf(p)
    ks = np.arange(len(pmf))
    mu = float((ks * pmf).sum())
    mask = np.abs(ks - mu) >= abs(k_obs - mu) - 1e-12
    return float(pmf[mask].sum()), mu


def clustered_ci(values, clusters, n_boot=BOOT, seed=SEED):
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    if len(values) == 0:
        return None
    uniq = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([groups[j] for j in pick[b]])
        means[b] = values[sel].mean()
    return {"ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "method": f"scenario-clustered ({len(uniq)} clusters)", "n_boot": int(n_boot)}


def paired_clustered_ci(delta, clusters, n_boot=BOOT, seed=SEED):
    delta = np.asarray(delta, float)
    clusters = np.asarray(clusters)
    if len(delta) == 0:
        return None
    uniq = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([groups[j] for j in pick[b]])
        means[b] = delta[sel].mean()
    return {"ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "mean": float(delta.mean()),
            "method": f"paired scenario-clustered ({len(uniq)} clusters)", "n_boot": int(n_boot)}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _render_options(q):
    if q["type"] == "choice":
        crit = q["criteria"]
        opts = [f"{k}: {v}" if v else k for k, v in crit.items()]
        keys = list(crit.keys())
    elif q["type"] == "score":
        crit = q["criteria"]
        opts = [f"level {i}: {v}" for i, v in enumerate(crit)]
        keys = [str(i) for i in range(len(opts))]
    else:  # noul
        crit = q["criteria"]
        opts = ["false: " + (crit.get("false") or "no, the statement does not hold"),
                "true: " + (crit.get("true") or "yes, the statement holds")]
        keys = ["false", "true"]
    return opts, keys


def evaluate_scenarios(model, adapter, scenarios, checkpoint_id, split_name):
    """One forward_case per scenario; one record per question."""
    records = []
    with torch.inference_mode():
        for sc in scenarios:
            state_text = json.dumps(sc["state"], ensure_ascii=False)
            questions, keys_by_q, gold_by_q = [], {}, {}
            for q in sc["questions"]:
                opts, keys = _render_options(q)
                questions.append(Question(type=q["type"], instruction=q["instructions"], options=opts))
                keys_by_q[q["id"]] = keys
                gold_by_q[q["id"]] = q["gold_label"]
            sids, smask = adapter._encode_state(state_text)
            sufids, sufmask, counts = adapter._encode_options(questions)
            qids = [QTYPE_ID[q.type] for q in questions]
            logits_list = model.forward_case(sids, sufids, sufmask, counts, qids, smask)
            for qi, q in enumerate(sc["questions"]):
                lf = logits_list[qi].float()
                probs = F.softmax(lf, dim=-1).cpu().numpy()
                keys = keys_by_q[q["id"]]
                gold = gold_by_q[q["id"]]
                pi = int(probs.argmax())
                pred = keys[pi]
                gold_soft = soft_target(gold, keys)
                rec = {
                    "scenario_id": sc["scenario_id"],
                    "family": sc["family"],
                    "depth": sc["depth"],
                    "topology": sc.get("topology", "chain"),
                    "qtype": q["type"],
                    "qid": q["id"],
                    "n_options": len(keys),
                    "gold": gold,
                    "pred": pred,
                    "correct": int(pred == gold),
                    "probs": [float(x) for x in probs],
                    "keys": keys,
                    "gold_soft": gold_soft,
                    "state_text": state_text,
                    "option_texts": [f"{q['instructions']} [SEP] {o} [SEP]"
                                     for o in _render_options(q)[0]],
                    "oracle_reachable": sc["oracle"].get("reachable", sc["oracle"].get("status") == "REACHABLE"),
                    "oracle_status": sc["oracle"].get("status"),
                    "oracle_action": sc["oracle"].get("action", sc["oracle"].get("action")),
                    "oracle_chain": sc["oracle"].get("chain"),
                    "checkpoint_id": checkpoint_id,
                    "split": split_name,
                }
                rec["error_category"] = _error_category(rec, sc)
                records.append(rec)
    return records


def _error_category(rec, sc):
    if rec["correct"]:
        return "correct"
    qt = rec["qtype"]
    if qt == "noul":
        return "false_when_reachable" if rec["oracle_reachable"] else "true_when_unreachable"
    # choice
    gold = rec["gold"]
    if rec["oracle_reachable"]:
        if gold == "report_unreachable":
            return "other"
        return "picked_wrong_live_or_dead" if not gold.startswith("invoke") and not gold.startswith("use") else "picked_wrong_stage"
    else:
        # unreachable: gold is report_*
        if rec["pred"].startswith(("invoke", "use")):
            return "picked_stage_when_unreachable"
        if rec["pred"] in ("reboot", "rebuild", "reboot_rebuild", "reboot_fresh"):
            return "picked_reboot_rebuild"
        return "other"


# ---------------------------------------------------------------------------
# baselines (fit on TRAIN only)
# ---------------------------------------------------------------------------
def compute_baselines(train_scenarios, eval_scenarios):
    # uniform-guess per question
    def uniform(recs):
        p = [1.0 / r["n_options"] for r in recs]
        return float(np.mean(p)), p

    # majority per qtype fit on TRAIN
    by_qt = defaultdict(list)
    for sc in train_scenarios:
        for q in sc["questions"]:
            by_qt[q["type"]].append(q["gold_label"])
    maj = {qt: Counter(v).most_common(1)[0][0] for qt, v in by_qt.items()}

    # label-prior per qtype fit on TRAIN
    prior = {qt: Counter(v).most_common(1)[0][0] for qt, v in by_qt.items()}

    # build eval records skeleton (gold + qtype + n_options) without a model
    skel = []
    for sc in eval_scenarios:
        for q in sc["questions"]:
            n = len(q["criteria"]) if q["type"] != "noul" else 2
            skel.append({"qtype": q["type"], "gold": q["gold_label"], "n_options": n})

    def maj_acc():
        return float(np.mean([int(maj[r["qtype"]] == r["gold"]) for r in skel]))

    def prior_acc():
        return float(np.mean([int(prior[r["qtype"]] == r["gold"]) for r in skel]))

    u_mean, u_p = uniform(skel)
    return {
        "uniform_guess_accuracy": u_mean,
        "uniform_guess_p": u_p,
        "majority_labels": maj,
        "majority_accuracy": maj_acc(),
        "label_prior_labels": prior,
        "label_prior_accuracy": prior_acc(),
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _slice_accuracy(records):
    if not records:
        return None
    return float(np.mean([r["correct"] for r in records]))


def _slice_uniform(records):
    if not records:
        return None
    return float(np.mean([1.0 / r["n_options"] for r in records]))


def summarize(records, baselines, label):
    n = len(records)
    clusters = [r["scenario_id"] for r in records]
    correct = np.array([r["correct"] for r in records], float)

    # overall
    acc = float(correct.mean())
    p = baselines["uniform_guess_p"]
    k = int(correct.sum())
    pval, mu = poisson_binomial_p(k, p)

    # by family / depth / family x depth / qtype
    def group(keyfn):
        g = defaultdict(list)
        for r in records:
            g[keyfn(r)].append(r)
        return {str(k): {
            "n": len(v),
            "accuracy": _slice_accuracy(v),
            "uniform": _slice_uniform(v),
            "accuracy_ci95": clustered_ci([x["correct"] for x in v],
                                          [x["scenario_id"] for x in v]),
        } for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    by_family = group(lambda r: r["family"])
    by_depth = group(lambda r: r["depth"])
    by_family_depth = group(lambda r: f"{r['family']}|d{r['depth']}")
    by_qtype = group(lambda r: r["qtype"])
    by_family_qtype = group(lambda r: f"{r['family']}|{r['qtype']}")

    # calibration (hard-label) separate from discrimination
    conf = np.array([max(r["probs"]) for r in records], float)
    hard_ece = _ece(conf, correct.astype(float))
    signed = float(conf.mean() - acc)

    # soft CE / Brier / hard NLL (FP32)
    ce_soft, brier, nll = [], [], []
    for r in records:
        pr = np.asarray(r["probs"], float)
        pr = pr / pr.sum()
        g = np.asarray(r["gold_soft"], float)
        gi = r["keys"].index(r["gold"])
        ce_soft.append(float(-(g * np.log(np.clip(pr, 1e-12, None))).sum()))
        brier.append(float(((pr - g) ** 2).sum()))
        nll.append(float(-np.log(max(float(pr[gi]), 1e-12))))

    # noul AUC
    noul = [r for r in records if r["qtype"] == "noul"]
    auc = None
    if noul:
        y = [1 if r["gold"] == "true" else 0 for r in noul]
        if len(set(y)) == 2:
            s = [r["probs"][r["keys"].index("true")] for r in noul]
            auc = float(roc_auc_score(y, s))

    # error categories
    err = Counter(r["error_category"] for r in records if not r["correct"])

    return {
        "panel": label,
        "n_decisions": n,
        "n_scenarios": len(set(clusters)),
        "accuracy": acc,
        "accuracy_ci95": clustered_ci(correct, clusters),
        "uniform_guess_accuracy": baselines["uniform_guess_accuracy"],
        "uniform_guess_expected_count": float(mu),
        "model_correct_count": k,
        "poisson_binomial_p_vs_uniform": pval,
        "majority_accuracy_trainfit": baselines["majority_accuracy"],
        "label_prior_accuracy_trainfit": baselines["label_prior_accuracy"],
        "by_family": by_family,
        "by_depth": by_depth,
        "by_family_depth": by_family_depth,
        "by_qtype": by_qtype,
        "by_family_qtype": by_family_qtype,
        "calibration": {
            "hard_label_ece_15bin": hard_ece,
            "mean_confidence": float(conf.mean()),
            "accuracy": acc,
            "signed_conf_minus_acc": signed,
            "direction": ("underconfident" if signed < 0 else "overconfident" if signed > 0 else "balanced"),
        },
        "log_loss_soft_ce": float(np.mean(ce_soft)),
        "brier_soft": float(np.mean(brier)),
        "log_loss_hard_label": float(np.mean(nll)),
        "noul_binary_auc_p_true": auc,
        "error_categories": dict(err),
    }


def _ece(conf, correct, n_bins=15):
    if len(conf) == 0:
        return None
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for b in range(n_bins):
        m = (conf >= edges[b]) & ((conf < edges[b + 1]) if b < n_bins - 1 else (conf <= edges[b + 1]))
        if m.any():
            e += float(m.mean()) * abs(float(correct[m].mean()) - float(conf[m].mean()))
    return float(e)


# ---------------------------------------------------------------------------
# robustness analysis (paired base vs variant)
# ---------------------------------------------------------------------------
def robustness_analysis(base_records, rob_records):
    """base_records: records for the base scenarios.
    rob_records: list of (variant_name, records)."""
    base_by_sc = defaultdict(dict)
    for r in base_records:
        base_by_sc[r["scenario_id"]][r["qtype"]] = r

    out = {}
    by_variant = defaultdict(list)
    for vname, recs in rob_records:
        by_variant[vname].append(recs)
    for vname, groups in by_variant.items():
        # each group is the records for one variant instance; pair with base by base scenario id
        agree, total, base_correct, var_correct = 0, 0, 0, 0
        for recs in groups:
            if not recs:
                continue
            # find the base scenario id from the record's metadata (stored on variant records)
            bid = recs[0].get("base_id")
            if bid is None or bid not in base_by_sc:
                continue
            for r in recs:
                br = base_by_sc[bid].get(r["qtype"])
                if br is None:
                    continue
                total += 1
                base_correct += br["correct"]
                var_correct += r["correct"]
                # prediction agreement (same predicted key)
                if br["pred"] == r["pred"]:
                    agree += 1
        out[vname] = {
            "n": total,
            "base_accuracy": base_correct / total if total else None,
            "variant_accuracy": var_correct / total if total else None,
            "prediction_agreement": agree / total if total else None,
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt(path):
    model = AutoModel.from_pretrained(path, trust_remote_code=True,
                                      dtype=torch.float16, attn_implementation="eager",
                                      local_files_only=True).to("cuda:0")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    adapter = SystemOneInference(model, tok, max_state_len=256, max_option_len=64)
    return model, adapter


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    cur = json.load(open(os.path.join(PROBE_DIR, "curriculum_v1.json")))

    if mode == "baseline":
        ckpt_path = MODEL_DIR
        ckpt_id = "published_v3_" + _sha(os.path.join(ckpt_path, "model.safetensors"))[:12]
        out_name = "curriculum_v1_baseline"
    else:
        ckpt_path = sys.argv[2]
        ckpt_id = "exp_" + _sha(os.path.join(ckpt_path, "model.safetensors"))[:12]
        out_name = sys.argv[3] if len(sys.argv) > 3 else "curriculum_v1_exp_eval"

    model, adapter = load_ckpt(ckpt_path)

    splits = {
        "dev": cur["dev"],
        "test": cur["test"],
        "composition": cur["composition"],
    }

    all_records = {}
    for sname, scenarios in splits.items():
        recs = evaluate_scenarios(model, adapter, scenarios, ckpt_id, sname)
        all_records[sname] = recs
        print(f"{sname}: {len(recs)} decisions", flush=True)

    # baselines: fit on TRAIN, apply per split
    train_sc = cur["train"]
    base_baselines = compute_baselines(train_sc, cur["dev"])
    test_baselines = compute_baselines(train_sc, cur["test"])
    comp_baselines = compute_baselines(train_sc, cur["composition"])

    summary = {
        "dev": summarize(all_records["dev"], base_baselines, "dev (train-family, checkpoint selection)"),
        "test": summarize(all_records["test"], test_baselines, "test (held-out families)"),
        "composition": summarize(all_records["composition"], comp_baselines, "composition (novel topology)"),
    }

    # robustness: build base records for the robustness base scenarios
    rob_bases = {}
    for entry in cur["robustness"]:
        bid = entry["base_id"]
        if bid not in rob_bases:
            # find the base scenario in test
            for sc in cur["test"]:
                if sc["scenario_id"] == bid:
                    rob_bases[bid] = sc
                    break
    base_rob_scenarios = list(rob_bases.values())
    base_rob_records = evaluate_scenarios(model, adapter, base_rob_scenarios, ckpt_id, "robustness_base")

    rob_records = []
    for entry in cur["robustness"]:
        sc = entry["scenario"]
        recs = evaluate_scenarios(model, adapter, [sc], ckpt_id, f"rob_{entry['variant']}")
        for r in recs:
            r["base_id"] = entry["base_id"]
            r["variant"] = entry["variant"]
        rob_records.append((entry["variant"], recs))
    rob_summary = robustness_analysis(base_rob_records, rob_records)

    out = {
        "checkpoint_id": ckpt_id,
        "checkpoint_path": ckpt_path,
        "curriculum_sha256": _sha(os.path.join(PROBE_DIR, "curriculum_v1.json")),
        "summary": summary,
        "robustness": rob_summary,
        "predictions": {k: v for k, v in all_records.items()},
        "robustness_base_predictions": base_rob_records,
    }
    # strip the large prediction arrays from the summary file; write separately
    summary_only = {k: v for k, v in out.items() if k not in ("predictions", "robustness_base_predictions")}
    with open(os.path.join(PROBE_DIR, out_name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(summary_only, fh, indent=2)
    with open(os.path.join(PROBE_DIR, out_name + "_predictions.json"), "w", encoding="utf-8") as fh:
        json.dump({"predictions": all_records, "robustness_base": base_rob_records,
                   "robustness_variants": [[v, r] for v, r in rob_records]}, fh)

    # compact print
    for sname in ("dev", "test", "composition"):
        s = summary[sname]
        print(f"\n[{sname}] acc={s['accuracy']:.4f} uniform={s['uniform_guess_accuracy']:.4f} "
              f"p_vs_uniform={s['poisson_binomial_p_vs_uniform']:.2e} "
              f"maj={s['majority_accuracy_trainfit']:.4f} "
              f"softCE={s['log_loss_soft_ce']:.4f} brier={s['brier_soft']:.4f} "
              f"noul_auc={s['noul_binary_auc_p_true']} "
              f"ECE={s['calibration']['hard_label_ece_15bin']:.4f} "
              f"dir={s['calibration']['direction']}")
        print(f"    by_family={ {k: round(v['accuracy'],3) for k,v in s['by_family'].items()} }")
        print(f"    by_depth={ {k: round(v['accuracy'],3) for k,v in s['by_depth'].items()} }")
    print("\nrobustness:", json.dumps(rob_summary, indent=2))


if __name__ == "__main__":
    main()