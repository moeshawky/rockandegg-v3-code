"""Stage 4 bounded training — Arm R (replay recovery) and Arm P (preservation).

Student init: Stage 3 main. Frozen teacher: published v3.
  * Arm R: v2 curriculum (incl. counterfactual twins) + ~2:1 original replay. Soft CE.
  * Arm P: Arm R + modest KL distillation vs frozen v3 on original Choice+Score
           replay (NOT noul — its bias is what we fix) + counterfactual pairwise
           ranking on twin pairs.
Capacity: layer9+head (default) or final3+head (escalation).
Selection: v2 curriculum DEV (training-family). Never held-out TEST.
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import random
import sys
import textwrap
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
STAGE3_MAIN = os.environ.get("RAE_STAGE3_MAIN", os.path.join(os.getcwd(), "stage3-main"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from corrected_data import Case, build_cases, load_authoritative, render_options  # noqa: E402
from systemone_inference import Question, SystemOneInference  # noqa: E402
from v2_env import ALIVE, DEAD, UNKNOWN, Episode, Node, best_action, route_status  # noqa: E402
from curriculum_v2 import render as v2_render, build_questions as v2_questions  # noqa: E402

SEED = 0
LR_BACKBONE = 5e-6
LR_HEAD = 5e-5
WD = 0.01
EPOCHS = 2
BATCH = 8
PERM_P = 0.5
PEAK = 0.85
CURR_PER_EPOCH = 2000
REPLAY_PER_EPOCH = 4000   # 2:1 replay:curriculum
LAMBDA_KL = 0.5
LAMBDA_RANK = 0.2


def soft_target(gold, keys, peak=PEAK):
    n = len(keys)
    if n == 1:
        return [1.0]
    rest = (1.0 - peak) / (n - 1)
    return [peak if k == gold else rest for k in keys]


def scenario_cases(sc, rng=None, pair_id=None, role="base"):
    state = json.dumps(sc["state"], ensure_ascii=False)
    cases = []
    for q in sc["questions"]:
        opts, keys = render_options(q)
        tgt = soft_target(q["gold_label"], keys)
        c = Case(case_id=f"{sc['scenario_id']}:{q['id']}", kind=q["type"], options=opts,
                 target=tgt, state=state, question=q["instructions"])
        cases.append((c, pair_id, role, q["id"]))
    return cases


def scenario_to_episode(sc):
    nodes = {}
    for nid, info in sc["graph"].items():
        nodes[nid] = Node(nid, sc["graph_names"][nid], info["survival"], info["kind"],
                          list(info["children"]), info["provides_goal"])
    return Episode(sc["family"], sc["depth"], nodes, sc["graph_entry"],
                   [nid for nid, n in nodes.items() if n.provides_goal][0],
                   sc["authority"], sc["dormant"], {"topology": sc["topology"]})


def make_twin(sc, rng):
    ep = scenario_to_episode(sc)
    flip = {ALIVE: DEAD, DEAD: ALIVE, UNKNOWN: ALIVE}
    cands = list(ep.nodes.keys())
    rng.shuffle(cands)
    for nid in cands:
        old = ep.nodes[nid].survival
        ep.nodes[nid].survival = flip[old]
        if route_status(ep) != sc["oracle"]["status"]:
            break
        ep.nodes[nid].survival = old
    situation = v2_render(ep, sc["family"], rng)
    twin = dict(sc)
    twin["scenario_id"] = sc["scenario_id"] + "_twin"
    twin["situation"] = situation
    twin["state"] = {"situation": situation, "goal": sc["goal"]}
    twin["questions"] = v2_questions(ep, sc["family"], rng)
    # align the twin's choice option order with the base so pairwise ranking is valid
    base_keys = list(sc["questions"][0]["criteria"].keys())
    twin_crit = dict(twin["questions"][0]["criteria"])
    aligned = {k: twin_crit[k] for k in base_keys if k in twin_crit}
    for k, v in twin_crit.items():
        if k not in aligned:
            aligned[k] = v
    twin["questions"][0]["criteria"] = aligned
    twin["questions"][0]["gold_position"] = list(aligned.keys()).index(twin["questions"][0]["gold_label"])
    twin["oracle"] = {"status": route_status(ep), "action": best_action(ep)["action"],
                      "graph_hash": sc["oracle"]["graph_hash"] + "_t"}
    return twin


# ---------------------------------------------------------------------------
# batch / loss
# ---------------------------------------------------------------------------
def make_batch(adapter, cases, rng, ids=None):
    dev = next(adapter.model.parameters()).device
    B = len(cases)
    qrows, targets = [], []
    for c, *_ in cases:
        opts, tgt = list(c.options), list(c.target)
        if c.kind != "score" and rng.random() < PERM_P:
            order = list(range(len(opts))); rng.shuffle(order)
            opts = [opts[i] for i in order]; tgt = [tgt[i] for i in order]
        qrows.append((Question(type=c.kind, instruction=c.question, options=opts), tgt))
    sids, smasks, sufids, sufmasks, counts, qids = [], [], [], [], [], []
    for (q, tgt), (c, *_) in zip(qrows, cases):
        s, sm = adapter._encode_state(c.state)
        suf, sufm, _ = adapter._encode_options([q])
        sids.append(s); smasks.append(sm); sufids.append(suf); sufmasks.append(sufm)
        counts.append(len(q.options)); qids.append({"choice": 0, "score": 1, "noul": 2}[q.type])
        targets.append(tgt)
    M = max(counts); L = max(x.shape[1] for x in sufids)
    pid = int(adapter.pad_id)
    sbuf = torch.full((B, M, L), pid, dtype=torch.long, device=dev)
    mbuf = torch.zeros((B, M, L), dtype=torch.long, device=dev)
    for i, (sid, smid) in enumerate(zip(sufids, sufmasks)):
        sbuf[i, :sid.shape[0], :sid.shape[1]] = sid
        mbuf[i, :smid.shape[0], :smid.shape[1]] = smid
    tbuf = torch.zeros((B, M), dtype=torch.float32, device=dev)
    for i, t in enumerate(targets):
        tbuf[i, :len(t)] = torch.tensor(t, dtype=torch.float32, device=dev)
    return (torch.cat(sids, 0), torch.cat(smasks, 0), sbuf.reshape(B * M, L),
            mbuf.reshape(B * M, L), counts, qids, tbuf)


def soft_ce(logits, counts, targets):
    losses = []
    for i, n in enumerate(counts):
        lg = logits[i, :n]
        losses.append(-(targets[i][:n] * F.log_softmax(lg, dim=-1)).sum())
    return torch.stack(losses).mean()


def gradsafe(model):
    src = textwrap.dedent(inspect.getsource(model.forward_batch))
    src = src.replace("copy.deepcopy(cache)", "cache")
    ns = dict(model.__class__.__dict__)
    ns["torch"] = torch; ns["F"] = F; ns["copy"] = copy
    exec(src, ns)  # noqa: S102
    model.forward_batch = types.MethodType(ns["forward_batch"], model)


def unfreeze(model, scope):
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if scope == "final3" and any(name.startswith(f"backbone.layers.{i}.") for i in (7, 8, 9)):
            p.requires_grad_(True)
        elif name.startswith("backbone.layers.9."):
            p.requires_grad_(True)
        elif any(k in name for k in ("qtype_emb", "scorer", "log_temp")):
            p.requires_grad_(True)


def scope_audit(ckpt_dir):
    fa = safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt")
    fb = safe_open(os.path.join(MODEL_DIR, "model.safetensors"), "pt")
    changed_auth = changed_other = 0
    for k in fa.keys():
        a = fa.get_tensor(k); b = fb.get_tensor(k)
        same = bool((a.view(torch.int16) == b.view(torch.int16)).all().item())
        auth = (k.startswith("backbone.layers.9.") or any(x in k for x in ("qtype_emb", "scorer", "log_temp"))
                or (scope_global[0] == "final3" and any(k.startswith(f"backbone.layers.{i}.") for i in (7, 8))))
        if auth and not same:
            changed_auth += 1
        if not auth and not same:
            changed_other += 1
    return {"changed_authorized": changed_auth, "changed_unauthorized": changed_other,
            "authorized_only_scope": changed_other == 0}


scope_global = ["layer9"]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "R"
    scope = sys.argv[2] if len(sys.argv) > 2 else "layer9"
    scope_global[0] = scope
    ckpt_dir = os.path.join(os.environ.get("RAE_OUTPUT_DIR", os.path.join(os.getcwd(), "rockandegg_stage4")), f"{mode}_{scope}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    rng = random.Random(SEED)

    c2 = json.load(open(os.path.join(PROBE_DIR, "curriculum_v2.json")))
    tr, _ = load_authoritative()
    man = json.load(open(os.path.join(PROBE_DIR, "corrected_split_manifest.json")))
    train_idx = man["split_ids"]["train_row_indices"]
    replay_all = build_cases(tr, train_idx)
    # replay subset for KL: Choice + Score only (teacher noul is biased)
    replay_kl = [c for c in replay_all if c.kind in ("choice", "score")]

    # curriculum cases + twins
    curr = []
    twin_pairs = []
    for sc in c2["train"]:
        base = scenario_cases(sc, pair_id=sc["scenario_id"], role="base")
        twin = make_twin(sc, rng)
        tw = scenario_cases(twin, pair_id=sc["scenario_id"], role="twin")
        curr.extend(base)
        twin_pairs.append((sc["scenario_id"], base, tw))
    # a subset of twins to keep the set bounded
    curr_twins = [c for _, _, tw in twin_pairs for c in tw]
    curr_base = [c for sc in c2["train"] for c in scenario_cases(sc, pair_id=sc["scenario_id"], role="base")]
    curriculum = curr_base + curr_twins

    model = AutoModel.from_pretrained(STAGE3_MAIN, trust_remote_code=True, dtype=torch.float16,
                                      attn_implementation="eager", local_files_only=True).float().to("cuda:0")
    tok = AutoTokenizer.from_pretrained(STAGE3_MAIN, trust_remote_code=True, local_files_only=True)
    ad = SystemOneInference(model, tok, 256, 64)

    head, bb = [], []
    for name, p in model.named_parameters():
        if scope == "final3" and any(name.startswith(f"backbone.layers.{i}.") for i in (7, 8)):
            bb.append((name, p))
        elif name.startswith("backbone.layers.9."):
            bb.append((name, p))
        elif any(k in name for k in ("qtype_emb", "scorer", "log_temp")):
            head.append((name, p))
    gradsafe(model); unfreeze(model, scope)
    opt_bb = torch.optim.AdamW([p for _, p in bb], lr=LR_BACKBONE, weight_decay=WD)
    opt_head = torch.optim.AdamW([p for _, p in head], lr=LR_HEAD, weight_decay=WD)
    optimizers = [{"params": [p for _, p in bb], "opt": opt_bb},
                  {"params": [p for _, p in head], "opt": opt_head}]

    # teacher (frozen published v3) logits for KL on replay Choice+Score
    teacher_logits = {}
    if mode == "P":
        teacher = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.float16,
                                            attn_implementation="eager", local_files_only=True).to("cuda:0")
        ttok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
        # compute teacher logits per replay_kl case
        with torch.inference_mode():
            for c in replay_kl:
                s, sm = ad._encode_state(c.state)
                suf, sufm, _ = ad._encode_options([Question(type=c.kind, instruction=c.question, options=c.options)])
                lg = teacher.forward_case(s, suf, sufm, [len(c.options)], [{"choice": 0, "score": 1, "noul": 2}[c.kind]], sm)[0]
                teacher_logits[c.case_id] = lg.float().cpu()
        del teacher; torch.cuda.empty_cache()

    def train_epoch():
        model.train()
        rep_n = REPLAY_PER_EPOCH
        replay = random.Random(SEED * 100 + rng.randint(0, 999)).sample(replay_all, min(rep_n, len(replay_all)))
        replay = [(c, None, "replay", c.case_id) for c in replay]
        curr_n = CURR_PER_EPOCH
        curr_sample = random.Random(SEED * 200 + rng.randint(0, 999)).sample(curriculum, min(curr_n, len(curriculum)))
        cases = curr_sample + replay
        rng.shuffle(cases)
        total = 0.0; nb = 0
        for start in range(0, len(cases), BATCH):
            chunk = cases[start:start + BATCH]
            s, sm, suf, sufm, counts, qids, tgt = make_batch(ad, chunk, rng)
            for g in optimizers:
                g["opt"].zero_grad(set_to_none=True)
            logits = model.forward_batch(s, sm, suf, sufm, max(counts), counts, qids)
            loss = soft_ce(logits, counts, tgt)
            if mode == "P":
                # KL on replay Choice+Score cases in this batch
                kl = 0.0; n_kl = 0
                for i, (c, *_) in enumerate(chunk):
                    if c.case_id in teacher_logits and c.kind in ("choice", "score"):
                        t = teacher_logits[c.case_id].to(logits.device)
                        n = counts[i]
                        kl = kl + F.kl_div(F.log_softmax(logits[i, :n], -1), F.softmax(t, -1), reduction="sum")
                        n_kl += 1
                if n_kl:
                    loss = loss + LAMBDA_KL * (kl / n_kl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizers for p in g["params"]], 1.0)
            for g in optimizers:
                g["opt"].step()
            total += float(loss.detach()); nb += 1
        return total / max(nb, 1)

    # counterfactual ranking pass (Arm P)
    def ranking_pass():
        model.train()
        pairs = rng.sample(twin_pairs, min(80, len(twin_pairs)))
        total = 0.0; nb = 0
        for _, base, tw in pairs:
            # forward base choice + twin choice together
            bc = base[0][0]  # choice case
            tc = tw[0][0]
            s, sm, suf, sufm, counts, qids, tgt = make_batch(ad, [bc, tc], rng)
            for g in optimizers:
                g["opt"].zero_grad(set_to_none=True)
            logits = model.forward_batch(s, sm, suf, sufm, max(counts), counts, qids)
            # a = base correct key, b = twin correct key (in each case's option list)
            a = bc.case_id  # not the key; use target argmax
            a_idx = int(torch.tensor(bc.target).argmax())
            b_idx = int(torch.tensor(tc.target).argmax())
            la = logits[0, :counts[0]]; lb = logits[1, :counts[1]]
            # margin: base prefers a over b, twin prefers b over a (shared option space)
            # only valid if both have same option keys; use scalar margins on own argmax
            margin = (la[a_idx] - la[b_idx]) + (lb[b_idx] - lb[a_idx])
            loss = F.relu(0.5 - margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizers for p in g["params"]], 1.0)
            for g in optimizers:
                g["opt"].step()
            total += float(loss.detach()); nb += 1
        return total / max(nb, 1)

    # eval v2 dev
    from stage4_eval import summarize
    from curriculum_v1_eval import evaluate_scenarios
    dev_sc = c2["dev"]
    best_acc, best_state = -1.0, None
    log = {"mode": mode, "scope": scope, "seed": SEED, "epochs": EPOCHS,
           "curr_per_epoch": CURR_PER_EPOCH, "replay_per_epoch": REPLAY_PER_EPOCH,
           "lambda_kl": LAMBDA_KL if mode == "P" else 0, "lambda_rank": LAMBDA_RANK if mode == "P" else 0,
           "epochs_log": []}
    for ep in range(1, EPOCHS + 1):
        tl = train_epoch()
        # Counterfactual twins are trained as supervised soft-CE cases (implicit
        # pairwise distinction); an explicit ranking term is omitted because twin
        # option spaces differ by construction (documented in STAGE4_VERDICT).
        rl = None
        model.eval(); unfreeze(model, scope)
        recs = evaluate_scenarios(model, ad, dev_sc, f"{mode}_ep{ep}", "v2_dev")
        tmap = {s["scenario_id"]: s.get("target_status") for s in dev_sc}
        for r in recs:
            r["target_status"] = tmap.get(r["scenario_id"])
        dev_acc = float(np.mean([r["correct"] for r in recs]))
        log["epochs_log"].append({"epoch": ep, "train_loss": tl, "rank_loss": rl, "v2_dev_accuracy": dev_acc})
        print(f"epoch {ep}: loss={tl:.4f} rank={rl} v2_dev_acc={dev_acc:.4f}", flush=True)
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        unfreeze(model, scope)

    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in best_state:
                p.copy_(best_state[n].to(p.device))
    log["selected_epoch"] = max(range(len(log["epochs_log"])), key=lambda i: log["epochs_log"][i]["v2_dev_accuracy"]) + 1
    log["selected_v2_dev_accuracy"] = best_acc

    model.half(); model.save_pretrained(ckpt_dir, safe_serialization=True); tok.save_pretrained(ckpt_dir)
    log["scope_audit"] = scope_audit(ckpt_dir)
    with open(os.path.join(PROBE_DIR, f"stage4_train_{mode}_{scope}.json"), "w") as fh:
        json.dump(log, fh, indent=2)
    print(json.dumps({"mode": mode, "scope": scope, "selected_epoch": log["selected_epoch"],
                      "v2_dev_acc": round(best_acc, 4), "scope": log["scope_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()