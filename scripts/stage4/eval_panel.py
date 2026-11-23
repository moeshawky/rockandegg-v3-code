"""Fresh, independently-authored evaluation panel (Stage 4).

Different construction process from Curriculum v2: NEW entity-name pools, NEW
situation templates, NEW distractor phrasings, and a new minimum-cost question
type. The ORACLE is the same v2_env three-valued truth (the mechanism is the
capability; the surface is what must not be memorized). This panel is FROZEN and
NEVER used for training, calibration, threshold fitting, or checkpoint selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from v2_env import ALIVE, DEAD, UNKNOWN, best_action, route_status  # noqa: E402
from curriculum_v2 import sample_branch, sample_chain  # noqa: E402

# Fresh name pools (different register from v2)
NODE_POOL = ["crimson", "hollow", "ironwood", "jade", "knoll", "loam", "moss",
             "nadir", "opal", "pyre", "quay", "rune", "slate", "tarn", "umber",
             "vane", "wold", "xylem", "yew", "zinc"]
RES_POOL = ["matrix", "lattice", "scaffold", "trellis", "girder", "plinth"]
CLIENT_POOL = ["concierge", "herald", "pursuivant", "sentry", "courier"]
ALIVE_P = ["nominal", "live", "responding", "present", "up", "serving"]
DEAD_P = ["flat", "gone", "dead", "down", "broken", "seized"]
UNK_P = ["unconfirmed", "in doubt", "untested", "of unknown state", "not yet verified"]


def _nm(rng, n):
    p = list(NODE_POOL); rng.shuffle(p); return p[:n]


def _w(rng, s):
    return rng.choice({ALIVE: ALIVE_P, DEAD: DEAD_P, UNKNOWN: UNK_P}[s])


def _auth(rng, a):
    return {"sufficient": "confirmed adequate", "insufficient": "confirmed inadequate",
            "unknown": "of unknown adequacy"}[a]


def _order(ep):
    from collections import deque
    order, seen, q = [], {ep.entry}, deque([ep.entry])
    while q:
        cur = q.popleft(); order.append(cur)
        for ch in ep.nodes[cur].children:
            if ch not in seen:
                seen.add(ch); q.append(ch)
    for n in ep.nodes:
        if n not in seen:
            order.append(n)
    return order


def render(ep, family, rng):
    order = _order(ep)
    if ep.meta.get("topology") == "branch":
        a = [n for n in order if n.startswith("A")]
        b = [n for n in order if n.startswith("B")]
        pa = " -> ".join(f"{ep.nodes[n].name} ({_w(rng, ep.nodes[n].survival)})" for n in a)
        pb = " -> ".join(f"{ep.nodes[n].name} ({_w(rng, ep.nodes[n].survival)})" for n in b)
        return (f"Two routes can yield the capability. Route 1: {pa}. Route 2: {pb}. "
                f"Authority is {_auth(rng, ep.authority)}; a route must be entirely live and the authority adequate.")
    parts = " -> ".join(f"{ep.nodes[n].name} ({_w(rng, ep.nodes[n].survival)})" for n in order)
    res = rng.choice(RES_POOL); rw = _w(rng, rng.choice([ALIVE, DEAD]))
    auth = _auth(rng, ep.authority)
    if family == "substrate":
        return (f"Infrastructure stack, highest first: {parts}. A stage functions only when every stage beneath it functions. "
                f"Stage {ep.nodes[order[-1]].name} provides the capability iff {res} is {rw}.")
    if family == "handle":
        cl = rng.choice(CLIENT_POOL); cw = _w(rng, rng.choice([ALIVE, DEAD]))
        return (f"A directory client ({cl}) is {cw}. Requests actually traverse: {parts}. "
                f"{ep.nodes[order[-1]].name} supplies the capability iff {res} is {rw}; authority is {auth}.")
    if family == "config":
        if ep.dormant and ep.dormant.get("present"):
            dw = ("an alternate profile exists and is fate-independent of the fault"
                  if ep.dormant.get("fate_independent")
                  else "an alternate profile exists but is fate-bound to the fault")
        else:
            dw = "no alternate profile exists"
        return (f"A service is pinned to one profile. Its dependency chain: {parts}. {dw.capitalize()}. Authority is {auth}.")
    return (f"One element outlived a fault; its former purpose is void. Remnants: {parts}. "
            f"{ep.nodes[order[-1]].name} yields the capability iff {res} is {rw}; authority is {auth}.")


def _dedup(pool, exclude, rng, n=3):
    rng.shuffle(pool); seen, out = {exclude}, []
    for k, v in pool:
        if k not in seen:
            seen.add(k); out.append((k, v))
    assert len(out) >= n
    return out[:n]


def build_questions(ep, family, rng):
    status = route_status(ep); ba = best_action(ep)
    nm = {n.nid: n.name for n in ep.nodes.values()}
    if ba["action"] == "invoke":
        ck = f"invoke_{nm[ba['node']]}"; cd = f"Invoke {nm[ba['node']]}; it is the highest live stage on a working route."
        pool = [(f"invoke_{n.name}", f"Retry {n.name}, which is not live.")
                for n in ep.nodes.values() if n.survival == DEAD]
        pool += [("report_impossible", "Declare the capability verified impossible."),
                 ("reboot_rebuild", "Reboot and rebuild everything."),
                 ("conclude_unreachable", "Conclude unreachable because something is unverified.")]
    elif ba["action"] == "reconfigure":
        ck = "reconfigure"; cd = "Switch to the fate-independent alternate profile to expose the capability."
        pool = [("report_impossible", "Declare the capability verified impossible."),
                ("reboot_rebuild", "Reboot and rebuild everything."),
                ("accept_ceiling", "Accept the pinned profile as the limit.")]
        pool += [(f"invoke_{n.name}", f"Retry {n.name}, which is not live.")
                 for n in ep.nodes.values() if n.survival == DEAD]
    elif ba["action"] == "probe":
        ck = f"probe_{nm[ba['node']]}"; cd = f"Run a cheap read on {nm[ba['node']]} to resolve its unknown state."
        pool = [("report_impossible", "Declare the capability verified impossible (treat unknown as dead)."),
                ("reboot_rebuild", "Reboot and rebuild everything."),
                ("conclude_unreachable", "Conclude unreachable because something is unverified.")]
        pool += [(f"invoke_{n.name}", f"Invoke {n.name} directly.")
                 for n in ep.nodes.values() if n.survival == ALIVE]
    else:
        ck = "report_impossible"; cd = "Report that every route is verified blocked (dead or inadequate authority)."
        pool = [("reboot_rebuild", "Reboot and rebuild everything."),
                ("accept_ceiling", "Accept the pinned profile as the limit.")]
        pool += [(f"invoke_{n.name}", f"Invoke {n.name} to force the capability.")
                 for n in ep.nodes.values() if n.survival == DEAD]
        pool += [("conclude_unreachable", "Conclude unreachable because something is unverified.")]
    distract = _dedup(pool, ck, rng)
    opts = [(ck, cd)] + distract; rng.shuffle(opts)
    crit = {k: v for k, v in opts}; keys = [k for k, _ in opts]
    assert len(keys) == 4 and len(set(keys)) == 4 and ck in keys, (keys, ck)
    q_choice = {"id": "choice", "type": "choice", "instructions": "What is the correct next move?",
                "criteria": crit, "gold_label": ck, "gold_position": keys.index(ck)}
    prov = nm.get(ep.goal_node) or nm[list(nm)[-1]]
    q_reach = {"id": "noul_reach", "type": "noul",
               "instructions": "Is there a verified live route with adequate authority to the goal?",
               "criteria": {"false": f"{prov} lacks a verified live route to the goal",
                            "true": f"{prov} has a verified live route to the goal"},
               "gold_label": "true" if status == "REACHABLE" else "false"}
    q_imposs = {"id": "noul_impossible", "type": "noul",
                "instructions": "Is the goal verified impossible (every route confirmed blocked)?",
                "criteria": {"false": f"a route to the goal via {prov} remains possible",
                             "true": f"every route to the goal via {prov} is confirmed blocked"},
                "gold_label": "true" if status == "UNREACHABLE" else "false"}
    q_unknown = {"id": "noul_unknown", "type": "noul",
                 "instructions": "Does at least one route remain unverified (neither confirmed nor blocked)?",
                 "criteria": {"false": f"routes to the goal via {prov} are fully resolved",
                              "true": f"a route to the goal via {prov} remains unverified"},
                 "gold_label": "true" if status == "UNKNOWN" else "false"}
    return [q_choice, q_reach, q_imposs, q_unknown]


def make(family, depth, target, rng, topology="chain"):
    ep = sample_branch(family, depth, target, rng) if topology == "branch" else sample_chain(family, depth, target, rng)
    situation = render(ep, family, rng)
    return {
        "scenario_id": f"panel_{family}_{topology}_d{depth}_{rng.randrange(1 << 30):09d}",
        "family": family, "depth": depth, "topology": topology, "target_status": target,
        "state": {"situation": situation, "goal": "reach the goal capability"},
        "situation": situation, "goal": "reach the goal capability",
        "questions": build_questions(ep, family, rng),
        "oracle": {"status": route_status(ep), "action": best_action(ep)["action"],
                   "graph_hash": __import__("v2_env").canonical_hash(ep)},
        "authority": ep.authority, "dormant": ep.dormant,
    }


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True, local_files_only=True)
    rng = random.Random(90210)
    n = 20
    panel = []
    for fam in ["substrate", "handle", "config", "survivor"]:
        for d in (1, 2, 3):
            for t in ("REACHABLE", "UNREACHABLE", "UNKNOWN"):
                for _ in range(n):
                    panel.append(make(fam, d, t, rng))
    # a few branch composition cases
    for t in ("REACHABLE", "UNREACHABLE", "UNKNOWN"):
        for _ in range(30):
            panel.append(make("mixed", 2, t, rng, topology="branch"))
    # validate
    over_s = over_o = bad = 0
    for sc in panel:
        st = json.dumps(sc["state"], ensure_ascii=False)
        if len(tok(st, add_special_tokens=False)["input_ids"]) > 256:
            over_s += 1
        for q in sc["questions"]:
            if q["type"] == "choice":
                opts = [f"{k}: {v}" if v else k for k, v in q["criteria"].items()]; keys = list(q["criteria"])
            else:
                opts = ["false: " + q["criteria"]["false"], "true: " + q["criteria"]["true"]]; keys = ["false", "true"]
            if q["gold_label"] not in keys:
                bad += 1
            for o in opts:
                if len(tok(o, add_special_tokens=False)["input_ids"]) > 64:
                    over_o += 1
    out = {"panel": "Stage4 independent evaluation panel", "synthetic": True,
           "evidence_class": "SYNTHETIC independent panel; NOT OD-1. Never used for training/calibration/selection.",
           "counts": {"scenarios": len(panel), "decisions": sum(len(c["questions"]) for c in panel)},
           "validation": {"over_state": over_s, "over_option": over_o, "bad_gold": bad,
                          "passed": over_s == 0 and over_o == 0 and bad == 0},
           "scenarios": panel}
    path = os.path.join(PROBE_DIR, "eval_panel.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    out["sha256"] = h
    with open(os.path.join(PROBE_DIR, "eval_panel_manifest.json"), "w") as fh:
        json.dump({"counts": out["counts"], "validation": out["validation"], "sha256": h}, fh, indent=2)
    print(json.dumps({"counts": out["counts"], "validation": out["validation"], "sha256": h}, indent=2))


if __name__ == "__main__":
    main()