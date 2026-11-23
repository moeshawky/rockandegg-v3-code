"""Curriculum v2 — evidence-aware counterfactual mechanism curriculum.

Builds on v2_env.py (three-valued Kleene oracle). Each episode is a capability
route with survival (ALIVE/DEAD/UNKNOWN), authority (sufficient/insufficient/
unknown), and an optional dormant interface. Every scenario yields:
  * a best-action Choice question (invoke highest survivor / reconfigure /
    probe cheapest unknown / report verified-impossible),
  * three Noul questions that keep the three-valued distinction:
      verified_reachable, verified_impossible, unknown_remains.

Families (surface stories): substrate, handle, config, survivor. Novel composition
uses branch (OR) topologies. Every family spans 1/2/3-hop depths.

SYNTHETIC. NOT OD-1 evidence.
"""
from __future__ import annotations

import json
import os
import random
import sys

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from v2_env import (  # noqa: E402
    ALIVE, DEAD, UNKNOWN, Episode, best_action, branch_episode, canonical_hash,
    chain_episode, metamorphic_tests, route_status,
)

FAMILIES = ["substrate", "handle", "config", "survivor"]
TRAIN_FAMILIES = ["substrate", "handle"]
HELDOUT_FAMILIES = ["config", "survivor"]
DEPTHS = [1, 2, 3]
TARGETS = ["REACHABLE", "UNREACHABLE", "UNKNOWN"]

STAGE_NAMES = ["atlas", "birch", "cedar", "delta", "ember", "forge", "grove",
               "harbor", "iris", "juniper", "kestrel", "lumen", "mesa", "north",
               "onyx", "prairie", "quartz", "ridge", "summit", "timber",
               "umber", "vale", "willow", "yarrow", "zephyr"]
RESOURCE_NAMES = ["bedrock", "anchor", "reservoir", "backing", "core", "substrate"]
CLIENT_NAMES = ["resolver", "broker", "gateway", "dispatcher", "locator"]
ALIVE_WORDS = ["operational", "intact", "responding", "available", "live", "healthy"]
DEAD_WORDS = ["destroyed", "failed", "unreachable", "offline", "broken", "seized"]
UNKNOWN_WORDS = ["unverified", "unknown", "uncertain", "not yet checked"]

# Varied instruction phrasings (defeat question-text memorization)
REACH_Q = [
    "Is there a verified route of surviving components with sufficient authority to the goal?",
    "Can the goal be reached through components confirmed to be live, with adequate authority?",
    "Is a fully verified path to the goal available right now?",
    "Do the surviving components provide a confirmed route to the goal?",
    "Is the goal reachable through verified-live components and sufficient authority?",
]
IMPOSS_Q = [
    "Is the goal verified impossible (every route confirmed blocked)?",
    "Is it confirmed that no route can reach the goal?",
    "Are all routes to the goal definitively blocked?",
    "Is the goal provably out of reach, with every dependency resolved?",
    "Has every possible route been confirmed dead or authority-blocked?",
]
UNKNOWN_Q = [
    "Does at least one route remain unverified (neither confirmed nor blocked)?",
    "Is there any route whose status is still unknown?",
    "Are some dependencies still untested rather than confirmed dead?",
    "Is the goal's reachability still undetermined rather than impossible?",
    "Does uncertainty remain about whether a route can succeed?",
]
CHOICE_Q = [
    "What is the correct next move?",
    "Which action should the operator take?",
    "What should be done next?",
    "Choose the correct intervention.",
    "Which move is correct?",
]


def _names(rng, n):
    p = list(STAGE_NAMES); rng.shuffle(p); return p[:n]


def _sw(rng, s):
    return rng.choice({ALIVE: ALIVE_WORDS, DEAD: DEAD_WORDS, UNKNOWN: UNKNOWN_WORDS}[s])


def _auth_word(rng, a):
    return {"sufficient": "verified sufficient", "insufficient": "verified insufficient",
            "unknown": "unverified"}[a]


def _sanitize(x):
    return "".join(c if c.isalnum() else "_" for c in x).lower()


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------
def sample_chain(family, depth, target, rng):
    """Chain of depth+1 nodes; goal provider survival/authority/dormant set to hit target."""
    names = _names(rng, depth + 1)
    goal = ALIVE if target == "REACHABLE" else DEAD if target == "UNREACHABLE" else UNKNOWN
    survival = [goal] + [rng.choice([ALIVE, ALIVE, DEAD, UNKNOWN]) for _ in range(depth)]
    survival = survival[::-1]  # survival[0]=entry ... survival[depth]=goal
    survival[depth] = goal
    # authority
    if family == "handle":
        if target == "REACHABLE":
            auth = "sufficient"
        elif target == "UNREACHABLE":
            auth = "insufficient"
        else:
            auth = "unknown"
    elif target == "UNKNOWN":
        auth = "unknown"
    else:
        auth = "sufficient"
    # dormant (config family rescues unreachable; sometimes present elsewhere)
    dormant = None
    if family == "config":
        if target == "UNREACHABLE":
            dormant = {"present": True, "fate_independent": False, "provides_goal": True}
        elif target == "REACHABLE" and rng.random() < 0.5:
            # dormant rescue path: goal dead but dormant fate-independent
            survival[depth] = DEAD
            dormant = {"present": True, "fate_independent": True, "provides_goal": True}
        elif rng.random() < 0.5:
            dormant = {"present": True, "fate_independent": True, "provides_goal": True}
        else:
            dormant = {"present": False, "fate_independent": False, "provides_goal": False}
    ep = chain_episode(family, depth, survival, auth, dormant, names)
    # verify
    st = route_status(ep)
    assert st == target, (family, depth, target, st, survival, auth, dormant)
    return ep


def sample_branch(family, depth, target, rng):
    """Branch: entry OR (A-chain, B-chain) -> goal. Both routes end at a provider."""
    d = max(1, depth - 1)
    names = _names(rng, 1 + 2 * (d + 1))
    # route lengths d+1 each
    def route(target_route):
        # return survival list for a route of length d+1
        return [rng.choice([ALIVE, DEAD, UNKNOWN]) for _ in range(d + 1)]
    ra, rb = route(None), route(None)
    auth = "sufficient" if target != "UNREACHABLE" else "insufficient"
    if target == "UNKNOWN":
        auth = "unknown"
    ep = branch_episode(family, d, ra, rb, auth, None, names)
    # adjust to hit target
    if target == "REACHABLE":
        # ensure at least one route fully alive with sufficient authority
        ra = [ALIVE] * (d + 1)
        if auth == "insufficient":
            auth = "sufficient"
        ep = branch_episode(family, d, ra, rb, auth, None, names)
    elif target == "UNREACHABLE":
        ra = [DEAD] * (d + 1); rb = [DEAD] * (d + 1)
        ep = branch_episode(family, d, ra, rb, "insufficient", None, names)
    else:  # UNKNOWN
        ra = [UNKNOWN] * (d + 1); rb = [DEAD] * (d + 1)
        ep = branch_episode(family, d, ra, rb, "sufficient", None, names)
    st = route_status(ep)
    assert st == target, (family, d, target, st, ra, rb)
    ep.meta["topology"] = "branch"
    return ep


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(ep, family, rng):
    order = _bfs(ep)
    if ep.meta.get("topology") == "branch":
        # two alternative routes from the entry
        a = [n for n in order if n.startswith("A")]
        b = [n for n in order if n.startswith("B")]
        pa = " -> ".join(f"{ep.nodes[n].name} ({_sw(rng, ep.nodes[n].survival)})" for n in a)
        pb = " -> ".join(f"{ep.nodes[n].name} ({_sw(rng, ep.nodes[n].survival)})" for n in b)
        return (f"Two alternative routes deliver the capability. Route 1: {pa}. Route 2: {pb}. "
                f"The goal is delivered if any route is fully live and the authority ({_auth_word(rng, ep.authority)}) is sufficient.")
    parts = " -> ".join(f"{ep.nodes[n].name} ({_sw(rng, ep.nodes[n].survival)})" for n in order)
    res = rng.choice(RESOURCE_NAMES)
    res_word = _sw(rng, rng.choice([ALIVE, DEAD]))
    auth = _auth_word(rng, ep.authority)
    if family == "substrate":
        return (f"Layers, top to bottom: {parts}. Each layer relies on the one below it. "
                f"The lowest component {ep.nodes[order[-1]].name} delivers the goal capability "
                f"only if {res} is {res_word}.")
    if family == "handle":
        client = rng.choice(CLIENT_NAMES); cw = _sw(rng, rng.choice([ALIVE, DEAD]))
        return (f"A lookup client ({client}) is {cw}. The service itself is reached through: {parts}. "
                f"The provider {ep.nodes[order[-1]].name} yields the capability only if {res} is {res_word}. "
                f"The authority to act is {auth}.")
    if family == "config":
        if ep.dormant and ep.dormant.get("present"):
            dw = ("a dormant secondary interface exists and does not share fate with the broken layer"
                  if ep.dormant.get("fate_independent")
                  else "a dormant secondary interface exists but shares fate with the broken layer")
        else:
            dw = "no dormant secondary interface exists"
        return (f"A component runs in a fixed view. The dependency chain to the full capability is: {parts}. "
                f"{dw.capitalize()}. The authority to act is {auth}.")
    # survivor
    return (f"A component survived a failure; its original role is gone. What remains: {parts}. "
            f"The provider {ep.nodes[order[-1]].name} can deliver the capability only if {res} is {res_word}. "
            f"The authority to act is {auth}.")


def _bfs(ep):
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


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
def _dedup(pool, exclude, rng, n=3):
    rng.shuffle(pool)
    seen, out = {exclude}, []
    for k, v in pool:
        if k not in seen:
            seen.add(k); out.append((k, v))
    assert len(out) >= n, (exclude, len(out))
    return out[:n]


def build_questions(ep, family, rng):
    d = ep.meta  # not used
    status = route_status(ep)
    ba = best_action(ep)
    nm = {n.nid: n.name for n in ep.nodes.values()}
    # ---- choice ----
    if ba["action"] == "invoke":
        ck = f"invoke_{_sanitize(nm[ba['node']])}"
        cd = f"Invoke {nm[ba['node']]} directly; it is the highest surviving node with a live route."
        pool = []
        # dead nodes (retry dead = playbook gravity)
        for n in ep.nodes.values():
            if n.survival == DEAD:
                pool.append((f"invoke_{_sanitize(n.name)}", f"Retry through {n.name}, which is not live."))
        pool.append(("report_impossible", "Report that the goal is verified impossible."))
        pool.append(("reboot_rebuild", "Reboot and rebuild the whole stack from scratch."))
        pool.append(("conclude_unreachable", "Conclude unreachable because a dependency is unverified."))
    elif ba["action"] == "reconfigure":
        ck = "reconfigure"
        cd = "Enable the dormant, fate-independent interface to expose the capability."
        pool = [
            ("report_impossible", "Report that the goal is verified impossible."),
            ("reboot_rebuild", "Reboot and rebuild the whole stack from scratch."),
            ("accept_ceiling", "Accept the current fixed view as the capability limit."),
        ]
        for n in ep.nodes.values():
            if n.survival == DEAD:
                pool.append((f"invoke_{_sanitize(n.name)}", f"Retry through {n.name}, which is not live."))
    elif ba["action"] == "probe":
        ck = f"probe_{_sanitize(nm[ba['node']])}"
        cd = f"Probe {nm[ba['node']]} with a cheap read to resolve its unverified state."
        pool = [
            ("report_impossible", "Report that the goal is verified impossible (treat unknown as dead)."),
            ("reboot_rebuild", "Reboot and rebuild the whole stack from scratch."),
            ("conclude_unreachable", "Conclude unreachable because a dependency is unverified."),
        ]
        for n in ep.nodes.values():
            if n.survival == ALIVE:
                pool.append((f"invoke_{_sanitize(n.name)}", f"Invoke {n.name} directly."))
    else:  # report_impossible
        ck = "report_impossible"
        cd = "Report that every route is verified blocked (dead or insufficient authority)."
        pool = [("reboot_rebuild", "Reboot and rebuild the whole stack from scratch."),
                ("accept_ceiling", "Accept the current configuration as the limit.")]
        for n in ep.nodes.values():
            if n.survival == DEAD:
                pool.append((f"invoke_{_sanitize(n.name)}", f"Invoke {n.name} directly to force the capability."))
        pool.append(("conclude_unreachable", "Conclude unreachable because a dependency is unverified."))
    distractors = _dedup(pool, ck, rng)
    opts = [(ck, cd)] + distractors
    rng.shuffle(opts)
    crit = {k: v for k, v in opts}
    keys = [k for k, _ in opts]
    assert len(keys) == 4 and len(set(keys)) == 4 and ck in keys, (family, keys, ck, ba)
    q_choice = {"id": "choice", "type": "choice", "instructions": rng.choice(CHOICE_Q),
                "criteria": crit, "gold_label": ck, "gold_position": keys.index(ck)}
    # ---- three noul ----
    # state-specific noul options (name the provider) so the scorer must read the
    # state, not memorize a generic instruction->answer mapping
    prov = nm.get(ep.goal_node) or nm[list(nm)[-1]]
    q_reach = {"id": "noul_reach", "type": "noul",
               "instructions": rng.choice(REACH_Q),
               "criteria": {"false": f"{prov} does not have a verified live route to the goal",
                            "true": f"{prov} has a verified live route to the goal"},
               "gold_label": "true" if status == "REACHABLE" else "false"}
    q_imposs = {"id": "noul_impossible", "type": "noul",
                "instructions": rng.choice(IMPOSS_Q),
                "criteria": {"false": f"a route to the goal via {prov} is still possible",
                             "true": f"every route to the goal via {prov} is confirmed blocked"},
                "gold_label": "true" if status == "UNREACHABLE" else "false"}
    q_unknown = {"id": "noul_unknown", "type": "noul",
                 "instructions": rng.choice(UNKNOWN_Q),
                 "criteria": {"false": f"the routes to the goal via {prov} are fully resolved",
                              "true": f"at least one route to the goal via {prov} is still unverified"},
                 "gold_label": "true" if status == "UNKNOWN" else "false"}
    return [q_choice, q_reach, q_imposs, q_unknown]


def build_scenario(family, depth, target, rng, topology="chain"):
    if topology == "branch":
        ep = sample_branch(family, depth, target, rng)
    else:
        ep = sample_chain(family, depth, target, rng)
    status = route_status(ep)
    situation = render(ep, family, rng)
    state = {"situation": situation, "goal": "reach the goal capability"}
    questions = build_questions(ep, family, rng)
    return {
        "scenario_id": f"{family}_{topology}_d{depth}_{rng.randrange(1 << 30):09d}",
        "family": family, "depth": depth, "topology": topology,
        "target_status": target,
        "state": state, "situation": situation, "goal": "reach the goal capability",
        "questions": questions,
        "oracle": {"status": status, "action": best_action(ep)["action"],
                   "action_node": best_action(ep)["node"], "graph_hash": canonical_hash(ep)},
        "graph": {n.nid: {"survival": n.survival, "kind": n.kind, "children": list(n.children),
                          "provides_goal": n.provides_goal} for n in ep.nodes.values()},
        "graph_names": {n.nid: n.name for n in ep.nodes.values()},
        "graph_entry": ep.entry,
        "authority": ep.authority, "dormant": ep.dormant,
    }


# ---------------------------------------------------------------------------
# Dataset build + validation
# ---------------------------------------------------------------------------
def build_cells(families, depths, targets, n_per_cell, seed, topology="chain"):
    rng = random.Random(seed)
    out = []
    for fam in families:
        for d in depths:
            for ti, t in enumerate(targets):
                for i in range(n_per_cell):
                    out.append(build_scenario(fam, d, t, rng, topology))
    return out


def validate(scenarios, tok, max_state=256, max_option=64):
    over_s = over_o = bad = 0
    for sc in scenarios:
        st = json.dumps(sc["state"], ensure_ascii=False)
        if len(tok(st, add_special_tokens=False)["input_ids"]) > max_state:
            over_s += 1
        for q in sc["questions"]:
            if q["type"] == "choice":
                crit = q["criteria"]; opts = [f"{k}: {v}" if v else k for k, v in crit.items()]; keys = list(crit)
            else:
                crit = q["criteria"]; opts = ["false: " + crit["false"], "true: " + crit["true"]]; keys = ["false", "true"]
            if q["gold_label"] not in keys:
                bad += 1
            for o in opts:
                if len(tok(o, add_special_tokens=False)["input_ids"]) > max_option:
                    over_o += 1
    return {"n": len(scenarios), "over_state": over_s, "over_option": over_o, "bad_gold": bad,
            "passed": over_s == 0 and over_o == 0 and bad == 0}


def graph_disjoint(a, b):
    ha = {s["oracle"]["graph_hash"] for s in a}; hb = {s["oracle"]["graph_hash"] for s in b}
    return {"a": len(ha), "b": len(hb), "overlap": len(ha & hb), "disjoint": len(ha & hb) == 0}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True, local_files_only=True)
    metam = metamorphic_tests()
    print("metamorphic:", "PASS" if metam["all_passed"] else "FAIL", metam["failures"])
    if not metam["all_passed"]:
        raise SystemExit("metamorphic failed")

    if mode == "validate":
        ds = build_cells(TRAIN_FAMILIES, DEPTHS, TARGETS, 4, seed=1)
        ds += build_cells(HELDOUT_FAMILIES, DEPTHS, TARGETS, 4, seed=2)
        print("validate:", json.dumps(validate(ds, tok), indent=2))
        from collections import Counter
        bal = Counter((s["family"], s["depth"], s["target_status"]) for s in ds)
        for k in sorted(bal):
            print("  ", k, bal[k])
        return

    # full build
    n = 60
    train = build_cells(TRAIN_FAMILIES, DEPTHS, TARGETS, n, seed=4101)
    dev = build_cells(TRAIN_FAMILIES, DEPTHS, TARGETS, 15, seed=4102)
    test = build_cells(HELDOUT_FAMILIES, DEPTHS, TARGETS, n, seed=4103)
    comp = build_cells(["substrate", "handle", "config", "survivor"], [2, 3], TARGETS, 40, seed=4104, topology="branch")
    val = validate(train + dev + test + comp, tok)
    dis = {"train_vs_test": graph_disjoint(train, test), "train_vs_comp": graph_disjoint(train, comp)}
    out = {
        "pilot": "Hacker Mindset Curriculum v2", "synthetic": True,
        "evidence_class": "SYNTHETIC oracle-verified (three-valued). NOT OD-1 evidence.",
        "counts": {"train": len(train), "dev": len(dev), "test": len(test), "composition": len(comp),
                   "train_decisions": sum(len(c["questions"]) for c in train),
                   "test_decisions": sum(len(c["questions"]) for c in test)},
        "validation": val, "disjointness": dis, "metamorphic": metam["all_passed"],
        "train": train, "dev": dev, "test": test, "composition": comp,
    }
    path = os.path.join(PROBE_DIR, "curriculum_v2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print("counts:", out["counts"])
    print("validation:", val)
    print("disjointness:", dis)
    print("curriculum_v2 sha:", h)


if __name__ == "__main__":
    main()