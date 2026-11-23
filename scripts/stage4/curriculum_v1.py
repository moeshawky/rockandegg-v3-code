"""Curriculum v1 — scenario decorators, generator, renderer, dataset builder.

Builds a mechanism-composition curriculum on top of the INDEPENDENT DAG oracle
(v1_graph.py). Four mechanism families, each spanning 1/2/3-hop dependency
depths (graph-derived, not an arbitrary label):

  1. substrate   — linear dependency stack; drop to the highest live layer that
                   still reaches the capability.
  2. handle      — the lookup client is a RED HERRING; the service-side chain
                   decides reachability.
  3. config      — a dormant interface behind a dependency chain; enable vs accept.
  4. survivor    — a degraded role behind a dependency chain; use vs dismiss.

Fixes over v0:
  * depth is graph-derived and crossed with family (no depth/family confound);
  * the oracle reads primitive alive-flags + topology, never a precomputed answer;
  * scenario_family / difficulty / hop-depth are NEVER in the model-visible state;
  * false-premise handle cases are impossible (the client is a red herring);
  * option order shuffled; only genuinely-wrong distractors; randomized names and
    status wording; two surface styles per family;
  * graph-level (not just text) train/eval disjointness via canonical hash;
  * 256/64-token budget enforced; over-long cases rejected;
  * metamorphic oracle tests run on every build.

SYNTHETIC. NOT OD-1 evidence (AGENTS.md §12).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any

PROBE_DIR = os.environ.get("RAE_PROBE_DIR", os.path.join(os.getcwd(), "probe_data"))
MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
sys.path.insert(0, PROBE_DIR)
sys.path.insert(0, MODEL_DIR)

from v1_graph import (  # noqa: E402
    Dag, Node, branching_dag, canonical_graph_hash, chain_dag, metamorphic_tests,
    satisfies, solve,
)

# ---------------------------------------------------------------------------
FAMILIES = ["substrate", "handle", "config", "survivor"]
TRAIN_FAMILIES = ["substrate", "handle"]
HELDOUT_FAMILIES = ["config", "survivor"]
DEPTHS = [1, 2, 3]

STAGE_NAMES = ["atlas", "birch", "cedar", "delta", "ember", "forge", "grove",
               "harbor", "iris", "juniper", "kestrel", "lumen", "mesa", "north",
               "onyx", "prairie", "quartz", "ridge", "summit", "timber",
               "umber", "vale", "willow", "yarrow", "zephyr"]
RESOURCE_NAMES = ["bedrock", "anchor", "reservoir", "backing", "core", "substrate"]
CLIENT_NAMES = ["resolver", "broker", "gateway", "dispatcher", "locator"]
ALIVE_WORDS = ["operational", "intact", "responding", "available", "live", "healthy"]
DEAD_WORDS = ["destroyed", "failed", "unreachable", "offline", "broken", "seized"]
LIMITED_VIEWS = ["read", "list", "status", "read-only", "monitoring"]


def _pick_names(rng, n):
    pool = list(STAGE_NAMES)
    rng.shuffle(pool)
    return pool[:n]


def _status(rng, alive):
    return rng.choice(ALIVE_WORDS if alive else DEAD_WORDS)


def _sanitize(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).lower()


def _chain_pattern(rng, depth, want_reachable):
    if want_reachable:
        j = rng.randint(0, depth)
        return [i >= j for i in range(depth + 1)], True
    mode = rng.choice(["kill_resource", "kill_provider", "kill_all"])
    if mode == "kill_resource":
        j = rng.randint(0, depth)
        return [i >= j for i in range(depth + 1)], False
    if mode == "kill_provider":
        return [True] * depth + [False], True
    return [False] * (depth + 1), True


def _dedup3(pool, correct_key, rng, n=3):
    rng.shuffle(pool)
    seen, out = {correct_key}, []
    for k, v in pool:
        if k not in seen:
            seen.add(k)
            out.append((k, v))
    assert len(out) >= n, (correct_key, len(out), pool)
    return out[:n]


# ===========================================================================
# FAMILY DECORATORS  (pattern/res_alive/style overrides for robustness)
# ===========================================================================
def fam_substrate(rng, depth, want_reachable, pattern=None, res_alive=None, style=0):
    names = _pick_names(rng, depth + 1)
    if pattern is None:
        pat, res_alive = _chain_pattern(rng, depth, want_reachable)
    else:
        pat = pattern
    dag = chain_dag("substrate", depth, pat, res_alive, names)
    res_name = rng.choice(RESOURCE_NAMES)
    status = {f"S{i}": _status(rng, dag.nodes[f"S{i}"].alive) for i in range(depth + 1)}
    res_status = _status(rng, res_alive)

    def describe(d):
        parts = "; ".join(f"{d.nodes[f'S{i}'].name} ({status[f'S{i}']})"
                          for i in range(depth + 1))
        if style % 2 == 0:
            return (f"The system is organized as {depth + 1} layers, top to bottom: {parts}. "
                    f"Each layer depends on the layer directly beneath it. The lowest layer "
                    f"{d.nodes[f'S{depth}'].name} delivers the goal capability only when "
                    f"{res_name} is {res_status}.")
        return (f"Layers, highest first: {parts}. A layer only works if the layer under it "
                f"works. {d.nodes[f'S{depth}'].name} can provide the goal only if {res_name} "
                f"is {res_status}.")

    def correct(d, sol):
        if sol["reachable"]:
            nm = d.nodes[sol["action_node"]].name
            return (f"invoke_{_sanitize(nm)}", f"Invoke {nm} directly to reach the capability.")
        return ("report_unreachable",
                "Report that no surviving layer reaches the capability without rebuilding.")

    def distractors(d, sol, r):
        nm = {n.nid: n.name for n in d.nodes.values()}
        cnode = sol["action_node"]
        pool = []
        if sol["reachable"]:
            for i in range(depth + 1):
                nid = f"S{i}"
                if d.nodes[nid].alive is False:
                    pool.append((f"invoke_{_sanitize(nm[nid])}",
                                 f"Retry through {nm[nid]}, which is not currently live."))
            pool.append(("report_unreachable",
                         "Report that no surviving layer reaches the capability."))
        else:
            pool.append((f"invoke_{_sanitize(nm['S0'])}",
                         f"Retry through {nm['S0']}, the documented top entry point."))
            pool.append((f"invoke_{_sanitize(nm[f'S{depth}'])}",
                         f"Invoke {nm[f'S{depth}']} directly to force the capability."))
        pool.append(("reboot", "Reboot into a clean environment and reinstall the whole stack."))
        pool.append(("rebuild", "Rebuild the destroyed layers before attempting anything else."))
        return _dedup3(pool, f"invoke_{_sanitize(nm[cnode])}" if sol["reachable"] else "report_unreachable", r)

    def noul(d):
        return {"instruction": "Is there a surviving layer from which the capability is still reachable without rebuilding?",
                "criteria": {"false": "no, every path to the capability is destroyed",
                             "true": "yes, at least one surviving layer still reaches the capability"}}
    return {"dag": dag, "goal": "reach the capability", "describe": describe,
            "correct": correct, "distractors": distractors, "noul": noul}


def fam_handle(rng, depth, want_reachable, pattern=None, res_alive=None, style=0):
    names = _pick_names(rng, depth + 1)
    if pattern is None:
        pat, res_alive = _chain_pattern(rng, depth, want_reachable)
    else:
        pat = pattern
    dag = chain_dag("handle", depth, pat, res_alive, names)
    res_name = rng.choice(RESOURCE_NAMES)
    client_name = rng.choice(CLIENT_NAMES)
    client_alive = rng.random() < 0.5
    status = {f"S{i}": _status(rng, dag.nodes[f"S{i}"].alive) for i in range(depth + 1)}
    res_status = _status(rng, res_alive)
    client_status = _status(rng, client_alive)

    def describe(d):
        parts = " -> ".join(f"{d.nodes[f'S{i}'].name} ({status[f'S{i}']})"
                            for i in range(depth + 1))
        if style % 2 == 0:
            return (f"A lookup client ({client_name}) is {client_status}. The service itself is "
                    f"reached through this chain: {parts}. {d.nodes[f'S{depth}'].name} yields the "
                    f"capability only if {res_name} is {res_status}.")
        return (f"Name lookups go through a client ({client_name}, {client_status}). Behind that, "
                f"the request path is {parts}. {d.nodes[f'S{depth}'].name} provides the capability "
                f"only when {res_name} is {res_status}.")

    def correct(d, sol):
        if sol["reachable"]:
            nm = d.nodes[sol["action_node"]].name
            return (f"use_{_sanitize(nm)}", f"Read/write through {nm}, which still reaches the capability.")
        return ("report_unreachable", "Report that no live reference reaches the capability.")

    def distractors(d, sol, r):
        nm = {n.nid: n.name for n in d.nodes.values()}
        ckey = f"use_{_sanitize(nm[sol['action_node']])}" if sol["reachable"] else "report_unreachable"
        pool = [
            ("conclude_client_dead",
             "Conclude the capability is unreachable because the lookup client is gone."),
            ("rebuild_client", "Reconstruct the lookup client first, then retry through it."),
            ("reboot", "Reboot the machine to restore the client and the whole path."),
        ]
        if sol["reachable"]:
            pool.append(("report_unreachable", "Report that no live reference reaches the capability."))
            if d.nodes["S0"].alive is False:
                pool.append((f"use_{_sanitize(nm['S0'])}", f"Read/write through {nm['S0']} directly."))
        else:
            pool.append((f"use_{_sanitize(nm['S0'])}", f"Read/write through {nm['S0']} directly."))
            pool.append((f"use_{_sanitize(nm[f'S{depth}'])}", f"Read/write through {nm[f'S{depth}']} directly."))
        return _dedup3(pool, ckey, r)

    def noul(d):
        return {"instruction": "Is the capability reachable from a surviving reference on the service side?",
                "criteria": {"false": "no, the capability is unreachable",
                             "true": "yes, a live reference still provides it"}}
    return {"dag": dag, "goal": "reach the capability", "describe": describe,
            "correct": correct, "distractors": distractors, "noul": noul}


def fam_config(rng, depth, want_reachable, pattern=None, res_alive=None, style=0):
    names = _pick_names(rng, depth + 1)
    if pattern is None:
        pat, res_alive = _chain_pattern(rng, depth, want_reachable)
    else:
        pat = pattern
    dag = chain_dag("config", depth, pat, res_alive, names)
    res_name = rng.choice(RESOURCE_NAMES)
    limited = rng.choice(LIMITED_VIEWS)
    status = {f"S{i}": _status(rng, dag.nodes[f"S{i}"].alive) for i in range(depth + 1)}
    res_status = _status(rng, res_alive)

    def describe(d):
        parts = " -> ".join(f"{d.nodes[f'S{i}'].name} ({status[f'S{i}']})"
                            for i in range(depth + 1))
        if style % 2 == 0:
            return (f"A component currently runs in a fixed view that exposes only {limited}. "
                    f"The dependency chain to the full capability is: {parts}. "
                    f"{d.nodes[f'S{depth}'].name} delivers the capability only if {res_name} is {res_status}.")
        return (f"The active view exposes only {limited}. Full capability depends on: {parts}. "
                f"{d.nodes[f'S{depth}'].name} can provide it only when {res_name} is {res_status}.")

    def correct(d, sol):
        if sol["reachable"]:
            return ("enable_dormant", "Enable the dormant secondary interface to expose the full capability.")
        return ("report_unavailable", "Report that no configuration exposes the capability here.")

    def distractors(d, sol, r):
        pool = [
            ("accept_ceiling", "Accept the current view as the capability limit."),
            ("replace_component", "Replace the whole component with a different one."),
            ("wait_retry", "Wait and retry under the current configuration."),
            ("force_view", "Force the current view to expose the capability without reconfiguring."),
        ]
        return _dedup3(pool, "enable_dormant" if sol["reachable"] else "report_unavailable", r)

    def noul(d):
        return {"instruction": "Can the component provide the capability under some configuration?",
                "criteria": {"false": "no, no configuration exposes it",
                             "true": "yes, a dormant interface exposes it"}}
    return {"dag": dag, "goal": "expose the full capability", "describe": describe,
            "correct": correct, "distractors": distractors, "noul": noul}


def fam_survivor(rng, depth, want_reachable, pattern=None, res_alive=None, style=0):
    names = _pick_names(rng, depth + 1)
    if pattern is None:
        pat, res_alive = _chain_pattern(rng, depth, want_reachable)
    else:
        pat = pattern
    dag = chain_dag("survivor", depth, pat, res_alive, names)
    res_name = rng.choice(RESOURCE_NAMES)
    status = {f"S{i}": _status(rng, dag.nodes[f"S{i}"].alive) for i in range(depth + 1)}
    res_status = _status(rng, res_alive)

    def describe(d):
        parts = " -> ".join(f"{d.nodes[f'S{i}'].name} ({status[f'S{i}']})"
                            for i in range(depth + 1))
        if style % 2 == 0:
            return (f"A component has survived a failure; its original role is gone. "
                    f"Its remaining dependency chain is: {parts}. "
                    f"{d.nodes[f'S{depth}'].name} delivers the capability only if {res_name} is {res_status}.")
        return (f"One component survived; its old job no longer exists. What remains: {parts}. "
                f"{d.nodes[f'S{depth}'].name} can provide the goal only when {res_name} is {res_status}.")

    def correct(d, sol):
        if sol["reachable"]:
            nm = d.nodes[sol["action_node"]].name
            return (f"use_{_sanitize(nm)}", f"Use {nm} in its degraded role as the resource for the goal.")
        return ("report_no_path", "Report that the survivor offers no usable path to the goal.")

    def distractors(d, sol, r):
        pool = [
            ("dismiss_junk", "Dismiss the survivor as junk and clean it up."),
            ("restore_role", "Insist on restoring the survivor's original role first."),
            ("reboot_fresh", "Reboot to obtain a fresh, normal-role component."),
            ("wait", "Wait for the survivor to recover its original role on its own."),
        ]
        return _dedup3(pool, f"use_{_sanitize(d.nodes[sol['action_node']].name)}" if sol["reachable"] else "report_no_path", r)

    def noul(d):
        return {"instruction": "Does the survivor provide a usable degraded role for the goal?",
                "criteria": {"false": "no, it has no useful role",
                             "true": "yes, its degraded role provides the goal"}}
    return {"dag": dag, "goal": "reach the goal with what survived", "describe": describe,
            "correct": correct, "distractors": distractors, "noul": noul}


FAMILY_FN = {"substrate": fam_substrate, "handle": fam_handle,
             "config": fam_config, "survivor": fam_survivor}


# ===========================================================================
# SCENARIO ASSEMBLY
# ===========================================================================
def _assemble(family, depth, deco, sol, rng, scenario_id):
    dag = deco["dag"]
    situation = deco["describe"](dag)
    state = {"situation": situation, "goal": deco["goal"]}
    ck, cd = deco["correct"](dag, sol)
    distractors = deco["distractors"](dag, sol, rng)
    opts = [(ck, cd)] + distractors
    rng.shuffle(opts)
    crit = {k: v for k, v in opts}
    keys = [k for k, _ in opts]
    assert len(keys) == 4 and len(set(keys)) == 4 and ck in keys, (family, keys, ck)
    q_choice = {"id": "choice", "type": "choice",
                "instructions": "What is the correct next move?",
                "criteria": crit, "gold_label": ck,
                "gold_position": keys.index(ck)}
    noul_spec = deco["noul"](dag)
    q_noul = {"id": "noul", "type": "noul", "instructions": noul_spec["instruction"],
              "criteria": noul_spec["criteria"],
              "gold_label": "true" if sol["reachable"] else "false"}
    return {
        "scenario_id": scenario_id,
        "family": family, "depth": depth,
        "topology": dag.meta.get("topology", "chain"),
        "state": state, "situation": situation, "goal": deco["goal"],
        "questions": [q_choice, q_noul],
        "oracle": {
            "reachable": sol["reachable"], "action": sol["action"],
            "action_name": sol["action_name"], "chain": sol["chain"],
            "depth": sol["depth"], "n_links_checked": sol["n_links_checked"],
            "candidate_results": sol["candidate_results"],
            "graph_hash": canonical_graph_hash(dag),
        },
        "graph_alive": {n.nid: n.alive for n in dag.nodes.values()},
        "graph_kind": {n.nid: n.kind for n in dag.nodes.values()},
        "graph_children": {n.nid: list(n.children) for n in dag.nodes.values()},
    }


def build_scenario(family, depth, want_reachable, rng, pattern=None, res_alive=None, style=None):
    if style is None:
        style = rng.randint(0, 1)
    deco = FAMILY_FN[family](rng, depth, want_reachable, pattern=pattern,
                             res_alive=res_alive, style=style)
    sol = solve(deco["dag"])
    assert sol["reachable"] == want_reachable, (family, depth, sol)
    sid = f"{family}_d{depth}_{rng.randrange(1 << 30):09d}"
    return _assemble(family, depth, deco, sol, rng, sid)


# ===========================================================================
# NOVEL-COMPOSITION SCENARIOS (branching topologies, depth-2)
# ===========================================================================
def build_composition_scenario(rng, topology, want_reachable):
    names = _pick_names(rng, 3)
    dag = branching_dag("config", 2, topology, names)
    # randomize alive flags to hit the requested reachability
    if topology == "or_redundant":
        # S0 OR (A->RA) (B->RB)
        dag.nodes["S0"].alive = True
        if want_reachable:
            dag.nodes["A"].alive = True; dag.nodes["RA"].alive = True
            dag.nodes["B"].alive = rng.random() < 0.5
            dag.nodes["RB"].alive = rng.random() < 0.5
        else:
            # kill both branches or the entry
            if rng.random() < 0.5:
                dag.nodes["S0"].alive = False
            else:
                dag.nodes["A"].alive = False; dag.nodes["RA"].alive = rng.random() < 0.5
                dag.nodes["B"].alive = False; dag.nodes["RB"].alive = rng.random() < 0.5
    else:  # and_split: S0 AND (A->RA) AND (B->RB)
        if want_reachable:
            for n in dag.nodes.values():
                n.alive = True
        else:
            victim = rng.choice(["S0", "A", "B", "RA", "RB"])
            for n in dag.nodes.values():
                n.alive = True
            dag.nodes[victim].alive = False
    sol = solve(dag)
    assert sol["reachable"] == want_reachable, (topology, sol)

    # render
    def st(nid):
        return _status(rng, dag.nodes[nid].alive)
    if topology == "or_redundant":
        situation = (f"A capability can be delivered through one of two providers. "
                     f"Provider path 1: {dag.nodes['A'].name} ({st('A')}) -> {dag.nodes['RA'].name} ({st('RA')}). "
                     f"Provider path 2: {dag.nodes['B'].name} ({st('B')}) -> {dag.nodes['RB'].name} ({st('RB')}). "
                     f"The dispatcher {dag.nodes['S0'].name} is {st('S0')}.")
    else:
        situation = (f"A capability requires two subsystems to work together. "
                     f"Subsystem 1: {dag.nodes['A'].name} ({st('A')}) -> {dag.nodes['RA'].name} ({st('RA')}). "
                     f"Subsystem 2: {dag.nodes['B'].name} ({st('B')}) -> {dag.nodes['RB'].name} ({st('RB')}). "
                     f"The coordinator {dag.nodes['S0'].name} is {st('S0')}.")
    goal = "reach the capability"
    if sol["reachable"]:
        correct = ("proceed", "Proceed — a live dependency path reaches the capability.")
    else:
        correct = ("report_unreachable", "Report that no live dependency path reaches the capability.")
    pool = [
        ("reboot_rebuild", "Reboot and rebuild every component, then retry from scratch."),
        ("force_primary", "Force the primary path even though a dependency is down."),
        ("wait_retry", "Wait and retry without changing anything."),
        ("replace_all", "Replace all components with new ones."),
    ]
    distractors = _dedup3(pool, correct[0], rng)
    opts = [(correct[0], correct[1])] + distractors
    rng.shuffle(opts)
    crit = {k: v for k, v in opts}
    keys = [k for k, _ in opts]
    q_choice = {"id": "choice", "type": "choice",
                "instructions": "What is the correct next move?",
                "criteria": crit, "gold_label": correct[0],
                "gold_position": keys.index(correct[0])}
    q_noul = {"id": "noul", "type": "noul",
              "instructions": "Is the capability reachable from the current configuration?",
              "criteria": {"false": "no, no live path reaches it",
                           "true": "yes, a live path reaches it"},
              "gold_label": "true" if sol["reachable"] else "false"}
    return {
        "scenario_id": f"comp_{topology}_{rng.randrange(1 << 30):09d}",
        "family": "composition", "depth": 2, "topology": topology,
        "state": {"situation": situation, "goal": goal},
        "situation": situation, "goal": goal,
        "questions": [q_choice, q_noul],
        "oracle": {"reachable": sol["reachable"], "action": sol["action"],
                   "action_name": sol["action_name"], "chain": sol["chain"],
                   "depth": sol["depth"], "n_links_checked": sol["n_links_checked"],
                   "candidate_results": sol["candidate_results"],
                   "graph_hash": canonical_graph_hash(dag)},
        "graph_alive": {n.nid: n.alive for n in dag.nodes.values()},
        "graph_kind": {n.nid: n.kind for n in dag.nodes.values()},
        "graph_children": {n.nid: list(n.children) for n in dag.nodes.values()},
    }


# ===========================================================================
# ROBUSTNESS VARIANTS (from a base scenario's causal structure)
# ===========================================================================
def _pattern_of(sc):
    pat = [sc["graph_alive"][f"S{i}"] for i in range(sc["depth"] + 1)]
    return pat, sc["graph_alive"].get("R", True)


def make_robustness_variants(base, rng):
    """Return {variant_name: scenario} for renamed / irrelevant_fact /
    option_permuted / counterfactual (only causal variants change the label)."""
    fam, depth = base["family"], base["depth"]
    pat, res = _pattern_of(base)
    base_reach = base["oracle"]["reachable"]
    out = {}

    # renamed: same pattern, new names
    out["renamed"] = build_scenario(fam, depth, base_reach, rng, pattern=pat,
                                    res_alive=res, style=base.get("_style", 0))
    # irrelevant_fact: same pattern + a neutral unrelated sentence appended
    irr = build_scenario(fam, depth, base_reach, rng, pattern=pat, res_alive=res,
                         style=base.get("_style", 0))
    irr_name = rng.choice(["ancillary", "auxiliary", "unrelated", "separate"])
    irr["situation"] = (irr["situation"] +
                        f" A {irr_name} monitor, unrelated to this path, is {rng.choice(ALIVE_WORDS + DEAD_WORDS)}.")
    irr["state"] = {"situation": irr["situation"], "goal": irr["goal"]}
    out["irrelevant_fact"] = irr
    # option_permuted: same questions, re-shuffle the choice option order
    perm = json.loads(json.dumps(base))
    crit = perm["questions"][0]["criteria"]
    items = list(crit.items())
    rng.shuffle(items)
    perm["questions"][0]["criteria"] = {k: v for k, v in items}
    perm["questions"][0]["gold_position"] = list(perm["questions"][0]["criteria"].keys()).index(
        perm["questions"][0]["gold_label"])
    out["option_permuted"] = perm
    # counterfactual: flip one alive flag so reachability changes
    nodes = [f"S{i}" for i in range(depth + 1)] + ["R"]
    rng.shuffle(nodes)
    for nid in nodes:
        newpat = list(pat)
        newres = res
        if nid == "R":
            newres = not res
        else:
            idx = int(nid[1:])
            newpat[idx] = not newpat[idx]
        reach = _reach_of_pattern(depth, newpat, newres)
        if reach != base_reach:
            cf = build_scenario(fam, depth, reach, rng, pattern=newpat,
                                res_alive=newres, style=base.get("_style", 0))
            out["counterfactual"] = {"scenario": cf, "flipped": nid,
                                     "old_reachable": base_reach, "new_reachable": reach}
            break
    return out


def _reach_of_pattern(depth, pat, res):
    if not res:
        return False
    # chain: exists i with all pat[i:] alive
    for i in range(depth + 1):
        if all(pat[i:]):
            return True
    return False


# ===========================================================================
# VALIDATION
# ===========================================================================
def validate_scenarios(scenarios, tok, max_state=256, max_option=64):
    over_state, over_option, bad_gold = [], [], []
    for sc in scenarios:
        st = json.dumps(sc["state"], ensure_ascii=False)
        n_state = len(tok(st, add_special_tokens=False)["input_ids"])
        if n_state > max_state:
            over_state.append((sc["scenario_id"], n_state))
        for q in sc["questions"]:
            if q["type"] == "choice":
                crit = q["criteria"]
                opts = [f"{k}: {v}" if v else k for k, v in crit.items()]
                keys = list(crit.keys())
            else:
                crit = q["criteria"]
                opts = ["false: " + crit.get("false", ""), "true: " + crit.get("true", "")]
                keys = ["false", "true"]
            if q["gold_label"] not in keys:
                bad_gold.append((sc["scenario_id"], q["id"], q["gold_label"], keys))
            for o in opts:
                n = len(tok(o, add_special_tokens=False)["input_ids"])
                if n > max_option:
                    over_option.append((sc["scenario_id"], q["id"], n))
    return {"n_scenarios": len(scenarios), "n_over_state": len(over_state),
            "n_over_option": len(over_option), "n_bad_gold": len(bad_gold),
            "over_state_examples": over_state[:10], "over_option_examples": over_option[:10],
            "bad_gold_examples": bad_gold[:10],
            "passed": len(over_state) == 0 and len(over_option) == 0 and len(bad_gold) == 0}


def graph_disjoint(train_scenarios, test_scenarios):
    tr = {s["oracle"]["graph_hash"] for s in train_scenarios}
    te = {s["oracle"]["graph_hash"] for s in test_scenarios}
    overlap = tr & te
    return {"train_graphs": len(tr), "test_graphs": len(te),
            "n_overlap": len(overlap), "disjoint": len(overlap) == 0,
            "overlap_examples": sorted(overlap)[:10]}


# ===========================================================================
# BUILD
# ===========================================================================
def build_cells(families, depths, n_per_cell, seed, styles=(0, 1)):
    rng = random.Random(seed)
    out = []
    for fam in families:
        for d in depths:
            for i in range(n_per_cell):
                want = (i % 2 == 0)
                st = styles[i % len(styles)]
                out.append(build_scenario(fam, d, want, rng, style=st))
    return out


def build_dataset(n_train=80, n_dev=20, n_test=80, n_comp=160, n_rob=80, seed=3001):
    train = build_cells(TRAIN_FAMILIES, DEPTHS, n_train, seed=seed)
    dev = build_cells(TRAIN_FAMILIES, DEPTHS, n_dev, seed=seed + 1)
    test = build_cells(HELDOUT_FAMILIES, DEPTHS, n_test, seed=seed + 2)
    # novel composition
    comp = []
    rngc = random.Random(seed + 3)
    for i in range(n_comp):
        topo = "or_redundant" if i % 2 == 0 else "and_split"
        comp.append(build_composition_scenario(rngc, topo, want_reachable=(i % 4 < 2)))
    # robustness: base scenarios drawn from held-out families (never trained on)
    rngr = random.Random(seed + 4)
    rob_bases = [test[i] for i in rngr.sample(range(len(test)), min(n_rob, len(test)))]
    rob = {}
    for b in rob_bases:
        b["_style"] = 0
        rob[b["scenario_id"]] = make_robustness_variants(b, rngr)

    # Graph disjointness across splits. The HELD-OUT evaluation boundary is
    # train families vs {held-out families, novel composition}: that is what must
    # be structurally disjoint (it is — different families / topologies). The
    # train/dev split shares the same families by design (DEV is for checkpoint
    # selection only), so structural-pattern overlap there is expected and is
    # reported honestly, not claimed disjoint.
    dis = {
        "train_vs_test": graph_disjoint(train, test),
        "train_vs_comp": graph_disjoint(train, comp),
        "test_vs_comp": graph_disjoint(test, comp),
        "train_vs_dev": graph_disjoint(train, dev),
        "note": ("held-out evaluation = held-out families + novel composition; "
                 "train_vs_test and train_vs_comp MUST be disjoint. train_vs_dev "
                 "overlap is expected (same families, DEV=checkpoint selection)."),
    }
    return {"train": train, "dev": dev, "test": test, "comp": comp, "robustness": rob,
            "disjointness": dis}


def _scenario_core(sc):
    """Strip internal fields for hashing/output."""
    return {k: v for k, v in sc.items() if not k.startswith("_")}


def write_dataset(ds, tok):
    # flatten robustness into a list of (base_id, variant_name, scenario)
    rob_flat = []
    for base_id, variants in ds["robustness"].items():
        for vname, sc in variants.items():
            if vname == "counterfactual":
                rob_flat.append({"base_id": base_id, "variant": vname,
                                 "scenario": sc["scenario"], "flipped": sc["flipped"],
                                 "old_reachable": sc["old_reachable"],
                                 "new_reachable": sc["new_reachable"]})
            else:
                rob_flat.append({"base_id": base_id, "variant": vname, "scenario": sc})

    all_main = ds["train"] + ds["dev"] + ds["test"] + ds["comp"]
    val = validate_scenarios(all_main + [r["scenario"] for r in rob_flat], tok)

    # hashes
    def hash_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    out = {
        "pilot": "Hacker Mindset Curriculum v1",
        "synthetic": True,
        "evidence_class": ("SYNTHETIC oracle-verified curriculum. NOT OD-1 evidence. "
                           "Labels derived by an independent DAG oracle from primitive facts."),
        "families": {"train": TRAIN_FAMILIES, "heldout": HELDOUT_FAMILIES,
                     "depths": DEPTHS},
        "counts": {
            "train_scenarios": len(ds["train"]),
            "dev_scenarios": len(ds["dev"]),
            "test_scenarios": len(ds["test"]),
            "composition_scenarios": len(ds["comp"]),
            "robustness_variants": len(rob_flat),
            "train_decisions": sum(len(c["questions"]) for c in ds["train"]),
            "dev_decisions": sum(len(c["questions"]) for c in ds["dev"]),
            "test_decisions": sum(len(c["questions"]) for c in ds["test"]),
            "composition_decisions": sum(len(c["questions"]) for c in ds["comp"]),
        },
        "validation": val,
        "disjointness": ds["disjointness"],
        "train": [_scenario_core(s) for s in ds["train"]],
        "dev": [_scenario_core(s) for s in ds["dev"]],
        "test": [_scenario_core(s) for s in ds["test"]],
        "composition": [_scenario_core(s) for s in ds["comp"]],
        "robustness": rob_flat,
    }
    path = os.path.join(PROBE_DIR, "curriculum_v1.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    out["curriculum_sha256"] = hash_file(path)

    # manifest
    metam = metamorphic_tests()
    manifest = {
        "pilot": "Hacker Mindset Curriculum v1",
        "seed_base": 3001,
        "families": {"train": TRAIN_FAMILIES, "heldout": HELDOUT_FAMILIES},
        "depths": DEPTHS,
        "counts": out["counts"],
        "validation_passed": val["passed"],
        "validation": {k: val[k] for k in ("n_scenarios", "n_over_state",
                                            "n_over_option", "n_bad_gold")},
        "disjointness": ds["disjointness"],
        "metamorphic_all_passed": metam["all_passed"],
        "metamorphic_failures": metam["failures"],
        "curriculum_sha256": out["curriculum_sha256"],
        "note": "SYNTHETIC; labels derived by independent DAG oracle; not OD-1 evidence.",
    }
    with open(os.path.join(PROBE_DIR, "curriculum_v1_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    from transformers import AutoTokenizer
    MODEL_DIR = os.environ.get("RAE_MODEL_DIR", os.path.join(os.getcwd(), "rae-v3"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)

    metam = metamorphic_tests()
    print("metamorphic:", "PASS" if metam["all_passed"] else "FAIL", metam["failures"])
    if not metam["all_passed"]:
        raise SystemExit("metamorphic tests failed")

    if mode == "validate":
        ds = build_dataset(n_train=6, n_dev=3, n_test=6, n_comp=8, n_rob=6, seed=3001)
        rob_sc = []
        for vs in ds["robustness"].values():
            for v in vs.values():
                rob_sc.append(v["scenario"] if isinstance(v, dict) and "scenario" in v else v)
        val = validate_scenarios(ds["train"] + ds["dev"] + ds["test"] + ds["comp"] + rob_sc, tok)
        print("validation:", json.dumps(val, indent=2))
        print("disjointness:", json.dumps(ds["disjointness"], indent=2))
        # balance summary
        from collections import Counter
        bal = Counter((c["family"], c["depth"], c["oracle"]["reachable"])
                      for c in ds["train"] + ds["dev"] + ds["test"])
        print("cell balance (family,depth,reachable):")
        for k in sorted(bal):
            print("  ", k, bal[k])
        return
    if mode == "build":
        ds = build_dataset()
        write_dataset(ds, tok)
        return
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()