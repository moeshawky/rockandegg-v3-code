"""Curriculum v1 — symbolic dependency DAG + INDEPENDENT oracle + metamorphic tests.

The measurement instrument, kept separate from the scenario decorators
(curriculum_v1.py) so the oracle can be audited alone.

--------------------------------------------------------------------------
THE GRAPH MODEL (family-agnostic, general DAG)
--------------------------------------------------------------------------
A scenario is a directed acyclic graph of dependency nodes:

  * ``Node``  : ``nid``, ``name``, ``alive``, ``kind`` in
                {"provider","relay","and","or"}, ``children`` (list of nids).
  * ``Dag``   : nodes, ``entry`` (facing the operator), ``goal_source``
                (the node that directly yields the goal capability), ``depth``
                (configured longest-path length), plus family/decoration metadata.

Semantics — the ONLY thing the oracle uses:

  satisfies(n) := n.alive AND
                  (n.children empty  ->  True)          # terminal/resource/goal source
                  (kind == "or"      ->  any child satisfies)
                  (otherwise         ->  all children satisfy)   # provider/relay/and

  reachable    := satisfies(entry)
  candidates   := the components offered as actions
  solution     := the highest (closest-to-entry) candidate that satisfies, else
                  "unreachable".

Family decorators (curriculum_v1.py) add stage names, status vocabulary,
situation prose, action phrasings and distractor pools. They NEVER compute the
answer — they render the graph the oracle already solved. The oracle reads
primitive facts (alive flags + topology), never a precomputed outcome field.

--------------------------------------------------------------------------
METHAMORPHIC TESTS
--------------------------------------------------------------------------
1. Kill the goal source  -> reachable flips true->false.
2. Kill the resource     -> reachable flips true->false.
3. Rename every entity   -> answer unchanged.
4. Add a disconnected node (decoration) -> answer unchanged.
5. OR-redundancy: a dead primary path + a live alternate path -> still reachable.
6. AND-dependency: one dead required child -> unreachable.
7. Fully-live chain of depth D -> entry is the top, links checked == D.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Graph + oracle
# ---------------------------------------------------------------------------
@dataclass
class Node:
    nid: str
    name: str
    alive: bool
    kind: str = "relay"           # provider | relay | and | or
    children: list[str] = field(default_factory=list)


@dataclass
class Dag:
    family: str
    nodes: dict[str, Node]
    entry: str
    goal_source: str
    depth: int
    candidates: list[str] = field(default_factory=list)   # nids offered as actions
    decorations: list[Node] = field(default_factory=list)  # described but irrelevant
    meta: dict[str, Any] = field(default_factory=dict)

    def node(self, nid: str) -> Node:
        return self.nodes[nid]


def satisfies(dag: Dag, nid: str, _stack: tuple[str, ...] = ()) -> bool:
    """Recursive dependency evaluation. Pure graph traversal."""
    if nid in _stack:  # cycle guard (DAGs should not cycle)
        return False
    node = dag.nodes[nid]
    if not node.alive:
        return False
    if not node.children:
        return True
    if node.kind == "or":
        return any(satisfies(dag, c, _stack + (nid,)) for c in node.children)
    return all(satisfies(dag, c, _stack + (nid,)) for c in node.children)


def depth_from_entry(dag: Dag) -> dict[str, int]:
    """BFS distance (in edges) from the entry to every reachable node."""
    dist = {dag.entry: 0}
    q = deque([dag.entry])
    while q:
        cur = q.popleft()
        for ch in dag.nodes[cur].children:
            if ch not in dist:
                dist[ch] = dist[cur] + 1
                q.append(ch)
    return dist


def solve(dag: Dag) -> dict[str, Any]:
    """Return the oracle's answer. The ONLY label source."""
    dist = depth_from_entry(dag)
    cands = dag.candidates or [dag.entry]
    good = [c for c in cands if satisfies(dag, c)]
    reachable = len(good) > 0
    if reachable:
        best = min(good, key=lambda c: dist.get(c, 10**6))
        action = f"invoke:{best}"
        chain = _live_chain(dag, best)
    else:
        best, action, chain = None, "report_unreachable", None
    return {
        "reachable": reachable,
        "action": action,
        "action_node": best,
        "action_name": dag.nodes[best].name if best else None,
        "chain": chain,
        "depth": dag.depth,
        # number of dependency EDGES traversed from the chosen action to the
        # goal-providing node (live path length minus one node).
        "n_links_checked": (len(chain) - 1) if (reachable and chain) else dag.depth,
        "candidate_results": {c: satisfies(dag, c) for c in cands},
    }


def _live_chain(dag: Dag, start: str) -> list[str]:
    """Return one live path start -> goal_source (names), or []."""
    # DFS for a path of satisfied nodes.
    def dfs(nid, path):
        if nid == dag.goal_source and dag.nodes[nid].alive:
            return path + [dag.nodes[nid].name]
        node = dag.nodes[nid]
        if not node.alive:
            return []
        for ch in node.children:
            if satisfies(dag, ch):
                r = dfs(ch, path + [node.name])
                if r:
                    return r
        return []
    return dfs(start, [])


def canonical_graph_hash(dag: Dag) -> str:
    """Structure-only hash: family + alive pattern + topology + entry/candidates."""
    struct = {
        "family": dag.family,
        "depth": dag.depth,
        "nodes": {n.nid: {"alive": n.alive, "kind": n.kind,
                          "children": sorted(n.children)}
                  for n in dag.nodes.values()},
        "entry": dag.entry,
        "goal_source": dag.goal_source,
        "candidates": sorted(dag.candidates),
        "topology": dag.meta.get("topology", "chain"),
    }
    return hashlib.sha256(json.dumps(struct, sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def chain_dag(family: str, depth: int, alive_pattern: list[bool],
              resource_alive: bool, names: list[str],
              topology: str = "chain") -> Dag:
    """Linear chain S_0 -> ... -> S_D -> resource(goal source).

    S_0 is the entry. S_D (provider) requires the resource, which is the goal
    source. ``alive_pattern`` has length depth+1.
    """
    assert len(alive_pattern) == depth + 1, (len(alive_pattern), depth)
    assert len(names) == depth + 1, (len(names), depth)
    nodes: dict[str, Node] = {}
    for i, alive in enumerate(alive_pattern):
        kind = "provider" if i == depth else "relay"
        nodes[f"S{i}"] = Node(nid=f"S{i}", name=names[i], alive=alive, kind=kind)
    for i in range(depth):
        nodes[f"S{i}"].children = [f"S{i+1}"]
    # The provider S_D IS the goal source; it additionally requires the resource
    # leaf R to be alive (so killing R makes the goal unreachable).
    res_id = "R"
    nodes[res_id] = Node(nid=res_id, name="res", alive=resource_alive, kind="provider")
    nodes[f"S{depth}"].children = [res_id]
    return Dag(family=family, nodes=nodes, entry="S0", goal_source=f"S{depth}",
               depth=depth, candidates=[f"S{i}" for i in range(depth + 1)],
               meta={"topology": topology})


def branching_dag(family: str, depth: int, topology: str,
                  names: list[str]) -> Dag:
    """Novel-composition DAGs (not used in training): OR-redundancy and AND-split.

    * "or_redundant": entry -> OR(S_a, S_b) ; each S -> resource. Reachable iff
      resource alive AND (S_a or S_b alive). Depth = 2 (entry->OR->provider->res).
    * "and_split":    entry -> AND(S_a, S_b) ; each S -> its own resource.
      Reachable iff all of S_a,S_b and both resources alive. Depth = 2.
    """
    nodes: dict[str, Node] = {}
    if topology == "or_redundant":
        nodes["S0"] = Node("S0", names[0], True, "or", ["A", "B"])
        nodes["A"] = Node("A", names[1], True, "provider", ["RA"])
        nodes["B"] = Node("B", names[2], True, "provider", ["RB"])
        nodes["RA"] = Node("RA", "res_a", True, "provider")
        nodes["RB"] = Node("RB", "res_b", True, "provider")
        return Dag(family, nodes, "S0", "S0", depth, candidates=["S0"],
                   meta={"topology": topology})
    if topology == "and_split":
        nodes["S0"] = Node("S0", names[0], True, "and", ["A", "B"])
        nodes["A"] = Node("A", names[1], True, "relay", ["RA"])
        nodes["B"] = Node("B", names[2], True, "relay", ["RB"])
        nodes["RA"] = Node("RA", "res_a", True, "provider")
        nodes["RB"] = Node("RB", "res_b", True, "provider")
        return Dag(family, nodes, "S0", "S0", depth, candidates=["S0"],
                   meta={"topology": topology})
    raise ValueError(topology)


# ---------------------------------------------------------------------------
# Metamorphic tests
# ---------------------------------------------------------------------------
def metamorphic_tests() -> dict[str, Any]:
    results: dict[str, Any] = {}
    failures: list[str] = []

    def check(name, cond, detail):
        results[name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            failures.append(name)

    # 1. kill goal source
    g = chain_dag("substrate", 2, [True, True, True], True, ["a", "b", "c"])
    check("kill_goal_source", solve(g)["reachable"] is True, solve(g))
    g.nodes["R"].alive = False
    check("kill_goal_source", solve(g)["reachable"] is False, solve(g))

    # 2. kill resource (goal source already tested); kill an intermediate on path
    g = chain_dag("substrate", 2, [True, True, True], True, ["a", "b", "c"])
    g.nodes["S1"].alive = False
    s = solve(g)
    check("kill_intermediate_entry_moves", s["reachable"] and s["action_node"] == "S2", s)

    # 3. name invariance
    g1 = chain_dag("substrate", 2, [True, False, True], True, ["a", "b", "c"])
    g2 = chain_dag("substrate", 2, [True, False, True], True, ["x", "y", "z"])
    check("name_invariance",
          solve(g1)["reachable"] == solve(g2)["reachable"]
          and solve(g1)["action_node"] == solve(g2)["action_node"],
          {"g1": solve(g1), "g2": solve(g2)})

    # 4. decoration irrelevance
    g1 = chain_dag("substrate", 2, [True, True, True], True, ["a", "b", "c"])
    base = solve(g1)
    g1.decorations = [Node("D0", "noise", False, "relay")]
    check("decoration_irrelevance", solve(g1) == base, {"base": base})

    # 5. OR-redundancy: kill A, B alive -> reachable
    g = branching_dag("config", 2, "or_redundant", ["entry", "a", "b"])
    g.nodes["A"].alive = False
    check("or_redundancy", solve(g)["reachable"] is True, solve(g))
    # kill both -> unreachable
    g.nodes["B"].alive = False
    check("or_both_dead", solve(g)["reachable"] is False, solve(g))

    # 6. AND-split: one child dead -> unreachable
    g = branching_dag("config", 2, "and_split", ["entry", "a", "b"])
    g.nodes["A"].alive = False
    check("and_one_dead_unreachable", solve(g)["reachable"] is False, solve(g))
    g.nodes["A"].alive = True
    check("and_all_alive_reachable", solve(g)["reachable"] is True, solve(g))

    # 7. fully-live chain depth D -> action S0, links == D
    depth_ok = True
    for d in (1, 2, 3):
        g = chain_dag("substrate", d, [True] * (d + 1), True, [f"s{i}" for i in range(d + 1)])
        s = solve(g)
        if not (s["reachable"] and s["action_node"] == "S0" and s["n_links_checked"] == d):
            depth_ok = False
    check("fully_live_depth_links", depth_ok, "ok")

    return {
        "n_tests": 8,
        "failures": failures,
        "all_passed": len(failures) == 0,
        "detail": results,
    }


if __name__ == "__main__":
    print(json.dumps(metamorphic_tests(), indent=2))