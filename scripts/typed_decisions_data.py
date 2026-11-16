"""Canonical conversion between the typed-decisions corpus and the v3 format.

Single source of truth for how a ``LocalLLaMA/typed-decisions`` case becomes a
SystemOne request. Both the training data builder and the benchmark harness
import from here, so the *training* encoding and the *evaluation* encoding
cannot drift apart.

The option rendering mirrors the reference typed-decisions format of the Laya
reference implementation, so both systems see the same option text.

Verified corpus invariants (checked over both splits):

* ``choice`` — gold probability keys match the criteria keys, in the same order.
* ``score`` — ``gold["score"]`` is the expected level ``sum(i * p_i)``;
  ``gold["label"]`` is the arg-max level. Criteria length is 4 or 5.
* ``noul`` — ``gold["noul"] == gold["probabilities"]["true"]``; keys are
  ``{"false", "true"}``.
"""

from __future__ import annotations

import json
from typing import Any

from train_jev_distill import Case

DATASET_NAME = "LocalLLaMA/typed-decisions"


def serialize_state(state: Any) -> str:
    """Render a case state as the text the model actually sees.

    Mirrors the state serialization of the Laya reference
    implementation: a string passes through, anything else is JSON-encoded.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(qdef: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Render one question's option texts and their keys.

    Args:
        qdef: a question definition from the corpus (``type``, ``criteria``).

    Returns:
        The option strings in label-index order and the matching key list.
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
            "false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds"),
        ]
        keys = ["false", "true"]
    return options, keys


def target_from_gold(gold_q: dict[str, Any], keys: list[str]) -> list[float]:
    """Build the soft-label target for one question, aligned to ``keys``.

    Args:
        gold_q: the gold answer object for one question.
        keys: option keys in the order the options were rendered.

    Returns:
        A normalized probability vector over the options.
    """
    probs = gold_q["probabilities"]
    vals = [float(probs.get(k, 0.0)) for k in keys]
    total = sum(vals)
    if total <= 0.0:
        return [1.0 / len(keys)] * len(keys)
    return [v / total for v in vals]


def load_typed_decisions_cases(split: str, max_cases: int | None = None) -> list[Case]:
    """Load one typed-decisions split as v3 training cases.

    Each corpus case contributes one :class:`Case` per question, so a
    1,200-case split yields 6,000 typed-decision training cases.

    Args:
        split: dataset split name (``train`` or ``test``).
        max_cases: optional cap on the number of corpus cases read.

    Returns:
        The cases, each carrying its soft-label target and option texts.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, "all", split=split)
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
            target = target_from_gold(g, keys)
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


def corpus_stats(cases: list[Case]) -> dict[str, Any]:
    """Summarize a loaded split: kind counts and per-kind option widths."""
    kinds: dict[str, int] = {}
    widths: dict[str, set[int]] = {}
    for case in cases:
        kinds[case.kind] = kinds.get(case.kind, 0) + 1
        widths.setdefault(case.kind, set()).add(len(case.options))
    return {
        "n_cases": len(cases),
        "by_kind": kinds,
        "option_widths": {k: sorted(v) for k, v in widths.items()},
    }
