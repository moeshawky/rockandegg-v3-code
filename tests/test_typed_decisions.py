"""Tests that the typed-decisions conversion is a single source of truth.

The training data builder (``typed_decisions_data``) and the benchmark harness
(``benchmark_vs_laya``) must render identical option text, instruction text and
state text for the same corpus row. If they ever drift, a matched-data
comparison silently measures two different encodings.

These tests need the ``LocalLLaMA/typed-decisions`` corpus. They skip when it
is not available rather than failing a CPU-only test run.
"""

from __future__ import annotations

import json

import pytest
from typed_decisions_data import (
    load_typed_decisions_cases,
    render_options,
    serialize_state,
    target_from_gold,
)


def _corpus_rows():
    """Return a few corpus rows, or skip when the dataset is unavailable."""
    try:
        from datasets import load_dataset

        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"typed-decisions corpus unavailable: {exc}")
    return [ds[i] for i in range(3)]


def test_render_options_matches_benchmark_harness() -> None:
    """The trainer and the benchmark must render identical option text."""
    # The Laya harness depends on a private reference checkout and is not part
    # of the public candidate. Skip explicitly rather than fail misleadingly.
    build_v3_request = pytest.importorskip(
        "benchmark_vs_laya",
        reason="private Laya benchmark harness is not shipped in the public release",
    ).build_v3_request

    for row in _corpus_rows():
        questions = json.loads(row["questions"])
        request, bench_keys = build_v3_request(json.loads(row["state"]), questions)
        bench_options = [q.options for q in request.questions]
        bench_instructions = [q.instruction for q in request.questions]

        canon_options, canon_keys = [], []
        for qdef in questions.values():
            opts, keys = render_options(qdef)
            canon_options.append(opts)
            canon_keys.append(keys)

        assert canon_options == bench_options
        assert canon_keys == bench_keys
        assert canon_instructions_ok(bench_instructions, questions)


def canon_instructions_ok(bench_instructions: list[str], questions: dict) -> bool:
    """Instructions must match the corpus verbatim."""
    expected = [str(q.get("instructions", "")) for q in questions.values()]
    return bench_instructions == expected


def test_state_text_is_identical_for_train_and_eval() -> None:
    """Both paths must feed the model byte-identical state text."""
    # The Laya harness depends on a private reference checkout and is not part
    # of the public candidate. Skip explicitly rather than fail misleadingly.
    build_v3_request = pytest.importorskip(
        "benchmark_vs_laya",
        reason="private Laya benchmark harness is not shipped in the public release",
    ).build_v3_request

    for row in _corpus_rows():
        state = json.loads(row["state"])
        request, _ = build_v3_request(state, json.loads(row["questions"]))
        assert request.state == serialize_state(state)


def test_target_is_normalized_and_key_aligned() -> None:
    """Soft targets sum to one and follow the rendered option order."""
    for row in _corpus_rows():
        questions = json.loads(row["questions"])
        gold = json.loads(row["gold"])
        for qid, qdef in questions.items():
            options, keys = render_options(qdef)
            target = target_from_gold(gold[qid], keys)
            assert len(target) == len(options)
            assert abs(sum(target) - 1.0) < 1e-6
            assert all(t >= 0.0 for t in target)
            # The target must be aligned to the keys: the probability of the
            # gold label has to equal the corpus value for that key.
            gprobs = gold[qid]["probabilities"]
            for key, value in zip(keys, target, strict=True):
                expected = gprobs.get(key, 0.0)
                if expected > 0:
                    assert abs(value - expected / sum(gprobs.values())) < 1e-6


def test_loader_yields_one_case_per_question() -> None:
    """The loader flattens corpus cases into one typed decision per question."""
    cases = load_typed_decisions_cases("test", max_cases=5)
    assert cases
    for case in cases:
        assert case.kind in {"choice", "noul", "score"}
        assert len(case.target) == len(case.options)
        assert abs(sum(case.target) - 1.0) < 1e-6
        assert ":" in case.case_id
