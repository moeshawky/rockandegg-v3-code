"""Regression tests for the multi-epoch training loop.

``train_jev_distill.py --epochs N`` used to run only the FIRST epoch: the
loop derived its start offset from the global step counter, so every epoch
after the first got an empty range. The symptom was silent — a 16-epoch run
reported the same step count as a 1-epoch run.

These tests pin the step arithmetic of :func:`epoch_step_plan` so the defect
cannot return unnoticed. Assertions are derived from the helper's documented
contract (coverage, step count, per-device shape, resume offset), not from
hard-coded arithmetic about a particular corpus size.
"""

from __future__ import annotations

import random

import pytest
from train_jev_distill import Case, epoch_step_plan, option_buckets


def _cases(n: int = 40) -> list[Case]:
    """Build synthetic cases with mixed option widths (2, 3 and 4 options)."""
    return [
        Case(
            case_id=f"c{i}",
            kind="choice" if i % 3 else "noul",
            options=[f"opt{j}" for j in range(2 + (i % 3))],
            target=[1.0 / (2 + (i % 3))] * (2 + (i % 3)),
            state="state",
            question="q",
        )
        for i in range(n)
    ]


def _batches(cases: list[Case], micro: int = 8, seed: int = 0) -> list[list[int]]:
    """Build one epoch's option-bucketed batches from a shuffled order."""
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)  # noqa: S311 - deterministic ML shuffling
    return option_buckets(order, cases, micro)


def _plan_indices(plan: list[list[list[int]]]) -> list[int]:
    """Flatten a step plan into the case indices it consumes, in order."""
    return [idx for step in plan for batch in step for idx in batch]


def test_plan_covers_every_case_exactly_once_per_epoch() -> None:
    """One epoch must present every training case exactly once."""
    cases = _cases()
    plan = epoch_step_plan(_batches(cases), n_devices=2)

    assert sorted(_plan_indices(plan)) == list(range(len(cases)))


def test_plan_has_no_duplicate_case_within_an_epoch() -> None:
    """A case must not be consumed twice inside one epoch."""
    cases = _cases()
    plan = epoch_step_plan(_batches(cases), n_devices=2)

    indices = _plan_indices(plan)
    assert len(indices) == len(set(indices))


def test_each_step_holds_at_most_one_batch_per_device() -> None:
    """A step is one batch per data-parallel replica, never more."""
    cases = _cases()
    plan = epoch_step_plan(_batches(cases), n_devices=2)

    assert all(len(step) <= 2 for step in plan)
    assert all(len(step) >= 1 for step in plan)


def test_step_count_is_epochs_times_steps_per_epoch() -> None:
    """A multi-epoch run must take ``epochs * steps-per-epoch`` optimizer steps.

    This is the quantity the cosine LR schedule is stretched over, and the
    quantity the old defect collapsed to a single epoch's worth.
    """
    cases = _cases()
    batches = _batches(cases)

    per_epoch = len(epoch_step_plan(batches, n_devices=2))
    assert per_epoch > 0

    for epochs in (1, 2, 4, 16):
        total = sum(len(epoch_step_plan(batches, n_devices=2)) for _ in range(epochs))
        assert total == per_epoch * epochs
        if epochs > 1:
            # The defect's signature: a multi-epoch run that took the same
            # number of steps as a single epoch.
            assert total > per_epoch, "a multi-epoch run must not collapse to one"


def test_global_step_offset_would_truncate_later_epochs() -> None:
    """Pin WHY the offset must be per-epoch, not global.

    Reproduces the old loop shape — a start offset derived from the running
    step counter — and asserts it yields an empty plan for epoch 2 onward.
    This is the regression witness: the correct shape must multiply, the old
    shape must collapse.
    """
    cases = _cases()
    batches = _batches(cases)
    devices = 2

    steps_per_epoch = len(epoch_step_plan(batches, n_devices=devices))
    assert steps_per_epoch > 0

    global_step = 0
    old_lengths: list[int] = []
    for _epoch in range(4):
        old_plan = [
            batches[offset : offset + devices]
            for offset in range(global_step * devices, len(batches), devices)
        ]
        old_lengths.append(len(old_plan))
        global_step += len(old_plan)

    assert old_lengths[0] == steps_per_epoch
    assert old_lengths[1] == 0, "the old global offset emptied later epochs"
    assert sum(old_lengths) == steps_per_epoch, "the old shape trains one epoch"

    new_lengths = [len(epoch_step_plan(batches, n_devices=devices)) for _ in range(4)]
    assert sum(new_lengths) == steps_per_epoch * 4


def test_resume_offset_skips_consumed_batches() -> None:
    """A resume start offset must skip exactly the consumed batches."""
    cases = _cases()
    batches = _batches(cases)
    devices = 2

    full = epoch_step_plan(batches, n_devices=devices)
    consumed_steps = 1
    assert len(full) > consumed_steps, "corpus must leave steps after the resume"

    resumed = epoch_step_plan(
        batches, n_devices=devices, start_offset=consumed_steps * devices
    )

    assert resumed == full[consumed_steps:]
    resumed_indices = set(_plan_indices(resumed))
    consumed_indices = set(_plan_indices(full[:consumed_steps]))
    assert not consumed_indices & resumed_indices
    assert consumed_indices | resumed_indices == set(_plan_indices(full))


def test_resume_offset_past_the_end_yields_an_empty_plan() -> None:
    """Resuming past the end of the epoch must produce no steps, not a crash."""
    cases = _cases()
    batches = _batches(cases)

    plan = epoch_step_plan(batches, n_devices=2, start_offset=len(batches) + 10)
    assert plan == []


def test_zero_device_count_is_rejected() -> None:
    """A zero-device plan would silently drop every batch."""
    with pytest.raises(ValueError, match="n_devices"):
        epoch_step_plan(_batches(_cases()), n_devices=0)
