"""CPU unit tests for the opt-in semantic rendering helper and stable keys.

These tests need no model weights. They lock the exact archived rendering
strings, the false-first / prob_true-index-1 alignment, the stable-key mapping,
and the deliberate exclusion of ``critical_condition``.
"""

from __future__ import annotations

import pytest

from systemone_inference import Question
from systemone_semantics import (
    DEFAULT_ANSWER_KEYS,
    family_instruction,
    generic_noul_options,
    render_state_specific_noul,
    render_state_specific_options,
    supported_families,
)


def test_supported_families_are_exactly_the_validated_two() -> None:
    """Only safe_to_continue and human_review have a validated rendering."""
    assert supported_families() == ("human_review", "safe_to_continue")


@pytest.mark.parametrize(
    ("family", "generic"),
    [
        (
            "safe_to_continue",
            [
                "false: the state shows an unresolved failure or an unapproved change",
                "true: the state shows all checks passed and continuation was approved",
            ],
        ),
        (
            "human_review",
            [
                "false: the state shows a routine deterministic case with no ambiguity",
                "true: the state shows conflicting instructions or an irreversible action",
            ],
        ),
    ],
)
def test_generic_options_match_archived_canonical_strings(
    family: str, generic: list[str]
) -> None:
    """Arm A rendering is byte-identical to the archived intervention lock."""
    assert generic_noul_options(family) == generic


def test_family_instruction_is_canonical() -> None:
    """The helper uses the exact instruction held constant in the study."""
    assert family_instruction("safe_to_continue") == (
        "Based on the state, is it safe to continue?"
    )
    assert family_instruction("human_review") == (
        "Based on the state, does this case require human review?"
    )


def test_state_specific_options_are_false_first() -> None:
    """prob_true is index 1; false must stay first, true second."""
    opts = render_state_specific_options(
        "safe_to_continue",
        "the canary is crashing and no fix is recorded",
        "every canary metric is green and the owner approved",
    )
    assert opts == [
        "false: the canary is crashing and no fix is recorded",
        "true: every canary metric is green and the owner approved",
    ]


def test_rendered_question_has_stable_keys_and_canonical_instruction() -> None:
    """Rendering changes scoring text, never the caller's answer identity."""
    q = render_state_specific_noul("safe_to_continue", "checks failed", "checks passed")
    assert q.type == "noul"
    assert q.answer_keys == list(DEFAULT_ANSWER_KEYS)
    assert q.instruction == family_instruction("safe_to_continue")
    assert q.observable_key(0) == "false"
    assert q.observable_key(1) == "true"


def test_custom_stable_keys_are_preserved() -> None:
    """A caller can keep its own identity vocabulary across renderings."""
    q = render_state_specific_noul(
        "human_review",
        "routine deterministic",
        "conflicting instructions",
        answer_keys=("no_review", "needs_review"),
    )
    assert q.observable_key(0) == "no_review"
    assert q.observable_key(1) == "needs_review"


def test_legacy_question_without_keys_reports_option_text() -> None:
    """Omitting answer_keys preserves the original observable key exactly."""
    q = Question(type="noul", options=["a", "b"])
    assert q.answer_keys is None
    assert q.observable_key(0) == "a"
    assert q.observable_key(1) == "b"


def test_answer_keys_length_mismatch_rejected() -> None:
    """A broken mapping fails at construction, before any inference."""
    with pytest.raises(ValueError, match="one entry per option"):
        Question(type="noul", options=["a", "b"], answer_keys=["only-one"])


@pytest.mark.parametrize("family", ["noul", "choice", ""])
def test_unknown_family_is_rejected_everywhere(family: str) -> None:
    """Non-families are rejected for both generic and rendered rendering."""
    with pytest.raises(ValueError, match="unknown decision family"):
        family_instruction(family)
    with pytest.raises(ValueError, match="unsupported decision family"):
        render_state_specific_options(family, "a", "b")


def test_critical_condition_generic_is_available_but_renderer_refuses() -> None:
    """Arm A generic is legitimate; only the trivializing renderer is blocked."""
    assert family_instruction("critical_condition") == (
        "Based on the state, is there a critical condition?"
    )
    assert generic_noul_options("critical_condition")[1].startswith("true:")
    with pytest.raises(ValueError, match="trivially separable") as exc:
        render_state_specific_options("critical_condition", "a", "b")
    assert "both the previous and R_final3" in str(exc.value)


@pytest.mark.parametrize("condition", ["", "   ", None])
def test_empty_condition_is_rejected(condition: str | None) -> None:
    """Empty decisive conditions cannot be rendered into meaningful options."""
    with pytest.raises(ValueError, match="non-empty string"):
        render_state_specific_options("safe_to_continue", condition, "ok")  # type: ignore[arg-type]


def test_generic_and_rendered_share_the_same_key_space() -> None:
    """A caller switching from Arm A to the renderer keeps the same keys."""
    a = Question(
        type="noul", options=generic_noul_options("safe_to_continue"), answer_keys=list(DEFAULT_ANSWER_KEYS)
    )
    b = render_state_specific_noul("safe_to_continue", "failed", "passed")
    for idx in (0, 1):
        assert a.observable_key(idx) == b.observable_key(idx)
