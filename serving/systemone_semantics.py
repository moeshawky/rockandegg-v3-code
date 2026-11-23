"""First-party semantic rendering helpers for typed SystemOne decisions.

Why this module exists
----------------------
The frozen-weight intervention study (private archive
``rockandegg-v3-postrelease-audit``, revision ``38ae9149…``) reproduced on
R_final3 weights (SHA-256 ``bd30d80e…``) with *exact* logit parity showed that
the residual ``safe_to_continue`` / ``human_review`` errors were substantially
an **option-rendering** problem, not a defect in the scoring engine:

* generic canonical options: balanced accuracy 0.673 / 0.800 (false approvals
  concentrated in ``safe_to_continue``);
* state-grounded options naming the concrete decisive condition: 0.942 / 0.880;
* state-blind control (masked state, same rendered options): 0.423 / 0.540 —
  i.e. the gain is genuinely state-conditioned, not answer text that carries
  the label on its own;
* counterfactual pairs flip 5/6 and 4/6 respectively.

The generic ``SystemOneInference`` engine was correct throughout: it scores
exactly the strings a caller supplies. The failure belonged to the *caller's*
request construction — underspecified option text. This module is the narrow,
explicit interface-boundary correction: it renders caller-supplied decisive
conditions into the validated state-specific format while preserving stable
answer identity through ``Question.answer_keys``.

Scope and non-goals
-------------------
* The caller supplies the two decisive-condition phrases. This helper does
  **not** infer a family from instruction text, does **not** read or parse the
  state, and never silently rewrites a caller's generic Noul options.
* ``critical_condition`` is deliberately **not** supported *for state-specific
  rendering*. Its state-specific rendering scored 1.000 on *both* the previous
  and the R_final3 checkpoint: a trivialization guard, not recovery evidence.
  Its canonical **generic** instruction and options remain available (Arm A is
  legitimate) — only :func:`render_state_specific_options` and
  :func:`render_state_specific_noul` refuse it.
* No weights, token budgets, precision, sharing mechanism, probability
  semantics or default request behaviour change.
"""

from __future__ import annotations

from collections.abc import Sequence

from systemone_inference import Question

# Canonical strings from the archived intervention lock (intervention_lock.json)
# and the production typed-decision renderer. Kept literal so the public helper
# reproduces the validated rendering exactly and can be regression-tested.
_FAMILY_INSTRUCTIONS: dict[str, str] = {
    "safe_to_continue": "Based on the state, is it safe to continue?",
    "human_review": "Based on the state, does this case require human review?",
    "critical_condition": "Based on the state, is there a critical condition?",
}

_GENERIC_OPTIONS: dict[str, tuple[str, str]] = {
    "safe_to_continue": (
        "false: the state shows an unresolved failure or an unapproved change",
        "true: the state shows all checks passed and continuation was approved",
    ),
    "human_review": (
        "false: the state shows a routine deterministic case with no ambiguity",
        "true: the state shows conflicting instructions or an irreversible action",
    ),
    "critical_condition": (
        "false: the state shows normal metrics within operating bounds",
        "true: the state shows severe sustained degradation of core metrics",
    ),
}

# Families with a validated state-specific rendering. critical_condition is
# excluded by the trivialization guard documented in the module docstring.
_RENDERABLE_FAMILIES = ("human_review", "safe_to_continue")

_UNSUPPORTED_FAMILIES: dict[str, str] = {
    "critical_condition": (
        "state-specific critical_condition options are trivially separable on "
        "both the previous and R_final3 checkpoints (1.000 balanced accuracy "
        "on each); that is a rendering trivialization, not recovery evidence"
    ),
}

DEFAULT_ANSWER_KEYS: tuple[str, str] = ("false", "true")


def supported_families() -> tuple[str, ...]:
    """Return the families with a validated state-specific rendering."""
    return _RENDERABLE_FAMILIES


def family_instruction(family: str) -> str:
    """Return the canonical instruction for any known decision family.

    This is the Arm A instruction and is available for all three families,
    including ``critical_condition``. Only the state-specific renderer is
    restricted.

    Raises:
        ValueError: If the family is unknown.
    """
    _require_known(family)
    return _FAMILY_INSTRUCTIONS[family]


def generic_noul_options(family: str) -> list[str]:
    """Return the validated generic (criterion-only) Noul options for a family.

    This is the Arm A rendering; it is available for all three families and is
    not subject to the critical_condition rendering guard.
    """
    _require_known(family)
    return list(_GENERIC_OPTIONS[family])


def render_state_specific_options(
    family: str,
    false_condition: str,
    true_condition: str,
) -> list[str]:
    """Render state-grounded Noul options (the validated Arm B format).

    Args:
        family: ``safe_to_continue`` or ``human_review`` (the validated
            renderable families; ``critical_condition`` is refused).
        false_condition: concrete, state-grounded phrase describing the
            condition that makes the decision false (e.g. "the migration
            failed halfway and continuation was not approved").
        true_condition: concrete, state-grounded phrase describing the
            condition that makes the decision true.

    Returns:
        ``["false: <false_condition>", "true: <true_condition>"]`` — the exact
        string format used by the archived frozen-weight study, with false
        first and true second so probability index 1 remains ``prob_true``.

    Raises:
        ValueError: If the family is not renderable or a condition is empty.
    """
    _require_renderable(family)
    _require_condition(false_condition, "false_condition")
    _require_condition(true_condition, "true_condition")
    return [f"false: {false_condition}", f"true: {true_condition}"]


def render_state_specific_noul(
    family: str,
    false_condition: str,
    true_condition: str,
    *,
    answer_keys: Sequence[str] = DEFAULT_ANSWER_KEYS,
) -> Question:
    """Build a Noul ``Question`` with rendered text and stable answer keys.

    The scoring text is the state-grounded rendering; the observable
    ``chosen_key`` is the caller's stable identity (``"false"`` / ``"true"``
    by default). This is the difference between a rendering change and an
    answer-identity change: a caller that approved on ``"true"`` keeps seeing
    ``"true"`` no matter how the option text is worded.

    Args:
        family: supported semantic family.
        false_condition: concrete false-decision condition phrase.
        true_condition: concrete true-decision condition phrase.
        answer_keys: two stable identities, false then true.

    Returns:
        A frozen ``Question`` ready for ``SystemOneInference`` or, via its
        ``options`` / ``instruction`` / ``answer_keys`` fields, an HTTP request.
    """
    options = render_state_specific_options(family, false_condition, true_condition)
    if len(answer_keys) != 2:
        msg = "answer_keys must contain exactly two identities (false, true)"
        raise ValueError(msg)
    return Question(
        type="noul",
        options=options,
        instruction=family_instruction(family),
        answer_keys=list(answer_keys),
    )


def _require_known(family: str) -> None:
    """Reject unknown families for canonical instruction/options lookups."""
    if family not in _FAMILY_INSTRUCTIONS:
        msg = f"unknown decision family {family!r}"
        raise ValueError(msg)


def _require_renderable(family: str) -> None:
    """Reject families without a validated state-specific rendering."""
    if family in _RENDERABLE_FAMILIES:
        return
    reason = _UNSUPPORTED_FAMILIES.get(
        family, "no validated state-specific rendering is available for this family"
    )
    msg = f"unsupported decision family for state-specific rendering {family!r}: {reason}"
    raise ValueError(msg)


def _require_condition(value: str, name: str) -> None:
    """Reject empty or whitespace-only decisive-condition phrases."""
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string"
        raise ValueError(msg)
