# Current-release supplement — opt-in semantic rendering with stable answer keys

**Status:** source-only release supplement (2026-11-23). It does **not** rewrite
the frozen historical release dossier in `docs/release/`; it adds the current
identity and behaviour on top of it.

## 1. What was wrong, and who owned it

The frozen-weight intervention archive (`moeshawky/rockandegg-v3-postrelease-audit`,
revision `38ae9149…`, archive SHA-256 `07001ae4…`) reported that Arm B —
state-grounded Noul option text — improved `safe_to_continue` balanced accuracy
from 0.692 to 0.942 (false approvals 16 → 2) and counterfactual flips from
0.167 to 0.833 on locked R_final3 weights.

We reproduced both arms through the **canonical production adapter** on the
downloaded pinned `main` weights (SHA-256 `bd30d80e…`):

| Check | Result |
| --- | --- |
| Rows scored (locked 152-case test panel, both renderings) | 304 |
| Max absolute logit difference vs archived raw predictions | **0.0** |
| Max absolute `prob_true` difference | **0.0** |
| Decision mismatches | **0** |

So the experiment and the production scoring path are equivalent. The generic
engine was never the defect: it scores exactly the strings a caller supplies.

The failure was **caller-side semantic underspecification**. The bundled
generic options compress the decision criterion ("unresolved failure or
unapproved change"); callers obtained false approvals precisely on
surface-misleading states where checks passed but approval was absent.

Two alternative explanations were tested and rejected or bounded:

* **Pure benchmark shortcut (H3/H4) — rejected as the whole story.** With the
  state replaced by a masked, information-free string while keeping the same
  rendered options, accuracy collapsed to 0.423 / 0.540 / 0.280 for the three
  families (near or below chance). The improvement is therefore genuinely
  state-conditioned; the option text alone does not carry the label.
* **Genuine representation failure (H5) — bounded.** Counterfactual pairs
  flipped 5/6 (`safe_to_continue`), 4/6 (`human_review`) and 6/6
  (`critical_condition`) with the frozen weights, so the information needed
  is reachable when the criterion is expressed concretely.

`critical_condition` remains **unresolved**: its state-specific rendering scored
1.000 on *both* the previous and R_final3 checkpoints, which the archive flags
as a rendering trivialization. It is not recovery evidence and is deliberately
excluded from the new renderer.

## 2. What changed

Three serving files and two test files changed; **no weights, architecture,
token budgets, precision policy, sharing mechanism or default semantics changed**.

1. `serving/systemone_inference.py` — `Question.answer_keys`, an optional
   backward-compatible stable-identity facility. When set, `chosen_key` is the
   matching answer key instead of the scoring string. Omitting it keeps the
   legacy behaviour byte-for-byte.
2. `serving/systemone_server.py` — `/v1/systemone` accepts optional
   `answer_keys` per question; a length mismatch is a canonical 400-class
   four-field error, never an inference 500.
3. `serving/systemone_semantics.py` (new, first-party helper) — canonical family
   instructions, canonical generic options, and a state-specific Noul renderer
   for `safe_to_continue` and `human_review`. It never inspects the state, never
   auto-detects a family from instruction text, and never rewrites a caller's
   generic options. It refuses `critical_condition` state-specific rendering
   with an explicit trivialization reason.
4. Tests and a reproducible validation script lock the rendering strings, the
   false-first / `prob_true`-index-1 mapping, key stability, HTTP behaviour, and
   real-model parity.

### How consumers use it

```python
from systemone_semantics import render_state_specific_noul
from systemone_inference import SystemOneRequest

q = render_state_specific_noul(
    "safe_to_continue",
    false_condition="checks passed but the operator has not signed off",
    true_condition="checks passed and the operator signed off",
)
response = adapter(SystemOneRequest(state=state, questions=[q]))
# chosen_key is "false" or "true", never the rewritten scoring text
```

Over HTTP, send the same `instruction` / `options` plus
`"answer_keys": ["false", "true"]`. Existing callers that omit `answer_keys`
are unaffected.

## 3. Evidence

* **Archive parity:** `evidence/semantic_rendering_validation.json` — 254 scored
  rows through the new adapter/helper, max `prob_true` difference **0.0**, zero
  decision mismatches, 254/254 stable keys.
* **Prospective panel (independently authored, deterministic truth conditions):**
  15 renderable cases, generic 12/15 → rendered 13/15. By family:
  `safe_to_continue` 5/8 → 7/8; `human_review` 7/7 → 6/7. The small panel
  supports the safe-to-continue rendering and is **inconclusive to negative**
  for human_review; it does not override the larger locked panel but is reported
  in full rather than hidden. `critical_condition` 2/2 guard refusals.
* **Live HTTP smoke:** `evidence/semantic_http_smoke.json` — ready on `cuda:0`,
  legacy generic request 200 with option-text key, rendered request 200 with
  stable key `false` (false approval removed on the smoke state), malformed key
  mapping 422 with the canonical error block.
* **Regression:** full CPU suite 41 passed, 2 explicitly skipped (private Laya
  harness not shipped).

## 4. Limitations (do not promote past these)

* Confidence remains **uncalibrated**. ECE ≤ 0.05 is not met; the renderer
  improves separation/ranking, not probability calibration. No calibrated
  confidence claim is made.
* Evidence is synthetic/semi-synthetic and small. There is **no established
  real-world generalization** and no second seed. The prospective panel is
  author-constructed, with deterministic checkable truth conditions, not an
  independent production panel.
* `critical_condition` is unresolved; generic over-rejection persists.
* The helper is opt-in. It does not magically derive decisive conditions from a
  state: the caller must supply truthful, concrete condition phrases. Its
  value is a validated rendering contract plus stable answer identity.

## 5. Rollback

No model artifact changed. To return to the pre-change serving behaviour:

1. Deploy the previous `systemone_inference.py` and `systemone_server.py`
   (or simply omit `answer_keys` in requests — the default path is unchanged).
2. Remove `systemone_semantics.py` from the serving directory.
3. Restore the previous release commit/tag; weights and tokenizer remain
   byte-identical to `bd30d80e…`.

## 6. Gate record

| Gate | Status |
| --- | --- |
| G1 Identity | PASS — remote refs, weight hash, archive hash verified |
| G2 Reproduction | PASS — exact 0.0-difference parity, 304 rows |
| G3 Causality | PASS — state-blind collapse + counterfactual flips; shortcut rejected as whole story |
| G4 Interface | PASS — first-party helper + live HTTP smoke with stable key |
| G5 Regression | PASS — full test suite, unchanged generic semantics |
| G6 Integrity | PASS — 313,161,733 params, no lm_head, weights unchanged |
| G7 Qualification | PASS — synthetic/prospective/production-path evidence distinguished |
| G8 Publication | see release report |
| G9 Documentation | see release report |
| G10 Persistence | evidence committed to source repo and HF evidence repo |
