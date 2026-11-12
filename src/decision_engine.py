"""Stage 6: turn rule results into one decision.

    BLOCK   any rule returned BLOCK (hard violation)
    REVIEW  any rule returned REVIEW or could not be evaluated
    PASS    every required rule ran and passed
    ERROR   is never produced here - it is set by the pipeline when a
            technical step (PDF, AI, database) fails before rules can run.
"""

from __future__ import annotations

from src.rules import BLOCK, NOT_EVALUATED, PASS, REQUIRED_RULE_CODES, REVIEW


class IncompleteValidationError(Exception):
    """A mandatory rule did not run. Never allowed to become a PASS."""


def make_decision(checks: list[dict]) -> str:
    ran = [c["rule_code"] for c in checks]
    missing = [code for code in REQUIRED_RULE_CODES if code not in ran]
    if missing:
        raise IncompleteValidationError(f"Mandatory rules did not run: {', '.join(missing)}")
    if len(ran) != len(set(ran)):
        raise IncompleteValidationError("A rule ran more than once")

    outcomes = {c["outcome"] for c in checks}
    unknown = outcomes - {PASS, REVIEW, BLOCK, NOT_EVALUATED}
    if unknown:
        raise IncompleteValidationError(f"Unknown rule outcome(s): {unknown}")
    if BLOCK in outcomes:
        return "BLOCK"
    if REVIEW in outcomes or NOT_EVALUATED in outcomes:
        return "REVIEW"
    return "PASS"
