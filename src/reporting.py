"""Builds the human-readable explanation from rule results.

No language model writes the explanation: it is assembled from fixed rule
reasons and verified numbers, so it can never say something the checks did
not establish.
"""

from __future__ import annotations

from src.money import format_money

DECISION_MEANING = {
    "PASS": "Ready for AP approval",
    "REVIEW": "Needs human review",
    "BLOCK": "Blocked — hard rule violation",
    "ERROR": "Processing could not be completed",
}


HEADLINE_PHRASES = {
    "REQUIRED_FIELDS": "required details are missing",
    "SOURCE_EVIDENCE": "some values could not be found in the PDF",
    "DATA_FORMAT": "an amount or the invoice date is invalid or out of range",
    "VENDOR_IDENTITY": "the vendor could not be confirmed",
    "VENDOR_APPROVED": "the vendor is not approved",
    "PO_EXISTS": "there is no valid PO to match",
    "PO_VENDOR_MATCH": "the PO belongs to a different vendor",
    "DUPLICATE_INVOICE": "it was already submitted",
    "CURRENCY": "the currency is not supported",
    "ARITHMETIC": "the totals don't add up",
    "LINE_ITEMS": "line items don't match the PO",
    "TAX": "tax doesn't match the PO rates",
    "NON_PO_CHARGES": "it includes charges not on the PO",
    "PO_BALANCE": "it exceeds the remaining PO balance",
    "EMBEDDED_INSTRUCTIONS": "it contains instructions aimed at AI",
    "BILLED_TO_US": "it is not addressed to our company",
}


# What an AP clerk needs to hear first.
HEADLINE_PRIORITY = (
    "BILLED_TO_US", "VENDOR_APPROVED", "PO_VENDOR_MATCH", "DUPLICATE_INVOICE", "VENDOR_IDENTITY", "PO_EXISTS",
    "PO_BALANCE", "LINE_ITEMS", "CURRENCY", "ARITHMETIC", "TAX", "REQUIRED_FIELDS",
    "EMBEDDED_INSTRUCTIONS", "NON_PO_CHARGES", "DATA_FORMAT", "SOURCE_EVIDENCE",
)


def _phrases(checks: list[dict]) -> str:
    order = {code: i for i, code in enumerate(HEADLINE_PRIORITY)}
    checks = sorted(checks, key=lambda c: order.get(c["rule_code"], len(order)))
    items = [HEADLINE_PHRASES.get(c["rule_code"], c["rule_name"].lower()) for c in checks]
    if len(items) <= 2:
        return " and ".join(items)
    return f"{items[0]}, {items[1]} and {len(items) - 2} more"


def build_explanation(
    decision: str,
    checks: list[dict],
    invoice_number: str | None,
    vendor_name: str | None,
    po_id: str | None,
    amount_paise: int | None,
    reservation_note: str | None = None,
    currency: str | None = "INR",
) -> tuple[str, str]:
    """Returns (headline, explanation_markdown)."""
    inv = f"Invoice {invoice_number}" if invoice_number else "Invoice (no number)"
    who = f" from {vendor_name}" if vendor_name else ""
    amt = f" for {format_money(amount_paise, currency)}" if amount_paise is not None else ""
    against = f" against {po_id}" if po_id else ""

    if decision == "PASS":
        headline = f"{inv}{who}{amt}{against} passed all {len(checks)} checks."
        lines = []
        if reservation_note:
            lines.append(reservation_note)
        notes = [c for c in checks if "Note:" in (c.get("reason") or "")]
        for c in notes:
            lines.append(f"- {c['rule_number']} {c['rule_name']}: {c['reason']}")
        lines.append("Next step: AP approver confirms in the Review queue. A reservation is not a payment.")
        return headline, "\n\n".join(lines)

    blocking = [c for c in checks if c["outcome"] == "BLOCK"]
    review = [c for c in checks if c["outcome"] == "REVIEW"]
    skipped = [c for c in checks if c["outcome"] == "NOT_EVALUATED"]
    n_pass = sum(1 for c in checks if c["outcome"] == "PASS")

    if decision == "BLOCK":
        headline = f"{inv}{who}{amt} is blocked because {_phrases(blocking)}."
    else:
        headline = f"{inv}{who}{amt} needs review because {_phrases(review or skipped)}."

    parts = [f"{n_pass} of {len(checks)} checks passed."]
    if blocking:
        parts.append("**Blocking issues**\n" + "\n".join(f"- {c['rule_number']} {c['rule_name']}: {c['reason']}" for c in blocking))
    if review:
        parts.append("**Needs attention**\n" + "\n".join(f"- {c['rule_number']} {c['rule_name']}: {c['reason']}" for c in review))
    if skipped:
        parts.append(f"{len(skipped)} further check{'s' if len(skipped) != 1 else ''} could not run until "
                     "these are resolved (see the Checks tab).")
    parts.append(reservation_note or "No PO balance has been reserved.")
    return headline, "\n\n".join(parts)
