"""Human-in-the-loop actions on processed invoices.

PASS runs hold a RESERVED amount; an AP approver either approves it (-> APPROVED)
or releases it (-> RELEASED, balance freed).

REVIEW runs hold nothing; a reviewer can approve them (choosing the PO when the invoice
did not print a valid one, or typing a missing invoice number) or reject them.

Human review is a control, not a way around controls. A reviewer may accept a judgement
call - a price variance, a missing PO reference, an old invoice date, embedded text - with a
written justification. A reviewer can NEVER approve:

  * an amount that is zero or negative, or a total that was not verified in the PDF
  * invoice arithmetic that does not reconcile
  * an invoice without a verified INR currency
  * an unconfirmed or unapproved vendor (fix the vendor master, then re-process)
  * any value the system could not verify in the PDF (number, date, supplier, amounts, item rows)
  * a different invoice number than the one printed (a number may only be entered if none is printed)
  * a different PO than the one printed - even when the printed PO is unknown (the vendor must correct it);
    a reviewer may only choose a PO when the invoice prints none
  * the same file, or a same-vendor/same-amount/same-date invoice, that is already active
  * a PO belonging to another vendor, or whose items do not match the invoice lines
  * more quantity than the PO line has left, or more money than the PO has left
  * a duplicate of an invoice that is already active

Every approval re-checks all of this inside ONE database transaction together with the
reservation and the audit record, so the outcome is all-or-nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src import config
from src import database as db
from src.evidence_validator import ParsedLine
from src.matcher import match_lines
from src.money import format_inr
from src.rules import RULES_BY_CODE

@dataclass
class ActionResult:
    ok: bool
    message: str


class _Refused(Exception):
    pass


def _parsed_lines(verification: dict) -> list[ParsedLine]:
    out = []
    for l in verification.get("lines", []):
        out.append(ParsedLine(
            index=l["index"], item_code=l["item_code"], description=l["description"],
            quantity=Decimal(l["quantity"]) if l["quantity"] is not None else None,
            unit_price_paise=l["unit_price_paise"], line_total_paise=l["line_total_paise"],
            raw=l["raw"], verified=l["verified"], issues=l["issues"],
            description_verified=l.get("description_verified", False),
        ))
    return out


def _checks_by_code(conn, run_id: str) -> dict[str, dict]:
    return {c["rule_code"]: c for c in db.get_run_checks(conn, run_id)}


def printed_po(run: dict, checks: dict[str, dict]) -> str | None:
    """The PO the invoice verifiably printed and that exists - a reviewer may not swap it."""
    po_check = checks.get("PO_EXISTS")
    if po_check and po_check["outcome"] == "PASS":
        return run["po_id"]
    return None


FIELD_LABELS = {
    "invoice_number": "invoice number", "invoice_date": "invoice date", "vendor_name": "supplier name",
    "vendor_tax_id": "supplier GSTIN", "po_number": "PO number", "currency": "currency", "subtotal": "subtotal",
    "tax_amount": "tax", "shipping_amount": "shipping", "discount_amount": "discount", "total_amount": "total",
    "buyer_name": "bill-to name", "buyer_tax_id": "bill-to GSTIN",
}


def approval_blockers(conn, run: dict, po_id: str | None, invoice_number: str | None) -> list[str]:
    """Every reason this REVIEW item cannot be approved as requested (empty list = allowed)."""
    blockers: list[str] = []
    checks = _checks_by_code(conn, run["run_id"])
    verification = json.loads(run["verification_json"] or "{}")
    fields = verification.get("field_checks", {})

    if run["decision"] != "REVIEW" or run["review_status"] != "OPEN":
        return ["This item is no longer waiting for review."]

    # 1. Nothing the system could not verify in the PDF can be approved by a note.
    for name, fc in fields.items():
        if fc.get("status") in ("UNVERIFIED", "INVALID"):
            blockers.append(f"The {FIELD_LABELS.get(name, name)} could not be verified in the PDF "
                            f"({fc.get('reason')}). Ask the vendor for a clear, corrected invoice.")
    if (fields.get("invoice_date") or {}).get("status") != "VERIFIED":
        blockers.append("The invoice date is missing or unverified.")

    # 2. Money must be real, verified and consistent.
    amount = run["amount_paise"]
    if amount is None or amount <= 0:
        blockers.append("The invoice total is missing, zero or negative. Ask the vendor for a corrected invoice.")
    if (fields.get("total_amount") or {}).get("status") != "VERIFIED":
        blockers.append("The invoice total was not verified in the PDF.")
    if checks.get("ARITHMETIC", {}).get("outcome") != "PASS":
        blockers.append("The invoice arithmetic does not reconcile. A corrected invoice is required.")
    if run["currency"] != "INR" or (fields.get("currency") or {}).get("status") != "VERIFIED":
        blockers.append("The currency is not a verified INR amount.")

    # 3. Identity is established by the system, not chosen by the reviewer.
    if checks.get("VENDOR_IDENTITY", {}).get("outcome") != "PASS" or not run["vendor_id"]:
        blockers.append("The vendor could not be confirmed. Add or fix it in the vendor master, then re-process.")
    for code in ("VENDOR_APPROVED", "PO_VENDOR_MATCH", "DUPLICATE_INVOICE", "CURRENCY", "ARITHMETIC", "BILLED_TO_US"):
        c = checks.get(code)
        if c and c["outcome"] == "BLOCK":
            blockers.append(f"{c['rule_name']}: {c['reason']}")
    billed = checks.get("BILLED_TO_US")
    if billed and billed["outcome"] != "PASS" and (billed.get("details") or {}).get("other_company"):
        blockers.append(f"{billed['rule_name']}: {billed['reason']}")

    printed_number = (run["invoice_number"] or "").strip()
    typed = (invoice_number or "").strip()
    if printed_number and typed and db.normalize_invoice_number(typed) != db.normalize_invoice_number(printed_number):
        blockers.append(f"The invoice prints number '{printed_number}'; it cannot be replaced.")
    if not (printed_number or typed):
        blockers.append("Enter the invoice number (it was missing on the document).")

    # 4. Every billed line must be fully verified and positive.
    lines = _parsed_lines(verification)
    if not lines:
        blockers.append("No line items were read from the invoice.")
    for l in lines:
        if l.line_total_paise is None or l.quantity is None or l.unit_price_paise is None:
            blockers.append(f"Line {l.index} has missing amounts.")
        elif l.quantity <= 0 or l.line_total_paise <= 0:
            blockers.append(f"Line {l.index} has a zero or negative amount.")
        if l.issues:
            blockers.append(f"Line {l.index} could not be verified: " + "; ".join(l.issues))

    # 5. PO: the printed one, or - only if none is printed - one of the vendor's open POs whose items match.
    fixed = printed_po(run, checks)
    po_field = fields.get("po_number") or {}
    if not fixed and po_field.get("status") == "VERIFIED":
        blockers.append(f"The invoice prints PO {po_field.get('raw_value')}, which is not a valid open PO for this "
                        "vendor. The vendor must correct the reference; it cannot be swapped for another PO.")
    if fixed and po_id and po_id != fixed:
        blockers.append(f"The invoice prints {fixed}; it cannot be allocated to a different PO.")
    chosen = po_id or fixed
    po = db.get_po(conn, chosen) if chosen else None
    if not chosen:
        blockers.append("Choose the PO to allocate this invoice to.")
    elif po is None:
        blockers.append(f"{chosen} does not exist.")
    else:
        if po["vendor_id"] != run["vendor_id"]:
            blockers.append(f"{chosen} belongs to another vendor.")
        if po["status"] not in config.OPEN_PO_STATUSES:
            blockers.append(f"{chosen} is {po['status']}, not open for invoicing.")
        if po["currency"] != "INR":
            blockers.append(f"{chosen} is not an INR purchase order.")
        unmatched = [m.invoice_line for m in match_lines(lines, db.get_po_lines(conn, chosen)) if m.po_line_no is None]
        if unmatched:
            blockers.append(f"Invoice line(s) {', '.join(map(str, unmatched))} do not match any item on {chosen}.")
        if amount and amount > 0:
            remaining = po["gross_total_paise"] - db.po_committed_paise(conn, chosen)
            if amount > remaining + config.BALANCE_ROUNDING_TOLERANCE_PAISE:
                blockers.append(f"{format_inr(amount)} exceeds the {format_inr(remaining)} remaining on {chosen}. "
                                "The PO must be amended first.")

    # 6. Duplicates: the identical file, or a typed-in number that hides a same-amount/same-date invoice.
    same_file = db.find_active_by_file_hash(conn, run["file_hash"])
    if same_file:
        blockers.append(f"This exact file is already {same_file['allocation_status'].lower()} as "
                        f"'{same_file['invoice_number']}' (ledger #{same_file['ledger_id']}).")
    if run["vendor_id"] and not printed_number:
        similar = db.find_active_similar(conn, run["vendor_id"], amount, verification.get("invoice_date"))
        if similar:
            blockers.append(f"A typed-in number cannot be used: '{similar['invoice_number']}' from this vendor has the "
                            f"same amount and date and is already {similar['allocation_status'].lower()}.")

    vendor = db.get_vendor(conn, run["vendor_id"]) if run["vendor_id"] else None
    if vendor is not None and vendor["approval_status"] != "APPROVED":
        blockers.append(f"{vendor['legal_name']} is not an approved vendor.")
    return list(dict.fromkeys(blockers))


def overridden_rules(conn, run_id: str) -> list[str]:
    checks = _checks_by_code(conn, run_id)
    return [f"{RULES_BY_CODE[c].number} {RULES_BY_CODE[c].name}" for c, chk in checks.items()
            if chk["outcome"] != "PASS" and c in RULES_BY_CODE]


def approve_review(
    engine: Engine,
    run_id: str,
    actor: str,
    note: str | None,
    po_id: str | None = None,
    vendor_id: str | None = None,
    invoice_number: str | None = None,
) -> ActionResult:
    if not note or not note.strip():
        return ActionResult(False, "A justification is required to approve an invoice that needed review.")
    try:
        with engine.begin() as conn:
            run = db.lock_run(conn, run_id)
            if run is None:
                return ActionResult(False, f"Run {run_id} not found.")
            if vendor_id and vendor_id != run["vendor_id"]:
                return ActionResult(False, "The vendor is set by the system from the invoice; a reviewer cannot change it.")
            blockers = approval_blockers(conn, run, po_id, invoice_number)
            if blockers:
                return ActionResult(False, "Cannot approve: " + " ".join(blockers))

            chosen = po_id or printed_po(run, _checks_by_code(conn, run_id))
            verification = json.loads(run["verification_json"] or "{}")
            lines = _parsed_lines(verification)
            alloc: dict[int, dict] = {}
            for m in match_lines(lines, db.get_po_lines(conn, chosen)):
                line = lines[m.invoice_line - 1]
                a = alloc.setdefault(m.po_line_no, {"line_no": m.po_line_no, "quantity_milli": 0, "amount_paise": 0})
                a["quantity_milli"] += int((line.quantity * 1000).to_integral_value())
                a["amount_paise"] += line.line_total_paise
            number = (run["invoice_number"] or invoice_number).strip()
            req = db.ReservationRequest(
                run_id=run_id, vendor_id=run["vendor_id"], invoice_number=number,
                invoice_number_norm=db.normalize_invoice_number(number), po_id=chosen,
                invoice_date=verification.get("invoice_date"), amount_paise=run["amount_paise"],
                file_hash=run["file_hash"], lines=list(alloc.values()), status="APPROVED", source="REVIEWER",
            )
            res = db.reserve_in_transaction(conn, req, enforce_line_quantities=True)  # re-checks everything
            if not res.ok:
                raise _Refused(res.message)
            overridden = overridden_rules(conn, run_id)
            db.update_run(conn, run_id, review_status="APPROVED", po_id=chosen,
                          invoice_number=number, invoice_number_norm=req.invoice_number_norm)
            db.record_review_action(
                conn, run_id, "APPROVE_OVERRIDE", actor,
                f"{note.strip()} | allocated to {chosen} | accepted exceptions: {', '.join(overridden) or 'none'}",
            )
    except _Refused as exc:
        return ActionResult(False, f"Cannot approve: {exc}.")
    except IntegrityError:
        return ActionResult(False, "Cannot approve: this invoice is already active in the ledger.")
    return ActionResult(True, f"Approved by {actor}: {format_inr(req.amount_paise)} allocated to {chosen} "
                              f"(ledger #{res.ledger_id}).")


def approve_reserved(engine: Engine, run_id: str, actor: str, note: str | None) -> ActionResult:
    with engine.begin() as conn:
        run = db.lock_run(conn, run_id)
        if run is None or run["decision"] != "PASS" or run["review_status"] != "OPEN" or not run["ledger_id"]:
            return ActionResult(False, "Only an open PASS with a reservation can be approved here.")
        entry = db.get_ledger_entry(conn, run["ledger_id"])
        if entry["allocation_status"] != "RESERVED":
            return ActionResult(False, f"Ledger entry is {entry['allocation_status']}, not RESERVED.")
        vendor = db.get_vendor(conn, entry["vendor_id"])
        if vendor is None or vendor["approval_status"] != "APPROVED":
            return ActionResult(False, "The vendor is no longer approved; release the reservation instead.")
        po = db.get_po(conn, entry["po_id"])
        if po is None or po["status"] not in config.OPEN_PO_STATUSES:
            return ActionResult(False, f"{entry['po_id']} is no longer open; release the reservation instead.")
        db.set_ledger_status(conn, run["ledger_id"], "APPROVED")
        db.update_run(conn, run_id, review_status="APPROVED")
        db.record_review_action(conn, run_id, "APPROVE", actor, note)
    return ActionResult(True, f"Approved for payment: {format_inr(entry['amount_paise'])} on {entry['po_id']}.")


def release_reservation(engine: Engine, run_id: str, actor: str, note: str | None) -> ActionResult:
    if not note or not note.strip():
        return ActionResult(False, "A reason is required to release a reservation.")
    with engine.begin() as conn:
        run = db.lock_run(conn, run_id)
        if run is None or run["decision"] != "PASS" or run["review_status"] != "OPEN" or not run["ledger_id"]:
            return ActionResult(False, "Only an open PASS with a reservation can be released.")
        entry = db.get_ledger_entry(conn, run["ledger_id"])
        db.set_ledger_status(conn, run["ledger_id"], "RELEASED")
        db.update_run(conn, run_id, review_status="RELEASED")
        db.record_review_action(conn, run_id, "RELEASE", actor, note)
    return ActionResult(True, f"Reservation released; {format_inr(entry['amount_paise'])} returned to {entry['po_id']}.")


def reject_review(engine: Engine, run_id: str, actor: str, note: str | None) -> ActionResult:
    if not note or not note.strip():
        return ActionResult(False, "A reason is required to reject an invoice.")
    with engine.begin() as conn:
        run = db.lock_run(conn, run_id)
        if run is None or run["decision"] != "REVIEW" or run["review_status"] != "OPEN":
            return ActionResult(False, "Only open REVIEW items can be rejected.")
        db.update_run(conn, run_id, review_status="REJECTED")
        db.record_review_action(conn, run_id, "REJECT", actor, note)
    return ActionResult(True, "Invoice rejected. Nothing was reserved; the vendor can resubmit a corrected invoice.")
