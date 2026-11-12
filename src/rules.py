"""Stage 5: the business rules engine.

Every rule is a plain Python function of (verified invoice, procurement match)
that returns a structured result with an explicit outcome:

    PASS          the condition holds
    REVIEW        a human needs to look (missing data, discrepancy, uncertainty)
    BLOCK         a hard violation (duplicate, unapproved vendor, wrong vendor's PO)
    NOT_EVALUATED a prerequisite is missing, so the rule could not be checked
                  (treated like REVIEW - an unchecked rule can never become a PASS)

No AI is involved. Same inputs + same database state -> same result, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from src import config
from src.evidence_validator import VerifiedInvoice
from src.matcher import MatchResult
from src.money import format_inr, format_money

PASS, REVIEW, BLOCK, NOT_EVALUATED = "PASS", "REVIEW", "BLOCK", "NOT_EVALUATED"


@dataclass(frozen=True)
class RuleDef:
    code: str
    number: str
    name: str
    fail_outcome: str
    description: str


RULES: tuple[RuleDef, ...] = (
    RuleDef("REQUIRED_FIELDS", "R01", "Required fields present", REVIEW,
            "Invoice number, date, vendor, total and at least one line item are present."),
    RuleDef("SOURCE_EVIDENCE", "R02", "Extracted values backed by the PDF", REVIEW,
            "Every extracted value was found in the document text; nothing ambiguous."),
    RuleDef("DATA_FORMAT", "R03", "Amounts and dates are valid", REVIEW,
            "Amounts parse exactly; invoice date is not in the future and not older than 1 year."),
    RuleDef("VENDOR_IDENTITY", "R04", "Vendor identified in vendor master", REVIEW,
            "Vendor confirmed by GSTIN or exact approved name/alias. Fuzzy matches are suggestions only."),
    RuleDef("VENDOR_APPROVED", "R05", "Vendor is approved", BLOCK,
            "A confirmed vendor that is not approved is blocked."),
    RuleDef("PO_EXISTS", "R06", "PO exists and is open", REVIEW,
            "The printed PO exists and is approved. Missing PO -> candidate POs suggested for review."),
    RuleDef("PO_VENDOR_MATCH", "R07", "PO belongs to this vendor", BLOCK,
            "Billing against another vendor's PO is blocked."),
    RuleDef("DUPLICATE_INVOICE", "R08", "Not a duplicate", BLOCK,
            "Same vendor + normalised invoice number is not already reserved, approved, paid or awaiting review."),
    RuleDef("CURRENCY", "R09", "Currency supported and matches PO", REVIEW,
            "MVP supports INR only."),
    RuleDef("ARITHMETIC", "R10", "Invoice arithmetic reconciles", REVIEW,
            "qty x price = line total; lines sum to subtotal; subtotal + tax + shipping - discount = total."),
    RuleDef("LINE_ITEMS", "R11", "Lines match PO items, prices and quantities", REVIEW,
            "Each line maps to a PO item; price within tolerance; cumulative quantity within PO quantity."),
    RuleDef("TAX", "R12", "Tax matches PO tax rates", REVIEW,
            "Invoice tax equals line amounts x PO tax rates (within Rs 1)."),
    RuleDef("NON_PO_CHARGES", "R13", "No charges outside the PO", REVIEW,
            "Shipping or discounts not on the PO need a human to decide the accounting treatment."),
    RuleDef("PO_BALANCE", "R14", "Fits remaining PO balance", REVIEW,
            "Invoice total <= PO value - already reserved/approved/paid invoices."),
    RuleDef("EMBEDDED_INSTRUCTIONS", "R15", "No instructions aimed at automation", REVIEW,
            "Invoices containing text addressed to AI/automated systems are routed to a human."),
    RuleDef("BILLED_TO_US", "R16", "Addressed to our company", BLOCK,
            "The bill-to party is one of our legal entities (GSTIN, or name). Another company's invoice is blocked."),
)

RULES_BY_CODE = {r.code: r for r in RULES}
REQUIRED_RULE_CODES = tuple(r.code for r in RULES)


def _result(code: str, outcome: str, reason: str, expected: str = "", actual: str = "", details=None) -> dict:
    r = RULES_BY_CODE[code]
    return {
        "rule_code": code,
        "rule_number": r.number,
        "rule_name": r.name,
        "outcome": outcome,
        "expected": expected,
        "actual": actual,
        "reason": reason,
        "details": details,
    }


def _pct(n: Decimal) -> str:
    return f"{n.quantize(Decimal('0.01'))}%"


def _qty(milli: int) -> str:
    q = Decimal(milli) / 1000
    return str(q.normalize()) if q != q.to_integral_value() else str(int(q))


# --------------------------------------------------------------------------
# Individual rules
# --------------------------------------------------------------------------


def check_required_fields(inv: VerifiedInvoice, m: MatchResult) -> dict:
    missing = [
        label for label, val in (
            ("invoice number", inv.invoice_number),
            ("invoice date", inv.field_checks["invoice_date"].raw_value),
            ("vendor name", inv.vendor_name),
            ("total amount", inv.field_checks["total_amount"].raw_value),
        ) if not val
    ]
    if not inv.lines:
        missing.append("line items")
    if missing:
        return _result("REQUIRED_FIELDS", REVIEW, "Missing on invoice: " + ", ".join(missing),
                       "All required fields present", "Missing: " + ", ".join(missing))
    return _result("REQUIRED_FIELDS", PASS, "All required fields present",
                   "All required fields present", f"{len(inv.lines)} line item(s) and all header fields")


def check_source_evidence(inv: VerifiedInvoice, m: MatchResult) -> dict:
    problems = []
    for name, fc in inv.field_checks.items():
        if fc.status in ("UNVERIFIED", "INVALID"):
            problems.append(f"{name.replace('_', ' ')}: {fc.reason}")
    for line in inv.lines:
        if not line.verified:
            problems.append(f"line {line.index}: {'; '.join(line.issues)}")
    for f in inv.ambiguous_fields:
        problems.append(f"{f.replace('_', ' ')}: marked ambiguous by extractor")
    verified = sum(1 for fc in inv.field_checks.values() if fc.status == "VERIFIED")
    if problems:
        return _result("SOURCE_EVIDENCE", REVIEW, "Some extracted values could not be verified against the PDF",
                       "Every value found in the document", f"{len(problems)} problem(s)", {"problems": problems})
    return _result("SOURCE_EVIDENCE", PASS, "All extracted values were found in the document",
                   "Every value found in the document",
                   f"{verified} header fields + {len(inv.lines)} line(s) verified")


def check_data_format(inv: VerifiedInvoice, m: MatchResult, today: date) -> dict:
    problems = list(inv.parse_errors)
    for line in inv.lines:
        if line.quantity is not None and line.quantity <= 0:
            problems.append(f"line {line.index}: non-positive quantity")
        if line.line_total_paise is not None and line.line_total_paise < 0:
            problems.append(f"line {line.index}: negative amount (credit notes not supported)")
    if inv.total_paise is not None and inv.total_paise <= 0:
        problems.append("total is zero or negative")
    if inv.invoice_date:
        if inv.invoice_date > today + timedelta(days=config.MAX_FUTURE_DAYS):
            problems.append(f"invoice date {inv.invoice_date.isoformat()} is in the future")
        if inv.invoice_date < today - timedelta(days=config.MAX_INVOICE_AGE_DAYS):
            problems.append(f"invoice date {inv.invoice_date.isoformat()} is older than {config.MAX_INVOICE_AGE_DAYS} days")
    if problems:
        return _result("DATA_FORMAT", REVIEW, "; ".join(problems), "Valid amounts and a plausible date",
                       f"{len(problems)} problem(s)")
    date_txt = inv.invoice_date.strftime("%d %b %Y") if inv.invoice_date else "no date"
    return _result("DATA_FORMAT", PASS, "Amounts and date are valid", "Valid amounts and a plausible date",
                   f"Invoice dated {date_txt}")


def check_vendor_identity(inv: VerifiedInvoice, m: MatchResult) -> dict:
    v = m.vendor
    if v.status == "CONFIRMED":
        return _result("VENDOR_IDENTITY", PASS, v.reason, "Vendor confirmed in vendor master",
                       f"{v.vendor['legal_name']} ({v.vendor['vendor_id']}) via {v.method.replace('_', ' ')}")
    sugg = ", ".join(f"{s['legal_name']} ({s.get('score', '-')})" for s in v.suggestions) if v.suggestions else "none"
    return _result("VENDOR_IDENTITY", REVIEW, v.reason, "Vendor confirmed in vendor master",
                   f"'{inv.vendor_name or '—'}' — {v.status.lower()}; suggestions: {sugg}",
                   {"suggestions": v.suggestions})


def check_vendor_approved(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if m.vendor.status != "CONFIRMED":
        return _result("VENDOR_APPROVED", NOT_EVALUATED, "Vendor identity not confirmed", "Vendor status APPROVED", "—")
    v = m.vendor.vendor
    if v["approval_status"] == "APPROVED":
        return _result("VENDOR_APPROVED", PASS, f"{v['legal_name']} is an approved vendor",
                       "Vendor status APPROVED", v["approval_status"])
    return _result("VENDOR_APPROVED", BLOCK, f"{v['legal_name']} is not an approved vendor ({v['approval_status']})",
                   "Vendor status APPROVED", v["approval_status"])


def check_po_exists(inv: VerifiedInvoice, m: MatchResult) -> dict:
    po = m.po
    cands = [{"po_id": c.po_id, "score": c.score, "remaining": format_inr(c.remaining_paise), "reasons": c.reasons}
             for c in m.candidates]
    cand_txt = ""
    if cands:
        cand_txt = " Best candidate: " + cands[0]["po_id"] + f" (fit score {cands[0]['score']}/100)."
    if po.status == "FOUND":
        if po.po["status"] in config.OPEN_PO_STATUSES:
            return _result("PO_EXISTS", PASS, f"{po.po['po_id']} exists and is {po.po['status']}",
                           "Existing, approved PO", f"{po.po['po_id']} — {po.po['status']}")
        return _result("PO_EXISTS", REVIEW, f"{po.po['po_id']} exists but is {po.po['status']}, not open for invoicing",
                       "Existing, approved PO", f"{po.po['po_id']} — {po.po['status']}")
    if po.status == "NOT_PROVIDED":
        return _result("PO_EXISTS", REVIEW, "No PO reference on the invoice." + cand_txt,
                       "Existing, approved PO", "No PO printed", {"candidates": cands})
    if po.status == "NOT_FOUND":
        return _result("PO_EXISTS", REVIEW, f"PO {inv.po_number} does not exist in procurement records." + cand_txt,
                       "Existing, approved PO", f"{inv.po_number} — not found", {"candidates": cands})
    return _result("PO_EXISTS", REVIEW, po.reason + "." + cand_txt, "Existing, approved PO", inv.po_number or "—",
                   {"candidates": cands})


def check_po_vendor_match(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if m.vendor.status != "CONFIRMED" or m.po.status != "FOUND":
        return _result("PO_VENDOR_MATCH", NOT_EVALUATED, "Needs both a confirmed vendor and an existing PO",
                       "PO owner = invoice vendor", "—")
    vid = m.vendor.vendor["vendor_id"]
    owner = m.po.po["vendor_id"]
    if vid == owner:
        return _result("PO_VENDOR_MATCH", PASS, f"{m.po.po['po_id']} belongs to {vid}",
                       f"PO owner = {vid}", f"PO owner = {owner}")
    return _result("PO_VENDOR_MATCH", BLOCK,
                   f"{m.po.po['po_id']} belongs to vendor {owner}, not to {m.vendor.vendor['legal_name']} ({vid})",
                   f"PO owner = {vid}", f"PO owner = {owner}")


def check_duplicate(inv: VerifiedInvoice, m: MatchResult) -> dict:
    d = m.duplicates
    same_file = d.same_file_runs
    file_note = f" Identical file was uploaded before ({len(same_file)} earlier run(s))." if same_file else ""
    details = {"same_file_runs": same_file} if same_file else None
    if d.active_same_file:
        e = d.active_same_file
        return _result("DUPLICATE_INVOICE", BLOCK,
                       f"This exact file was already {e['allocation_status'].lower()} as '{e['invoice_number']}' "
                       f"({format_inr(e['amount_paise'])} on {e['po_id']}, ledger #{e['ledger_id']}).",
                       "No active invoice with the same identity", f"Same file as ledger #{e['ledger_id']}", details)
    if d.open_same_file:
        return _result("DUPLICATE_INVOICE", BLOCK,
                       f"This exact file is already waiting in the review queue ({d.open_same_file[0]['run_id']}).",
                       "No active invoice with the same identity", "Same file already open", details)
    if not inv.invoice_number_norm:
        return _result("DUPLICATE_INVOICE", NOT_EVALUATED, "No invoice number to check" + file_note,
                       "No active invoice with the same identity", "—", details)
    if d.active_ledger_entry:
        e = d.active_ledger_entry
        return _result("DUPLICATE_INVOICE", BLOCK,
                       f"Invoice '{inv.invoice_number}' is already {e['allocation_status']} as '{e['invoice_number']}' "
                       f"({format_inr(e['amount_paise'])} on {e['po_id']}, ledger #{e['ledger_id']})." + file_note,
                       "No active invoice with the same identity",
                       f"Matches ledger #{e['ledger_id']} ({e['allocation_status']})", details)
    if d.open_review_run:
        r = d.open_review_run
        return _result("DUPLICATE_INVOICE", BLOCK,
                       f"Invoice '{inv.invoice_number}' is already waiting in the review queue ({r['run_id']})." + file_note,
                       "No active invoice with the same identity", f"Open review {r['run_id']}", details)
    if d.same_number_other_vendor:
        e = d.same_number_other_vendor[0]
        return _result("DUPLICATE_INVOICE", REVIEW,
                       f"Vendor unconfirmed and invoice number already exists for vendor {e['vendor_id']}." + file_note,
                       "No active invoice with the same identity", f"Ledger #{e['ledger_id']} ({e['vendor_id']})", details)
    if d.similar_active:
        e = d.similar_active
        return _result("DUPLICATE_INVOICE", REVIEW,
                       f"Possible duplicate: '{e['invoice_number']}' from the same vendor has the same amount "
                       f"({format_inr(e['amount_paise'])}) and date and is already {e['allocation_status'].lower()}.",
                       "No active invoice with the same identity", f"Similar to ledger #{e['ledger_id']}", details)
    if m.vendor.status != "CONFIRMED":
        return _result("DUPLICATE_INVOICE", PASS,
                       "No active invoice with this number for any vendor." + file_note,
                       "No active invoice with the same identity", f"'{inv.invoice_number_norm}' not found", details)
    return _result("DUPLICATE_INVOICE", PASS, "No earlier invoice with this vendor + invoice number." + file_note,
                   "No active invoice with the same identity",
                   f"{m.vendor.vendor['vendor_id']} + {inv.invoice_number_norm} is new", details)


def check_currency(inv: VerifiedInvoice, m: MatchResult) -> dict:
    fc = inv.field_checks["currency"]
    if fc.status != "VERIFIED" or not inv.currency:
        return _result("CURRENCY", REVIEW, "Currency missing or not verifiable", "INR", inv.currency or "missing")
    if inv.currency not in config.SUPPORTED_CURRENCIES:
        return _result("CURRENCY", REVIEW, f"{inv.currency} invoices are outside the MVP (INR only)", "INR", inv.currency)
    if m.po.status == "FOUND" and m.po.po["currency"] != inv.currency:
        return _result("CURRENCY", REVIEW, "Invoice currency differs from PO currency", m.po.po["currency"], inv.currency)
    return _result("CURRENCY", PASS, f"{inv.currency} is supported" + (" and matches the PO" if m.po.status == "FOUND" else ""),
                   "INR", inv.currency)


def check_arithmetic(inv: VerifiedInvoice, m: MatchResult) -> dict:
    tol = config.ARITHMETIC_TOLERANCE_PAISE
    fmt = lambda p: format_money(p, inv.currency)
    if inv.total_paise is None or not inv.lines:
        return _result("ARITHMETIC", NOT_EVALUATED, "Total or line items missing", "Totals reconcile", "—")
    problems, rows = [], []
    line_sum = 0
    for line in inv.lines:
        if line.quantity is None or line.unit_price_paise is None or line.line_total_paise is None:
            problems.append(f"line {line.index}: incomplete amounts")
            continue
        calc = int((line.quantity * line.unit_price_paise).to_integral_value(ROUND_HALF_UP))
        line_sum += line.line_total_paise
        ok = abs(calc - line.line_total_paise) <= tol
        rows.append({"line": line.index, "calc": calc, "printed": line.line_total_paise, "ok": ok})
        if not ok:
            problems.append(f"line {line.index}: {line.quantity} x {fmt(line.unit_price_paise)} = "
                            f"{fmt(calc)}, invoice shows {fmt(line.line_total_paise)}")
    subtotal = inv.subtotal_paise if inv.subtotal_paise is not None else line_sum
    if inv.subtotal_paise is not None and abs(line_sum - inv.subtotal_paise) > tol:
        problems.append(f"lines add up to {fmt(line_sum)}, subtotal shows {fmt(inv.subtotal_paise)}")
    tax = inv.tax_paise or 0
    expected_total = subtotal + tax + inv.shipping_paise - inv.discount_paise
    if abs(expected_total - inv.total_paise) > tol:
        problems.append(f"subtotal {fmt(subtotal)} + tax {fmt(tax)}"
                        + (f" + shipping {fmt(inv.shipping_paise)}" if inv.shipping_paise else "")
                        + (f" - discount {fmt(inv.discount_paise)}" if inv.discount_paise else "")
                        + f" = {fmt(expected_total)}, but total shows {fmt(inv.total_paise)}")
    expected = f"Total = {fmt(expected_total)}"
    if problems:
        return _result("ARITHMETIC", REVIEW, "; ".join(problems), expected, f"Total = {fmt(inv.total_paise)}",
                       {"lines": rows, "difference_paise": inv.total_paise - expected_total})
    return _result("ARITHMETIC", PASS, "Line totals, subtotal, tax and grand total all reconcile",
                   expected, f"Total = {fmt(inv.total_paise)}", {"lines": rows})


def _currency_mismatch(inv: VerifiedInvoice, m: MatchResult) -> bool:
    """Invoice and PO in different currencies: amounts cannot be compared at all."""
    return m.po.status == "FOUND" and bool(inv.currency) and inv.currency != m.po.po["currency"]


def check_line_items(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if _currency_mismatch(inv, m):
        return _result("LINE_ITEMS", NOT_EVALUATED,
                       f"Invoice is in {inv.currency} but the PO is in {m.po.po['currency']}; prices cannot be compared",
                       "Lines match PO", "—")
    if m.po.status != "FOUND":
        return _result("LINE_ITEMS", NOT_EVALUATED, "No PO to compare line items against",
                       "Lines match PO", "—")
    if not inv.lines:
        return _result("LINE_ITEMS", NOT_EVALUATED, "No line items extracted", "Lines match PO", "—")
    problems, notes, comparison = [], [], []
    qty_this_invoice: dict[int, int] = {}
    for lm in m.line_matches:
        line = inv.lines[lm.invoice_line - 1]
        row = {
            "invoice_line": line.index,
            "description": line.description or line.item_code,
            "po_line": lm.po_line_no,
            "po_description": lm.po_line["description"] if lm.po_line else None,
            "match_method": lm.method,
            "invoice_qty": str(line.quantity) if line.quantity is not None else None,
            "invoice_unit_price": line.unit_price_paise,
            "po_unit_price": lm.po_line["unit_price_paise"] if lm.po_line else None,
            "status": "OK",
        }
        if lm.po_line is None:
            hint = f" (closest PO item: '{lm.suggestion['description']}', similarity {lm.suggestion_score})" if lm.suggestion else ""
            problems.append(f"line {line.index} '{line.description}' does not match any item on {m.po.po['po_id']}{hint}")
            row["status"] = "NO MATCH"
            comparison.append(row)
            continue
        pl = lm.po_line
        if line.unit_price_paise is not None and pl["unit_price_paise"] <= 0:
            if line.unit_price_paise > 0:
                problems.append(f"line {line.index}: PO item '{pl['description']}' has no approved price "
                                f"but is billed at {format_inr(line.unit_price_paise)}")
                row["status"] = "PRICE OVER TOLERANCE"
        elif line.unit_price_paise is not None:
            diff = line.unit_price_paise - pl["unit_price_paise"]
            pct = Decimal(diff) * 100 / Decimal(pl["unit_price_paise"])
            row["price_variance_pct"] = str(pct.quantize(Decimal("0.01")))
            if diff > 0:
                within = pct <= Decimal(str(config.PRICE_TOLERANCE_PERCENT)) and diff <= config.PRICE_TOLERANCE_MAX_PAISE_PER_UNIT
                if within:
                    notes.append(f"line {line.index}: price {format_inr(line.unit_price_paise)} is {_pct(pct)} above PO "
                                 f"{format_inr(pl['unit_price_paise'])}, within the {config.PRICE_TOLERANCE_PERCENT}% tolerance")
                    row["status"] = "WITHIN TOLERANCE"
                else:
                    problems.append(f"line {line.index}: price {format_inr(line.unit_price_paise)} is {_pct(pct)} above "
                                    f"approved {format_inr(pl['unit_price_paise'])} (tolerance {config.PRICE_TOLERANCE_PERCENT}%)")
                    row["status"] = "PRICE OVER TOLERANCE"
            elif diff < 0:
                notes.append(f"line {line.index}: billed below PO price ({_pct(pct)})")
        if line.quantity is not None:
            qm = int((line.quantity * 1000).to_integral_value())
            qty_this_invoice[pl["line_no"]] = qty_this_invoice.get(pl["line_no"], 0) + qm
        comparison.append(row)

    for line_no, this_qty in qty_this_invoice.items():
        pl = next(p for p in m.po.lines if p["line_no"] == line_no)
        before = m.po.invoiced_qty_milli.get(line_no, 0)
        available = pl["quantity_milli"] - before
        for row in comparison:
            if row["po_line"] == line_no:
                row["po_qty"] = _qty(pl["quantity_milli"])
                row["previously_invoiced"] = _qty(before)
                row["available_qty"] = _qty(available)
        if this_qty > available:
            problems.append(f"PO line {line_no} '{pl['description']}': invoicing {_qty(this_qty)} but only "
                            f"{_qty(available)} of {_qty(pl['quantity_milli'])} remain ({_qty(before)} already invoiced)")
            for row in comparison:
                if row["po_line"] == line_no and row["status"] == "OK":
                    row["status"] = "QTY EXCEEDS PO"

    details = {"comparison": comparison, "notes": notes}
    matched = sum(1 for lm in m.line_matches if lm.po_line)
    if problems:
        return _result("LINE_ITEMS", REVIEW, "; ".join(problems), "All lines match PO items, prices and quantities",
                       f"{matched}/{len(inv.lines)} lines matched, {len(problems)} issue(s)", details)
    reason = f"All {len(inv.lines)} line(s) match {m.po.po['po_id']} items, prices and available quantities"
    if notes:
        reason += ". Note: " + "; ".join(notes)
    return _result("LINE_ITEMS", PASS, reason, "All lines match PO items, prices and quantities",
                   f"{matched}/{len(inv.lines)} lines matched", details)


def check_tax(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if _currency_mismatch(inv, m):
        return _result("TAX", NOT_EVALUATED, f"Invoice is in {inv.currency}, PO in {m.po.po['currency']}",
                       "Tax = lines x PO tax rates", "—")
    if m.po.status != "FOUND":
        return _result("TAX", NOT_EVALUATED, "No PO to take tax rates from", "Tax = lines x PO tax rates", "—")
    if not m.line_matches or any(lm.po_line is None for lm in m.line_matches):
        return _result("TAX", NOT_EVALUATED, "Tax rates come from matched PO lines; not all lines matched",
                       "Tax = lines x PO tax rates", "—")
    if any(l.line_total_paise is None for l in inv.lines):
        return _result("TAX", NOT_EVALUATED, "Line amounts missing", "Tax = lines x PO tax rates", "—")
    exact = sum(
        Decimal(inv.lines[lm.invoice_line - 1].line_total_paise) * lm.po_line["tax_rate_bp"] / 10000
        for lm in m.line_matches
    )
    expected = int(exact.to_integral_value(ROUND_HALF_UP))
    rates = sorted({lm.po_line["tax_rate_bp"] / 100 for lm in m.line_matches})
    rate_txt = "/".join(f"{r:g}%" for r in rates)
    if inv.tax_paise is None:
        if expected == 0:
            return _result("TAX", PASS, "No tax expected or charged", format_inr(0), format_inr(0))
        return _result("TAX", REVIEW, f"No tax shown on invoice; PO rates ({rate_txt}) imply {format_inr(expected)}",
                       format_inr(expected), "No tax printed")
    diff = inv.tax_paise - expected
    src = " (sum of CGST/SGST/IGST components)" if inv.tax_source == "components" else ""
    if abs(diff) <= config.TAX_TOLERANCE_PAISE:
        return _result("TAX", PASS, f"Tax{src} matches PO GST rate {rate_txt}",
                       f"{format_inr(expected)} at {rate_txt}", format_inr(inv.tax_paise))
    return _result("TAX", REVIEW, f"Tax{src} {format_inr(inv.tax_paise)} differs from {format_inr(expected)} "
                   f"expected at PO rate {rate_txt} (difference {format_inr(diff)})",
                   f"{format_inr(expected)} at {rate_txt}", format_inr(inv.tax_paise))


def check_non_po_charges(inv: VerifiedInvoice, m: MatchResult) -> dict:
    extra = []
    if inv.shipping_paise:
        extra.append(f"shipping/freight {format_money(inv.shipping_paise, inv.currency)}")
    if inv.discount_paise:
        extra.append(f"discount {format_money(inv.discount_paise, inv.currency)}")
    if extra:
        return _result("NON_PO_CHARGES", REVIEW, "Charges not on the PO: " + ", ".join(extra),
                       "Only PO items billed", ", ".join(extra))
    return _result("NON_PO_CHARGES", PASS, "No shipping, discounts or other non-PO charges",
                   "Only PO items billed", "None")


def check_po_balance(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if _currency_mismatch(inv, m):
        return _result("PO_BALANCE", NOT_EVALUATED,
                       f"Invoice is in {inv.currency} but the PO is in {m.po.po['currency']}; balances cannot be compared",
                       "Invoice <= remaining PO balance", "—")
    if m.po.status != "FOUND" or inv.total_paise is None:
        return _result("PO_BALANCE", NOT_EVALUATED, "Needs an existing PO and an invoice total",
                       "Invoice <= remaining PO balance", "—")
    po = m.po
    remaining = po.remaining_paise
    exp = (f"<= {format_inr(remaining)} remaining "
           f"({format_inr(po.po['gross_total_paise'])} PO - {format_inr(po.committed_paise)} committed)")
    details = {"po_total": po.po["gross_total_paise"], "committed": po.committed_paise,
               "remaining": remaining, "invoice": inv.total_paise}
    if inv.total_paise <= remaining:
        return _result("PO_BALANCE", PASS,
                       f"{format_inr(inv.total_paise)} fits within the remaining {format_inr(remaining)} on {po.po['po_id']}",
                       exp, format_inr(inv.total_paise), details)
    if inv.total_paise - remaining <= config.BALANCE_ROUNDING_TOLERANCE_PAISE:
        return _result("PO_BALANCE", PASS,
                       f"{format_inr(inv.total_paise)} is {format_inr(inv.total_paise - remaining)} above the remaining "
                       f"{format_inr(remaining)} on {po.po['po_id']}: within the "
                       f"{format_inr(config.BALANCE_ROUNDING_TOLERANCE_PAISE)} rounding tolerance. Note: rounding",
                       exp, format_inr(inv.total_paise), details)
    over = inv.total_paise - remaining
    return _result("PO_BALANCE", REVIEW,
                   f"Invoice {format_inr(inv.total_paise)} exceeds the remaining {format_inr(remaining)} on "
                   f"{po.po['po_id']} by {format_inr(over)}",
                   exp, format_inr(inv.total_paise), details)


def check_embedded_instructions(inv: VerifiedInvoice, m: MatchResult) -> dict:
    if inv.suspicious_instructions:
        return _result("EMBEDDED_INSTRUCTIONS", REVIEW,
                       "Invoice contains text addressed to AI/automated systems. It was ignored, "
                       "but the invoice is routed to a human as a possible manipulation attempt.",
                       "No instructions to automation", "Instructions detected")
    return _result("EMBEDDED_INSTRUCTIONS", PASS, "No embedded instructions detected",
                   "No instructions to automation", "None")


def check_billed_to_us(inv: VerifiedInvoice, m: MatchResult) -> dict:
    import difflib

    from src.database import normalize_name, our_company_entities

    ours = our_company_entities()
    our_gstins = {"".join(c for c in e["gstin"].upper() if c.isalnum()) for e in ours}
    our_names = [normalize_name(e["legal_name"]) or "" for e in ours]
    exp = "Bill-to is " + (ours[0]["legal_name"] if ours else "our company")
    gst = inv.buyer_tax_id if inv.is_verified("buyer_tax_id") else None
    name = inv.buyer_name if inv.is_verified("buyer_name") else None
    name_is_ours = bool(name) and max(
        (difflib.SequenceMatcher(None, normalize_name(name) or "", n).ratio() for n in our_names), default=0) >= 0.8
    if gst:
        if "".join(c for c in gst.upper() if c.isalnum()) in our_gstins:
            return _result("BILLED_TO_US", PASS, f"Billed to our GSTIN {gst}", exp, f"{name or '—'} · {gst}")
        if name_is_ours:
            return _result("BILLED_TO_US", REVIEW,
                           f"Bill-to name is ours but GSTIN {gst} is not one of our registrations",
                           exp, f"{name} · {gst}")
        return _result("BILLED_TO_US", BLOCK, f"The invoice is addressed to {name or 'another company'} "
                       f"(GSTIN {gst}), not to us", exp, f"{name or '—'} · {gst}", {"other_company": True})
    if name:
        if name_is_ours:
            return _result("BILLED_TO_US", PASS, f"Billed to {name}", exp, name)
        return _result("BILLED_TO_US", BLOCK, f"The invoice is addressed to {name}, not to us",
                       exp, name, {"other_company": True})
    return _result("BILLED_TO_US", REVIEW, "No bill-to party is printed, so it can't be confirmed the invoice is ours",
                   exp, "Not printed")


def validate_invoice(inv: VerifiedInvoice, m: MatchResult, today: date | None = None) -> list[dict]:
    today = today or date.today()
    return [
        check_required_fields(inv, m),
        check_source_evidence(inv, m),
        check_data_format(inv, m, today),
        check_vendor_identity(inv, m),
        check_vendor_approved(inv, m),
        check_po_exists(inv, m),
        check_po_vendor_match(inv, m),
        check_duplicate(inv, m),
        check_currency(inv, m),
        check_arithmetic(inv, m),
        check_line_items(inv, m),
        check_tax(inv, m),
        check_non_po_charges(inv, m),
        check_po_balance(inv, m),
        check_embedded_instructions(inv, m),
        check_billed_to_us(inv, m),
    ]


# Kept as a tiny standalone example of a deterministic rule (used in tests/docs).
def check_po_balance_simple(invoice_amount_paise: int, remaining_balance_paise: int) -> dict:
    passed = invoice_amount_paise <= remaining_balance_paise
    return {
        "rule_code": "PO_BALANCE",
        "passed": passed,
        "reason": "Within remaining PO balance" if passed else "Invoice exceeds remaining PO balance",
    }
