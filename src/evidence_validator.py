"""Stage 3: independently verify what the AI extracted.

The model proposes values plus a verbatim quote for each. This module, with
no AI involved, checks that:
  * the quote really exists in the PDF text (whitespace-insensitive),
  * the value really appears inside that quote as a whole token,
  * the value parses as a real amount / date / quantity.
Anything that fails is marked UNVERIFIED and can never lead to a PASS.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal

from src import config
from src.database import normalize_invoice_number
from src.money import AmountError, parse_amount, parse_quantity, to_paise
from src.pdf_reader import DocumentText
from src.schemas import InvoiceExtraction

AMOUNT_FIELDS = ("subtotal", "tax_amount", "shipping_amount", "discount_amount", "total_amount")
# Identifiers are distinctive enough to be located directly if the model's quote is off.
LOCATABLE_FIELDS = ("invoice_number", "vendor_tax_id", "po_number", "vendor_name", "buyer_name", "buyer_tax_id")

DATE_FORMATS = (
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b, %Y", "%d %B, %Y", "%d-%b-%y", "%d/%m/%y",
)

CURRENCY_ALIASES = {
    "INR": "INR", "₹": "INR", "RS": "INR", "RS.": "INR", "RUPEES": "INR", "INDIAN RUPEE": "INR",
    "INDIAN RUPEES": "INR", "USD": "USD", "$": "USD", "US$": "USD", "EUR": "EUR", "€": "EUR",
    "GBP": "GBP", "£": "GBP",
}

# Printed markers that count as evidence for a currency (word-bounded, so "hours." is not "Rs.").
CURRENCY_PATTERNS = {
    "INR": re.compile(r"₹|\bINR\b|\bRs\.?(?=\s?\d)|\bRupees?\b|\bIndian Rupees?\b", re.IGNORECASE),
    "USD": re.compile(r"\bUSD\b|US\$|\$\s?\d", re.IGNORECASE),
    "EUR": re.compile(r"€|\bEUR\b", re.IGNORECASE),
    "GBP": re.compile(r"£|\bGBP\b", re.IGNORECASE),
}


# Explicit labels that introduce the seller's invoice number. A value must follow the
# label on the same line and contain a digit; "Invoice Date" etc. never match.
INVOICE_NUMBER_LABEL = re.compile(
    r"(?i)\b(invoice\s*(?:no\.?|number|num\.?|#)|inv\.?\s*(?:no\.?|#)|bill\s*(?:no\.?|number|reference|ref\.?)"
    r"|tax\s+invoice\s+(?:no\.?|number)|document\s*(?:no\.?|number))\s*[:#.\-]?\s*"
    r"(?P<value>[A-Z0-9][A-Z0-9/\-_.]{1,40})"
)


def labelled_invoice_numbers(text: str) -> dict[str, str]:
    """Distinct invoice numbers printed next to an explicit label -> the line they appear on."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        for m in INVOICE_NUMBER_LABEL.finditer(line):
            value = m.group("value").rstrip(".-/")
            if any(ch.isdigit() for ch in value):
                found.setdefault(value, line.strip())
    return found


def currencies_in(text: str) -> dict[str, str]:
    """Currencies whose markers appear in the text -> the first line showing each."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        for code, pattern in CURRENCY_PATTERNS.items():
            if code not in found and pattern.search(line):
                found[code] = line.strip()
    return found


@dataclass
class FieldCheck:
    field: str
    raw_value: str | None
    status: str  # VERIFIED / UNVERIFIED / MISSING / INVALID
    source_text: str | None = None
    page: int | None = None
    reason: str = ""


@dataclass
class ParsedLine:
    index: int
    item_code: str | None
    description: str | None
    quantity: Decimal | None
    unit_price_paise: int | None
    line_total_paise: int | None
    raw: dict
    verified: bool
    issues: list[str] = field(default_factory=list)
    # True only if the description itself is printed on the invoice. Description-based PO matching
    # requires this, so an invented description can never select a PO item.
    description_verified: bool = False


@dataclass
class VerifiedInvoice:
    field_checks: dict[str, FieldCheck]
    invoice_number: str | None
    invoice_number_norm: str | None
    invoice_date: date | None
    vendor_name: str | None
    vendor_tax_id: str | None
    po_number: str | None
    currency: str | None
    buyer_name: str | None
    buyer_tax_id: str | None
    subtotal_paise: int | None
    tax_paise: int | None
    tax_source: str | None  # "single" / "components" / None
    shipping_paise: int
    discount_paise: int
    total_paise: int | None
    lines: list[ParsedLine]
    ambiguous_fields: list[str]
    suspicious_instructions: bool
    parse_errors: list[str]

    def is_verified(self, name: str) -> bool:
        fc = self.field_checks.get(name)
        return fc is not None and fc.status == "VERIFIED"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["invoice_date"] = self.invoice_date.isoformat() if self.invoice_date else None
        for line in d["lines"]:
            line["quantity"] = str(line["quantity"]) if line["quantity"] is not None else None
        return d


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.replace(" ", " ").split()).casefold()


_NUMBER_TOKEN = re.compile(r"\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _numbers_in(text: str) -> set[Decimal]:
    out = set()
    for tok in _NUMBER_TOKEN.findall(text):
        try:
            out.add(Decimal(tok.replace(",", "")))
        except Exception:  # pragma: no cover
            pass
    return out


def _token_in(value: str, haystack: str, numeric: bool) -> bool:
    """Whole-token containment.

    Numeric: the parsed number must equal a complete number printed in the
    text ('1,000.00' never matches inside '11,000.00'; '4500.00' matches '4,500.00').
    Text: case/whitespace-insensitive match on word boundaries.
    """
    if numeric:
        try:
            v = parse_amount(value)
        except AmountError:
            try:
                v = parse_quantity(value)
            except AmountError:
                return False
        return v is not None and v in _numbers_in(haystack)
    v = re.escape(_norm(value))
    return re.search(rf"(?<![0-9a-z]){v}(?![0-9a-z])", _norm(haystack)) is not None


def _find_quote(quote: str, doc: DocumentText) -> int | None:
    q = _norm(quote)
    if not q:
        return None
    for page in doc.pages:
        if q in _norm(page.text):
            return page.page_number
    return None


def _locate_value(value: str, doc: DocumentText, numeric: bool) -> tuple[int, str] | None:
    for page in doc.pages:
        for line in page.text.splitlines():
            if _token_in(value, line, numeric):
                return page.page_number, line.strip()
    return None


DESCRIPTION_MATCH_THRESHOLD = 0.8


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.casefold()) if len(w) >= 2 or w.isdigit()]


def description_in_document(description: str, doc: DocumentText) -> bool:
    """At least 80% of the description's words appear together within 3 consecutive lines.

    Tolerates wrapped descriptions and spacing changes, but an invented product name
    (words that are not printed near each other on the invoice) fails.
    """
    want = _words(description)
    if not want:
        return False
    for page in doc.pages:
        lines = page.text.splitlines()
        for i in range(len(lines)):
            have = set(_words(" ".join(lines[i:i + 3])))
            if sum(w in have for w in want) / len(want) >= DESCRIPTION_MATCH_THRESHOLD:
                return True
    return False


ROW_WINDOW_LINES = 3


@dataclass
class RowEvidence:
    page: int
    method: str        # how that page was read: "text-layer", "ocr" or "text+ocr"
    code_in_row: bool
    desc_in_row: bool
    text: str


def find_item_row(numbers: list[Decimal], item_code: str | None, description: str | None,
                  doc: DocumentText) -> RowEvidence | None:
    """Find the invoice row this line came from.

    The quantity, unit price and line total must all be printed within ROW_WINDOW_LINES consecutive
    lines of one page, together with the item's code or (most of) its description. This stops an
    extraction from assembling a line out of unrelated numbers printed elsewhere on the invoice.
    """
    want = _words(description or "")
    for page in doc.pages:
        lines = page.text.splitlines()
        for i in range(len(lines)):
            window = "\n".join(lines[i:i + ROW_WINDOW_LINES])
            have = _numbers_in(window)
            if any(n not in have for n in numbers):
                continue
            code_in = bool(item_code) and _token_in(item_code, window, numeric=False)
            words = set(_words(window))
            desc_in = bool(want) and sum(w in words for w in want) / len(want) >= DESCRIPTION_MATCH_THRESHOLD
            if code_in or desc_in:
                return RowEvidence(page.page_number, page.method, code_in, desc_in, window.strip())
    return None


def parse_invoice_date(raw: str) -> date:
    text = " ".join(raw.replace(",", ", ").split()).replace(" ,", ",")
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {raw!r}")


def normalize_currency(raw: str | None) -> str | None:
    """'INR', '₹', 'Rs.', 'Indian Rupees (INR)' -> 'INR'. Unknown text is returned uppercased."""
    if raw is None:
        return None
    key = " ".join(raw.strip().upper().split())
    if key in CURRENCY_ALIASES:
        return CURRENCY_ALIASES[key]
    hits = {code for code, pattern in CURRENCY_PATTERNS.items() if pattern.search(raw) or code in key}
    if "RUPEE" in key:
        hits.add("INR")
    return hits.pop() if len(hits) == 1 else key


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def verify_extraction(ex: InvoiceExtraction, doc: DocumentText) -> VerifiedInvoice:
    # If the AI returned no invoice number, use one printed next to an explicit label
    # ("Invoice No:", "Bill Reference:"...) - only when exactly one such value exists.
    number_from_label = False
    if not (ex.invoice_number or "").strip() and "invoice_number" not in ex.ambiguous_fields:
        candidates = labelled_invoice_numbers(doc.text)
        if len(candidates) == 1:
            value, line = next(iter(candidates.items()))
            from src.schemas import FieldEvidence

            ex = ex.model_copy(update={
                "invoice_number": value,
                "evidence": [e for e in ex.evidence if e.field != "invoice_number"]
                + [FieldEvidence(field="invoice_number", source_text=line)],
            })
            number_from_label = True

    quotes: dict[str, str] = {}
    for ev in ex.evidence:
        quotes.setdefault(ev.field, ev.source_text)

    checks: dict[str, FieldCheck] = {}
    parse_errors: list[str] = []

    for name in config.EVIDENCE_FIELDS:
        raw = getattr(ex, name)
        numeric = name in AMOUNT_FIELDS
        if raw is None or str(raw).strip() == "":
            status = "MISSING"
            reason = "Listed as ambiguous by extractor" if name in ex.ambiguous_fields else "Not present on invoice"
            checks[name] = FieldCheck(name, None, status, reason=reason)
            continue
        raw = str(raw).strip()
        quote = quotes.get(name)
        page = _find_quote(quote, doc) if quote else None
        if quote and page and _token_in(raw, quote, numeric):
            checks[name] = FieldCheck(name, raw, "VERIFIED", quote.strip(), page, "Quote found in document and contains value")
        elif name in LOCATABLE_FIELDS and (hit := _locate_value(raw, doc, numeric)):
            checks[name] = FieldCheck(name, raw, "VERIFIED", hit[1], hit[0], "Value located in document by verifier")
        else:
            if not quote:
                reason = "Extractor gave no source quote"
            elif page is None:
                reason = "Quoted text does not exist in the document"
            else:
                reason = "Value does not appear in the quoted text"
            checks[name] = FieldCheck(name, raw, "UNVERIFIED", quote, page, reason)

    if number_from_label and checks["invoice_number"].status == "VERIFIED":
        checks["invoice_number"].reason = "Not returned by the AI; read by code from the labelled line on the invoice"

    # Currency: verify against markers printed on the invoice. If the extractor gave none,
    # determine it deterministically from those markers (only when exactly one currency appears).
    cur_raw = ex.currency.strip() if ex.currency else None
    currency = normalize_currency(cur_raw)
    # Placeholder text ("null", "N/A", "not specified"...) is not a currency: treat as not given.
    if cur_raw is not None and not re.fullmatch(r"[A-Z]{3}", currency or ""):
        cur_raw, currency = None, None
    printed = currencies_in(doc.text)
    if cur_raw is None:
        if len(printed) == 1:
            currency, line = next(iter(printed.items()))
            checks["currency"] = FieldCheck(
                "currency", currency, "VERIFIED", line,
                reason="Not stated by the extractor; determined by code from the only currency marker on the invoice")
        elif len(printed) > 1:
            checks["currency"] = FieldCheck("currency", None, "MISSING",
                                            reason=f"Several currencies printed ({', '.join(sorted(printed))})")
        else:
            checks["currency"] = FieldCheck("currency", None, "MISSING", reason="No currency printed on invoice")
    elif currency in printed or _norm(cur_raw) in _norm(doc.text):
        checks["currency"] = FieldCheck("currency", cur_raw, "VERIFIED", printed.get(currency),
                                        reason="Currency marker found in document")
    else:
        checks["currency"] = FieldCheck("currency", cur_raw, "UNVERIFIED", reason="Currency marker not found in document")

    # Parse amounts (only meaningful when verified; parse anyway to report format errors).
    amounts: dict[str, int | None] = {}
    for name in AMOUNT_FIELDS:
        raw = getattr(ex, name)
        try:
            amounts[name] = to_paise(raw) if raw is not None else None
        except AmountError as exc:
            amounts[name] = None
            checks[name].status = "INVALID"
            checks[name].reason = str(exc)
            parse_errors.append(f"{name}: {exc}")

    # Tax: a single printed figure, or the sum of printed components.
    tax_paise = amounts["tax_amount"]
    tax_source = "single" if tax_paise is not None else None
    if tax_paise is None and ex.tax_components:
        total = 0
        ok = True
        quoted = []
        for comp in ex.tax_components:
            try:
                p = to_paise(comp.amount)
            except AmountError as exc:
                parse_errors.append(f"tax component {comp.label}: {exc}")
                ok = False
                break
            if p is None or not _locate_value(comp.amount, doc, numeric=True):
                ok = False
                break
            total += p
            quoted.append(f"{comp.label}: {comp.amount}")
        if ok:
            tax_paise, tax_source = total, "components"
            checks["tax_amount"] = FieldCheck(
                "tax_amount", "; ".join(quoted), "VERIFIED", reason="Sum of tax components found in document"
            )
        else:
            checks["tax_amount"] = FieldCheck(
                "tax_amount", "; ".join(f"{c.label}: {c.amount}" for c in ex.tax_components), "UNVERIFIED",
                reason="One or more tax components could not be found in the document",
            )

    # Invoice date
    inv_date = None
    if ex.invoice_date:
        try:
            inv_date = parse_invoice_date(ex.invoice_date)
        except ValueError as exc:
            checks["invoice_date"].status = "INVALID"
            checks["invoice_date"].reason = str(exc)
            parse_errors.append(f"invoice_date: {exc}")

    # Line items: every number must exist in the document and parse.
    lines: list[ParsedLine] = []
    for i, li in enumerate(ex.line_items, start=1):
        issues: list[str] = []
        qty = unit = total = None
        try:
            qty = parse_quantity(li.quantity)
        except AmountError as exc:
            issues.append(str(exc))
        try:
            unit = to_paise(li.unit_price)
        except AmountError as exc:
            issues.append(f"unit price: {exc}")
        try:
            total = to_paise(li.line_total)
        except AmountError as exc:
            issues.append(f"line total: {exc}")
        for label, raw in (("quantity", li.quantity), ("unit price", li.unit_price), ("line total", li.line_total)):
            if raw is None:
                issues.append(f"{label} missing")
            elif not _locate_value(raw, doc, numeric=True):
                issues.append(f"{label} '{raw}' not found in document")
        code_found = bool(li.item_code) and _locate_value(li.item_code, doc, numeric=False) is not None
        if li.item_code and not code_found:
            issues.append(f"item code '{li.item_code}' not found in document")
        if not li.description and not li.item_code:
            issues.append("no description or item code")
        if qty is not None and qty != qty.quantize(Decimal("0.001")):
            issues.append(f"quantity {li.quantity} has more than 3 decimal places (not supported)")

        row = None
        if qty is not None and unit is not None and total is not None and not issues:
            row = find_item_row([qty, Decimal(unit) / 100, Decimal(total) / 100], li.item_code, li.description, doc)
            if row is None:
                issues.append("quantity, unit price and line total are not printed together on one row with this item")

        description_verified = bool(li.description) and description_in_document(li.description, doc)
        if li.description and not description_verified:
            # OCR garbles words. Only on a row that was itself read by OCR may the printed item code
            # identify the product instead - never because some other page needed OCR.
            ocr_row_with_code = row is not None and row.method != "text-layer" and row.code_in_row
            if not ocr_row_with_code:
                issues.append(f"description '{li.description[:60]}' not found in document")
        lines.append(
            ParsedLine(
                index=i, item_code=li.item_code, description=li.description, quantity=qty,
                unit_price_paise=unit, line_total_paise=total, raw=li.model_dump(),
                verified=not issues, issues=issues, description_verified=description_verified,
            )
        )

    return VerifiedInvoice(
        field_checks=checks,
        invoice_number=ex.invoice_number.strip() if ex.invoice_number else None,
        invoice_number_norm=normalize_invoice_number(ex.invoice_number),
        invoice_date=inv_date,
        vendor_name=ex.vendor_name.strip() if ex.vendor_name else None,
        vendor_tax_id=ex.vendor_tax_id.strip() if ex.vendor_tax_id else None,
        po_number=ex.po_number.strip() if ex.po_number else None,
        currency=currency,
        buyer_name=ex.buyer_name.strip() if ex.buyer_name else None,
        buyer_tax_id=ex.buyer_tax_id.strip() if ex.buyer_tax_id else None,
        subtotal_paise=amounts["subtotal"],
        tax_paise=tax_paise,
        tax_source=tax_source,
        shipping_paise=amounts["shipping_amount"] or 0,
        discount_paise=amounts["discount_amount"] or 0,
        total_paise=amounts["total_amount"],
        lines=lines,
        ambiguous_fields=list(ex.ambiguous_fields),
        suspicious_instructions=ex.suspicious_instructions or _contains_ai_instructions(doc.text),
        parse_errors=parse_errors,
    )


_INJECTION_PATTERNS = re.compile(
    r"(?i)(ignore (all|any|previous|prior) (instructions|rules)|"
    r"(note|message|instruction)s? (to|for) (the )?(ai|llm|automated|assistant)|"
    r"mark (this )?(invoice )?as (approved|pass(ed)?|paid)|pre-?approved by|"
    r"skip (the )?validation|you are an? (ai|assistant))"
)


def _contains_ai_instructions(text: str) -> bool:
    """Deterministic backstop: flag injection attempts even if the model doesn't."""
    return bool(_INJECTION_PATTERNS.search(text))
