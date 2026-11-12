"""Stage 4: look the invoice up against known procurement records.

Pure database queries and deterministic comparisons. Nothing here asks the
AI anything. Only VERIFIED fields are used to confirm identities.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field

from sqlalchemy.engine import Connection

from src import database as db
from src.evidence_validator import ParsedLine, VerifiedInvoice

FUZZY_SUGGESTION_THRESHOLD = 0.6
LINE_FUZZY_THRESHOLD = 0.75
# Printed name must be at least this similar to the GSTIN owner's legal name or an alias.
NAME_CONSISTENCY_THRESHOLD = 0.6


@dataclass
class VendorMatch:
    status: str  # CONFIRMED / UNCERTAIN / CONFLICT / NOT_FOUND
    vendor: dict | None = None
    method: str | None = None  # TAX_ID / NAME
    reason: str = ""
    suggestions: list[dict] = field(default_factory=list)


@dataclass
class LineMatch:
    invoice_line: int
    po_line_no: int | None
    method: str  # CODE / DESCRIPTION / NONE
    po_line: dict | None = None
    suggestion: dict | None = None  # closest PO line when no confident match
    suggestion_score: float | None = None


@dataclass
class POCandidate:
    po_id: str
    score: int
    remaining_paise: int
    reasons: list[str]


@dataclass
class POMatch:
    status: str  # FOUND / NOT_FOUND / NOT_PROVIDED / UNVERIFIED
    po: dict | None = None
    lines: list[dict] = field(default_factory=list)
    committed_paise: int = 0
    remaining_paise: int | None = None
    invoiced_qty_milli: dict[int, int] = field(default_factory=dict)
    reason: str = ""


@dataclass
class DuplicateInfo:
    active_ledger_entry: dict | None = None
    open_review_run: dict | None = None
    same_number_other_vendor: list[dict] = field(default_factory=list)
    same_file_runs: list[dict] = field(default_factory=list)
    active_same_file: dict | None = None        # identical PDF already reserved/approved
    open_same_file: list[dict] = field(default_factory=list)  # identical PDF waiting in the queue
    similar_active: dict | None = None          # same vendor + amount + date, different number


@dataclass
class MatchResult:
    vendor: VendorMatch
    po: POMatch
    line_matches: list[LineMatch]
    candidates: list[POCandidate]
    duplicates: DuplicateInfo

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------


def _code(s: str | None) -> str | None:
    return "".join(ch for ch in s.upper() if ch.isalnum()) if s else None


def _desc(s: str | None) -> str:
    return " ".join((s or "").casefold().replace("-", " ").split())


def _code_pattern(code: str) -> re.Pattern:
    """'LAP-5440' matches 'LAP-5440', 'LAP 5440', 'LAP5440' as a whole token, never inside 'XLAP-54401'."""
    chars = [c for c in code.upper() if c.isalnum()]
    body = r"[\s\-_/.]?".join(re.escape(c) for c in chars)
    return re.compile(rf"(?<![A-Z0-9]){body}(?![A-Z0-9])")


def _code_in_text(code: str | None, text: str | None) -> bool:
    if not code or not text or not any(c.isalnum() for c in code):
        return False
    return _code_pattern(code).search(text.upper()) is not None


def match_vendor(conn: Connection, inv: VerifiedInvoice) -> VendorMatch:
    by_tax = by_name = None
    if inv.vendor_tax_id and inv.is_verified("vendor_tax_id"):
        by_tax = db.find_vendor_by_tax_id(conn, inv.vendor_tax_id)
    if inv.vendor_name and inv.is_verified("vendor_name"):
        by_name = db.find_vendor_by_name(conn, inv.vendor_name)

    if by_tax and by_name and by_tax["vendor_id"] != by_name["vendor_id"]:
        return VendorMatch(
            "CONFLICT", None, None,
            f"GSTIN belongs to {by_tax['legal_name']} but the name matches {by_name['legal_name']}",
            suggestions=[by_tax, by_name],
        )
    if by_tax:
        # A GSTIN can be copied from a genuine supplier: the printed name must also be consistent.
        if inv.vendor_name and inv.is_verified("vendor_name") and not by_name:
            target = db.normalize_name(inv.vendor_name) or ""
            known = [a["alias_norm"] for a in db.list_aliases(conn) if a["vendor_id"] == by_tax["vendor_id"]]
            best = max((difflib.SequenceMatcher(None, target, k).ratio() for k in known), default=0.0)
            if best < NAME_CONSISTENCY_THRESHOLD:
                return VendorMatch(
                    "CONFLICT", None, None,
                    f"GSTIN {inv.vendor_tax_id} belongs to {by_tax['legal_name']}, but the invoice is issued by "
                    f"'{inv.vendor_name}'", suggestions=[by_tax],
                )
        return VendorMatch("CONFIRMED", by_tax, "TAX_ID", f"GSTIN {inv.vendor_tax_id} matches vendor master")
    if inv.vendor_tax_id and inv.is_verified("vendor_tax_id") and by_name:
        return VendorMatch(
            "CONFLICT", None, None,
            f"Name matches {by_name['legal_name']} but GSTIN {inv.vendor_tax_id} is not the one on file ({by_name['tax_id']})",
            suggestions=[by_name],
        )
    if by_name:
        return VendorMatch("CONFIRMED", by_name, "NAME", f"Name '{inv.vendor_name}' matches an approved name/alias")

    # Nothing exact: offer suggestions for a human, never confirm.
    suggestions = []
    if inv.vendor_name:
        target = db.normalize_name(inv.vendor_name) or ""
        for alias in db.list_aliases(conn):
            score = difflib.SequenceMatcher(None, target, alias["alias_norm"]).ratio()
            if score >= FUZZY_SUGGESTION_THRESHOLD:
                v = db.get_vendor(conn, alias["vendor_id"])
                suggestions.append({**v, "score": round(score, 2), "matched_alias": alias["alias"]})
        suggestions.sort(key=lambda s: -s["score"])
        seen, unique = set(), []
        for s in suggestions:
            if s["vendor_id"] not in seen:
                seen.add(s["vendor_id"])
                unique.append(s)
        suggestions = unique[:3]
    if not inv.is_verified("vendor_name") and inv.vendor_name:
        return VendorMatch("UNCERTAIN", None, None, "Vendor name could not be verified in the document", suggestions)
    if suggestions:
        return VendorMatch("UNCERTAIN", None, None, "No exact vendor match; similar names found", suggestions)
    return VendorMatch("NOT_FOUND", None, None, "Vendor is not in the vendor master")


def match_lines(inv_lines: list[ParsedLine], po_lines: list[dict]) -> list[LineMatch]:
    """Map invoice lines to PO lines by item code (whole token) or by a description that is
    verifiably printed on the invoice. Near matches are only suggestions."""
    out = []
    for line in inv_lines:
        hit, method = None, "NONE"
        if line.item_code:
            hit = next((p for p in po_lines if p["item_code"] and _code(p["item_code"]) == _code(line.item_code)), None)
            method = "CODE" if hit else "NONE"
        if hit is None and line.description and line.description_verified:
            # A printed code may sit inside the description column ("... [LAP-5440]").
            hit = next((p for p in po_lines if _code_in_text(p["item_code"], line.description)), None)
            if hit:
                method = "CODE"
            else:
                hit = next((p for p in po_lines if _desc(p["description"]) == _desc(line.description)), None)
                method = "DESCRIPTION" if hit else "NONE"
        if hit:
            out.append(LineMatch(line.index, hit["line_no"], method, hit))
            continue
        best, best_score = None, 0.0
        for p in po_lines:
            score = difflib.SequenceMatcher(None, _desc(line.description), _desc(p["description"])).ratio()
            if score > best_score:
                best, best_score = p, score
        out.append(
            LineMatch(line.index, None, "NONE", None,
                      suggestion=best if best_score >= LINE_FUZZY_THRESHOLD else None,
                      suggestion_score=round(best_score, 2) if best else None)
        )
    return out


def load_po(conn: Connection, po: dict) -> POMatch:
    committed = db.po_committed_paise(conn, po["po_id"])
    return POMatch(
        status="FOUND",
        po=po,
        lines=db.get_po_lines(conn, po["po_id"]),
        committed_paise=committed,
        remaining_paise=po["gross_total_paise"] - committed,
        invoiced_qty_milli=db.po_invoiced_qty_milli(conn, po["po_id"]),
        reason=f"PO {po['po_id']} found",
    )


def suggest_po_candidates(conn: Connection, vendor_id: str, inv: VerifiedInvoice) -> list[POCandidate]:
    """Rank the vendor's open POs by how well they fit the invoice. Suggestions only."""
    out = []
    for po in db.list_open_pos_for_vendor(conn, vendor_id):
        pm = load_po(conn, po)
        lm = match_lines(inv.lines, pm.lines)
        matched = [m for m in lm if m.po_line_no is not None]
        reasons = []
        score = 0
        if inv.lines:
            frac = len(matched) / len(inv.lines)
            score += int(60 * frac)
            reasons.append(f"{len(matched)}/{len(inv.lines)} invoice lines match PO items")
            price_ok = sum(
                1 for m in matched
                if inv.lines[m.invoice_line - 1].unit_price_paise == m.po_line["unit_price_paise"]
            )
            if matched:
                score += int(20 * price_ok / len(matched))
                reasons.append(f"{price_ok}/{len(matched)} matched lines at the approved price")
        if inv.total_paise is not None and pm.remaining_paise is not None:
            if inv.total_paise <= pm.remaining_paise:
                score += 20
                reasons.append("Invoice total fits the remaining PO balance")
            else:
                reasons.append("Invoice total exceeds the remaining PO balance")
        if score > 0:
            out.append(POCandidate(po["po_id"], score, pm.remaining_paise or 0, reasons))
    out.sort(key=lambda c: -c.score)
    return out[:3]


def match_procurement(conn: Connection, inv: VerifiedInvoice, run_id: str, file_hash: str | None) -> MatchResult:
    vendor = match_vendor(conn, inv)

    if inv.po_number is None:
        po = POMatch("NOT_PROVIDED", reason="No PO reference printed on the invoice")
    elif not inv.is_verified("po_number"):
        po = POMatch("UNVERIFIED", reason="PO reference could not be verified in the document")
    else:
        found = db.find_po_normalized(conn, inv.po_number)
        po = load_po(conn, found) if found else POMatch("NOT_FOUND", reason=f"PO {inv.po_number} does not exist")

    line_matches = match_lines(inv.lines, po.lines) if po.status == "FOUND" else []

    candidates: list[POCandidate] = []
    if po.status != "FOUND" and vendor.status == "CONFIRMED":
        candidates = suggest_po_candidates(conn, vendor.vendor["vendor_id"], inv)

    dup = DuplicateInfo()
    if inv.invoice_number_norm:
        if vendor.status == "CONFIRMED":
            vid = vendor.vendor["vendor_id"]
            dup.active_ledger_entry = db.find_active_invoice(conn, vid, inv.invoice_number_norm)
            dup.open_review_run = db.find_open_review_run(conn, vid, inv.invoice_number_norm, run_id)
        else:
            dup.same_number_other_vendor = db.find_active_invoice_any_vendor(conn, inv.invoice_number_norm)
    if file_hash:
        dup.same_file_runs = db.runs_with_file_hash(conn, file_hash, run_id)
        dup.active_same_file = db.find_active_by_file_hash(conn, file_hash)
        dup.open_same_file = db.find_open_runs_by_file_hash(conn, file_hash, run_id)
    if vendor.status == "CONFIRMED" and inv.invoice_date:
        dup.similar_active = db.find_active_similar(conn, vendor.vendor["vendor_id"], inv.total_paise,
                                                    inv.invoice_date.isoformat(), inv.invoice_number_norm)

    return MatchResult(vendor, po, line_matches, candidates, dup)
