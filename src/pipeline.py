"""The orchestrator: one function that runs an uploaded PDF through every stage.

    1 Receive upload   -> run record + SHA-256
    2 Read PDF         -> page text (OCR for scanned pages)
    3 AI extraction    -> structured fields + quotes          (only AI step)
    4 Evidence check   -> every value verified against the PDF
    5 PO matching      -> vendor, PO, line items, duplicates from the database
    6 Business rules   -> 16 deterministic checks
    7 Decide & save    -> PASS/REVIEW/BLOCK; PASS re-checked + reserved atomically

Every stage is logged to the database as it finishes, and reported to the UI
through `on_stage` so the live view shows real progress, not an animation.
A technical failure at any stage ends the run as ERROR with no ledger change.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from datetime import date
from typing import Callable

from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from src import config
from src import database as db
from src.decision_engine import IncompleteValidationError, make_decision
from src.evidence_validator import VerifiedInvoice, verify_extraction
from src.extractor import ExtractionError, ExtractionResult, extract_invoice
from src.matcher import MatchResult, match_procurement
from src.money import format_inr
from src.pdf_reader import PdfReadError, file_sha256, read_pdf
from src.reporting import build_explanation
from src.rules import BLOCK, REVIEW, RULES_BY_CODE, validate_invoice

STAGES = (
    ("UPLOAD", "Receive upload"),
    ("PDF_READ", "Read PDF"),
    ("EXTRACTION", "AI field extraction"),
    ("EVIDENCE", "Verify against source"),
    ("MATCHING", "Match vendor & PO"),
    ("RULES", "Business rules"),
    ("DECISION", "Decide & save"),
)
STAGE_LABELS = dict(STAGES)

StageCallback = Callable[[str, str, str, int | None], None]


@dataclass
class PipelineResult:
    run_id: str
    decision: str
    headline: str


class _StageFailed(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def _noop(*_args) -> None:
    pass


def process_invoice(
    pdf_bytes: bytes,
    file_name: str = "upload.pdf",
    engine: Engine | None = None,
    extractor: Callable[[str], ExtractionResult] = extract_invoice,
    on_stage: StageCallback = _noop,
    today: date | None = None,
) -> PipelineResult:
    engine = engine or db.get_engine()
    started = time.perf_counter()
    run_id = db.new_run_id()
    seq = 0

    def stage(key: str, status: str, detail: str, t0: float | None) -> None:
        nonlocal seq
        ms = int((time.perf_counter() - t0) * 1000) if t0 is not None else None
        seq += 1
        db.log_stage(engine, run_id, seq, key, status, detail, ms)
        on_stage(key, status, detail, ms)

    # ---- 1. Receive upload -------------------------------------------------
    t0 = time.perf_counter()
    on_stage("UPLOAD", "RUNNING", "", None)
    file_hash = file_sha256(pdf_bytes)
    model = config.extraction_model()
    db.create_run(engine, run_id, file_name, file_hash, len(pdf_bytes), model)
    stage("UPLOAD", "SUCCESS", f"{file_name} · {len(pdf_bytes) / 1024:.1f} KB · SHA-256 {file_hash[:12]}…", t0)

    current = "PDF_READ"
    try:
        # ---- 2. Read PDF ---------------------------------------------------
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        try:
            doc = read_pdf(pdf_bytes)
        except PdfReadError as exc:
            raise _StageFailed(current, str(exc)) from exc
        db.update_run(engine, run_id, page_count=doc.page_count, ocr_used=doc.ocr_used, document_text=doc.text)
        methods = ", ".join(sorted({p.method for p in doc.pages}))
        stage(current, "WARNING" if doc.ocr_used else "SUCCESS",
              f"{doc.page_count} page(s), {doc.total_chars} characters via {methods}"
              + (" — scanned page read with OCR" if doc.ocr_used else ""), t0)

        # ---- 3. AI extraction ---------------------------------------------
        current = "EXTRACTION"
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        try:
            ex_result = extractor(doc.text)
        except ExtractionError as exc:
            raise _StageFailed(current, str(exc)) from exc
        ex = ex_result.extraction
        db.update_run(engine, run_id, model=ex_result.model, extraction_json=ex.model_dump())
        filled = sum(1 for k, v in ex.model_dump().items()
                     if v not in (None, [], False) and k not in ("evidence", "ambiguous_fields", "suspicious_instructions"))
        tokens = f" · {ex_result.input_tokens}+{ex_result.output_tokens} tokens" if ex_result.input_tokens else ""
        stage(current, "SUCCESS", f"{filled} fields and {len(ex.line_items)} line item(s) extracted by {ex_result.model}{tokens}", t0)

        # ---- 4. Evidence verification -------------------------------------
        current = "EVIDENCE"
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        inv: VerifiedInvoice = verify_extraction(ex, doc)
        n_ver = sum(1 for f in inv.field_checks.values() if f.status == "VERIFIED")
        n_bad = sum(1 for f in inv.field_checks.values() if f.status in ("UNVERIFIED", "INVALID"))
        n_bad += sum(1 for l in inv.lines if not l.verified)
        db.update_run(
            engine, run_id,
            verification_json=inv.to_dict(),
            invoice_number=inv.invoice_number,
            invoice_number_norm=inv.invoice_number_norm,
            vendor_name=inv.vendor_name,
            amount_paise=inv.total_paise,
            currency=inv.currency,
        )
        stage(current, "WARNING" if n_bad else "SUCCESS",
              f"{n_ver} fields verified in the PDF" + (f", {n_bad} could not be verified" if n_bad else ""), t0)

        # ---- 5. Procurement matching --------------------------------------
        current = "MATCHING"
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        with engine.connect() as conn:
            match: MatchResult = match_procurement(conn, inv, run_id, file_hash)
        vendor_id = match.vendor.vendor["vendor_id"] if match.vendor.vendor else None
        po_id = match.po.po["po_id"] if match.po.po else None
        db.update_run(engine, run_id, vendor_id=vendor_id, po_id=po_id or inv.po_number, match_json=match.to_dict())
        v_txt = f"Vendor {match.vendor.status.lower()}" + (f" ({vendor_id})" if vendor_id else "")
        p_txt = f"PO {po_id} found, {format_inr(match.po.remaining_paise)} available" if po_id else match.po.reason
        if match.candidates:
            p_txt += f"; suggested {', '.join(c.po_id for c in match.candidates)}"
        ok = match.vendor.status == "CONFIRMED" and match.po.status == "FOUND"
        stage(current, "SUCCESS" if ok else "WARNING", f"{v_txt} · {p_txt}", t0)

        # ---- 6. Rules -----------------------------------------------------
        current = "RULES"
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        checks = validate_invoice(inv, match, today=today)
        counts = {o: sum(1 for c in checks if c["outcome"] == o) for o in ("PASS", "REVIEW", "BLOCK", "NOT_EVALUATED")}
        stage(current, "SUCCESS" if counts["PASS"] == len(checks) else "WARNING",
              f"{counts['PASS']}/{len(checks)} passed"
              + (f", {counts['REVIEW']} review" if counts["REVIEW"] else "")
              + (f", {counts['BLOCK']} block" if counts["BLOCK"] else "")
              + (f", {counts['NOT_EVALUATED']} not evaluated" if counts["NOT_EVALUATED"] else ""), t0)

        # ---- 7. Decide and persist ----------------------------------------
        current = "DECISION"
        t0 = time.perf_counter()
        on_stage(current, "RUNNING", "", None)
        try:
            decision = make_decision(checks)
        except IncompleteValidationError as exc:
            raise _StageFailed(current, str(exc)) from exc

        vendor_label = match.vendor.vendor["legal_name"] if match.vendor.vendor else inv.vendor_name
        req = _reservation_request(run_id, inv, match, file_hash) if decision == "PASS" else None

        def finalize(conn, decision, checks, forced_duplicate=False):
            """Reservation (if PASS), final decision, rule checks: one transaction, all or nothing."""
            note = None
            if decision == "PASS":
                res = (db.ReservationResult(False, conflict="DUPLICATE",
                                            message="Unique constraint: invoice identity already active")
                       if forced_duplicate else db.reserve_in_transaction(conn, req))
                if res.ok:
                    note = (f"Reserved {format_inr(inv.total_paise)} on {req.po_id} (ledger #{res.ledger_id}); "
                            f"{format_inr(res.remaining_paise)} remains on the PO.")
                else:
                    decision, checks, note = _downgrade(checks, res)
            headline, explanation = build_explanation(
                decision, checks, inv.invoice_number, vendor_label, po_id or inv.po_number,
                inv.total_paise, note, currency=inv.currency,
            )
            db.save_checks(conn, run_id, checks)
            db.update_run(
                conn, run_id,
                status="COMPLETED", decision=decision, headline=headline, explanation=explanation,
                finished_at=db.now_iso(), duration_ms=int((time.perf_counter() - started) * 1000),
                review_status="OPEN" if decision in ("PASS", "REVIEW") else "NONE",
            )
            return decision, headline, note

        try:
            with engine.begin() as conn:
                decision, headline, reservation_note = finalize(conn, decision, checks)
        except IntegrityError:
            # A concurrent upload activated the same invoice first; the whole transaction rolled back.
            with engine.begin() as conn:
                decision, headline, reservation_note = finalize(conn, decision, checks, forced_duplicate=True)

        # The financial decision is now committed. Nothing after this point (stage log, UI callback)
        # is allowed to change it: a display failure must never turn a saved decision into ERROR.
        committed = PipelineResult(run_id, decision, headline)
        duration = int((time.perf_counter() - started) * 1000)
        try:
            stage(current, {"PASS": "SUCCESS", "REVIEW": "WARNING"}.get(decision, "FAILED"),
                  f"{decision} — " + (reservation_note or "no reservation created"), t0)
            on_stage("DONE", decision, headline, duration)
        except Exception:
            traceback.print_exc()
        return committed

    except _StageFailed as exc:
        return _fail(engine, run_id, exc.stage, exc.message, started, stage, on_stage)
    except Exception as exc:  # unexpected bug: fail closed, never PASS
        detail = f"Unexpected error: {exc.__class__.__name__}: {exc}"
        traceback.print_exc()
        return _fail(engine, run_id, current, detail, started, stage, on_stage)


def _reservation_request(run_id, inv: VerifiedInvoice, match: MatchResult, file_hash) -> db.ReservationRequest:
    agg: dict[int, dict] = {}
    for lm in match.line_matches:
        line = inv.lines[lm.invoice_line - 1]
        a = agg.setdefault(lm.po_line_no, {"line_no": lm.po_line_no, "quantity_milli": 0, "amount_paise": 0})
        a["quantity_milli"] += int((line.quantity * 1000).to_integral_value())
        a["amount_paise"] += line.line_total_paise
    return db.ReservationRequest(
        run_id=run_id,
        vendor_id=match.vendor.vendor["vendor_id"],
        invoice_number=inv.invoice_number,
        invoice_number_norm=inv.invoice_number_norm,
        po_id=match.po.po["po_id"],
        invoice_date=inv.invoice_date.isoformat() if inv.invoice_date else None,
        amount_paise=inv.total_paise,
        file_hash=file_hash,
        lines=list(agg.values()),
    )


def _downgrade(checks, res):
    """State changed between validation and commit (e.g. a parallel upload): never keep a stale PASS."""
    checks = [dict(c) for c in checks]
    target, outcome = {
        "DUPLICATE": ("DUPLICATE_INVOICE", BLOCK),
        "BALANCE": ("PO_BALANCE", REVIEW),
        "VENDOR": ("VENDOR_APPROVED", BLOCK),
    }.get(res.conflict, ("PO_EXISTS", REVIEW))
    for c in checks:
        if c["rule_code"] == target:
            c["outcome"] = outcome
            c["reason"] = f"Final re-check at commit time: {res.message}"
            c["actual"] = "Changed since validation"
    return make_decision(checks), checks, f"Commit-time re-check failed ({res.conflict}); nothing was reserved."


def _fail(engine, run_id, stage_key, message, started, stage, on_stage) -> PipelineResult:
    headline = f"Processing failed at '{STAGE_LABELS.get(stage_key, stage_key)}': {message}"
    try:  # defensive: never overwrite a decision that was already committed
        with engine.connect() as conn:
            run = db.get_run(conn, run_id)
        if run and run.get("status") == "COMPLETED" and run.get("decision"):
            return PipelineResult(run_id, run["decision"], run.get("headline") or "")
    except Exception:
        traceback.print_exc()
    try:
        try:
            stage(stage_key, "FAILED", message, None)
        except Exception:
            traceback.print_exc()
        db.update_run(
            engine, run_id, status="FAILED", decision="ERROR", headline=headline,
            explanation=headline + "\n\nNo decision was made and nothing was reserved. "
            "A technical failure is not evidence about the invoice — retry or process manually.",
            error_stage=stage_key, error_message=message, finished_at=db.now_iso(),
            duration_ms=int((time.perf_counter() - started) * 1000), review_status="NONE",
        )
    except Exception:  # database itself is down: still report ERROR to the UI
        traceback.print_exc()
    try:
        on_stage("DONE", "ERROR", headline, None)
    except Exception:
        traceback.print_exc()
    return PipelineResult(run_id, "ERROR", headline)


def load_run_bundle(run_id: str, engine: Engine | None = None) -> dict | None:
    """Everything the UI needs to render one run."""
    engine = engine or db.get_engine()
    with engine.connect() as conn:
        run = db.get_run(conn, run_id)
        if run is None:
            return None
        bundle = {
            "run": run,
            "stages": db.get_run_stages(conn, run_id),
            "checks": db.get_run_checks(conn, run_id),
            "actions": db.get_review_actions(conn, run_id),
            "ledger": db.get_ledger_entry(conn, run["ledger_id"]) if run.get("ledger_id") else None,
        }
    for key in ("extraction_json", "verification_json", "match_json"):
        bundle[key.replace("_json", "")] = json.loads(run[key]) if run.get(key) else None
    for c in bundle["checks"]:
        r = RULES_BY_CODE.get(c["rule_code"])
        c["rule_number"] = r.number if r else ""
    return bundle
