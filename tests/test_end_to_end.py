"""Every sample invoice through the full pipeline, checking decisions AND database state.

The AI extractor is replaced by a deterministic stand-in returning the known
ground truth, so these tests exercise the real PDF reading, OCR, evidence
verification, matching, rules, decisions, reservations and audit trail.
"""

from sqlalchemy import func, select

from src import database as db
from src import review
from src.pipeline import load_run_bundle, process_invoice
from tests.conftest import TODAY, run_sample
from tests.fake_extractor import FakeExtractor, load_manifest
from src import config


def committed(engine, po):
    with engine.connect() as conn:
        return db.po_committed_paise(conn, po)


def ledger_rows(engine):
    with engine.connect() as conn:
        return [e for e in db.list_ledger(conn) if e["source"] != "SEED"]


def test_full_demo_sequence_matches_expected_decisions(engine):
    for entry in load_manifest():
        if entry["ground_truth"] is None:
            pdf = (config.SAMPLE_INVOICE_DIR / entry["file"]).read_bytes()
            result = process_invoice(pdf, entry["file"], engine=engine,
                                     extractor=FakeExtractor({}, fail="must not be called"), today=TODAY)
        else:
            result = run_sample(engine, entry["file"])
        assert result.decision == entry["expected"], f"{entry['file']}: {result.headline}"

    # Exactly the four PASS invoices hold reservations; nothing else touched a PO balance.
    rows = ledger_rows(engine)
    assert sorted(r["invoice_number"] for r in rows) == sorted(
        ["INV-ABC-2041", "SFS-2026-1102", "ZOS/26-27/0457", "KL/INV/3385"])
    assert all(r["allocation_status"] == "RESERVED" for r in rows)
    assert committed(engine, "PO-1001") == 47_495_000
    assert committed(engine, "PO-2001") == 47_200_000 + 11_800_000
    assert committed(engine, "PO-1002") == 9_735_000
    assert committed(engine, "PO-1005") == 8_512_000
    assert committed(engine, "PO-1003") == 0      # over-tolerance + arithmetic error reserved nothing
    assert committed(engine, "PO-1004") == 0      # missing-PO invoice waits for a human


def test_happy_path_audit_trail_is_complete(engine):
    result = run_sample(engine, "01_happy_path.pdf")
    b = load_run_bundle(result.run_id, engine)
    assert b["run"]["status"] == "COMPLETED" and b["run"]["decision"] == "PASS"
    assert [s["stage"] for s in b["stages"]] == ["UPLOAD", "PDF_READ", "EXTRACTION", "EVIDENCE", "MATCHING", "RULES", "DECISION"]
    assert len(b["checks"]) == 16 and all(c["outcome"] == "PASS" for c in b["checks"])
    assert b["ledger"]["allocation_status"] == "RESERVED" and b["ledger"]["amount_paise"] == 47_495_000
    assert b["run"]["policy_version"] == config.POLICY_VERSION and b["run"]["model"] == "fake-extractor"
    assert "Reserved ₹4,74,950.00 on PO-1001" in b["run"]["explanation"]


def test_split_billing_consumes_quantity_then_reviews_overrun(engine):
    assert run_sample(engine, "02_split_billing_part1.pdf").decision == "PASS"
    over = run_sample(engine, "03_split_billing_overrun.pdf")
    assert over.decision == "REVIEW"
    b = load_run_bundle(over.run_id, engine)
    checks = {c["rule_code"]: c for c in b["checks"]}
    assert "exceeds the remaining ₹1,18,000.00" in checks["PO_BALANCE"]["reason"]
    assert "only 2 of 12 remain (10 already invoiced)" in checks["LINE_ITEMS"]["reason"]
    assert b["ledger"] is None


def test_scanned_invoice_passes_via_ocr(engine):
    result = run_sample(engine, "12_scanned_invoice.pdf")
    b = load_run_bundle(result.run_id, engine)
    assert b["run"]["ocr_used"] and result.decision == "PASS"


def test_ai_outage_is_error_with_no_side_effects(engine):
    pdf = (config.SAMPLE_INVOICE_DIR / "01_happy_path.pdf").read_bytes()
    result = process_invoice(pdf, "01_happy_path.pdf", engine=engine,
                             extractor=FakeExtractor({}, fail="AI provider timed out."), today=TODAY)
    assert result.decision == "ERROR"
    b = load_run_bundle(result.run_id, engine)
    assert b["run"]["error_stage"] == "EXTRACTION" and b["run"]["status"] == "FAILED"
    assert ledger_rows(engine) == []
    # The invoice can still be processed once the provider is back.
    assert run_sample(engine, "01_happy_path.pdf").decision == "PASS"


def test_unexpected_bug_fails_closed(engine):
    class Broken:
        def __call__(self, text):
            raise RuntimeError("boom")
    pdf = (config.SAMPLE_INVOICE_DIR / "01_happy_path.pdf").read_bytes()
    result = process_invoice(pdf, "x.pdf", engine=engine, extractor=Broken(), today=TODAY)
    assert result.decision == "ERROR" and ledger_rows(engine) == []


def test_injected_instruction_does_not_change_outcome(engine):
    result = run_sample(engine, "11_prompt_injection.pdf")
    assert result.decision == "REVIEW"
    assert ledger_rows(engine) == []


# ---- human review --------------------------------------------------------


def test_reviewer_allocates_missing_po_invoice_to_suggested_po(engine):
    r = run_sample(engine, "06_missing_po.pdf")
    res = review.approve_review(engine, r.run_id, "Asha (AP)", "Confirmed with buyer: PO-1004", po_id="PO-1004")
    assert res.ok, res.message
    assert committed(engine, "PO-1004") == 32_450_000
    b = load_run_bundle(r.run_id, engine)
    assert b["run"]["review_status"] == "APPROVED" and b["actions"][0]["actor"] == "Asha (AP)"
    # It is now a known invoice: a resend is blocked.
    assert run_sample(engine, "06_missing_po.pdf").decision == "BLOCK"


def test_reviewer_cannot_overspend_po(engine):
    run_sample(engine, "02_split_billing_part1.pdf")
    over = run_sample(engine, "03_split_billing_overrun.pdf")
    res = review.approve_review(engine, over.run_id, "Asha", "vendor says ok")
    assert not res.ok and "exceeds" in res.message
    assert committed(engine, "PO-2001") == 59_000_000


def test_reviewer_can_accept_price_variance(engine):
    r = run_sample(engine, "05_price_over_tolerance.pdf")
    res = review.approve_review(engine, r.run_id, "Asha", "Price increase agreed by procurement (email 4 Nov)")
    assert res.ok and committed(engine, "PO-1003") == 11_682_000


def test_justification_is_mandatory(engine):
    r = run_sample(engine, "05_price_over_tolerance.pdf")
    assert not review.approve_review(engine, r.run_id, "Asha", "").ok
    assert not review.reject_review(engine, r.run_id, "Asha", None).ok


def test_missing_invoice_number_must_be_supplied_by_reviewer(engine):
    r = run_sample(engine, "13_missing_invoice_number.pdf")
    assert not review.approve_review(engine, r.run_id, "Asha", "ok").ok
    res = review.approve_review(engine, r.run_id, "Asha", "Number confirmed by vendor email", invoice_number="INV-ABC-2063")
    assert res.ok


def test_release_frees_balance_and_reject_reserves_nothing(engine):
    p = run_sample(engine, "01_happy_path.pdf")
    assert not review.release_reservation(engine, p.run_id, "Asha", "").ok
    assert review.release_reservation(engine, p.run_id, "Asha", "Goods not received").ok
    assert committed(engine, "PO-1001") == 0
    # After release, a corrected resubmission can pass again.
    assert run_sample(engine, "07_duplicate_resend.pdf").decision == "PASS"

    r = run_sample(engine, "10_arithmetic_error.pdf")
    assert review.reject_review(engine, r.run_id, "Asha", "Total wrong; asked vendor for corrected invoice").ok
    assert committed(engine, "PO-1003") == 0


def test_approve_reserved_moves_to_approved(engine):
    p = run_sample(engine, "04_price_within_tolerance.pdf")
    assert review.approve_reserved(engine, p.run_id, "Asha", None).ok
    with engine.connect() as conn:
        assert db.get_ledger_entry(conn, db.get_run(conn, p.run_id)["ledger_id"])["allocation_status"] == "APPROVED"
    assert not review.approve_reserved(engine, p.run_id, "Asha", None).ok  # not twice


def test_runs_persist_across_engine_restarts(engine, tmp_path):
    r = run_sample(engine, "01_happy_path.pdf")
    url = engine.url.render_as_string(hide_password=False)
    engine.dispose()
    db.get_engine.cache_clear()
    fresh = db.get_engine(url)
    with fresh.connect() as conn:
        assert db.get_run(conn, r.run_id)["decision"] == "PASS"
        assert conn.execute(select(func.count()).select_from(db.processing_runs)).scalar_one() == 1


def test_foreign_vendor_usd_invoice_is_reviewed_with_clear_reasons(engine):
    """Shaped like a typical overseas SaaS invoice: unknown vendor, no PO, dollars, old date."""
    from src.extractor import ExtractionResult
    from src.pdf_reader import DocumentText, PageText
    from tests.fake_extractor import build_extraction
    import src.pipeline as pipeline

    text = ("Invoice\nInvoice Number INV-9001\nFrom:\nAcme Web Studio LLC\nInvoice Date January 25, 2024\n"
            "Hrs/Qty Service Rate/Price Sub Total\n1.00 Web Design $85.00 $85.00\n"
            "Sub Total $85.00\nTax $8.50\nTotal $93.50\n")
    ai = {"invoice_number": "INV-9001", "invoice_date": "January 25, 2024", "vendor_name": "Acme Web Studio LLC",
          "vendor_tax_id": None, "po_number": None, "currency": "$",
          "line_items": [{"item_code": None, "description": "Web Design", "quantity": "1.00",
                          "unit_price": "$85.00", "line_total": "$85.00"}],
          "subtotal": "$85.00", "tax_amount": "$8.50", "tax_components": [], "shipping_amount": None,
          "discount_amount": None, "total_amount": "$93.50", "suspicious_instructions": False}
    doc = DocumentText([PageText(1, text, "text-layer", len(text))], 1, False)
    orig = pipeline.read_pdf
    pipeline.read_pdf = lambda b: doc
    try:
        r = process_invoice(b"%PDF-fake", "usd.pdf", engine=engine,
                            extractor=lambda t: ExtractionResult(build_extraction(ai, t), "fake"), today=TODAY)
    finally:
        pipeline.read_pdf = orig
    b = load_run_bundle(r.run_id, engine)
    outcomes = {c["rule_code"]: c["outcome"] for c in b["checks"]}
    assert r.decision == "REVIEW" and "$93.50" in r.headline and "₹" not in r.headline
    assert outcomes["SOURCE_EVIDENCE"] == "PASS" and outcomes["ARITHMETIC"] == "PASS"
    assert outcomes["VENDOR_IDENTITY"] == "REVIEW" and outcomes["CURRENCY"] == "REVIEW"
    assert ledger_rows(engine) == []


# ---- A real-world GST invoice layout (CloudZen / GSTZen demo) ----------------

CLOUDZEN_TEXT = """Tax Invoice Original for Recipient
Supplier: Details of Invoice:
CloudZen Software Labs Pvt. Ltd. Invoice Number: 17-18/JH/97
State: Jharkhand Invoice Date: July 26, 2017
State Code: JH - 20 Invoice Amount: INR 47,925.00
GSTIN: 20QXOCC9424D1Z5 Reverse charge applicable: No
Details of Buyer (Bill To): Cipla Ltd GSTIN: 08AKOCX6349P1ZL
# Description Taxable Value IGST Rate Amount Total
1 OTHR BLCHD WOVN FBRCS WGHNG >200 G/M2 12,096.00 18.00 2,177.28 14,273.28
521222 9.00 GMS 1,344.00
2 OTER BEANS DRIED & SHLD 7,742.00 18.00 1,393.56 9,135.56
071339 7.00 UNT 1,106.00
3 GLAZIERS & GRAFTING PUTY, RESIN ELEMNTS NON RFRCTRY SRFCNG PRPN FR FLOORS, WALL ETC 9,345.00 28.00 2,616.60 11,961.60
321490 7.00 CCM 1,335.00
4 OTHER, FRESH OR CHILLED 7,567.00 28.00 2,118.76 9,685.76
020735 7.00 BTL 1,081.00
5 OTHER COSMETIC & TOILT PRPN N E S 2,241.00 28.00 627.48 2,868.48
33079090 3.00 MLT 747.00
Total Amounts (INR) 38,991.00 8,933.68 47,924.68
Rounding 0.32
Invoice Total (in figures): INR 47,925.00
"""

CLOUDZEN_AI = {
    "invoice_number": "17-18/JH/97", "invoice_date": "July 26, 2017",
    "vendor_name": "CloudZen Software Labs Pvt. Ltd.", "vendor_tax_id": "20QXOCC9424D1Z5", "po_number": None,
    "currency": "INR", "buyer_name": "Cipla Ltd", "buyer_tax_id": "08AKOCX6349P1ZL",
    "line_items": [
        {"item_code": "521222", "description": "OTHR BLCHD WOVN FBRCS WGHNG >200 G/M2", "quantity": "9.00 GMS", "unit_price": "1,344.00", "line_total": "12,096.00"},
        {"item_code": "071339", "description": "OTER BEANS DRIED & SHLD", "quantity": "7.00 UNT", "unit_price": "1,106.00", "line_total": "7,742.00"},
        {"item_code": "321490", "description": "GLAZIERS & GRAFTING PUTY, RESIN ELEMNTS NON RFRCTRY SRFCNG PRPN FR FLOORS, WALL ETC", "quantity": "7.00 CCM", "unit_price": "1,335.00", "line_total": "9,345.00"},
        {"item_code": "020735", "description": "OTHER, FRESH OR CHILLED", "quantity": "7.00 BTL", "unit_price": "1,081.00", "line_total": "7,567.00"},
        {"item_code": "33079090", "description": "OTHER COSMETIC & TOILT PRPN N E S", "quantity": "3.00 MLT", "unit_price": "747.00", "line_total": "2,241.00"},
    ],
    "subtotal": "38,991.00", "tax_amount": "8,933.68", "tax_components": [], "shipping_amount": None,
    "discount_amount": None, "total_amount": "47,925.00", "suspicious_instructions": False,
}


def _run_text_invoice(engine, text, ai, today=TODAY):
    from src.extractor import ExtractionResult
    from src.pdf_reader import DocumentText, PageText
    from tests.fake_extractor import build_extraction
    import src.pipeline as pipeline

    doc = DocumentText([PageText(1, text, "text-layer", len(text))], 1, False)
    orig = pipeline.read_pdf
    pipeline.read_pdf = lambda b: doc
    try:
        return process_invoice(b"%PDF-x" + text.encode()[:50], "cloudzen.pdf", engine=engine,
                               extractor=lambda t: ExtractionResult(build_extraction(ai, t), "fake"), today=today)
    finally:
        pipeline.read_pdf = orig


def test_cloudzen_invoice_billed_to_another_company_is_blocked(engine):
    """The real GSTZen demo invoice is addressed to Cipla Ltd, not to us: it must not be paid."""
    r = _run_text_invoice(engine, CLOUDZEN_TEXT, CLOUDZEN_AI)
    b = load_run_bundle(r.run_id, engine)
    out = {c["rule_code"]: c for c in b["checks"]}
    assert r.decision == "BLOCK"
    assert out["BILLED_TO_US"]["outcome"] == "BLOCK" and "Cipla Ltd" in out["BILLED_TO_US"]["reason"]
    assert "not addressed to our company" in r.headline
    # Everything else is still read and checked correctly
    for code in ("VENDOR_IDENTITY", "VENDOR_APPROVED", "SOURCE_EVIDENCE", "ARITHMETIC", "CURRENCY"):
        assert out[code]["outcome"] == "PASS", (code, out[code]["reason"])
    assert out["PO_EXISTS"]["details"]["candidates"][0]["po_id"] == "PO-1007"
    res = review.approve_review(engine, r.run_id, "Asha (AP)", "Confirmed", po_id="PO-1007")
    assert not res.ok and committed(engine, "PO-1007") == 0


def test_cloudzen_invoice_within_policy_date_and_quoting_po_passes(engine):
    """Same invoice, recent and quoting its PO: everything (5 lines, mixed 18%/28% GST, rounding) checks out."""
    from datetime import date
    text = (CLOUDZEN_TEXT.replace("July 26, 2017", "November 5, 2026")
            .replace("Cipla Ltd GSTIN: 08AKOCX6349P1ZL", "Northwind Retail India Pvt Ltd GSTIN: 29NWRIN1234Q1Z3")
            + "PO Number: PO-1007\n")
    ai = {**CLOUDZEN_AI, "invoice_date": "November 5, 2026", "po_number": "PO-1007",
          "buyer_name": "Northwind Retail India Pvt Ltd", "buyer_tax_id": "29NWRIN1234Q1Z3"}
    r = _run_text_invoice(engine, text, ai, today=date(2026, 11, 9))
    b = load_run_bundle(r.run_id, engine)
    assert r.decision == "PASS", [(c["rule_code"], c["reason"]) for c in b["checks"] if c["outcome"] != "PASS"]
    # PO-1007 is worth the exact line total Rs 47,924.68; the invoice rounds up to Rs 47,925.00.
    bal = {c["rule_code"]: c for c in b["checks"]}["PO_BALANCE"]
    assert "rounding tolerance" in bal["reason"]
    assert committed(engine, "PO-1007") == 4_792_500
