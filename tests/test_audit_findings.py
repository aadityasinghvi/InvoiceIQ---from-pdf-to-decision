"""Regression tests for the findings of an external review of InvoiceIQ.

Each test reproduces a reported loophole and asserts it is closed. The principle:
human review is a control, not a way around controls - and the AI can never cause an approval.
"""

from __future__ import annotations

import pytest
from sqlalchemy import update

from datetime import date

from src import config, database as db, review
from src.extractor import ExtractionResult
from src.pdf_reader import DocumentText, PageText
from src.pipeline import load_run_bundle, process_invoice
from tests.conftest import TODAY, run_sample
from tests.fake_extractor import build_extraction, load_manifest

TRUTH = {m["file"]: m["ground_truth"] for m in load_manifest()}


def committed(engine, po):
    with engine.connect() as conn:
        return db.po_committed_paise(conn, po)


def pipeline_ledger(engine):
    with engine.connect() as conn:
        return [e for e in db.list_ledger(conn) if e["source"] != "SEED"]


def run_with(engine, file_name, **overrides):
    """Process a real sample PDF, but with the AI returning altered values (a misread / hallucination)."""
    pdf = (config.SAMPLE_INVOICE_DIR / file_name).read_bytes()
    gt = {**TRUTH[file_name], **overrides}
    return process_invoice(pdf, file_name, engine=engine,
                           extractor=lambda t: ExtractionResult(build_extraction(gt, t), "fake"), today=TODAY)


def run_text(engine, text, ai, name="text.pdf"):
    import src.pipeline as pipeline
    doc = DocumentText([PageText(1, text, "text-layer", len(text))], 1, False)
    orig = pipeline.read_pdf
    pipeline.read_pdf = lambda b: doc
    try:
        return process_invoice(b"%PDF-" + name.encode(), name, engine=engine,
                               extractor=lambda t: ExtractionResult(build_extraction(ai, t), "fake"), today=TODAY)
    finally:
        pipeline.read_pdf = orig


WIDGET_TEXT = ("Tax Invoice\nABC Technologies Pvt Ltd\nGSTIN: 29ABCDE1234F1Z5\nInvoice No: INV-ABC-9100\n"
               "Invoice Date: 05 Nov 2026\nPO Number: PO-1001\n1 Unapproved Widget 1 70,000.00 70,000.00\n"
               "Sub Total 70,000.00\nCGST @ 9% 6,300.00\nSGST @ 9% 6,300.00\nGrand Total (INR) 82,600.00\n")
WIDGET_AI = {"invoice_number": "INV-ABC-9100", "invoice_date": "05 Nov 2026", "vendor_name": "ABC Technologies Pvt Ltd",
             "vendor_tax_id": "29ABCDE1234F1Z5", "po_number": "PO-1001", "currency": "INR",
             "line_items": [{"item_code": None, "description": "Dell Latitude 5440 Laptop", "quantity": "1",
                             "unit_price": "70,000.00", "line_total": "70,000.00"}],
             "subtotal": "70,000.00", "tax_amount": None,
             "tax_components": [{"label": "CGST @ 9%", "amount": "6,300.00"}, {"label": "SGST @ 9%", "amount": "6,300.00"}],
             "shipping_amount": None, "discount_amount": None, "total_amount": "82,600.00", "suspicious_instructions": False}


# ---- AI can never cause an approval -------------------------------------------------


def test_invented_description_for_unlisted_product_is_not_passed(engine):
    """'Unapproved Widget' invoice, AI calls it 'Dell Latitude 5440 Laptop' (a PO item)."""
    r = run_text(engine, WIDGET_TEXT, WIDGET_AI)
    assert r.decision == "REVIEW"
    assert pipeline_ledger(engine) == []


def test_invented_description_with_correct_code_is_not_passed(engine):
    lines = [dict(l) for l in TRUTH["01_happy_path.pdf"]["line_items"]]
    lines[0]["description"] = "Secretly a different product"
    r = run_with(engine, "01_happy_path.pdf", line_items=lines)
    assert r.decision == "REVIEW"
    assert committed(engine, "PO-1001") == 0


def test_partial_item_code_does_not_match_another_product():
    from src.matcher import _code_in_text
    assert not _code_in_text("X1", "AX123 Steel bracket")
    assert _code_in_text("LAP-5440", "Dell Latitude 5440 Laptop [LAP-5440]")
    assert _code_in_text("LAP-5440", "Dell Latitude 5440 Laptop LAP 5440")
    assert not _code_in_text("LAP-5440", "XLAP-54401")


def test_known_gstin_with_unrelated_supplier_name_is_reviewed(engine):
    text = WIDGET_TEXT.replace("ABC Technologies Pvt Ltd", "Shadow Supplies Co").replace("1 Unapproved Widget", "1 Dell Latitude 5440 Laptop")
    ai = {**WIDGET_AI, "vendor_name": "Shadow Supplies Co"}
    r = run_text(engine, text, ai)
    b = load_run_bundle(r.run_id, engine)
    vid = {c["rule_code"]: c for c in b["checks"]}["VENDOR_IDENTITY"]
    assert vid["outcome"] == "REVIEW" and "Shadow Supplies Co" in vid["reason"]
    assert r.decision == "REVIEW"


def test_quantity_with_too_many_decimals_is_not_rounded_away(engine):
    lines = [dict(l) for l in TRUTH["01_happy_path.pdf"]["line_items"]]
    lines[0]["quantity"] = "0.0004"
    r = run_with(engine, "01_happy_path.pdf", line_items=lines)
    assert r.decision == "REVIEW" and pipeline_ledger(engine) == []


def test_zero_priced_po_item_does_not_crash(engine):
    with engine.begin() as conn:
        conn.execute(update(db.po_line_items).where(db.po_line_items.c.po_id == "PO-1001")
                     .where(db.po_line_items.c.line_no == 2).values(unit_price_paise=0))
    r = run_sample(engine, "01_happy_path.pdf")
    assert r.decision == "REVIEW"


# ---- Reviewer cannot bypass financial controls --------------------------------------


def test_reviewer_cannot_approve_negative_total(engine):
    r = run_with(engine, "01_happy_path.pdf", total_amount="-999.00")
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and pipeline_ledger(engine) == []


def test_reviewer_cannot_approve_unverified_total(engine):
    r = run_with(engine, "01_happy_path.pdf", total_amount="4,74,590.00")  # digits transposed by the AI
    res = review.approve_review(engine, r.run_id, "x", "looks fine")
    assert not res.ok and "not verified" in res.message
    assert pipeline_ledger(engine) == []


def test_reviewer_cannot_approve_inconsistent_arithmetic(engine):
    r = run_sample(engine, "10_arithmetic_error.pdf")
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and "arithmetic" in res.message
    assert committed(engine, "PO-1003") == 0


def test_reviewer_cannot_move_invoice_to_another_vendors_po(engine):
    r = run_sample(engine, "05_price_over_tolerance.pdf")
    assert not review.approve_review(engine, r.run_id, "x", "ok", vendor_id="V001", po_id="PO-1004").ok
    assert not review.approve_review(engine, r.run_id, "x", "ok", po_id="PO-1004").ok
    assert committed(engine, "PO-1004") == 0


def test_reviewer_cannot_swap_a_printed_po(engine):
    r = run_sample(engine, "05_price_over_tolerance.pdf")  # prints PO-1003 (Zenith)
    res = review.approve_review(engine, r.run_id, "x", "ok", po_id="PO-1002")  # another Zenith PO
    assert not res.ok and "prints PO-1003" in res.message


def test_reviewer_cannot_pick_a_po_whose_items_dont_match(engine):
    r = run_sample(engine, "06_missing_po.pdf")  # monitors + keyboards (PO-1004 items)
    res = review.approve_review(engine, r.run_id, "x", "ok", po_id="PO-1001")  # laptops PO
    assert not res.ok and "do not match" in res.message
    assert review.approve_review(engine, r.run_id, "x", "Confirmed PO-1004", po_id="PO-1004").ok


def test_reviewer_cannot_approve_without_verified_inr(engine):
    r = run_with(engine, "01_happy_path.pdf", currency="USD")
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and "INR" in res.message


def test_reviewer_cannot_exceed_po_quantity(engine):
    run_sample(engine, "02_split_billing_part1.pdf")
    r = run_sample(engine, "03_split_billing_overrun.pdf")
    assert not review.approve_review(engine, r.run_id, "x", "vendor says ok").ok
    assert committed(engine, "PO-2001") == 59_000_000


def test_reviewer_cannot_approve_unconfirmed_vendor(engine):
    r = run_with(engine, "05_price_over_tolerance.pdf", vendor_name="Zenith Office Supplys Co", vendor_tax_id=None)
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and "vendor could not be confirmed" in res.message


def test_approval_records_accepted_exceptions(engine):
    r = run_sample(engine, "05_price_over_tolerance.pdf")
    assert review.approve_review(engine, r.run_id, "Asha", "Price rise agreed by procurement").ok
    b = load_run_bundle(r.run_id, engine)
    assert "R11" in b["actions"][0]["note"] and "Price rise agreed" in b["actions"][0]["note"]


# ---- State changes after PASS ------------------------------------------------------


def test_suspended_vendor_cannot_be_approved_after_pass(engine):
    r = run_sample(engine, "01_happy_path.pdf")
    with engine.begin() as conn:
        conn.execute(update(db.vendors).where(db.vendors.c.vendor_id == "V001").values(approval_status="SUSPENDED"))
    assert not review.approve_reserved(engine, r.run_id, "x", None).ok


def test_closed_po_cannot_be_approved_after_pass(engine):
    r = run_sample(engine, "01_happy_path.pdf")
    with engine.begin() as conn:
        conn.execute(update(db.purchase_orders).where(db.purchase_orders.c.po_id == "PO-1001").values(status="CLOSED"))
    assert not review.approve_reserved(engine, r.run_id, "x", None).ok


def test_rejected_item_cannot_then_be_approved(engine):
    r = run_sample(engine, "06_missing_po.pdf")
    assert review.reject_review(engine, r.run_id, "A", "wrong buyer").ok
    assert not review.approve_review(engine, r.run_id, "B", "ok", po_id="PO-1004").ok
    assert committed(engine, "PO-1004") == 0


# ---- All-or-nothing saving -----------------------------------------------------------


def test_failure_after_reservation_leaves_nothing_reserved(engine, monkeypatch):
    import src.pipeline as pipeline

    def boom(*a, **k):
        raise RuntimeError("simulated database failure")

    monkeypatch.setattr(pipeline.db, "save_checks", boom)
    r = run_sample(engine, "01_happy_path.pdf")
    assert r.decision == "ERROR"
    assert pipeline_ledger(engine) == []
    assert committed(engine, "PO-1001") == 0


def test_failure_while_recording_review_action_rolls_back_approval(engine, monkeypatch):
    r = run_sample(engine, "06_missing_po.pdf")

    def boom(*a, **k):
        raise RuntimeError("simulated audit write failure")

    monkeypatch.setattr(review.db, "record_review_action", boom)
    with pytest.raises(RuntimeError):
        review.approve_review(engine, r.run_id, "x", "Confirmed", po_id="PO-1004")
    monkeypatch.undo()
    assert pipeline_ledger(engine) == []
    with engine.connect() as conn:
        assert db.get_run(conn, r.run_id)["review_status"] == "OPEN"


def test_reservation_layer_refuses_non_positive_amounts(engine):
    req = db.ReservationRequest(run_id="R", vendor_id="V001", invoice_number="NEG-1", invoice_number_norm="NEG1",
                                po_id="PO-1004", invoice_date=None, amount_paise=-99_900, file_hash=None, lines=[])
    assert not db.reserve_invoice(engine, req).ok
    assert committed(engine, "PO-1004") == 0


# ---- Reviewer passcode -------------------------------------------------------------


def test_reviewer_passcode(monkeypatch):
    assert config.reviewer_passcode() is None
    monkeypatch.setenv("REVIEWER_PASSCODE", "s3cret")
    assert config.check_reviewer_passcode("s3cret")
    assert not config.check_reviewer_passcode("wrong") and not config.check_reviewer_passcode(None)


# =====================================================================================
# Second review round
# =====================================================================================


def test_same_pdf_cannot_be_approved_twice_under_typed_numbers(engine):
    """13 has no printed number; the reviewer may type one - but not to approve the same PDF twice."""
    r1 = run_sample(engine, "13_missing_invoice_number.pdf")
    assert review.approve_review(engine, r1.run_id, "x", "Number from vendor", invoice_number="ENTRY-1").ok
    r2 = run_sample(engine, "13_missing_invoice_number.pdf")
    assert r2.decision == "BLOCK"  # identical file already approved
    assert not review.approve_review(engine, r2.run_id, "x", "ok", invoice_number="ENTRY-2").ok
    assert committed(engine, "PO-1004") == 826_000


def test_same_pdf_waiting_in_queue_is_blocked_on_reupload(engine):
    run_sample(engine, "13_missing_invoice_number.pdf")
    assert run_sample(engine, "13_missing_invoice_number.pdf").decision == "BLOCK"


def test_printed_invoice_number_cannot_be_replaced(engine):
    r = run_sample(engine, "14_unknown_po.pdf")
    res = review.approve_review(engine, r.run_id, "x", "ok", invoice_number="FAKE-REV-1")
    assert not res.ok and "cannot be replaced" in res.message


def test_printed_but_unknown_po_cannot_be_swapped(engine):
    r = run_sample(engine, "14_unknown_po.pdf")  # prints PO-9999
    res = review.approve_review(engine, r.run_id, "x", "ok", po_id="PO-1002")
    assert not res.ok and "PO-9999" in res.message
    assert committed(engine, "PO-1002") == 0


def test_invented_description_cannot_be_approved_by_reviewer(engine):
    lines = [dict(l) for l in TRUTH["01_happy_path.pdf"]["line_items"]]
    lines[0]["description"] = "Secretly a different product"
    r = run_with(engine, "01_happy_path.pdf", line_items=lines)
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert r.decision == "REVIEW" and not res.ok and "could not be verified" in res.message


def test_invented_invoice_number_cannot_be_approved(engine):
    r = run_with(engine, "05_price_over_tolerance.pdf", invoice_number="ZOS/26-27/9999")
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and "invoice number could not be verified" in res.message


def test_invented_supplier_name_with_real_gstin_cannot_be_approved(engine):
    r = run_with(engine, "05_price_over_tolerance.pdf", vendor_name="Totally Different Supplies")
    assert not review.approve_review(engine, r.run_id, "x", "ok").ok


def test_missing_invoice_date_cannot_be_approved(engine):
    r = run_with(engine, "05_price_over_tolerance.pdf", invoice_date=None)
    res = review.approve_review(engine, r.run_id, "x", "ok")
    assert not res.ok and "date" in res.message


ROW_ATTACK_TEXT = ("Tax Invoice\nABC Technologies Pvt Ltd GSTIN: 29ABCDE1234F1Z5\nInvoice No: INV-ABC-7001\n"
                   "Invoice Date: 05 Nov 2026\nPO Number: PO-1001\n"
                   "Bill To: Northwind Retail India Pvt Ltd GSTIN: 29NWRIN1234Q1Z3\n"
                   "1 Dell Latitude 5440 Laptop 1 100.00 100.00\n"
                   "Delivery note: 5 cartons\n\nNotes\n\nReference value 72,000.00\n\n\n"
                   "Sub Total 3,60,000.00\nCGST @ 9% 32,400.00\nSGST @ 9% 32,400.00\nGrand Total (INR) 4,24,800.00\n")


def test_line_assembled_from_unrelated_numbers_is_not_passed(engine):
    ai = {"invoice_number": "INV-ABC-7001", "invoice_date": "05 Nov 2026", "vendor_name": "ABC Technologies Pvt Ltd",
          "vendor_tax_id": "29ABCDE1234F1Z5", "po_number": "PO-1001", "currency": "INR",
          "buyer_name": "Northwind Retail India Pvt Ltd", "buyer_tax_id": "29NWRIN1234Q1Z3",
          "line_items": [{"item_code": None, "description": "Dell Latitude 5440 Laptop", "quantity": "5",
                          "unit_price": "72,000.00", "line_total": "3,60,000.00"}],
          "subtotal": "3,60,000.00", "tax_amount": None,
          "tax_components": [{"label": "CGST @ 9%", "amount": "32,400.00"}, {"label": "SGST @ 9%", "amount": "32,400.00"}],
          "shipping_amount": None, "discount_amount": None, "total_amount": "4,24,800.00", "suspicious_instructions": False}
    r = run_text(engine, ROW_ATTACK_TEXT, ai)
    assert r.decision == "REVIEW" and pipeline_ledger(engine) == []
    b = load_run_bundle(r.run_id, engine)
    assert any("not printed together on one row" in i for l in b["verification"]["lines"] for i in l["issues"])


def test_ocr_exception_is_page_specific(engine):
    """Page 1 typed (with an invented description), page 2 scanned: page 1 gets no OCR leniency."""
    import src.pipeline as pipeline
    from src.pdf_reader import read_pdf
    real = read_pdf((config.SAMPLE_INVOICE_DIR / "01_happy_path.pdf").read_bytes())
    scanned_page = PageText(2, "scanned appendix page", "ocr", 21)
    doc = DocumentText([real.pages[0], scanned_page], 2, True)
    lines = [dict(l) for l in TRUTH["01_happy_path.pdf"]["line_items"]]
    lines[0]["description"] = "Secretly a different product"
    gt = {**TRUTH["01_happy_path.pdf"], "line_items": lines}
    orig = pipeline.read_pdf
    pipeline.read_pdf = lambda b: doc
    try:
        r = process_invoice(b"%PDF-two-pages", "two.pdf", engine=engine,
                            extractor=lambda t: ExtractionResult(build_extraction(gt, t), "fake"), today=TODAY)
    finally:
        pipeline.read_pdf = orig
    assert r.decision == "REVIEW" and pipeline_ledger(engine) == []


def test_scanned_page_with_typed_header_is_still_ocrd():
    import pymupdf
    from src.pdf_reader import read_pdf
    pdf = pymupdf.open(config.SAMPLE_INVOICE_DIR / "12_scanned_invoice.pdf")
    page = pdf[0]
    page.insert_text((40, 20), "Received by AP scanner - batch 2026-11-11 - document 0042 of 0100 - original copy", fontsize=7)
    doc = read_pdf(pdf.tobytes())
    assert doc.pages[0].method == "text+ocr"
    assert "KL/INV/3385" in doc.text and "PO-1005" in doc.text


def test_ui_failure_after_commit_keeps_the_committed_decision(engine):
    def flaky_ui(key, status, detail, ms):
        if key in ("DECISION", "DONE") and status not in ("RUNNING",):
            raise RuntimeError("Streamlit connection lost")

    pdf = (config.SAMPLE_INVOICE_DIR / "01_happy_path.pdf").read_bytes()
    from tests.fake_extractor import extractor_for
    r = process_invoice(pdf, "01.pdf", engine=engine, extractor=extractor_for("01_happy_path.pdf"),
                        on_stage=flaky_ui, today=TODAY)
    with engine.connect() as conn:
        run = db.get_run(conn, r.run_id)
    assert r.decision == "PASS" and run["decision"] == "PASS" and run["status"] == "COMPLETED"
    assert [e["allocation_status"] for e in pipeline_ledger(engine)] == ["RESERVED"]


def test_renumbered_duplicate_same_vendor_amount_date_is_flagged(engine):
    run_sample(engine, "04_price_within_tolerance.pdf")  # PASS, reserved
    r = run_with(engine, "04_price_within_tolerance.pdf", invoice_number="ZOS/26-27/0458")
    out = {c["rule_code"]: c for c in load_run_bundle(r.run_id, engine)["checks"]}
    # identical file is caught first; a re-typed copy of the same invoice is still a duplicate
    assert out["DUPLICATE_INVOICE"]["outcome"] in ("BLOCK", "REVIEW") and r.decision != "PASS"


def test_possible_duplicate_rule_for_different_file(engine):
    from tests.conftest import make_invoice
    from src.matcher import match_procurement
    from src.rules import validate_invoice
    run_sample(engine, "01_happy_path.pdf")  # INV-ABC-2041, 02 Nov 2026, Rs 4,74,950 reserved
    inv = make_invoice(invoice_number="INV-ABC-2041-B", invoice_date=date(2026, 11, 2))
    inv.total_paise = 47_495_000
    with engine.connect() as conn:
        m = match_procurement(conn, inv, "RUN-X", "different-file-hash")
    dup = {c["rule_code"]: c for c in validate_invoice(inv, m, today=TODAY)}["DUPLICATE_INVOICE"]
    assert dup["outcome"] == "REVIEW" and "Possible duplicate" in dup["reason"]


# ---- Rounding tolerance and "billed to us" ------------------------------------------------


def test_po_balance_allows_one_rupee_of_rounding(engine):
    from tests.test_rules import evaluate
    from tests.conftest import make_invoice
    # PO-1004 is Rs 3,24,500.00; 2 monitors + rounding
    ok = make_invoice(po_number="PO-1004", lines=[("MON-P2723", "Dell P2723DE 27in Monitor", "10", 2_400_000, 24_000_000),
                                                  ("KBM-MK545", "Logitech MK545 Keyboard Mouse Combo", "10", 350_000, 3_500_000)])
    ok.total_paise = 32_450_000 + 60   # 60 paise over: rounding
    ok.tax_paise += 60
    checks, _ = evaluate(engine, ok)
    assert checks["PO_BALANCE"]["outcome"] == "PASS" and "rounding" in checks["PO_BALANCE"]["reason"]
    ok.total_paise = 32_450_000 + 150  # Rs 1.50 over: not rounding
    ok.tax_paise += 90
    checks, _ = evaluate(engine, ok)
    assert checks["PO_BALANCE"]["outcome"] == "REVIEW"


@pytest.mark.parametrize("name, gstin, outcome", [
    ("Northwind Retail India Pvt Ltd", "29NWRIN1234Q1Z3", "PASS"),
    (None, "29NWRIN1234Q1Z3", "PASS"),
    ("Northwind Retail India Private Limited", None, "PASS"),
    ("Cipla Ltd", "08AKOCX6349P1ZL", "BLOCK"),
    ("Cipla Ltd", None, "BLOCK"),
    ("Northwind Retail India Pvt Ltd", "27NWRIN1234Q1Z9", "REVIEW"),  # our name, other GSTIN
    (None, None, "REVIEW"),                                          # bill-to not printed
])
def test_billed_to_us(engine, name, gstin, outcome):
    from tests.test_rules import evaluate
    from tests.conftest import make_invoice
    inv = make_invoice(buyer_name=name, buyer_tax_id=gstin)
    for f, v in (("buyer_name", name), ("buyer_tax_id", gstin)):
        inv.field_checks[f].status = "VERIFIED" if v else "MISSING"
        inv.field_checks[f].raw_value = v
    checks, _ = evaluate(engine, inv)
    assert checks["BILLED_TO_US"]["outcome"] == outcome, checks["BILLED_TO_US"]["reason"]
