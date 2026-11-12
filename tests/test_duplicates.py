"""Duplicate protection and atomic reservations, including concurrent submissions."""

import threading

from sqlalchemy import func, select

from src import database as db
from tests.conftest import run_sample


def active_ledger_count(engine, vendor_id, norm):
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(db.invoice_ledger)
            .where(db.invoice_ledger.c.vendor_id == vendor_id)
            .where(db.invoice_ledger.c.invoice_number_norm == norm)
            .where(db.invoice_ledger.c.allocation_status.in_(db.ACTIVE))
        ).scalar_one()


def test_invoice_number_normalisation():
    assert db.normalize_invoice_number("INV/ABC/2041") == db.normalize_invoice_number(" inv-abc-2041 ") == "INVABC2041"


def test_identical_reupload_is_blocked_and_reserves_nothing(engine):
    first = run_sample(engine, "01_happy_path.pdf")
    second = run_sample(engine, "01_happy_path.pdf")
    assert (first.decision, second.decision) == ("PASS", "BLOCK")
    assert active_ledger_count(engine, "V001", "INVABC2041") == 1
    with engine.connect() as conn:
        assert db.po_committed_paise(conn, "PO-1001") == 47_495_000
        checks = {c["rule_code"]: c for c in db.get_run_checks(conn, second.run_id)}
    assert "This exact file was already reserved" in checks["DUPLICATE_INVOICE"]["reason"]


def test_resend_with_different_bytes_and_formatting_is_blocked(engine):
    run_sample(engine, "01_happy_path.pdf")
    resend = run_sample(engine, "07_duplicate_resend.pdf")
    assert resend.decision == "BLOCK"
    assert active_ledger_count(engine, "V001", "INVABC2041") == 1


def test_duplicate_of_item_waiting_in_review_is_blocked(engine):
    a = run_sample(engine, "05_price_over_tolerance.pdf")
    b = run_sample(engine, "05_price_over_tolerance.pdf")
    assert (a.decision, b.decision) == ("REVIEW", "BLOCK")


def _request(run_id, number, amount, qty=1):
    return db.ReservationRequest(
        run_id=run_id, vendor_id="V001", invoice_number=number,
        invoice_number_norm=db.normalize_invoice_number(number), po_id="PO-1004", invoice_date="2026-11-05",
        amount_paise=amount, file_hash=None, lines=[{"line_no": 1, "quantity_milli": qty * 1000, "amount_paise": amount}],
    )


def _race(engine, requests):
    results = [None] * len(requests)
    barrier = threading.Barrier(len(requests))

    def worker(i, req):
        barrier.wait()
        results[i] = db.reserve_invoice(engine, req)

    threads = [threading.Thread(target=worker, args=(i, r)) for i, r in enumerate(requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_same_invoice_reserves_once(engine):
    results = _race(engine, [_request(f"R{i}", "INV-RACE-1", 2_000_000) for i in range(6)])
    assert sum(r.ok for r in results) == 1
    assert all(r.conflict == "DUPLICATE" for r in results if not r.ok)
    assert active_ledger_count(engine, "V001", "INVRACE1") == 1


def test_concurrent_different_invoices_cannot_overspend_po(engine):
    # PO-1004 is Rs 3,24,500. Five invoices of Rs 1,00,000 (1 monitor each) race; only 3 can fit.
    results = _race(engine, [_request(f"R{i}", f"INV-RACE-{i}", 10_000_000) for i in range(5)])
    assert sum(r.ok for r in results) == 3
    with engine.connect() as conn:
        assert db.po_committed_paise(conn, "PO-1004") == 30_000_000
        assert db.po_committed_paise(conn, "PO-1004") <= 32_450_000


def test_unique_index_is_the_last_line_of_defence(engine):
    req = _request("R1", "INV-IDX-1", 1_000)
    assert db.reserve_invoice(engine, req).ok
    from sqlalchemy import insert
    from sqlalchemy.exc import IntegrityError
    import pytest
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(insert(db.invoice_ledger).values(
                vendor_id="V001", invoice_number="inv idx 1", invoice_number_norm="INVIDX1", po_id="PO-1004",
                amount_paise=1_000, allocation_status="RESERVED", source="TEST", created_at="x", updated_at="x"))
