from __future__ import annotations

import os

from datetime import date
from decimal import Decimal

import pytest

from src import database as db
from src.evidence_validator import FieldCheck, ParsedLine, VerifiedInvoice
from src.pipeline import process_invoice
from src import config
from tests.fake_extractor import extractor_for

TODAY = date(2026, 11, 9)


# Settings that would make a test talk to a real AI provider. A developer's .env is
# loaded on import, so clear them for every test: tests must never make real API calls.
_PROVIDER_SETTINGS = (
    "LLM_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
    "GEMINI_THINKING_LEVEL", "GROQ_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL", "GROQ_REASONING_EFFORT",
    "REVIEWER_PASSCODE",
)


@pytest.fixture(autouse=True)
def _no_real_ai_providers(monkeypatch):
    for name in _PROVIDER_SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine(tmp_path):
    """A fresh, seeded database per test.

    SQLite by default. Set TEST_DATABASE_URL=postgresql://... to run the whole
    suite against PostgreSQL (the deployment database) instead.
    """
    db.get_engine.cache_clear()
    url = os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path / 'test.db'}"
    eng = db.get_engine(url)
    db.metadata.drop_all(eng)
    db.reset_demo(eng)
    yield eng
    eng.dispose()
    db.get_engine.cache_clear()


def run_sample(engine, file_name: str, **kw):
    pdf = (config.SAMPLE_INVOICE_DIR / file_name).read_bytes()
    return process_invoice(pdf, file_name, engine=engine, extractor=extractor_for(file_name, **kw), today=TODAY)


def make_invoice(
    lines: list[tuple[str | None, str, str, int, int]] | None = None,
    **overrides,
) -> VerifiedInvoice:
    """Build a fully verified invoice for rule tests.

    lines: (item_code, description, quantity, unit_price_paise, line_total_paise)
    """
    lines = lines if lines is not None else [("LAP-5440", "Dell Latitude 5440 Laptop", "5", 7_200_000, 36_000_000)]
    parsed = [
        ParsedLine(i, code, desc, Decimal(q), unit, total, {}, True, [])
        for i, (code, desc, q, unit, total) in enumerate(lines, start=1)
    ]
    subtotal = sum(l.line_total_paise for l in parsed)
    tax = round(subtotal * 18 / 100)
    values = dict(
        invoice_number="INV-T-1",
        invoice_date=date(2026, 11, 5),
        vendor_name="ABC Technologies Pvt Ltd",
        vendor_tax_id="29ABCDE1234F1Z5",
        po_number="PO-1001",
        currency="INR",
        subtotal_paise=subtotal,
        tax_paise=tax,
        tax_source="single",
        shipping_paise=0,
        discount_paise=0,
        total_paise=subtotal + tax,
        suspicious_instructions=False,
        buyer_name="Northwind Retail India Pvt Ltd",
        buyer_tax_id="29NWRIN1234Q1Z3",
    )
    values.update(overrides)
    checks = {}
    for f in config.EVIDENCE_FIELDS + ("currency",):
        key = {"subtotal": "subtotal_paise", "tax_amount": "tax_paise", "total_amount": "total_paise",
               "shipping_amount": "shipping_paise", "discount_amount": "discount_paise",
               "invoice_date": "invoice_date"}.get(f, f)
        v = values.get(key)
        present = v not in (None, 0) or f in ("total_amount",)
        checks[f] = FieldCheck(f, str(v) if present and v is not None else None,
                               "VERIFIED" if present and v is not None else "MISSING")
    return VerifiedInvoice(
        field_checks=checks,
        invoice_number=values["invoice_number"],
        invoice_number_norm=db.normalize_invoice_number(values["invoice_number"]),
        invoice_date=values["invoice_date"],
        vendor_name=values["vendor_name"],
        vendor_tax_id=values["vendor_tax_id"],
        po_number=values["po_number"],
        currency=values["currency"],
        buyer_name=values["buyer_name"],
        buyer_tax_id=values["buyer_tax_id"],
        subtotal_paise=values["subtotal_paise"],
        tax_paise=values["tax_paise"],
        tax_source=values["tax_source"],
        shipping_paise=values["shipping_paise"],
        discount_paise=values["discount_paise"],
        total_paise=values["total_paise"],
        lines=parsed,
        ambiguous_fields=[],
        suspicious_instructions=values["suspicious_instructions"],
        parse_errors=[],
    )
