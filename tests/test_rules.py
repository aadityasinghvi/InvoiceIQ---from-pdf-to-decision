"""Business rules and decision logic. No AI, no PDFs: verified data + seeded database."""

from datetime import date

import pytest

from src.decision_engine import IncompleteValidationError, make_decision
from src.matcher import match_procurement
from src.rules import (
    REQUIRED_RULE_CODES,
    check_po_balance_simple,
    validate_invoice,
)
from tests.conftest import TODAY, make_invoice


def evaluate(engine, inv):
    with engine.connect() as conn:
        m = match_procurement(conn, inv, "RUN-TEST", None)
    checks = validate_invoice(inv, m, today=TODAY)
    return {c["rule_code"]: c for c in checks}, make_decision(checks)


def test_simple_balance_rule():
    assert check_po_balance_simple(600_000, 600_000)["passed"] is True
    assert check_po_balance_simple(700_000, 600_000)["passed"] is False


def test_clean_invoice_passes_every_rule(engine):
    checks, decision = evaluate(engine, make_invoice())
    assert decision == "PASS", {k: c["reason"] for k, c in checks.items() if c["outcome"] != "PASS"}
    assert set(checks) == set(REQUIRED_RULE_CODES)


def test_absolute_cap_applies_even_within_percent(engine):
    # Laptop Rs 72,000 -> Rs 73,440 is exactly +2%, but Rs 1,440/unit breaches the Rs 500/unit cap.
    inv = make_invoice(lines=[("LAP-5440", "Dell Latitude 5440 Laptop", "1", 7_344_000, 7_344_000)])
    checks, _ = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "REVIEW"


def test_price_within_both_tolerances_passes(engine):
    # Toner: Rs 3,200 on PO; Rs 3,264 is +2.0% and Rs 64 over -> within both limits
    inv = make_invoice(po_number="PO-1002", vendor_name="Zenith Office Supplies", vendor_tax_id="27ZENTH5678K1Z2",
                       lines=[("TNR-HP-26A", "HP 26A Toner Cartridge", "1", 326_400, 326_400)])
    checks, decision = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "PASS" and "within the 2.0% tolerance" in checks["LINE_ITEMS"]["reason"]
    assert decision == "PASS"


def test_price_one_paisa_over_percent_tolerance_reviews(engine):
    inv = make_invoice(po_number="PO-1002", vendor_name="Zenith Office Supplies", vendor_tax_id="27ZENTH5678K1Z2",
                       lines=[("TNR-HP-26A", "HP 26A Toner Cartridge", "1", 326_401, 326_401)])
    checks, decision = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "REVIEW" and decision == "REVIEW"


def test_lower_price_is_fine(engine):
    inv = make_invoice(lines=[("LAP-5440", "Dell Latitude 5440 Laptop", "1", 7_000_000, 7_000_000)])
    checks, _ = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "PASS"


def test_quantity_beyond_po_reviews(engine):
    inv = make_invoice(lines=[("LAP-5440", "Dell Latitude 5440 Laptop", "6", 7_200_000, 43_200_000)])
    checks, _ = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "REVIEW"
    assert "only 5 of 5 remain" in checks["LINE_ITEMS"]["reason"]


def test_unknown_item_reviews_with_suggestion(engine):
    inv = make_invoice(lines=[(None, "Dell Latitude 5440 Laptops", "1", 7_200_000, 7_200_000)])
    checks, _ = evaluate(engine, inv)
    assert checks["LINE_ITEMS"]["outcome"] == "REVIEW"
    assert "closest PO item" in checks["LINE_ITEMS"]["reason"]


def test_arithmetic_mismatch_reviews(engine):
    inv = make_invoice()
    inv.total_paise += 100_000  # Rs 1,000 extra
    checks, decision = evaluate(engine, inv)
    assert checks["ARITHMETIC"]["outcome"] == "REVIEW" and decision == "REVIEW"


def test_rounding_within_one_rupee_is_tolerated(engine):
    inv = make_invoice()
    inv.total_paise += 50
    inv.tax_paise += 50
    checks, _ = evaluate(engine, inv)
    assert checks["ARITHMETIC"]["outcome"] == "PASS" and checks["TAX"]["outcome"] == "PASS"


def test_wrong_tax_rate_reviews(engine):
    inv = make_invoice()
    inv.tax_paise = inv.subtotal_paise * 12 // 100
    inv.total_paise = inv.subtotal_paise + inv.tax_paise
    checks, _ = evaluate(engine, inv)
    assert checks["TAX"]["outcome"] == "REVIEW"


def test_unapproved_vendor_blocks(engine):
    inv = make_invoice(vendor_name="Orion Traders", vendor_tax_id="07ORION9012L1Z8", po_number="PO-1006",
                       lines=[("STL-RCK-5", "Steel Storage Rack 5-shelf", "1", 720_000, 720_000)])
    checks, decision = evaluate(engine, inv)
    assert checks["VENDOR_APPROVED"]["outcome"] == "BLOCK" and decision == "BLOCK"


def test_other_vendors_po_blocks(engine):
    inv = make_invoice(vendor_name="Zenith Office Supplies", vendor_tax_id="27ZENTH5678K1Z2")
    checks, decision = evaluate(engine, inv)
    assert checks["PO_VENDOR_MATCH"]["outcome"] == "BLOCK" and decision == "BLOCK"


def test_gstin_and_name_disagree_reviews(engine):
    inv = make_invoice(vendor_name="Zenith Office Supplies")  # GSTIN is ABC's
    checks, decision = evaluate(engine, inv)
    assert checks["VENDOR_IDENTITY"]["outcome"] == "REVIEW"
    assert decision == "REVIEW"


def test_fuzzy_vendor_name_is_only_a_suggestion(engine):
    inv = make_invoice(vendor_name="ABC Technology Solutions", vendor_tax_id=None)
    checks, decision = evaluate(engine, inv)
    assert checks["VENDOR_IDENTITY"]["outcome"] == "REVIEW"
    assert checks["VENDOR_APPROVED"]["outcome"] == "NOT_EVALUATED"
    assert decision == "REVIEW"


def test_alias_name_confirms_vendor(engine):
    inv = make_invoice(vendor_name="ABC Technologies Private Limited", vendor_tax_id=None)
    checks, _ = evaluate(engine, inv)
    assert checks["VENDOR_IDENTITY"]["outcome"] == "PASS"


def test_missing_po_suggests_candidates_but_never_assigns(engine):
    inv = make_invoice(po_number=None, lines=[("MON-P2723", "Dell P2723DE 27in Monitor", "2", 2_400_000, 4_800_000)])
    checks, decision = evaluate(engine, inv)
    assert decision == "REVIEW"
    cands = checks["PO_EXISTS"]["details"]["candidates"]
    assert cands[0]["po_id"] == "PO-1004"
    assert checks["PO_BALANCE"]["outcome"] == "NOT_EVALUATED"


def test_shipping_not_on_po_reviews(engine):
    inv = make_invoice()
    inv.shipping_paise = 150_000
    inv.total_paise += 150_000
    checks, _ = evaluate(engine, inv)
    assert checks["NON_PO_CHARGES"]["outcome"] == "REVIEW"


def test_future_dated_invoice_reviews(engine):
    checks, _ = evaluate(engine, make_invoice(invoice_date=date(2026, 12, 25)))
    assert checks["DATA_FORMAT"]["outcome"] == "REVIEW"


def test_foreign_currency_reviews(engine):
    checks, _ = evaluate(engine, make_invoice(currency="USD"))
    assert checks["CURRENCY"]["outcome"] == "REVIEW"


def test_embedded_instructions_review(engine):
    checks, decision = evaluate(engine, make_invoice(suspicious_instructions=True))
    assert checks["EMBEDDED_INSTRUCTIONS"]["outcome"] == "REVIEW" and decision == "REVIEW"


def test_same_inputs_same_state_same_result(engine):
    inv = make_invoice(lines=[("LAP-5440", "Dell Latitude 5440 Laptop", "6", 7_200_000, 43_200_000)])
    results = [evaluate(engine, inv) for _ in range(5)]
    assert len({(d, tuple((k, c["outcome"], c["reason"]) for k, c in sorted(ch.items()))) for ch, d in results}) == 1


# ---- decision engine ------------------------------------------------------


def _checks(**outcomes):
    return [{"rule_code": c, "outcome": outcomes.get(c, "PASS")} for c in REQUIRED_RULE_CODES]


def test_block_beats_review():
    assert make_decision(_checks(PO_BALANCE="REVIEW", DUPLICATE_INVOICE="BLOCK")) == "BLOCK"


def test_not_evaluated_never_passes():
    assert make_decision(_checks(TAX="NOT_EVALUATED")) == "REVIEW"


def test_missing_rule_cannot_pass():
    checks = [c for c in _checks() if c["rule_code"] != "PO_BALANCE"]
    with pytest.raises(IncompleteValidationError, match="PO_BALANCE"):
        make_decision(checks)


def test_unknown_outcome_rejected():
    with pytest.raises(IncompleteValidationError):
        make_decision(_checks(TAX="MAYBE"))


def test_foreign_currency_invoice_is_never_compared_with_inr_po(engine):
    # A USD invoice that quotes an INR PO: price, tax and balance checks must not compare $ with ₹.
    checks, decision = evaluate(engine, make_invoice(currency="USD"))
    assert checks["CURRENCY"]["outcome"] == "REVIEW"
    for code in ("LINE_ITEMS", "TAX", "PO_BALANCE"):
        assert checks[code]["outcome"] == "NOT_EVALUATED" and "USD" in checks[code]["reason"]
    assert decision == "REVIEW"


def test_headline_leads_with_vendor_and_po_problems():
    from src.reporting import build_explanation
    checks = [
        {"rule_code": "DATA_FORMAT", "rule_number": "R03", "rule_name": "x", "outcome": "REVIEW", "reason": "old date"},
        {"rule_code": "CURRENCY", "rule_number": "R09", "rule_name": "x", "outcome": "REVIEW", "reason": "USD"},
        {"rule_code": "VENDOR_IDENTITY", "rule_number": "R04", "rule_name": "x", "outcome": "REVIEW", "reason": "unknown"},
    ]
    headline, _ = build_explanation("REVIEW", checks, "INV-1", "Acme", None, 9_350, currency="USD")
    assert "for $93.50" in headline
    assert headline.index("vendor could not be confirmed") < headline.index("and 1 more")
