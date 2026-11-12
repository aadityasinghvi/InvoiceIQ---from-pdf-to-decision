from decimal import Decimal

import pytest

from src.money import AmountError, format_inr, format_inr_short, parse_amount, parse_quantity, to_paise


@pytest.mark.parametrize(
    "raw, paise",
    [
        ("11,800.00", 1_180_000),
        ("1,18,000.00", 11_800_000),      # Indian grouping
        ("118,000.00", 11_800_000),       # Western grouping
        ("INR 4,500", 450_000),
        ("Rs. 250.50", 25_050),
        ("₹97,350.00", 9_735_000),
        ("₹ 1,000/-", 100_000),
        ("0.01", 1),
    ],
)
def test_to_paise_accepts_real_invoice_formats(raw, paise):
    assert to_paise(raw) == paise


@pytest.mark.parametrize("raw", ["12.345", "abc", "1,2,3", "NaN", "Infinity", "12..5", "--5"])
def test_to_paise_rejects_ambiguous_values(raw):
    with pytest.raises(AmountError):
        to_paise(raw)


def test_none_and_empty_are_missing_not_zero():
    assert to_paise(None) is None
    assert parse_amount("  ") is None


def test_decimal_exactness():
    # 0.1 + 0.2 style float errors cannot happen in paise
    assert to_paise("0.10") + to_paise("0.20") == to_paise("0.30")


def test_quantity_with_units():
    assert parse_quantity("20 Nos") == Decimal("20")
    assert parse_quantity("2.5") == Decimal("2.5")


def test_indian_formatting():
    assert format_inr(47_495_000) == "₹4,74,950.00"
    assert format_inr(12_345) == "₹123.45"
    assert format_inr(-100) == "-₹1.00"
    assert format_inr_short(77_500_000) == "₹7.75 L"


@pytest.mark.parametrize("raw, minor", [("$85.00", 8_500), ("US$ 1,200.50", 120_050), ("USD 93.50", 9_350),
                                        ("€19.99", 1_999), ("£ 7.00", 700), ("EUR 1,000", 100_000)])
def test_foreign_currency_amounts_parse(raw, minor):
    assert to_paise(raw) == minor


def test_amounts_formatted_in_invoice_currency():
    from src.money import format_money
    assert format_money(9_350, "USD") == "$93.50"
    assert format_money(47_495_000, "INR") == "₹4,74,950.00"
    assert format_money(120_000, "AED") == "AED 1,200.00"
    assert format_money(100, None) == "₹1.00"


@pytest.mark.parametrize("raw, qty", [("9.00 GMS", "9.00"), ("7.00 BTL", "7.00"), ("3.00 MLT", "3.00"),
                                      ("20 Nos.", "20"), ("2 months", "2"), ("12", "12")])
def test_quantities_with_units_of_measure(raw, qty):
    assert parse_quantity(raw) == Decimal(qty)
