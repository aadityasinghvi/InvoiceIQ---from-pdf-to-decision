"""Exact money handling.

Amounts are stored as integer paise in the database and handled as Decimal in
Python. Floats never touch a financial comparison.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_CURRENCY_TOKENS = re.compile(r"(?i)(₹|\binr\b|\brs\.?|/-|us\$|\busd\b|\$|€|\beur\b|£|\bgbp\b)")


class AmountError(ValueError):
    """Raised when text cannot be read as a valid money amount."""


def parse_amount(value: str | None) -> Decimal | None:
    """Parse an amount exactly as written on an invoice.

    Accepts '11,800.00', '1,18,000', 'INR 4,500', 'Rs. 250.50', '₹ 1,000/-'.
    Returns None for None/empty. Raises AmountError for anything ambiguous.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1]
    text = _CURRENCY_TOKENS.sub("", text).replace(" ", "").replace(" ", "")
    if text.startswith("-"):
        negative, text = True, text[1:]
    if not re.fullmatch(r"\d{1,3}(,\d{2,3})*(\.\d+)?|\d+(\.\d+)?", text):
        raise AmountError(f"Not a valid amount: {value!r}")
    text = text.replace(",", "")
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - regex already guards
        raise AmountError(f"Not a valid amount: {value!r}") from exc
    if not amount.is_finite():
        raise AmountError(f"Not a finite amount: {value!r}")
    return -amount if negative else amount


def to_paise(value: str | Decimal | int | None) -> int | None:
    """Convert an amount to integer paise. Rejects more than 2 decimal places."""
    if value is None:
        return None
    amount = value if isinstance(value, Decimal) else parse_amount(str(value))
    if amount is None:
        return None
    paise = amount * 100
    if paise != paise.to_integral_value():
        raise AmountError(f"More than two decimal places: {value!r}")
    return int(paise)


def parse_quantity(value: str | None) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace(",", "")
    # Drop a trailing unit of measure: "9.00 GMS", "7 BTL", "20 Nos.", "3.00 MLT", "2 months".
    text = re.sub(r"\s*[A-Za-z][A-Za-z./]*\.?$", "", text)
    try:
        qty = Decimal(text)
    except InvalidOperation as exc:
        raise AmountError(f"Not a valid quantity: {value!r}") from exc
    if not qty.is_finite():
        raise AmountError(f"Not a valid quantity: {value!r}")
    return qty


def format_inr(paise: int | None, symbol: str = "₹") -> str:
    """Format paise in Indian digit grouping: 47495000 -> '₹4,74,950.00'."""
    if paise is None:
        return "—"
    sign = "-" if paise < 0 else ""
    paise = abs(int(paise))
    rupees, p = divmod(paise, 100)
    s = str(rupees)
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        s = ",".join(groups) + "," + tail
    return f"{sign}{symbol}{s}.{p:02d}"


def format_inr_short(paise: int | None) -> str:
    """Compact Indian notation for headline numbers: ₹7.75 L, ₹1.20 Cr."""
    if paise is None:
        return "—"
    rupees = paise / 100
    if abs(rupees) >= 1_00_00_000:
        return f"₹{rupees / 1_00_00_000:.2f} Cr"
    if abs(rupees) >= 1_00_000:
        return f"₹{rupees / 1_00_000:.2f} L"
    return format_inr(paise).split(".")[0]


_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


def format_money(paise: int | None, currency: str | None = "INR") -> str:
    """Format minor units in the invoice's own currency: ₹4,74,950.00 / $93.50 / AED 1,200.00."""
    if paise is None:
        return "—"
    if not currency or currency == "INR":
        return format_inr(paise)
    sign = "-" if paise < 0 else ""
    major, minor = divmod(abs(int(paise)), 100)
    prefix = _SYMBOLS.get(currency, f"{currency} ")
    return f"{sign}{prefix}{major:,}.{minor:02d}"
