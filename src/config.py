"""Central configuration and the explicit business policy InvoiceIQ implements.

Every threshold the rules engine uses lives here, so the policy can be read,
reviewed and changed in one place. POLICY_VERSION is stored with every run so
an old decision can always be traced back to the rules that produced it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_INVOICE_DIR = DATA_DIR / "sample_invoices"
# English OCR language data shipped with the project (PyMuPDF has the OCR engine built in).
BUNDLED_TESSDATA_DIR = PROJECT_ROOT / "tessdata"

# override=True: the .env file always wins over leftover values in the terminal (VS Code can inject
# an old copy of .env when a terminal opens). On Streamlit Cloud there is no .env; secrets are used.
load_dotenv(PROJECT_ROOT / ".env", override=True)

# --------------------------------------------------------------------------
# Runtime settings (environment variables / Streamlit secrets)
# --------------------------------------------------------------------------


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def database_url() -> str:
    url = get_setting("DATABASE_URL", f"sqlite:///{DATA_DIR / 'invoiceiq.db'}")
    # Supabase / Heroku style URLs use "postgres://", SQLAlchemy wants "postgresql://"
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Pin the driver (psycopg 3) so behaviour doesn't depend on the SQLAlchemy version's default.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def openai_model() -> str:
    return get_setting("OPENAI_MODEL", "gpt-4.1-mini")


def gemini_model() -> str:
    return get_setting("GEMINI_MODEL", "gemini-3.8-flash")


def groq_model() -> str:
    return get_setting("GROQ_MODEL", "openai/gpt-oss-120b")


def reviewer_passcode() -> str | None:
    """If set (recommended for the hosted app), approve/reject/release/reset require this passcode."""
    return get_setting("REVIEWER_PASSCODE")


def check_reviewer_passcode(attempt: str | None) -> bool:
    import hmac

    expected = reviewer_passcode()
    return bool(expected) and hmac.compare_digest((attempt or "").encode(), expected.encode())


PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}


def _real_key(name: str) -> str | None:
    """A configured API key, ignoring template placeholders such as 'paste-your-gemini-key-here' or 'gsk_...'."""
    value = (get_setting(name) or "").strip().strip("'\"")
    if not value or value.endswith("...") or "paste-" in value.lower() or "your-" in value.lower():
        return None
    return value


def llm_provider() -> str:
    """'groq', 'gemini' or 'openai'. Explicit LLM_PROVIDER wins; otherwise the first provider with a real key
    (Groq first: it is the provider this project is tested and demonstrated on)."""
    explicit = (get_setting("LLM_PROVIDER") or "").strip().lower()
    if explicit in PROVIDER_KEYS:
        return explicit
    for provider in ("groq", "gemini", "openai"):
        if _real_key(PROVIDER_KEYS[provider]):
            return provider
    return "groq"


def extraction_model() -> str:
    return {"gemini": gemini_model, "groq": groq_model, "openai": openai_model}[llm_provider()]()


def has_llm_key() -> bool:
    return bool(_real_key(PROVIDER_KEYS[llm_provider()]))


# --------------------------------------------------------------------------
# Business policy (documented in README -> "Assumptions")
# --------------------------------------------------------------------------

POLICY_VERSION = "2026.11-v1"

SUPPORTED_CURRENCIES = {"INR"}

# Arithmetic: allowed absolute difference when reconciling
# qty x price = line total, sum(lines) = subtotal, subtotal + tax + shipping - discount = total.
ARITHMETIC_TOLERANCE_PAISE = 100  # Rs 1.00 (covers per-line rounding)

# Unit price variance vs approved PO price. Invoiced price may be up to this
# much ABOVE the approved PO price and still pass. Anything above -> REVIEW.
PRICE_TOLERANCE_PERCENT = 2.0
# ...but never more than this absolute amount per unit.
PRICE_TOLERANCE_MAX_PAISE_PER_UNIT = 50_000  # Rs 500 per unit

# Invoices are often rounded to the rupee: allow this much above the remaining PO balance.
BALANCE_ROUNDING_TOLERANCE_PAISE = 100  # Rs 1.00

# Tax on the invoice must match the tax implied by PO tax rates within this.
TAX_TOLERANCE_PAISE = 100  # Rs 1.00

# Invoices dated more than this many days in the future are suspicious.
MAX_FUTURE_DAYS = 1
# Invoices older than this are stale and need a human look.
MAX_INVOICE_AGE_DAYS = 365

# PO statuses that can accept invoices.
OPEN_PO_STATUSES = {"APPROVED", "OPEN"}

# Ledger statuses that consume PO balance / count as an existing invoice.
ACTIVE_LEDGER_STATUSES = ("RESERVED", "APPROVED", "PAID")

# Upload limits.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 10
# A page with fewer extractable characters than this is treated as scanned -> OCR.
MIN_TEXT_CHARS_PER_PAGE = 30
OCR_DPI = 300

# AI call
OPENAI_TIMEOUT_SECONDS = 60
OPENAI_MAX_RETRIES = 2
# Free-tier Gemini returns 429 when the per-minute quota is hit; wait and retry.
RATE_LIMIT_RETRY_WAITS_SECONDS = (15, 30, 45)
# Gemini: server-side timeouts (504) are retried; per-request timeout in seconds.
GEMINI_TIMEOUT_SECONDS = 120
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)

# Fields that must be present and backed by source evidence.
CRITICAL_FIELDS = (
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "total_amount",
)
# Fields we verify against the source text when they are present.
EVIDENCE_FIELDS = (
    "invoice_number",
    "invoice_date",
    "vendor_name",
    "vendor_tax_id",
    "po_number",
    "subtotal",
    "tax_amount",
    "shipping_amount",
    "discount_amount",
    "total_amount",
    "buyer_name",
    "buyer_tax_id",
)
