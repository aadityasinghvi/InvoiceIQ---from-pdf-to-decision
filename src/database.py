"""Database layer: schema, seed data, queries and the atomic reservation.

All SQL lives in this module so switching from local SQLite to hosted
PostgreSQL is only a DATABASE_URL change.

Money is stored as integer paise. Quantities are stored as integer
thousandths ("milli") so fractional quantities stay exact.
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any, Iterable

from sqlalchemy import (
    Boolean,
    Column,
    Index,
    Integer,
    BigInteger,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    insert,
    select,
    text,
    update,
    delete,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError

from src import config
from src.money import to_paise

metadata = MetaData()

ACTIVE = config.ACTIVE_LEDGER_STATUSES

vendors = Table(
    "vendors",
    metadata,
    Column("vendor_id", String(20), primary_key=True),
    Column("legal_name", String(200), nullable=False),
    Column("tax_id", String(30)),
    Column("approval_status", String(20), nullable=False),
    Column("contact_email", String(200)),
)

vendor_aliases = Table(
    "vendor_aliases",
    metadata,
    Column("alias_norm", String(200), primary_key=True),
    Column("alias", String(200), nullable=False),
    Column("vendor_id", String(20), nullable=False),
)

purchase_orders = Table(
    "purchase_orders",
    metadata,
    Column("po_id", String(30), primary_key=True),
    Column("vendor_id", String(20), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("gross_total_paise", BigInteger, nullable=False),
    Column("status", String(20), nullable=False),
    Column("po_date", String(10)),
    Column("description", String(300)),
)

po_line_items = Table(
    "po_line_items",
    metadata,
    Column("po_id", String(30), primary_key=True),
    Column("line_no", Integer, primary_key=True),
    Column("item_code", String(50)),
    Column("description", String(300), nullable=False),
    Column("quantity_milli", BigInteger, nullable=False),
    Column("unit_price_paise", BigInteger, nullable=False),
    Column("tax_rate_bp", Integer, nullable=False),  # basis points, 1800 = 18%
)

invoice_ledger = Table(
    "invoice_ledger",
    metadata,
    Column("ledger_id", Integer, primary_key=True, autoincrement=True),
    Column("vendor_id", String(20), nullable=False),
    Column("invoice_number", String(100), nullable=False),
    Column("invoice_number_norm", String(100), nullable=False),
    Column("po_id", String(30), nullable=False),
    Column("invoice_date", String(10)),
    Column("amount_paise", BigInteger, nullable=False),
    Column("allocation_status", String(20), nullable=False),
    Column("run_id", String(40)),
    Column("file_hash", String(64)),
    Column("source", String(20), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)

# Database-level guarantee: one active invoice per vendor + invoice number.
_active_where = text("allocation_status IN ('RESERVED','APPROVED','PAID')")
Index(
    "uq_active_invoice_identity",
    invoice_ledger.c.vendor_id,
    invoice_ledger.c.invoice_number_norm,
    unique=True,
    sqlite_where=_active_where,
    postgresql_where=_active_where,
)

ledger_lines = Table(
    "ledger_lines",
    metadata,
    Column("ledger_id", Integer, primary_key=True),
    Column("po_id", String(30), nullable=False),
    Column("line_no", Integer, primary_key=True),
    Column("quantity_milli", BigInteger, nullable=False),
    Column("amount_paise", BigInteger, nullable=False),
)

processing_runs = Table(
    "processing_runs",
    metadata,
    Column("run_id", String(40), primary_key=True),
    Column("file_name", String(300)),
    Column("file_hash", String(64)),
    Column("file_size", Integer),
    Column("page_count", Integer),
    Column("ocr_used", Boolean, default=False),
    Column("started_at", String(40), nullable=False),
    Column("finished_at", String(40)),
    Column("duration_ms", Integer),
    Column("status", String(20), nullable=False),  # RUNNING / COMPLETED / FAILED
    Column("decision", String(10)),  # PASS / REVIEW / BLOCK / ERROR
    Column("headline", Text),
    Column("explanation", Text),
    Column("invoice_number", String(100)),
    Column("invoice_number_norm", String(100)),
    Column("vendor_id", String(20)),
    Column("vendor_name", String(200)),
    Column("po_id", String(30)),
    Column("amount_paise", BigInteger),
    Column("currency", String(3)),
    Column("extraction_json", Text),
    Column("verification_json", Text),
    Column("match_json", Text),
    Column("document_text", Text),
    Column("model", String(60)),
    Column("policy_version", String(30)),
    Column("error_stage", String(40)),
    Column("error_message", Text),
    Column("review_status", String(20), nullable=False, default="NONE"),
    Column("ledger_id", Integer),
)

run_stages = Table(
    "run_stages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, index=True),
    Column("seq", Integer, nullable=False),
    Column("stage", String(40), nullable=False),
    Column("status", String(20), nullable=False),  # SUCCESS / WARNING / FAILED
    Column("detail", Text),
    Column("finished_at", String(40), nullable=False),
    Column("duration_ms", Integer),
)

rule_checks = Table(
    "rule_checks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, index=True),
    Column("seq", Integer, nullable=False),
    Column("rule_code", String(40), nullable=False),
    Column("rule_name", String(120), nullable=False),
    Column("outcome", String(20), nullable=False),
    Column("expected", Text),
    Column("actual", Text),
    Column("reason", Text),
    Column("details_json", Text),
)

review_actions = Table(
    "review_actions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(40), nullable=False, index=True),
    Column("action", String(30), nullable=False),
    Column("actor", String(100), nullable=False),
    Column("note", Text),
    Column("created_at", String(40), nullable=False),
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def is_sqlite(engine_or_conn) -> bool:
    return engine_or_conn.dialect.name == "sqlite"


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or config.database_url()
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"timeout": 30, "check_same_thread": False})

        # Take full control of SQLite transactions so every write transaction
        # starts with BEGIN IMMEDIATE (acquires the write lock up front). This
        # serialises concurrent reservations instead of letting two requests
        # read the same balance.
        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, _record):
            dbapi_conn.isolation_level = None
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn):
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=2)
    return engine


def init_db(engine: Engine | None = None, seed_if_empty: bool = True) -> Engine:
    engine = engine or get_engine()
    metadata.create_all(engine)
    if seed_if_empty:
        with engine.connect() as conn:
            count = conn.execute(select(func.count()).select_from(vendors)).scalar_one()
        if count == 0:
            seed_reference_data(engine)
    return engine


# --------------------------------------------------------------------------
# Normalisation helpers (deterministic, shared by matcher and ledger)
# --------------------------------------------------------------------------


def normalize_invoice_number(value: str | None) -> str | None:
    """'inv / abc-2041 ' -> 'INVABC2041'. Used for duplicate identity."""
    if value is None:
        return None
    norm = "".join(ch for ch in value.upper() if ch.isalnum())
    return norm or None


_LEGAL_SUFFIXES = (
    "PRIVATE LIMITED", "PVT LTD", "PVT. LTD.", "PVT. LTD", "LIMITED", "LTD", "LLP", "INC", "CO",
)


def normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    s = " ".join(value.upper().replace(".", " ").replace(",", " ").split())
    changed = True
    while changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            suf = " ".join(suffix.replace(".", " ").split())
            if s.endswith(" " + suf):
                s = s[: -len(suf) - 1].strip()
                changed = True
    return s or None


def qty_to_milli(qty: Decimal) -> int:
    return int((qty * 1000).to_integral_value())


# --------------------------------------------------------------------------
# Seed / reset
# --------------------------------------------------------------------------


def _read_csv(name: str) -> list[dict[str, str]]:
    with open(config.DATA_DIR / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def seed_reference_data(engine: Engine) -> None:
    ts = now_iso()
    with engine.begin() as conn:
        for table in (ledger_lines, invoice_ledger, po_line_items, purchase_orders, vendor_aliases, vendors):
            conn.execute(delete(table))
        conn.execute(insert(vendors), _read_csv("vendors.csv"))
        alias_rows = []
        for r in _read_csv("vendors.csv"):
            alias_rows.append({"alias_norm": normalize_name(r["legal_name"]), "alias": r["legal_name"], "vendor_id": r["vendor_id"]})
        for r in _read_csv("vendor_aliases.csv"):
            alias_rows.append({"alias_norm": normalize_name(r["alias"]), "alias": r["alias"], "vendor_id": r["vendor_id"]})
        dedup = {r["alias_norm"]: r for r in alias_rows}
        conn.execute(insert(vendor_aliases), list(dedup.values()))
        conn.execute(
            insert(purchase_orders),
            [
                {
                    "po_id": r["po_id"],
                    "vendor_id": r["vendor_id"],
                    "currency": r["currency"],
                    "gross_total_paise": to_paise(r["gross_total"]),
                    "status": r["status"],
                    "po_date": r["po_date"],
                    "description": r["description"],
                }
                for r in _read_csv("purchase_orders.csv")
            ],
        )
        conn.execute(
            insert(po_line_items),
            [
                {
                    "po_id": r["po_id"],
                    "line_no": int(r["line_no"]),
                    "item_code": r["item_code"] or None,
                    "description": r["description"],
                    "quantity_milli": qty_to_milli(Decimal(r["quantity"])),
                    "unit_price_paise": to_paise(r["unit_price"]),
                    "tax_rate_bp": int(Decimal(r["tax_rate_percent"]) * 100),
                }
                for r in _read_csv("po_items.csv")
            ],
        )
        po_lines = {
            (r["po_id"], int(r["line_no"])): r for r in _read_csv("po_items.csv")
        }
        for r in _read_csv("invoice_history.csv"):
            result = conn.execute(
                insert(invoice_ledger).values(
                    vendor_id=r["vendor_id"],
                    invoice_number=r["invoice_number"],
                    invoice_number_norm=normalize_invoice_number(r["invoice_number"]),
                    po_id=r["po_id"],
                    invoice_date=r["invoice_date"],
                    amount_paise=to_paise(r["amount"]),
                    allocation_status=r["allocation_status"],
                    source="SEED",
                    created_at=ts,
                    updated_at=ts,
                )
            )
            ledger_id = result.inserted_primary_key[0]
            for part in r["lines"].split(";"):
                line_no, qty = part.split(":")
                pl = po_lines[(r["po_id"], int(line_no))]
                q = Decimal(qty)
                conn.execute(
                    insert(ledger_lines).values(
                        ledger_id=ledger_id,
                        po_id=r["po_id"],
                        line_no=int(line_no),
                        quantity_milli=qty_to_milli(q),
                        amount_paise=to_paise(q * Decimal(pl["unit_price"])),
                    )
                )


def reset_demo(engine: Engine | None = None) -> None:
    """Wipe all runs and ledger activity and restore the seeded reference data."""
    engine = engine or get_engine()
    metadata.create_all(engine)
    with engine.begin() as conn:
        for table in (review_actions, rule_checks, run_stages, processing_runs):
            conn.execute(delete(table))
    seed_reference_data(engine)


def our_company_entities() -> list[dict]:
    """The legal entities invoices must be addressed to (data/company.csv)."""
    path = config.DATA_DIR / "company.csv"
    if not path.exists():
        return []
    return [{"legal_name": r["legal_name"], "gstin": r["gstin"]} for r in _read_csv("company.csv")]


# --------------------------------------------------------------------------
# Reference data queries
# --------------------------------------------------------------------------


def _row(r) -> dict[str, Any] | None:
    return dict(r._mapping) if r is not None else None


def _rows(rs) -> list[dict[str, Any]]:
    return [dict(r._mapping) for r in rs]


def get_vendor(conn: Connection, vendor_id: str) -> dict | None:
    return _row(conn.execute(select(vendors).where(vendors.c.vendor_id == vendor_id)).first())


def find_vendor_by_tax_id(conn: Connection, tax_id: str) -> dict | None:
    norm = "".join(ch for ch in tax_id.upper() if ch.isalnum())
    for v in _rows(conn.execute(select(vendors))):
        if v["tax_id"] and "".join(ch for ch in v["tax_id"].upper() if ch.isalnum()) == norm:
            return v
    return None


def find_vendor_by_name(conn: Connection, name: str) -> dict | None:
    norm = normalize_name(name)
    if not norm:
        return None
    hit = conn.execute(select(vendor_aliases.c.vendor_id).where(vendor_aliases.c.alias_norm == norm)).first()
    return get_vendor(conn, hit[0]) if hit else None


def list_vendors(conn: Connection) -> list[dict]:
    return _rows(conn.execute(select(vendors).order_by(vendors.c.vendor_id)))


def list_aliases(conn: Connection) -> list[dict]:
    return _rows(conn.execute(select(vendor_aliases).order_by(vendor_aliases.c.vendor_id)))


def get_po(conn: Connection, po_id: str) -> dict | None:
    return _row(conn.execute(select(purchase_orders).where(purchase_orders.c.po_id == po_id)).first())


def find_po_normalized(conn: Connection, po_ref: str) -> dict | None:
    """Exact match first, then a match ignoring punctuation/case ('po 1001' -> 'PO-1001')."""
    po = get_po(conn, po_ref.strip())
    if po:
        return po
    norm = "".join(ch for ch in po_ref.upper() if ch.isalnum())
    for p in _rows(conn.execute(select(purchase_orders))):
        if "".join(ch for ch in p["po_id"].upper() if ch.isalnum()) == norm:
            return p
    return None


def get_po_lines(conn: Connection, po_id: str) -> list[dict]:
    return _rows(
        conn.execute(select(po_line_items).where(po_line_items.c.po_id == po_id).order_by(po_line_items.c.line_no))
    )


def list_pos(conn: Connection) -> list[dict]:
    return _rows(conn.execute(select(purchase_orders).order_by(purchase_orders.c.po_id)))


def list_open_pos_for_vendor(conn: Connection, vendor_id: str) -> list[dict]:
    return _rows(
        conn.execute(
            select(purchase_orders)
            .where(purchase_orders.c.vendor_id == vendor_id)
            .where(purchase_orders.c.status.in_(sorted(config.OPEN_PO_STATUSES)))
            .order_by(purchase_orders.c.po_id)
        )
    )


def po_committed_paise(conn: Connection, po_id: str) -> int:
    return int(
        conn.execute(
            select(func.coalesce(func.sum(invoice_ledger.c.amount_paise), 0))
            .where(invoice_ledger.c.po_id == po_id)
            .where(invoice_ledger.c.allocation_status.in_(ACTIVE))
        ).scalar_one()
    )


def po_invoiced_qty_milli(conn: Connection, po_id: str) -> dict[int, int]:
    rs = conn.execute(
        select(ledger_lines.c.line_no, func.sum(ledger_lines.c.quantity_milli))
        .select_from(ledger_lines.join(invoice_ledger, ledger_lines.c.ledger_id == invoice_ledger.c.ledger_id))
        .where(ledger_lines.c.po_id == po_id)
        .where(invoice_ledger.c.allocation_status.in_(ACTIVE))
        .group_by(ledger_lines.c.line_no)
    )
    return {int(line_no): int(q) for line_no, q in rs}


def find_active_invoice(conn: Connection, vendor_id: str, invoice_number_norm: str) -> dict | None:
    return _row(
        conn.execute(
            select(invoice_ledger)
            .where(invoice_ledger.c.vendor_id == vendor_id)
            .where(invoice_ledger.c.invoice_number_norm == invoice_number_norm)
            .where(invoice_ledger.c.allocation_status.in_(ACTIVE))
        ).first()
    )


def find_active_invoice_any_vendor(conn: Connection, invoice_number_norm: str) -> list[dict]:
    return _rows(
        conn.execute(
            select(invoice_ledger)
            .where(invoice_ledger.c.invoice_number_norm == invoice_number_norm)
            .where(invoice_ledger.c.allocation_status.in_(ACTIVE))
        )
    )


def find_active_by_file_hash(conn: Connection, file_hash: str | None) -> dict | None:
    """An active (reserved/approved/paid) ledger entry created from the identical file."""
    if not file_hash:
        return None
    return _row(conn.execute(
        select(invoice_ledger).where(invoice_ledger.c.file_hash == file_hash)
        .where(invoice_ledger.c.allocation_status.in_(ACTIVE))).first())


def find_open_runs_by_file_hash(conn: Connection, file_hash: str | None, exclude_run_id: str) -> list[dict]:
    if not file_hash:
        return []
    return _rows(conn.execute(
        select(processing_runs.c.run_id, processing_runs.c.decision)
        .where(processing_runs.c.file_hash == file_hash)
        .where(processing_runs.c.review_status == "OPEN")
        .where(processing_runs.c.run_id != exclude_run_id)))


def find_active_similar(conn: Connection, vendor_id: str, amount_paise: int | None, invoice_date: str | None,
                        exclude_number_norm: str | None = None) -> dict | None:
    """Same vendor, same amount, same date, different number: a likely re-numbered duplicate."""
    if not vendor_id or amount_paise is None or not invoice_date:
        return None
    q = (select(invoice_ledger).where(invoice_ledger.c.vendor_id == vendor_id)
         .where(invoice_ledger.c.amount_paise == amount_paise)
         .where(invoice_ledger.c.invoice_date == invoice_date)
         .where(invoice_ledger.c.allocation_status.in_(ACTIVE)))
    if exclude_number_norm:
        q = q.where(invoice_ledger.c.invoice_number_norm != exclude_number_norm)
    return _row(conn.execute(q).first())


def find_open_review_run(conn: Connection, vendor_id: str | None, invoice_number_norm: str, exclude_run_id: str) -> dict | None:
    q = (
        select(processing_runs)
        .where(processing_runs.c.invoice_number_norm == invoice_number_norm)
        .where(processing_runs.c.review_status == "OPEN")
        .where(processing_runs.c.run_id != exclude_run_id)
    )
    if vendor_id:
        q = q.where(processing_runs.c.vendor_id == vendor_id)
    return _row(conn.execute(q.order_by(processing_runs.c.started_at)).first())


def runs_with_file_hash(conn: Connection, file_hash: str, exclude_run_id: str) -> list[dict]:
    return _rows(
        conn.execute(
            select(processing_runs.c.run_id, processing_runs.c.decision, processing_runs.c.started_at)
            .where(processing_runs.c.file_hash == file_hash)
            .where(processing_runs.c.run_id != exclude_run_id)
            .order_by(processing_runs.c.started_at)
        )
    )


def list_ledger(conn: Connection) -> list[dict]:
    return _rows(conn.execute(select(invoice_ledger).order_by(invoice_ledger.c.ledger_id)))


# --------------------------------------------------------------------------
# Processing runs and audit trail
# --------------------------------------------------------------------------


def new_run_id() -> str:
    return "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6].upper()


def create_run(engine: Engine, run_id: str, file_name: str, file_hash: str, file_size: int, model: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(processing_runs).values(
                run_id=run_id,
                file_name=file_name,
                file_hash=file_hash,
                file_size=file_size,
                started_at=now_iso(),
                status="RUNNING",
                model=model,
                policy_version=config.POLICY_VERSION,
                review_status="NONE",
                ocr_used=False,
            )
        )


def update_run(engine_or_conn, run_id: str, **values: Any) -> None:
    for key in ("extraction_json", "verification_json", "match_json"):
        if key in values and not isinstance(values[key], (str, type(None))):
            values[key] = json.dumps(values[key], default=str)
    stmt = update(processing_runs).where(processing_runs.c.run_id == run_id).values(**values)
    if isinstance(engine_or_conn, Engine):
        with engine_or_conn.begin() as conn:
            conn.execute(stmt)
    else:
        engine_or_conn.execute(stmt)


def log_stage(engine: Engine, run_id: str, seq: int, stage: str, status: str, detail: str, duration_ms: int | None) -> None:
    with engine.begin() as conn:
        conn.execute(
            insert(run_stages).values(
                run_id=run_id, seq=seq, stage=stage, status=status, detail=detail,
                finished_at=now_iso(), duration_ms=duration_ms,
            )
        )


def save_checks(conn: Connection, run_id: str, checks: Iterable[dict]) -> None:
    conn.execute(delete(rule_checks).where(rule_checks.c.run_id == run_id))
    rows = []
    for i, c in enumerate(checks, start=1):
        rows.append(
            {
                "run_id": run_id,
                "seq": i,
                "rule_code": c["rule_code"],
                "rule_name": c["rule_name"],
                "outcome": c["outcome"],
                "expected": c.get("expected"),
                "actual": c.get("actual"),
                "reason": c.get("reason"),
                "details_json": json.dumps(c.get("details"), default=str) if c.get("details") is not None else None,
            }
        )
    if rows:
        conn.execute(insert(rule_checks), rows)


def get_run(conn: Connection, run_id: str) -> dict | None:
    return _row(conn.execute(select(processing_runs).where(processing_runs.c.run_id == run_id)).first())


def get_run_stages(conn: Connection, run_id: str) -> list[dict]:
    return _rows(conn.execute(select(run_stages).where(run_stages.c.run_id == run_id).order_by(run_stages.c.seq)))


def get_run_checks(conn: Connection, run_id: str) -> list[dict]:
    rows = _rows(conn.execute(select(rule_checks).where(rule_checks.c.run_id == run_id).order_by(rule_checks.c.seq)))
    for r in rows:
        r["details"] = json.loads(r["details_json"]) if r.get("details_json") else None
    return rows


def get_review_actions(conn: Connection, run_id: str) -> list[dict]:
    return _rows(conn.execute(select(review_actions).where(review_actions.c.run_id == run_id).order_by(review_actions.c.id)))


def list_runs(conn: Connection, limit: int = 500) -> list[dict]:
    cols = [c for c in processing_runs.c if c.name not in ("document_text", "extraction_json", "verification_json", "match_json")]
    return _rows(conn.execute(select(*cols).order_by(processing_runs.c.started_at.desc()).limit(limit)))


def get_ledger_entry(conn: Connection, ledger_id: int) -> dict | None:
    return _row(conn.execute(select(invoice_ledger).where(invoice_ledger.c.ledger_id == ledger_id)).first())


# --------------------------------------------------------------------------
# The atomic reservation
# --------------------------------------------------------------------------


@dataclass
class ReservationRequest:
    run_id: str
    vendor_id: str
    invoice_number: str
    invoice_number_norm: str
    po_id: str
    invoice_date: str | None
    amount_paise: int
    file_hash: str | None
    lines: list[dict]  # {"line_no", "quantity_milli", "amount_paise"}
    status: str = "RESERVED"
    source: str = "PIPELINE"


@dataclass
class ReservationResult:
    ok: bool
    ledger_id: int | None = None
    conflict: str | None = None  # DUPLICATE / BALANCE / PO_STATE
    message: str = ""
    remaining_paise: int | None = None
    existing: dict | None = None


def _lock_po(conn: Connection, po_id: str) -> dict | None:
    """Lock the PO row for the rest of the transaction.

    PostgreSQL: SELECT ... FOR UPDATE row lock.
    SQLite: the transaction already holds the database write lock (BEGIN IMMEDIATE).
    """
    q = select(purchase_orders).where(purchase_orders.c.po_id == po_id)
    if not is_sqlite(conn):
        q = q.with_for_update()
    return _row(conn.execute(q).first())


def lock_run(conn: Connection, run_id: str) -> dict | None:
    """Read a run and lock its row until the transaction ends (PostgreSQL); SQLite already holds the write lock."""
    q = select(processing_runs).where(processing_runs.c.run_id == run_id)
    if not is_sqlite(conn):
        q = q.with_for_update()
    return _row(conn.execute(q).first())


def reserve_in_transaction(conn: Connection, req: ReservationRequest,
                           enforce_line_quantities: bool = True) -> ReservationResult:
    """Re-check every financial invariant and insert the allocation, inside the CALLER's transaction.

    Nothing is committed here: the caller saves the run's final decision and audit records on the
    same connection, so the reservation and the decision are committed together or not at all.
    May raise IntegrityError if a concurrent duplicate slipped past the read (unique index).
    """
    if req.amount_paise is None or req.amount_paise <= 0:
        return ReservationResult(False, conflict="AMOUNT", message="Amount must be greater than zero")
    po = _lock_po(conn, req.po_id)
    if po is None or po["status"] not in config.OPEN_PO_STATUSES:
        return ReservationResult(False, conflict="PO_STATE", message=f"PO {req.po_id} is not open for invoicing")
    if po["vendor_id"] != req.vendor_id:
        return ReservationResult(False, conflict="PO_STATE", message=f"PO {req.po_id} does not belong to vendor {req.vendor_id}")
    vendor = get_vendor(conn, req.vendor_id)
    if vendor is None or vendor["approval_status"] != "APPROVED":
        return ReservationResult(False, conflict="VENDOR", message=f"Vendor {req.vendor_id} is not approved")

    existing = find_active_invoice(conn, req.vendor_id, req.invoice_number_norm)
    if existing:
        return ReservationResult(
            False, conflict="DUPLICATE", existing=existing,
            message=f"Invoice already active in ledger (entry #{existing['ledger_id']}, {existing['allocation_status']})",
        )
    same_file = find_active_by_file_hash(conn, req.file_hash)
    if same_file:
        return ReservationResult(
            False, conflict="DUPLICATE", existing=same_file,
            message=f"The identical file is already active in the ledger (entry #{same_file['ledger_id']}, "
                    f"as '{same_file['invoice_number']}')",
        )

    committed = po_committed_paise(conn, req.po_id)
    remaining = po["gross_total_paise"] - committed
    if req.amount_paise > remaining + config.BALANCE_ROUNDING_TOLERANCE_PAISE:
        return ReservationResult(False, conflict="BALANCE", remaining_paise=remaining,
                                 message="Invoice exceeds remaining PO balance at commit time")

    if enforce_line_quantities:
        invoiced = po_invoiced_qty_milli(conn, req.po_id)
        po_lines = {l["line_no"]: l for l in get_po_lines(conn, req.po_id)}
        for line in req.lines:
            pl = po_lines.get(line["line_no"])
            if pl is None:
                return ReservationResult(False, conflict="PO_STATE", message=f"PO line {line['line_no']} no longer exists")
            if line["quantity_milli"] <= 0:
                return ReservationResult(False, conflict="AMOUNT", message="Line quantities must be greater than zero")
            if invoiced.get(line["line_no"], 0) + line["quantity_milli"] > pl["quantity_milli"]:
                return ReservationResult(False, conflict="BALANCE", remaining_paise=remaining,
                                         message=f"PO line {line['line_no']} quantity already consumed")

    ts = now_iso()
    result = conn.execute(
        insert(invoice_ledger).values(
            vendor_id=req.vendor_id, invoice_number=req.invoice_number, invoice_number_norm=req.invoice_number_norm,
            po_id=req.po_id, invoice_date=req.invoice_date, amount_paise=req.amount_paise,
            allocation_status=req.status, run_id=req.run_id, file_hash=req.file_hash, source=req.source,
            created_at=ts, updated_at=ts,
        )
    )
    ledger_id = result.inserted_primary_key[0]
    if req.lines:
        conn.execute(insert(ledger_lines), [
            {"ledger_id": ledger_id, "po_id": req.po_id, "line_no": l["line_no"],
             "quantity_milli": l["quantity_milli"], "amount_paise": l["amount_paise"]}
            for l in req.lines
        ])
    conn.execute(update(processing_runs).where(processing_runs.c.run_id == req.run_id).values(ledger_id=ledger_id))
    return ReservationResult(True, ledger_id=ledger_id, remaining_paise=remaining - req.amount_paise)


def reserve_invoice(engine: Engine, req: ReservationRequest, enforce_line_quantities: bool = True) -> ReservationResult:
    """Stand-alone reservation in its own transaction (used by tests and simple callers)."""
    try:
        with engine.begin() as conn:
            return reserve_in_transaction(conn, req, enforce_line_quantities)
    except IntegrityError:
        return ReservationResult(False, conflict="DUPLICATE", message="Unique constraint: invoice identity already active")


def record_review_action(conn: Connection, run_id: str, action: str, actor: str, note: str | None) -> None:
    conn.execute(
        insert(review_actions).values(run_id=run_id, action=action, actor=actor, note=note, created_at=now_iso())
    )


def set_ledger_status(conn: Connection, ledger_id: int, status: str) -> None:
    conn.execute(
        update(invoice_ledger)
        .where(invoice_ledger.c.ledger_id == ledger_id)
        .values(allocation_status=status, updated_at=now_iso())
    )
