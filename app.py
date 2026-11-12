"""InvoiceIQ — invoice to decision, with every step visible.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="InvoiceIQ", page_icon="🧾", layout="wide")

# Streamlit Cloud secrets -> environment, before config is read anywhere.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from src import config  # noqa: E402
from src import database as db  # noqa: E402
from src import extractor as ai  # noqa: E402
from src import review  # noqa: E402
from src.money import format_inr, format_inr_short, format_money  # noqa: E402
from src.pipeline import load_run_bundle, process_invoice  # noqa: E402
from src.rules import RULES  # noqa: E402
from src.ui import (  # noqa: E402
    DECISION_COLORS,
    blank_stage_state,
    esc,
    inject_css,
    render_run_detail,
    rules_table_html,
    timeline_html,
)


@st.cache_resource
def engine():
    return db.init_db(db.get_engine())


def manifest() -> list[dict]:
    path = config.SAMPLE_INVOICE_DIR / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


inject_css()
ENGINE = engine()


def settings_menu() -> None:
    """System status, reviewer sign-in and demo reset, tucked into one popover at the top right of every page."""
    signed_in = st.session_state.get("reviewer_ok")
    label = "Reviewer: signed in" if (config.reviewer_passcode() and signed_in) else "Settings"
    with st.popover(label, icon=":material/tune:", use_container_width=True):
        url = config.database_url()
        provider_name = {"openai": "OpenAI", "gemini": "Gemini", "groq": "Groq"}.get(config.llm_provider(), config.llm_provider())
        st.markdown(f"**Reading invoices with** {provider_name} · `{config.extraction_model()}`  \n"
                    f"**API key** {'configured' if config.has_llm_key() else 'missing'}  \n"
                    f"**Database** {'PostgreSQL' if url.startswith('postgres') else 'SQLite (local file)'}")
        if not url.startswith("postgres"):
            st.caption("A local SQLite file is wiped when Streamlit Cloud restarts; use PostgreSQL for the hosted demo.")
        st.divider()
        if config.reviewer_passcode():
            if signed_in:
                st.caption("You are signed in as a reviewer.")
                if st.button("Sign out", use_container_width=True):
                    st.session_state["reviewer_ok"] = False
                    st.rerun()
            else:
                code = st.text_input("Reviewer passcode", type="password")
                if st.button("Sign in as reviewer", use_container_width=True):
                    if config.check_reviewer_passcode(code):
                        st.session_state["reviewer_ok"] = True
                        st.rerun()
                    else:
                        st.error("Wrong passcode.")
        else:
            st.caption("Reviewer actions are open to everyone: no REVIEWER_PASSCODE is set.")
        if not config.reviewer_passcode() or signed_in:
            st.divider()
            st.markdown("**Reset demo data**")
            st.caption("Deletes all runs, reservations and reviewer actions and restores the original vendors, POs and history.")
            confirm = st.checkbox("I understand this clears the history")
            if st.button("Reset demo data", disabled=not confirm, use_container_width=True):
                db.reset_demo(ENGINE)
                st.session_state.pop("last_run_id", None)
                st.rerun()


def page_header(title: str, lede: str | None = None) -> None:
    left, right = st.columns([5, 1.3], vertical_alignment="top")
    with left:
        st.title(title)
        if lede:
            st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)
    with right:
        settings_menu()


def footer() -> None:
    url = config.database_url()
    st.markdown(
        f'<div class="foot"><span><b>InvoiceIQ</b> — AI reads the invoice, fixed rules make the decision</span>'
        f'<span>Policy {config.POLICY_VERSION}</span><span>{len(RULES)} rules</span>'
        f'<span>{"PostgreSQL" if url.startswith("postgres") else "SQLite"}</span></div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# Page 1 — Process an invoice
# ==========================================================================


def page_process() -> None:
    page_header("Process an invoice",
                "Upload a vendor invoice and get a decision you can defend: pass, review or block, "
                "with the evidence behind every check.")
    st.markdown(
        '<div class="flow">'
        '<div><b><i>1</i>AI reads the invoice</b><span>Every value comes back with the exact text it was read from.</span></div>'
        '<div><b><i>2</i>Code checks the evidence</b><span>Values that cannot be found in the PDF are never trusted.</span></div>'
        f'<div><b><i>3</i>{len(RULES)} fixed rules decide</b><span>Matched against vendors, purchase orders and past invoices.</span></div>'
        '</div>', unsafe_allow_html=True)

    has_key = config.has_llm_key()
    if not has_key:
        st.warning("No AI API key is configured, so invoices can't be read yet. Add `GROQ_API_KEY` or `GEMINI_API_KEY` "
                   "(both free) or `OPENAI_API_KEY` to `.env` (local) or the app's Secrets (Streamlit Cloud).")

    with st.container(border=True, key="intake"):
        src_col, sample_col = st.columns([3, 2], gap="large")
        with src_col:
            uploaded = st.file_uploader("Vendor invoice (PDF, up to 10 MB)", type=["pdf"])
        with sample_col:
            samples = manifest()
            labels = ["—"] + [f"{m['file'][:2]} · {m['title']}" for m in samples]
            choice = st.selectbox("Or try a prepared test invoice", labels, index=0,
                                  help="Synthetic invoices covering the happy path and each edge case.")
            sample = samples[labels.index(choice) - 1] if choice != "—" else None
            if sample:
                st.caption(f"{sample['story']} Expected: **{sample['expected']}**.")

        pdf_bytes, file_name = None, None
        if uploaded is not None:
            pdf_bytes, file_name = uploaded.getvalue(), uploaded.name
        elif sample is not None:
            path = config.SAMPLE_INVOICE_DIR / sample["file"]
            pdf_bytes, file_name = path.read_bytes(), sample["file"]

        b1, b2, _ = st.columns([1.2, 1.4, 3])
        go = b1.button("Process invoice", type="primary", disabled=pdf_bytes is None or not has_key,
                       use_container_width=True)
        if sample is not None and uploaded is None:
            b2.download_button("Download this test PDF", pdf_bytes, file_name=file_name, mime="application/pdf",
                               use_container_width=True)
    st.write("")

    left, right = st.columns([2, 5], gap="large")
    with left:
        with st.container(border=True, key="liverun"):
            st.markdown('<div class="panel-title">Live run</div>'
                        '<div class="panel-sub">Each stage updates the moment it finishes.</div>', unsafe_allow_html=True)
            timeline_box = st.empty()
    with right:
        result_box = st.container()

    if go and pdf_bytes is not None:
        state = blank_stage_state()
        timeline_box.markdown(timeline_html(state), unsafe_allow_html=True)
        with result_box:
            waiting = st.empty()
            waiting.markdown('<div class="empty">The decision appears here once every check has run.</div>',
                             unsafe_allow_html=True)

        def on_stage(key: str, status: str, detail: str, ms: int | None) -> None:
            if key == "DONE":
                return
            state[key] = {"status": status, "detail": detail, "ms": ms}
            timeline_box.markdown(timeline_html(state), unsafe_allow_html=True)

        result = process_invoice(pdf_bytes, file_name, engine=ENGINE, extractor=ai.extract_invoice, on_stage=on_stage)
        waiting.empty()
        st.session_state["last_run_id"] = result.run_id
        st.session_state["animate_run"] = result.run_id

    run_id = st.session_state.get("last_run_id")
    if run_id:
        bundle = load_run_bundle(run_id, ENGINE)
        if bundle:
            from src.ui import stage_state_from_db
            timeline_box.markdown(timeline_html(stage_state_from_db(bundle["stages"])), unsafe_allow_html=True)
            with result_box:
                animate = st.session_state.pop("animate_run", None) == run_id
                render_run_detail(bundle, animate=animate, show_timeline=False)
    else:
        timeline_box.markdown(timeline_html(blank_stage_state()), unsafe_allow_html=True)
        with result_box:
            st.markdown('<div class="empty">Choose an invoice above and select <b>Process invoice</b>.<br>'
                        'The decision, its reasons and the evidence for every check appear here.</div>',
                        unsafe_allow_html=True)
    footer()


# ==========================================================================
# Page 2 — Dashboard
# ==========================================================================


def page_dashboard() -> None:
    page_header("Dashboard", "Every invoice InvoiceIQ has processed, what it decided and where each one stands now.")
    with ENGINE.connect() as conn:
        runs = db.list_runs(conn)
        ledger = db.list_ledger(conn)
        vendor_names = {v["vendor_id"]: v["legal_name"] for v in db.list_vendors(conn)}
    if not runs:
        st.markdown('<div class="empty">No invoices processed yet. Process one on the <b>Process an invoice</b> page '
                    'and it will appear here.</div>', unsafe_allow_html=True)
        footer()
        return

    df = pd.DataFrame(runs)
    count = df["decision"].value_counts()
    open_queue = int(((df["review_status"] == "OPEN")).sum())
    reserved = sum(e["amount_paise"] for e in ledger if e["allocation_status"] == "RESERVED")
    approved = sum(e["amount_paise"] for e in ledger if e["allocation_status"] == "APPROVED" and e["source"] != "SEED")
    total = len(df)
    segs, keys = [], []
    for d in ("PASS", "REVIEW", "BLOCK", "ERROR"):
        n = int(count.get(d, 0))
        if n:
            segs.append(f'<span style="width:{n / total * 100:.2f}%;background:{DECISION_COLORS[d]}" title="{d}: {n}"></span>')
        keys.append(f'<span><i style="background:{DECISION_COLORS[d]}"></i><b>{n}</b>{d.capitalize()}</span>')
    st.markdown(
        f'<div class="band">'
        f'<div><div class="k">Invoices processed</div><div class="v">{total}</div></div>'
        f'<div><div class="k">Decisions</div><div class="mix">{"".join(segs)}</div><div class="mixkey">{"".join(keys)}</div></div>'
        f'<div><div class="k">In review queue</div><div class="v">{open_queue}</div><div class="s">waiting for a person</div></div>'
        f'<div><div class="k">Value committed</div><div class="v">{format_inr_short(reserved + approved)}</div>'
        f'<div class="s">reserved or approved on POs</div></div>'
        f'</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 3])
    decisions = f1.multiselect("Decision", ["PASS", "REVIEW", "BLOCK", "ERROR"], placeholder="All decisions")
    vendors = sorted({v for v in df["vendor_name"].dropna().unique()})
    vend = f2.multiselect("Vendor", vendors, placeholder="All vendors")
    q = f3.text_input("Search invoice number, PO or file", placeholder="e.g. PO-1001")
    view = df.copy()
    if decisions:
        view = view[view["decision"].isin(decisions)]
    if vend:
        view = view[view["vendor_name"].isin(vend)]
    if q:
        mask = pd.Series(False, index=view.index)
        for col in ("invoice_number", "po_id", "file_name", "run_id"):
            mask |= view[col].fillna("").str.contains(q, case=False, regex=False)
        view = view[mask]

    from src.ui import review_status_text
    table = pd.DataFrame({
        "Processed (UTC)": view["started_at"].str[:19].str.replace("T", " "),
        "Invoice": view["invoice_number"].fillna("—"),
        "Vendor": [vendor_names.get(v) if isinstance(v, str) else (n if isinstance(n, str) else "—")
                   for v, n in zip(view["vendor_id"], view["vendor_name"])],
        "PO": view["po_id"].fillna("—"),
        "Amount": [format_money(int(a), c if isinstance(c, str) else "INR") if pd.notna(a) else "—"
                   for a, c in zip(view["amount_paise"], view["currency"])],
        "Decision": view["decision"].fillna("RUNNING"),
        "Status": [review_status_text(r) for r in view.to_dict("records")],
        "Time (s)": [f"{d / 1000:.1f}" if pd.notna(d) else "—" for d in view["duration_ms"]],
        "File": view["file_name"],
        "Run ID": view["run_id"],
    })

    def color(v):
        return f"color:{DECISION_COLORS.get(v, '#1C2B30')};font-weight:700"

    st.caption("Select a row to open its full audit trail.")
    event = st.dataframe(
        table.style.map(color, subset=["Decision"]), hide_index=True, use_container_width=True,
        height=min(36 * (len(table) + 1) + 4, 560),
        on_select="rerun", selection_mode="single-row", key="runs_table",
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        run_id = table.iloc[rows[0]]["Run ID"]
        st.divider()
        bundle = load_run_bundle(run_id, ENGINE)
        if bundle:
            render_run_detail(bundle)
    footer()


# ==========================================================================
# Page 3 — Review queue
# ==========================================================================


def reviewer_signed_in() -> bool:
    """Review actions need the reviewer passcode when REVIEWER_PASSCODE is configured (hosted app)."""
    return not config.reviewer_passcode() or st.session_state.get("reviewer_ok", False)


def page_review() -> None:
    page_header("Review queue",
                "Every PASS waits here for an AP approver, and every REVIEW waits for a person to decide. "
                "A reviewer can accept a judgement call, such as a small price variance or a missing PO reference, "
                "but can never approve an unverified total, a duplicate, an unconfirmed vendor or more than the PO allows.")
    can_act = reviewer_signed_in()
    if not can_act:
        st.info("Sign in as a reviewer from **Settings** at the top right to approve, reject or release invoices.")
    actor = st.columns([2, 5])[0].text_input("Reviewing as", value=st.session_state.get("actor", "AP Reviewer"),
                                             disabled=not can_act)
    st.session_state["actor"] = actor

    with ENGINE.connect() as conn:
        runs = [r for r in db.list_runs(conn) if r["review_status"] == "OPEN"]
    passes = [r for r in runs if r["decision"] == "PASS"]
    reviews = [r for r in runs if r["decision"] == "REVIEW"]

    if msg := st.session_state.pop("review_msg", None):
        (st.success if msg[0] else st.error)(msg[1])

    st.markdown(f'<div class="qhead"><h3>Ready for approval</h3><span class="qcount">{len(passes)}</span></div>',
                unsafe_allow_html=True)
    if not passes:
        st.caption("Nothing waiting. Invoices that pass every check appear here with their amount already reserved.")
    for r in passes:
        with st.expander(f":green[**PASS**] &nbsp; {r['invoice_number']} · {r['vendor_name']} · "
                         f"{format_inr(r['amount_paise'])} · {r['po_id']}"):
            st.markdown(esc(r["headline"]))
            note = st.text_input("Note (required to release)", key=f"n_{r['run_id']}", disabled=not can_act)
            a, b, _ = st.columns([1, 1, 3])
            if a.button("Approve for payment", key=f"a_{r['run_id']}", type="primary", disabled=not can_act):
                res = review.approve_reserved(ENGINE, r["run_id"], actor, note or None)
                st.session_state["review_msg"] = (res.ok, res.message)
                st.rerun()
            if b.button("Release reservation", key=f"r_{r['run_id']}", disabled=not can_act):
                res = review.release_reservation(ENGINE, r["run_id"], actor, note or None)
                st.session_state["review_msg"] = (res.ok, res.message)
                st.rerun()

    st.markdown(f'<div class="qhead"><h3>Needs a decision</h3><span class="qcount">{len(reviews)}</span></div>',
                unsafe_allow_html=True)
    if not reviews:
        st.caption("Nothing waiting. Invoices with missing data, discrepancies or uncertainty appear here.")
    for r in reviews:
        amount = format_money(r["amount_paise"], r.get("currency") or "INR") if r["amount_paise"] is not None else "no total"
        with st.expander(f":orange[**REVIEW**] &nbsp; {r['invoice_number'] or '(no invoice number)'} · "
                         f"{r['vendor_name'] or 'unknown vendor'} · {amount}"):
            bundle = load_run_bundle(r["run_id"], ENGINE)
            failing = [c for c in bundle["checks"] if c["outcome"] in ("BLOCK", "REVIEW")]
            st.markdown(rules_table_html(failing), unsafe_allow_html=True)
            st.write("")
            with ENGINE.connect() as conn:
                run = db.get_run(conn, r["run_id"])
                checks = {c["rule_code"]: c for c in bundle["checks"]}
                fixed_po = review.printed_po(run, checks)
                vendor = db.get_vendor(conn, run["vendor_id"]) if run["vendor_id"] else None
                st.markdown(f"**Vendor:** {esc(vendor['legal_name']) + ' (' + vendor['vendor_id'] + ')' if vendor else 'not confirmed'}")
                po_id = fixed_po
                if fixed_po:
                    st.markdown(f"**PO:** {fixed_po} (printed on the invoice)")
                elif vendor:
                    pos = [p["po_id"] for p in db.list_open_pos_for_vendor(conn, vendor["vendor_id"])]
                    cands = ((checks.get("PO_EXISTS") or {}).get("details") or {}).get("candidates") or []
                    cand_ids = [c["po_id"] for c in cands]
                    ordered = [c for c in cand_ids if c in pos] + [p for p in pos if p not in cand_ids]
                    labels = {}
                    for pid in ordered:
                        po = db.get_po(conn, pid)
                        rem = po["gross_total_paise"] - db.po_committed_paise(conn, pid)
                        labels[pid] = f"{pid} · {format_inr(rem)} remaining" + (" · suggested" if pid in cand_ids else "")
                    if ordered:
                        po_id = st.selectbox("Allocate to PO (no valid PO printed: confirm with procurement)",
                                             ordered, format_func=lambda p: labels[p], key=f"p_{r['run_id']}",
                                             disabled=not can_act)
                inv_no = r["invoice_number"]
                if not inv_no:
                    inv_no = st.text_input("Invoice number (missing on the document: confirm with the vendor)",
                                           key=f"i_{r['run_id']}", disabled=not can_act)
                blockers = review.approval_blockers(conn, run, po_id, inv_no)
            if blockers:
                st.markdown('<div class="blockers"><b>Cannot be approved as it stands</b><ul>'
                            + "".join(f"<li>{esc(b)}</li>" for b in blockers) + "</ul></div>", unsafe_allow_html=True)
                st.caption("Fix the source (vendor master, PO, or a corrected invoice from the vendor) and process "
                           "the invoice again, or reject it.")
            else:
                st.caption("Approving accepts the exceptions listed above. Your justification is saved in the audit trail.")
            note = st.text_input("Justification / reason (required)", key=f"j_{r['run_id']}", disabled=not can_act)
            a, b, c3 = st.columns([1, 1, 3])
            if a.button("Approve and allocate", key=f"ap_{r['run_id']}", type="primary",
                        disabled=bool(blockers) or not can_act):
                res = review.approve_review(ENGINE, r["run_id"], actor, note or None, po_id=po_id,
                                            invoice_number=inv_no or None)
                st.session_state["review_msg"] = (res.ok, res.message)
                st.rerun()
            if b.button("Reject invoice", key=f"rj_{r['run_id']}", disabled=not can_act):
                res = review.reject_review(ENGINE, r["run_id"], actor, note or None)
                st.session_state["review_msg"] = (res.ok, res.message)
                st.rerun()
            c3.caption(f"Run {r['run_id']}")
    footer()


# ==========================================================================
# Page 4 — Records (vendors, POs, ledger, policy)
# ==========================================================================


def page_records() -> None:
    page_header("Procurement records",
                "The reference data every invoice is checked against. Balances update as invoices are reserved, "
                "approved or released.")
    with ENGINE.connect() as conn:
        pos = db.list_pos(conn)
        vendors = {v["vendor_id"]: v for v in db.list_vendors(conn)}
        ledger = db.list_ledger(conn)
        po_rows = []
        for p in pos:
            committed = db.po_committed_paise(conn, p["po_id"])
            lines = db.get_po_lines(conn, p["po_id"])
            po_rows.append({
                "PO": p["po_id"], "Vendor": vendors[p["vendor_id"]]["legal_name"], "Status": p["status"],
                "Items": "; ".join(f"{l['quantity_milli'] // 1000} × {l['description']} @ {format_inr(l['unit_price_paise'])}" for l in lines),
                "PO value": format_inr(p["gross_total_paise"]), "Committed": format_inr(committed),
                "Remaining": format_inr(p["gross_total_paise"] - committed),
                "Used": committed / p["gross_total_paise"] if p["gross_total_paise"] else 0,
            })
    t1, t2, t3, t4 = st.tabs(["Purchase orders", "Vendors", "Invoice ledger", "Rules & policy"])
    with t1:
        st.dataframe(pd.DataFrame(po_rows), hide_index=True, use_container_width=True,
                     column_config={"Used": st.column_config.ProgressColumn("Used", format="percent", min_value=0, max_value=1)})
    with t2:
        st.dataframe(pd.DataFrame([
            {"ID": v["vendor_id"], "Legal name": v["legal_name"], "GSTIN": v["tax_id"],
             "Status": v["approval_status"].replace("_", " ").capitalize()} for v in vendors.values()
        ]), hide_index=True, use_container_width=True)
    with t3:
        if ledger:
            st.dataframe(pd.DataFrame([
                {"#": e["ledger_id"], "Vendor": vendors[e["vendor_id"]]["legal_name"], "Invoice": e["invoice_number"],
                 "PO": e["po_id"], "Amount": format_inr(e["amount_paise"]), "Status": e["allocation_status"],
                 "Source": {"SEED": "History", "PIPELINE": "InvoiceIQ", "REVIEWER": "Reviewer"}.get(e["source"], e["source"]),
                 "Run": e["run_id"] or "—"} for e in ledger
            ]), hide_index=True, use_container_width=True)
        st.caption("Only RESERVED, APPROVED and PAID entries use up PO balance. Rejected, blocked and failed runs never do.")
    with t4:
        st.markdown(f"**Policy version {config.POLICY_VERSION}**")
        st.markdown(
            f"- Currency: INR only\n"
            f"- Price tolerance: up to {config.PRICE_TOLERANCE_PERCENT}% above the PO price, max "
            f"{format_inr(config.PRICE_TOLERANCE_MAX_PAISE_PER_UNIT)} per unit\n"
            f"- Arithmetic and tax tolerance: {format_inr(config.ARITHMETIC_TOLERANCE_PAISE)}\n"
            f"- PO values are gross (including GST); balance = PO value − reserved − approved − paid\n"
            f"- Duplicate identity: vendor + invoice number with punctuation and case removed\n"
            f"- Invoice dates more than {config.MAX_FUTURE_DAYS} day ahead or {config.MAX_INVOICE_AGE_DAYS} days old go to review\n"
            f"- A PASS reserves the amount; a person still approves payment"
        )
        st.dataframe(pd.DataFrame([
            {"Rule": r.number, "Check": r.name, "If it fails": r.fail_outcome.capitalize(), "Detail": r.description}
            for r in RULES
        ]), hide_index=True, use_container_width=True)
    footer()


# ==========================================================================
# Navigation
# ==========================================================================

pages = [
    st.Page(page_process, title="Process an invoice", icon=":material/upload_file:", default=True),
    st.Page(page_dashboard, title="Dashboard", icon=":material/monitoring:", url_path="dashboard"),
    st.Page(page_review, title="Review queue", icon=":material/fact_check:", url_path="review"),
    st.Page(page_records, title="Procurement records", icon=":material/inventory_2:", url_path="records"),
]

st.logo(str(config.PROJECT_ROOT / "assets" / "wordmark-dark.svg"), size="large")
try:
    nav = st.navigation(pages, position="top")   # website-style top bar (Streamlit 1.46+)
except TypeError:
    nav = st.navigation(pages)                   # older Streamlit: sidebar navigation
nav.run()
