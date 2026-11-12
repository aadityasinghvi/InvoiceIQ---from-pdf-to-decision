"""Streamlit rendering helpers: styles, the live stage timeline and the run detail view."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src.money import format_inr, format_money
from src.pipeline import STAGES

DECISION_COLORS = {"PASS": "#19764A", "REVIEW": "#B26B00", "BLOCK": "#B42318", "ERROR": "#5B6577"}
OUTCOME_COLORS = {"PASS": "#19764A", "REVIEW": "#B26B00", "BLOCK": "#B42318", "NOT_EVALUATED": "#5B6577"}
OUTCOME_LABELS = {"PASS": "Pass", "REVIEW": "Review", "BLOCK": "Block", "NOT_EVALUATED": "Not checked"}
STAGE_ICONS = {"PENDING": "○", "RUNNING": "◐", "SUCCESS": "✓", "WARNING": "!", "FAILED": "✕"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700;800&display=swap');

:root {
  --desk: #EDF0F3; --paper: #FFFFFF; --ink: #17223B; --ink-2: #3D4860; --muted: #5B6577;
  --rule: #D9DEE5; --rule-soft: #E8ECF1; --carbon: #3346C9; --carbon-soft: #EBEEFC;
  --pass: #19764A; --review: #B26B00; --block: #B42318; --error: #5B6577;
}
html, body, .stApp,
.stApp :where(p, span, div, label, input, textarea, button, li, td, th, h1, h2, h3, h4, small, b, summary)
  :not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {
  font-family: 'Schibsted Grotesk', system-ui, sans-serif;
}
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: var(--desk); color: var(--ink); }
.block-container { padding-top: 4.6rem; padding-bottom: 4rem; max-width: 1280px; }

/* ---- top bar ---- */
header[data-testid="stHeader"] { background: var(--paper); border-bottom: 1px solid var(--rule); height: 3.6rem; }
header[data-testid="stHeader"] a, header[data-testid="stHeader"] span { font-weight: 600; }
[data-testid="stTopNavLink"], [data-testid="stSidebarNavLink"] { border-radius: 6px; }
[data-testid="stTopNavLink"][aria-current="page"], [data-testid="stTopNavLink"]:has(+ *)[aria-current="page"] {
  background: var(--carbon-soft);
}
[data-testid="stTopNavLink"][aria-current="page"] span { color: var(--carbon); }
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--rule); }

/* ---- type ---- */
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.02em; }
h1 { font-weight: 800; font-size: 2.35rem !important; line-height: 1.1 !important; padding-bottom: .2rem !important; }
h2 { font-weight: 700; font-size: 1.45rem !important; }
h3 { font-weight: 700; font-size: 1.12rem !important; }
.stMarkdown p { line-height: 1.55; }
.lede { font-size: 17px; color: var(--ink-2); max-width: 64ch; line-height: 1.55; margin: 2px 0 6px 0; }
.num, .stDataFrame { font-variant-numeric: tabular-nums; }
a { color: var(--carbon); }

/* ---- page header ---- */
.page-kicker { color: var(--carbon); font-weight: 700; font-size: 14px; margin-bottom: 4px; }

/* ---- the "how it decides" strip on the process page (a real sequence) ---- */
.flow { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 18px 0 26px 0;
  border: 1px solid var(--rule); border-radius: 10px; background: var(--paper); overflow: hidden; }
.flow div { padding: 14px 18px; border-right: 1px solid var(--rule); }
.flow div:last-child { border-right: none; }
.flow b { display: block; font-size: 15px; color: var(--ink); }
.flow span { font-size: 13.5px; color: var(--muted); line-height: 1.45; display: block; margin-top: 2px; }
.flow i { font-style: normal; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%;
  background: var(--carbon); color: #fff; font-size: 12px; font-weight: 700; margin-right: 8px; vertical-align: 1px; }
@media (max-width: 760px) { .flow { grid-template-columns: 1fr; } .flow div { border-right: none; border-bottom: 1px solid var(--rule); } }

/* ---- paper panels (bordered st.container) ---- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  background: var(--paper);
}
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--rule) !important; border-radius: 10px !important; }
.panel-title { font-weight: 700; font-size: 15px; color: var(--ink); margin-bottom: 2px; }
.panel-sub { font-size: 13px; color: var(--muted); margin-bottom: 8px; }

/* ---- file uploader as a drop zone ---- */
[data-testid="stFileUploaderDropzone"] { background: var(--carbon-soft); border: 1.5px dashed #9AA6E8; border-radius: 10px; padding: 26px 18px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--carbon); }

/* ---- buttons, inputs ---- */
.stButton > button, .stDownloadButton > button, [data-testid="stPopover"] button { border-radius: 8px; font-weight: 600; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--carbon); border-color: var(--carbon); padding: .55rem 1.3rem;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background: #2636A8; border-color: #2636A8; }
.stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
  background: #C9CFE9 !important; border-color: #C9CFE9 !important; color: #FFFFFF !important; }
:focus-visible { outline: 2px solid var(--carbon) !important; outline-offset: 2px; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="input"] input,
[data-baseweb="base-input"] input { background: var(--paper) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] { border-radius: 8px !important; border-color: var(--rule) !important; }
[class*="st-key-intake"], [class*="st-key-liverun"] { background: var(--paper); }
[data-testid="stSelectbox"] [data-baseweb="select"] div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] [data-baseweb] { background-color: var(--paper) !important; }
[data-testid="stTextInput"] [data-baseweb] input:disabled { background-color: var(--desk) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] [data-baseweb="input"] { border: 1px solid #C5CCD6 !important; }
[data-testid="stCode"] pre, .stCode pre { background: #F6F8FA !important; border: 1px solid var(--rule); }

/* ---- tabs ---- */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--rule); }
[data-baseweb="tab"] { font-weight: 600; color: var(--muted); padding: 8px 12px !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--carbon); }
[data-baseweb="tab-highlight"] { background: var(--carbon) !important; }

/* ---- expanders as rows of paper ---- */
[data-testid="stExpander"] details { background: var(--paper); border: 1px solid var(--rule); border-radius: 10px; }
[data-testid="stExpander"] summary { font-weight: 600; }

/* ---- result sheet + stamp: the one loud element ---- */
.sheet { position: relative; background: var(--paper); border: 1px solid var(--rule); border-radius: 10px;
  padding: 22px 24px 18px 24px; margin-bottom: 14px; box-shadow: 0 1px 0 var(--rule-soft), 0 10px 24px -18px rgba(23,34,59,.35); }
.sheet .bar { position: absolute; left: 0; top: 0; bottom: 0; width: 5px; border-radius: 10px 0 0 10px; }
.stamp-wrap { display: flex; gap: 26px; align-items: center; }
.stamp {
  flex: none; transform: rotate(-5deg);
  border: 4px double currentColor; border-radius: 6px;
  padding: 7px 20px 5px 20px; font-weight: 800; font-size: 38px; letter-spacing: 0.08em;
  line-height: 1.02; text-align: center; background: rgba(255,255,255,.7); mix-blend-mode: multiply;
}
.stamp small { display: block; font-size: 11.5px; letter-spacing: 0.02em; font-weight: 600; opacity: .9; }
.stamp.animate { animation: thump .5s cubic-bezier(.2,1.6,.4,1) both; }
@keyframes thump { 0% { transform: rotate(-5deg) scale(1.7); opacity: 0; } 100% { transform: rotate(-5deg) scale(1); opacity: 1; } }
@media (prefers-reduced-motion: reduce) { .stamp.animate { animation: none; } }
.headline { font-size: 19px; font-weight: 700; color: var(--ink); line-height: 1.35; letter-spacing: -0.01em; }
.meta { color: var(--muted); font-size: 13px; margin-top: 6px; }
.meta span + span::before { content: ""; display: inline-block; width: 1px; height: 11px; background: var(--rule);
  margin: 0 10px; vertical-align: -1px; }
.status-chip { display: inline-block; margin-top: 10px; padding: 3px 10px; border-radius: 6px; font-size: 13px;
  font-weight: 600; background: var(--desk); color: var(--ink-2); }
.explain { font-size: 14.5px; color: var(--ink-2); line-height: 1.6; border-top: 1px solid var(--rule-soft);
  margin-top: 16px; padding-top: 14px; }
.explain p { margin: 0 0 6px 0; }

.explain-rule { border-top: 1px solid var(--rule-soft); margin: 4px 0 0 0; }
[class*="st-key-sheet_"] { background: var(--paper); border-left-width: 6px !important; padding-left: 6px;
  box-shadow: 0 10px 24px -18px rgba(23,34,59,.35); }
[class*="st-key-sheet_PASS"] { border-left-color: var(--pass) !important; }
[class*="st-key-sheet_REVIEW"] { border-left-color: var(--review) !important; }
[class*="st-key-sheet_BLOCK"] { border-left-color: var(--block) !important; }
[class*="st-key-sheet_ERROR"] { border-left-color: var(--error) !important; }

/* ---- stage timeline ---- */
.timeline { list-style: none; padding: 0; margin: 0; position: relative; }
.timeline li { display: grid; grid-template-columns: 30px 1fr auto; gap: 12px; padding: 9px 0; position: relative; }
.timeline li:not(:last-child)::before {
  content: ""; position: absolute; left: 14px; top: 36px; bottom: -6px; width: 2px; background: var(--rule);
}
.timeline li.SUCCESS:not(:last-child)::before { background: #A9C9B8; }
.timeline .dot {
  width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center;
  font-weight: 700; font-size: 14px; border: 2px solid #C5CCD6; color: #98A1B0; background: var(--paper); z-index: 1;
}
.timeline .SUCCESS .dot { background: var(--pass); border-color: var(--pass); color: #fff; }
.timeline .WARNING .dot { background: var(--review); border-color: var(--review); color: #fff; }
.timeline .FAILED .dot { background: var(--block); border-color: var(--block); color: #fff; }
.timeline .RUNNING .dot { border-color: var(--carbon); color: var(--carbon); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { box-shadow: 0 0 0 6px rgba(51,70,201,.16); } }
@media (prefers-reduced-motion: reduce) { .timeline .RUNNING .dot { animation: none; } }
.timeline .label { font-weight: 600; color: var(--ink); font-size: 14.5px; }
.timeline .PENDING .label { color: #98A1B0; font-weight: 500; }
.timeline .detail { color: var(--ink-2); font-size: 12.5px; margin-top: 2px; line-height: 1.4; }
.timeline .ms { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; padding-top: 5px; }

/* ---- rule table ---- */
table.rules { width: 100%; border-collapse: collapse; font-size: 13.5px; background: var(--paper); }
table.rules td, table.rules th { padding: 10px 10px; border-bottom: 1px solid var(--rule-soft); vertical-align: top; text-align: left; }
table.rules td, table.rules th, table.rules tr { border-left: none !important; border-right: none !important; border-top: none !important; }
table.rules { border: 1px solid var(--rule) !important; border-radius: 10px; border-collapse: separate; border-spacing: 0; overflow: hidden; }
table.rules tr:last-child td { border-bottom: none; }
table.rules th { background: #F6F8FA; }
table.rules th { color: var(--muted); font-weight: 600; font-size: 12.5px; border-bottom-color: var(--rule); }
table.rules .code { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 5px; font-size: 12px; font-weight: 700; white-space: nowrap;
  border: 1.5px solid currentColor; background: var(--paper); }
.muted { color: var(--muted); }
.reason { color: var(--ink); }
.ev { font-size: 12.5px; color: var(--ink-2); }
.quote { background: #EEF5F1; border-left: 3px solid var(--pass); padding: 2px 8px; font-size: 12.5px; display: inline-block; }
.quote.bad { border-left-color: var(--review); background: #FCF4E6; }

/* ---- balance bar ---- */
.bal { height: 14px; border-radius: 7px; background: var(--rule-soft); overflow: hidden; display: flex; }
.bal span { display: block; height: 100%; }
.legend { font-size: 12.5px; color: var(--ink-2); margin-top: 6px; }
.legend b { color: var(--ink); }
.empty { border: 1.5px dashed #C5CCD6; border-radius: 10px; padding: 34px 28px; color: var(--muted); text-align: center;
  background: rgba(255,255,255,.5); line-height: 1.55; }
.empty b { color: var(--ink); }

/* ---- dashboard summary band ---- */
.band { display: grid; grid-template-columns: 1.1fr 2.6fr 1fr 1.2fr; background: var(--paper); border: 1px solid var(--rule);
  border-radius: 10px; margin: 14px 0 22px 0; }
.band > div { padding: 18px 22px; border-right: 1px solid var(--rule-soft); }
.band > div:last-child { border-right: none; }
.band .k { font-size: 13px; color: var(--muted); font-weight: 600; }
.band .v { font-size: 34px; font-weight: 800; letter-spacing: -0.02em; color: var(--ink); line-height: 1.15; font-variant-numeric: tabular-nums; }
.band .s { font-size: 12.5px; color: var(--muted); }
.mix { display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: 12px 0 10px 0; background: var(--rule-soft); }
.mix span { display: block; height: 100%; }
.mix span + span { border-left: 2px solid var(--paper); }
.mixkey { display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px; color: var(--ink-2); }
.mixkey b { font-size: 16px; color: var(--ink); margin-right: 4px; font-variant-numeric: tabular-nums; }
.mixkey i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
@media (max-width: 900px) { .band { grid-template-columns: 1fr 1fr; } .band > div { border-bottom: 1px solid var(--rule-soft); } }

/* ---- review queue ---- */
.qhead { display: flex; align-items: baseline; gap: 10px; margin: 22px 0 8px 0; }
.qhead h3 { margin: 0; padding: 0; }
.qcount { font-weight: 700; font-size: 13px; padding: 1px 9px; border-radius: 999px; background: var(--paper);
  border: 1px solid var(--rule); color: var(--ink-2); }
.blockers { background: #FDF0EE; border: 1px solid #F1C9C3; border-radius: 8px; padding: 10px 14px; font-size: 13.5px; color: #6E1A12; }
.blockers ul { margin: 6px 0 0 0; padding-left: 18px; }

/* ---- footer ---- */
.foot { margin-top: 48px; padding-top: 14px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 12.5px;
  display: flex; gap: 22px; flex-wrap: wrap; }
.foot b { color: var(--ink-2); font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


# --------------------------------------------------------------------------
# Live timeline
# --------------------------------------------------------------------------


def blank_stage_state() -> dict:
    return {key: {"status": "PENDING", "detail": "", "ms": None} for key, _ in STAGES}


def timeline_html(state: dict) -> str:
    items = []
    for key, label in STAGES:
        s = state.get(key, {"status": "PENDING", "detail": "", "ms": None})
        status = s["status"]
        ms = f"{s['ms']:,} ms" if s.get("ms") is not None else ""
        detail = esc(s.get("detail") or ("Running…" if status == "RUNNING" else ""))
        items.append(
            f'<li class="{status}"><div class="dot">{STAGE_ICONS.get(status, "○")}</div>'
            f'<div><div class="label">{esc(label)}</div><div class="detail">{detail}</div></div>'
            f'<div class="ms">{ms}</div></li>'
        )
    return f'<ul class="timeline">{"".join(items)}</ul>'


def stage_state_from_db(stages: list[dict]) -> dict:
    state = blank_stage_state()
    for s in stages:
        state[s["stage"]] = {"status": s["status"], "detail": s["detail"], "ms": s["duration_ms"]}
    return state


# --------------------------------------------------------------------------
# Decision header
# --------------------------------------------------------------------------


def stamp_html(decision: str, animate: bool = False) -> str:
    color = DECISION_COLORS.get(decision, "#5B6577")
    sub = {"PASS": "Ready for AP approval", "REVIEW": "Human review", "BLOCK": "Do not pay", "ERROR": "Not processed"}
    return (f'<div class="stamp{" animate" if animate else ""}" style="color:{color}">{esc(decision)}'
            f'<small>{sub.get(decision, "")}</small></div>')


def review_status_text(run: dict) -> str:
    rs = run.get("review_status")
    d = run.get("decision")
    if d == "PASS":
        return {"OPEN": "Reserved, awaiting AP approval", "APPROVED": "Approved for payment",
                "RELEASED": "Reservation released"}.get(rs, "")
    if d == "REVIEW":
        return {"OPEN": "Waiting in review queue", "APPROVED": "Approved by reviewer",
                "REJECTED": "Rejected by reviewer"}.get(rs, "")
    if d == "BLOCK":
        return "Closed, nothing reserved"
    if d == "ERROR":
        return "Failed, nothing reserved"
    return ""


def render_decision_header(run: dict, animate: bool = False) -> None:
    decision = run.get("decision") or "ERROR"
    meta = [f"Run {run['run_id']}", esc(run.get("file_name"))]
    if run.get("duration_ms") is not None:
        meta.append(f"{run['duration_ms'] / 1000:.1f} s")
    if run.get("model"):
        meta.append(f"model {esc(run['model'])}")
    meta.append(f"policy {esc(run.get('policy_version'))}")
    status = review_status_text(run)
    st.markdown(
        f'<div class="stamp-wrap">{stamp_html(decision, animate)}'
        f'<div><div class="headline">{esc(run.get("headline"))}</div>'
        f'<div class="meta">{"".join(f"<span>{m}</span>" for m in meta)}</div>'
        + (f'<div class="status-chip">{esc(status)}</div>' if status else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Run detail tabs
# --------------------------------------------------------------------------


def rules_table_html(checks: list[dict]) -> str:
    rows = []
    for c in checks:
        color = OUTCOME_COLORS.get(c["outcome"], "#5B6577")
        rows.append(
            f'<tr><td class="code">{esc(c.get("rule_number"))}</td>'
            f'<td><b>{esc(c["rule_name"])}</b><div class="reason">{esc(c["reason"])}</div></td>'
            f'<td class="ev">{esc(c.get("expected"))}</td>'
            f'<td class="ev">{esc(c.get("actual"))}</td>'
            f'<td><span class="pill" style="color:{color}">{OUTCOME_LABELS.get(c["outcome"], c["outcome"])}</span></td></tr>'
        )
    return ('<table class="rules"><tr><th></th><th>Check</th><th>Expected</th><th>Found</th><th>Result</th></tr>'
            + "".join(rows) + "</table>")


def balance_bar_html(po_total: int, committed: int, invoice: int) -> str:
    po_total = max(po_total, 1)
    remaining = max(po_total - committed, 0)
    used = min(committed, po_total) / po_total * 100
    fits = invoice <= remaining
    inv_w = min(invoice, remaining) / po_total * 100
    over = invoice - remaining if not fits else 0
    inv_color = "#19764A" if fits else "#B26B00"
    return (
        f'<div class="bal"><span style="width:{used:.2f}%;background:#8C96A8"></span>'
        f'<span style="width:{inv_w:.2f}%;background:{inv_color}"></span></div>'
        f'<div class="legend"><b>PO value</b> {format_inr(po_total)} &nbsp;·&nbsp; '
        f'<span style="color:#8C96A8">■</span> already committed {format_inr(committed)} &nbsp;·&nbsp; '
        f'<span style="color:{inv_color}">■</span> this invoice {format_inr(invoice)} &nbsp;·&nbsp; '
        f'remaining before this invoice <b>{format_inr(remaining)}</b>'
        + (f' &nbsp;·&nbsp; <b style="color:#B42318">over by {format_inr(over)}</b>' if over else "")
        + "</div>"
    )


def render_invoice_vs_po(bundle: dict) -> None:
    checks = {c["rule_code"]: c for c in bundle["checks"]}
    match = bundle.get("match") or {}
    po = (match.get("po") or {}).get("po")
    bal = checks.get("PO_BALANCE", {}).get("details")
    if po:
        st.markdown(f"**{po['po_id']}** · {esc(po.get('description'))} · status {po['status']} · vendor {po['vendor_id']}")
    if bal:
        st.markdown(balance_bar_html(bal["po_total"], bal["committed"], bal["invoice"]), unsafe_allow_html=True)
        st.write("")
    li = (checks.get("LINE_ITEMS") or {}).get("details") or {}
    comp = li.get("comparison")
    if comp:
        df = pd.DataFrame([
            {
                "Invoice line": r["description"],
                "PO item": r.get("po_description") or "—",
                "Matched by": {"CODE": "Item code", "DESCRIPTION": "Description", "NONE": "—"}.get(r["match_method"], r["match_method"]),
                "Qty billed": r.get("invoice_qty"),
                "PO qty": r.get("po_qty", "—"),
                "Invoiced before": r.get("previously_invoiced", "—"),
                "Billed price": format_inr(r.get("invoice_unit_price")),
                "PO price": format_inr(r.get("po_unit_price")),
                "Variance": (r.get("price_variance_pct") + "%") if r.get("price_variance_pct") else "—",
                "Result": r["status"].capitalize(),
            }
            for r in comp
        ])

        def color(v):
            if v in ("Ok", "Within tolerance"):
                return "color:#19764A;font-weight:600"
            return "color:#B26B00;font-weight:600"

        st.dataframe(df.style.map(color, subset=["Result"]), hide_index=True, use_container_width=True)
        for n in li.get("notes") or []:
            st.caption(n)
    cands = (checks.get("PO_EXISTS", {}).get("details") or {}).get("candidates")
    if cands:
        st.markdown("**Candidate POs** — ranked by line items, prices and remaining balance. A reviewer confirms one; "
                    "the system never assigns a PO on its own.")
        st.dataframe(pd.DataFrame([
            {"PO": c["po_id"], "Fit score": f"{c['score']}/100", "Remaining": c["remaining"], "Why": "; ".join(c["reasons"])}
            for c in cands
        ]), hide_index=True, use_container_width=True)
    vend = (checks.get("VENDOR_IDENTITY", {}).get("details") or {}).get("suggestions")
    if vend:
        st.markdown("**Possible vendor matches** (suggestions only)")
        st.dataframe(pd.DataFrame([{"Vendor": v["legal_name"], "ID": v["vendor_id"], "Similarity": v.get("score")} for v in vend]),
                     hide_index=True)
    if not (po or comp or cands):
        st.markdown('<div class="empty">No purchase order could be matched to this invoice.</div>', unsafe_allow_html=True)


FIELD_LABELS = {
    "invoice_number": "Invoice number", "invoice_date": "Invoice date", "vendor_name": "Vendor",
    "vendor_tax_id": "Vendor GSTIN", "po_number": "PO number", "currency": "Currency", "subtotal": "Subtotal",
    "tax_amount": "Tax", "shipping_amount": "Shipping", "discount_amount": "Discount", "total_amount": "Total",
}


def render_extracted(bundle: dict) -> None:
    ver = bundle.get("verification")
    if not ver:
        st.markdown('<div class="empty">Nothing was extracted for this run.</div>', unsafe_allow_html=True)
        return
    rows = []
    for name, label in FIELD_LABELS.items():
        fc = ver["field_checks"].get(name)
        if not fc:
            continue
        status = fc["status"]
        if status == "MISSING" and name in ("shipping_amount", "discount_amount"):
            continue
        icon = {"VERIFIED": "✓ Verified", "UNVERIFIED": "! Not verified", "MISSING": "— Missing", "INVALID": "✕ Invalid"}[status]
        quote = ""
        if fc.get("source_text"):
            cls = "quote" if status == "VERIFIED" else "quote bad"
            page = f' <span class="muted">p.{fc["page"]}</span>' if fc.get("page") else ""
            quote = f'<span class="{cls}">{esc(fc["source_text"][:140])}</span>{page}'
        color = {"VERIFIED": "#19764A", "UNVERIFIED": "#B26B00", "MISSING": "#5B6577", "INVALID": "#B42318"}[status]
        rows.append(
            f'<tr><td><b>{label}</b></td><td>{esc(fc.get("raw_value") or "—")}</td>'
            f'<td style="color:{color};font-weight:600;white-space:nowrap">{icon}</td>'
            f'<td>{quote}<div class="ev">{esc(fc.get("reason"))}</div></td></tr>'
        )
    st.markdown('<table class="rules"><tr><th>Field</th><th>Extracted value</th><th>Check</th><th>Evidence in the PDF</th></tr>'
                + "".join(rows) + "</table>", unsafe_allow_html=True)
    if ver.get("lines"):
        cur = ver.get("currency") or "INR"
        st.markdown("**Line items**")
        st.dataframe(pd.DataFrame([
            {
                "#": l["index"], "Code": l["item_code"], "Description": l["description"], "Qty": l["quantity"],
                "Unit price": format_money(l["unit_price_paise"], cur), "Line total": format_money(l["line_total_paise"], cur),
                "Verified": "Yes" if l["verified"] else "No — " + "; ".join(l["issues"]),
            }
            for l in ver["lines"]
        ]), hide_index=True, use_container_width=True)
    if ver.get("suspicious_instructions"):
        st.warning("The document contains instructions aimed at AI or automated systems. They were treated as data and ignored.")


def render_audit(bundle: dict) -> None:
    run = bundle["run"]
    st.markdown("**Processing stages**")
    st.dataframe(pd.DataFrame([
        {"#": s["seq"], "Stage": dict(STAGES).get(s["stage"], s["stage"]), "Result": s["status"].capitalize(),
         "Detail": s["detail"], "Duration (ms)": s["duration_ms"], "Finished (UTC)": s["finished_at"][:23].replace("T", " ")}
        for s in bundle["stages"]
    ]), hide_index=True, use_container_width=True)
    if bundle.get("ledger"):
        e = bundle["ledger"]
        st.markdown(f"**Ledger entry #{e['ledger_id']}** — {e['allocation_status']} · {format_inr(e['amount_paise'])} on "
                    f"{e['po_id']} · created {e['created_at'][:19].replace('T', ' ')} UTC")
    else:
        st.markdown("**Ledger** — no PO balance reserved by this run.")
    if bundle.get("actions"):
        st.markdown("**Reviewer actions**")
        st.dataframe(pd.DataFrame([
            {"When (UTC)": a["created_at"][:19].replace("T", " "), "Action": a["action"].replace("_", " ").capitalize(),
             "By": a["actor"], "Note": a["note"]}
            for a in bundle["actions"]
        ]), hide_index=True, use_container_width=True)
    st.caption(f"File SHA-256 {run.get('file_hash')} · {run.get('page_count') or '?'} page(s)"
               + (" · OCR used" if run.get("ocr_used") else "") + f" · policy {run.get('policy_version')}")


def render_run_detail(bundle: dict, animate: bool = False, show_timeline: bool = True) -> None:
    run = bundle["run"]
    decision = run.get("decision") or "ERROR"
    with st.container(border=True, key=f"sheet_{decision}_{run['run_id']}"):
        render_decision_header(run, animate)
        if decision == "ERROR":
            st.error(f"Stopped at **{dict(STAGES).get(run.get('error_stage'), run.get('error_stage'))}**: "
                     f"{run.get('error_message')}")
            st.markdown("A technical failure says nothing about whether the invoice is valid. Nothing was reserved; "
                        "retry, or process this invoice manually.")
        else:
            st.markdown('<div class="explain-rule"></div>', unsafe_allow_html=True)
            st.markdown(run.get("explanation") or "")
    if decision == "ERROR":
        if show_timeline:
            st.markdown(timeline_html(stage_state_from_db(bundle["stages"])), unsafe_allow_html=True)
        return
    t1, t2, t3, t4, t5 = st.tabs(["Checks", "Invoice vs PO", "Extracted data", "Document text", "Audit trail"])
    with t1:
        problems = [c for c in bundle["checks"] if c["outcome"] in ("BLOCK", "REVIEW")]
        if problems:
            st.markdown(rules_table_html(problems), unsafe_allow_html=True)
            with st.expander(f"Show all {len(bundle['checks'])} checks"):
                st.markdown(rules_table_html(bundle["checks"]), unsafe_allow_html=True)
        else:
            st.markdown(rules_table_html(bundle["checks"]), unsafe_allow_html=True)
    with t2:
        render_invoice_vs_po(bundle)
    with t3:
        render_extracted(bundle)
    with t4:
        st.code(run.get("document_text") or "", language=None)
    with t5:
        render_audit(bundle)
