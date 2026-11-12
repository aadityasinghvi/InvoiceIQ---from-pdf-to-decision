# Demo script

## Before you record or present

1. Open the app 5 minutes early (wakes the Streamlit app and the Supabase database).
2. **Settings** (top right) → **Reset demo data** → tick the box → Reset.
3. Open **Settings** and check it shows **Reading invoices with Groq**, **API key configured** and **Database PostgreSQL**.
4. Have `data/sample_invoices/` open in a file window so you can **drag in real PDFs**.
   The dropdown is a backup, but uploading a file shows the process takes real input.
5. Rehearse once with `python scripts/run_samples.py`, then reset again.

## 5-minute video (≈ 650 words spoken)

| Time | Show | Say (in your own words) |
|---|---|---|
| 0:00–0:30 | Process page | "AP teams match every vendor invoice to a PO by hand. InvoiceIQ takes the PDF and returns PASS, REVIEW or BLOCK with the reasons. One design rule: **the AI only reads the invoice, code makes the decision.** So it can't be talked into approving something, and the same invoice and data always give the same answer." |
| 0:30–1:45 | Upload `01_happy_path.pdf` | Narrate the timeline as it fills: "PDF read… the model extracts fields **and quotes the line it read each one from**… Python checks every quote really exists in the PDF… then vendor by GSTIN, the PO, line prices, GST, balance." Point at the stamp: "PASS means **reserved, not paid**: ₹4.75 L is now held on PO-1001 and a person approves payment." Open **Extracted data** to show quotes beside values. |
| 1:45–2:45 | Upload `02`, then `03` | "Edge case one: a vendor splitting one PO across invoices. This 12-month PO has 8 months billed. Two more months pass. Now they bill three, but only two are left." Open **Invoice vs PO**: the balance bar shows the over-run. "Comparing each invoice to the PO total would miss this; we track cumulative value and quantity." |
| 2:45–3:30 | Upload `07_duplicate_resend.pdf` | "Edge case two: the same invoice re-sent in a different template, numbered `INV/ABC/2041` instead of `INV-ABC-2041`. Different file, same invoice → BLOCK. The match is on vendor plus the normalised number; the database also has a unique constraint, so two simultaneous uploads can't both reserve." |
| 3:30–4:15 | Upload `06_missing_po.pdf`, then Review queue | "Edge case three: no PO printed. The AI is told never to guess one. Instead the system ranks this vendor's open POs and suggests PO-1004. A reviewer confirms it, with a justification. Approval still re-checks everything: the reviewer can't pick a PO whose items don't match, change the vendor, or approve an unverified total." Click **Approve and allocate**. |
| 4:15–5:00 | Dashboard → click a row → Audit trail | "Every run is stored with each stage, all 15 checks, the model and the policy version, so any decision can be traced months later. Also handled: price tolerance, unapproved vendors, another vendor's PO, bad arithmetic, scanned paper invoices via OCR, and an invoice that tries to instruct the AI to approve it. What I'd build next: email intake and goods-receipt (three-way) matching." |

## Live interview: suggested order

1. `01` Happy path → PASS (explain the architecture on this run).
2. `02` → PASS, `03` → REVIEW (split billing).
3. `04` → PASS with the tolerance note, `05` → REVIEW (amounts close but not exact).
4. `06` → REVIEW, then approve with PO-1004 in the **Review queue** (missing PO).
5. `07` → BLOCK (duplicate re-sent).
6. If they want more: `11` prompt injection, `12` scanned (OCR), `15` corrupted → ERROR.
7. Records page: show the PO balances that changed during the demo.

## Questions you should expect, with short answers

- **"What if the AI misreads the total?"** Each value has to appear inside a quote that must exist in the PDF, and the arithmetic, tax and PO price checks give three independent cross-checks. A misread total fails at least one, and the invoice goes to REVIEW.
- **"Why not let the LLM decide?"** It isn't repeatable or auditable, and it can be manipulated by text in the invoice (see sample 11). Rules are cheap, testable and explainable to an auditor.
- **"Why 2%?"** It's a policy assumption, kept in one place (`config.py`) with a version number stored on every run. In practice it would be set per vendor or category by the finance controller.
- **"What happens at scale?"** Extraction is the only slow or expensive step (one API call per invoice). The rules take milliseconds. The next step would be a queue worker reading from the AP inbox, with this UI as the exception workbench.
- **"How do you know it works?"** 212 automated tests on both SQLite and PostgreSQL, covering six-way concurrent submissions, plus `run_samples.py` against the real model.
- **"Can a reviewer just override everything?"** No. A reviewer can accept judgement calls (price variance, missing PO reference, old date) with a justification. Unverified totals, broken arithmetic, unknown vendors, wrong-vendor POs, quantity or money beyond the PO, and duplicates can't be overridden; the screen lists the blocker and approval is refused.
- **"What if saving fails halfway?"** Reservation, decision and audit record are one database transaction. A failure rolls everything back, so an ERROR never leaves money reserved. There's a test that injects exactly that failure.
- **"An external review found holes. What did you do?"** Reproduced each one, fixed it, and added a regression test for it (`tests/test_audit_findings.py`).
- **"What stops us paying an invoice meant for someone else?"** R16 checks the bill-to GSTIN or name against our company. The public GSTZen demo invoice is billed to Cipla Ltd, so it's blocked.
- **"Could someone resubmit the same PDF with a new invoice number?"** No. The file's fingerprint (SHA-256) is checked at processing and again at approval; same vendor + amount + date under a new number is flagged too.
- **"Invoices round to the rupee. Does that break the PO check?"** No. Up to ₹1 above the remaining PO balance is accepted as round-off; anything more needs a PO amendment.
