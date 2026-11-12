"""Deterministic stand-in for the OpenAI extractor, used only by automated tests.

It returns the known ground truth for each sample invoice (as a perfect
extractor would) and builds evidence quotes by locating each value in the
actual PDF text, exactly like the real model is instructed to do. This
isolates business logic from AI variability; live extraction is tested
separately against the real API.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evidence_validator import _token_in
from src.extractor import ExtractionError, ExtractionResult
from src.schemas import InvoiceExtraction

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "sample_invoices" / "manifest.json"
NUMERIC = {"subtotal", "tax_amount", "shipping_amount", "discount_amount", "total_amount"}


def load_manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _quote_for(value: str, text: str, numeric: bool) -> str:
    for line in text.splitlines():
        if _token_in(value, line, numeric):
            return line.strip()
    return value


def build_extraction(truth: dict, document_text: str) -> InvoiceExtraction:
    t = copy.deepcopy(truth)
    evidence = []
    for field in ("invoice_number", "invoice_date", "vendor_name", "vendor_tax_id", "po_number",
                  "subtotal", "tax_amount", "total_amount", "buyer_name", "buyer_tax_id"):
        if t.get(field):
            evidence.append({"field": field, "source_text": _quote_for(t[field], document_text, field in NUMERIC)})
    t["evidence"] = evidence
    t["ambiguous_fields"] = []
    t.setdefault("tax_components", [])
    t.setdefault("buyer_name", None)    # hand-written test invoices may not set the bill-to fields
    t.setdefault("buyer_tax_id", None)
    return InvoiceExtraction.model_validate(t)


class FakeExtractor:
    """Callable(document_text) -> ExtractionResult, keyed by the file being processed."""

    def __init__(self, truth: dict | None = None, fail: str | None = None, overrides: dict | None = None):
        self.truth = truth
        self.fail = fail
        self.overrides = overrides or {}

    def __call__(self, document_text: str) -> ExtractionResult:
        if self.fail:
            raise ExtractionError(self.fail)
        truth = {**self.truth, **self.overrides}
        return ExtractionResult(build_extraction(truth, document_text), model="fake-extractor")


def extractor_for(file_name: str, **kwargs) -> FakeExtractor:
    for entry in load_manifest():
        if entry["file"] == file_name:
            return FakeExtractor(entry["ground_truth"], **kwargs)
    raise KeyError(file_name)
