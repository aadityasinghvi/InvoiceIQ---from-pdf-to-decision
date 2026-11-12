"""PDF reading, the extraction contract and independent evidence verification."""

import json
from types import SimpleNamespace

import openai
import pytest
from openai.lib._pydantic import to_strict_json_schema

from src import config, extractor
from src.evidence_validator import parse_invoice_date, verify_extraction
from src.pdf_reader import PdfReadError, read_pdf
from src.schemas import InvoiceExtraction
from tests.fake_extractor import build_extraction, load_manifest

SAMPLES = config.SAMPLE_INVOICE_DIR


def _truth(name):
    return next(m["ground_truth"] for m in load_manifest() if m["file"] == name)


def _doc(name):
    return read_pdf((SAMPLES / name).read_bytes())


# ---- PDF reading ----------------------------------------------------------


@pytest.mark.parametrize("name", ["01_happy_path.pdf", "04_price_within_tolerance.pdf",
                                  "05_price_over_tolerance.pdf", "02_split_billing_part1.pdf"])
def test_text_layer_pdfs_read_without_ocr(name):
    doc = _doc(name)
    assert not doc.ocr_used and doc.total_chars > 300


def test_scanned_pdf_uses_ocr_and_reads_key_values():
    doc = _doc("12_scanned_invoice.pdf")
    assert doc.ocr_used
    for value in ("KL/INV/3385", "PO-1005", "85,120.00", "24KRSHN7788P1Z9"):
        assert value in doc.text


@pytest.mark.parametrize("data, msg", [
    (b"", "empty"),
    (b"hello world, not a pdf", "not a PDF"),
    ((SAMPLES / "15_corrupted.pdf").read_bytes(), "corrupted"),
])
def test_unreadable_files_raise(data, msg):
    with pytest.raises(PdfReadError, match=msg):
        read_pdf(data)


# ---- extraction contract --------------------------------------------------


def test_schema_is_valid_for_openai_strict_mode():
    schema = to_strict_json_schema(InvoiceExtraction)
    assert set(schema["required"]) == set(schema["properties"])  # every field required (nullable)


def test_prompt_forbids_guessing_and_embedded_instructions():
    p = extractor.EXTRACTION_INSTRUCTIONS.lower()
    for phrase in ("never invent", "never infer a purchase order", "never calculate", "untrusted data"):
        assert phrase in p


class _FakeResponses:
    def __init__(self, exc=None, parsed=None):
        self.exc, self.parsed = exc, parsed

    def parse(self, **kwargs):
        if self.exc:
            raise self.exc
        return SimpleNamespace(output_parsed=self.parsed, model="m", usage=None)


def _patch_client(monkeypatch, responses):
    monkeypatch.setattr(extractor, "_client", lambda: SimpleNamespace(responses=responses))


def test_api_timeout_becomes_extraction_error(monkeypatch):
    exc = openai.APITimeoutError(request=SimpleNamespace(method="POST", url="x"))
    _patch_client(monkeypatch, _FakeResponses(exc=exc))
    with pytest.raises(extractor.ExtractionError, match="timed out"):
        extractor.extract_invoice_openai("text")


def test_refusal_becomes_extraction_error(monkeypatch):
    _patch_client(monkeypatch, _FakeResponses(parsed=None))
    with pytest.raises(extractor.ExtractionError, match="refused"):
        extractor.extract_invoice_openai("text")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(extractor.ExtractionError, match="OPENAI_API_KEY"):
        extractor._client()


# ---- evidence verification -----------------------------------------------


def test_perfect_extraction_is_fully_verified():
    doc = _doc("01_happy_path.pdf")
    inv = verify_extraction(build_extraction(_truth("01_happy_path.pdf"), doc.text), doc)
    bad = {k: v.reason for k, v in inv.field_checks.items() if v.status not in ("VERIFIED", "MISSING")}
    assert bad == {}
    assert all(l.verified for l in inv.lines)
    assert inv.total_paise == 47_495_000
    assert inv.tax_paise == 7_245_000 and inv.tax_source == "components"  # CGST + SGST summed


def test_invented_invoice_number_is_not_verified():
    doc = _doc("01_happy_path.pdf")
    ex = build_extraction({**_truth("01_happy_path.pdf"), "invoice_number": "INV-ABC-9999"}, doc.text)
    inv = verify_extraction(ex, doc)
    assert inv.field_checks["invoice_number"].status == "UNVERIFIED"


def test_misread_total_is_not_verified_even_with_a_real_quote():
    doc = _doc("01_happy_path.pdf")
    ex = build_extraction(_truth("01_happy_path.pdf"), doc.text)
    ex.total_amount = "4,74,590.00"  # digits transposed, quote still points at the real line
    inv = verify_extraction(ex, doc)
    assert inv.field_checks["total_amount"].status == "UNVERIFIED"
    assert "does not appear in the quoted text" in inv.field_checks["total_amount"].reason


def test_fabricated_quote_is_rejected():
    doc = _doc("01_happy_path.pdf")
    ex = build_extraction(_truth("01_happy_path.pdf"), doc.text)
    for ev in ex.evidence:
        if ev.field == "total_amount":
            ev.source_text = "Grand Total (INR) 4,74,950.00 APPROVED"
    inv = verify_extraction(ex, doc)
    assert inv.field_checks["total_amount"].status == "UNVERIFIED"


def test_number_inside_a_bigger_number_does_not_count():
    doc = _doc("04_price_within_tolerance.pdf")
    ex = build_extraction(_truth("04_price_within_tolerance.pdf"), doc.text)
    ex.line_items[0].line_total = "0,000.00"  # substring of ₹50,000.00
    inv = verify_extraction(ex, doc)
    assert not inv.lines[0].verified


def test_formatting_differences_still_verify():
    doc = _doc("04_price_within_tolerance.pdf")
    ex = build_extraction(_truth("04_price_within_tolerance.pdf"), doc.text)
    ex.line_items[1].unit_price = "3250.00"  # printed as ₹3,250.00
    inv = verify_extraction(ex, doc)
    assert inv.lines[1].verified and inv.lines[1].unit_price_paise == 325_000


def test_prompt_injection_detected_even_if_model_misses_it():
    doc = _doc("11_prompt_injection.pdf")
    ex = build_extraction({**_truth("11_prompt_injection.pdf"), "suspicious_instructions": False}, doc.text)
    assert verify_extraction(ex, doc).suspicious_instructions is True


@pytest.mark.parametrize("raw, iso", [
    ("02 Nov 2026", "2026-11-02"), ("05/11/2026", "2026-11-05"), ("2026-11-05", "2026-11-05"),
    ("5th November 2026", "2026-11-05"), ("Nov 5, 2026", "2026-11-05"),
])
def test_date_formats(raw, iso):
    assert parse_invoice_date(raw).isoformat() == iso


# ---- provider selection and Gemini -----------------------------------------


def test_provider_auto_selects_gemini_when_only_gemini_key(monkeypatch):
    from src import config as cfg
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert cfg.llm_provider() == "gemini" and cfg.has_llm_key()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert cfg.llm_provider() == "openai" and not cfg.has_llm_key()


def test_gemini_schema_is_self_contained():
    import json
    s = json.dumps(extractor.inline_json_schema(InvoiceExtraction.model_json_schema()))
    assert "$ref" not in s and "$defs" not in s


class _FakeGeminiModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(text=out, model_version="gemini-test", usage_metadata=None)


def _gemini(monkeypatch, outcomes):
    models = _FakeGeminiModels(outcomes)
    monkeypatch.setattr(extractor, "_gemini_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(config, "RATE_LIMIT_RETRY_WAITS_SECONDS", (0, 0))
    return models


def _api_error(code):
    from google.genai import errors
    return errors.APIError(code, {"error": {"message": "x", "status": "x"}})


def test_gemini_valid_json_is_parsed(monkeypatch):
    doc = _doc("01_happy_path.pdf")
    payload = build_extraction(_truth("01_happy_path.pdf"), doc.text).model_dump_json()
    _gemini(monkeypatch, [payload])
    result = extractor.extract_invoice_gemini(doc.text)
    assert result.extraction.invoice_number == "INV-ABC-2041" and result.model == "gemini-test"


def test_gemini_rate_limit_is_retried(monkeypatch):
    doc = _doc("01_happy_path.pdf")
    payload = build_extraction(_truth("01_happy_path.pdf"), doc.text).model_dump_json()
    models = _gemini(monkeypatch, [_api_error(429), payload])
    assert extractor.extract_invoice_gemini(doc.text).extraction.total_amount == "4,74,950.00"
    assert models.calls == 2


def test_gemini_quota_exhausted_becomes_error(monkeypatch):
    _gemini(monkeypatch, [_api_error(429)] * 3)
    with pytest.raises(extractor.ExtractionError, match="per-minute rate limit"):
        extractor.extract_invoice_gemini("text")


def test_gemini_bad_key_and_bad_json(monkeypatch):
    _gemini(monkeypatch, [_api_error(403)])
    with pytest.raises(extractor.ExtractionError, match="rejected the request"):
        extractor.extract_invoice_gemini("text")
    _gemini(monkeypatch, ['{"invoice_number": 5}'])
    with pytest.raises(extractor.ExtractionError, match="does not match the schema"):
        extractor.extract_invoice_gemini("text")


def test_gemini_server_timeout_is_retried(monkeypatch):
    doc = _doc("01_happy_path.pdf")
    payload = build_extraction(_truth("01_happy_path.pdf"), doc.text).model_dump_json()
    models = _gemini(monkeypatch, [_api_error(504), payload])
    assert extractor.extract_invoice_gemini(doc.text).extraction.invoice_number == "INV-ABC-2041"
    assert models.calls == 2


def test_gemini_unsupported_thinking_level_falls_back(monkeypatch):
    doc = _doc("01_happy_path.pdf")
    payload = build_extraction(_truth("01_happy_path.pdf"), doc.text).model_dump_json()
    from google.genai import errors
    bad = errors.APIError(400, {"error": {"message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}})
    models = _gemini(monkeypatch, [bad, payload])
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    assert extractor.extract_invoice_gemini(doc.text).extraction.total_amount == "4,74,950.00"
    assert models.calls == 2


def test_gemini_daily_quota_is_not_retried(monkeypatch):
    from google.genai import errors
    daily = errors.APIError(429, {"error": {"message": "Quota exceeded for metric: "
                                  "generate_content_free_tier_requests, quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                  "status": "RESOURCE_EXHAUSTED"}})
    models = _gemini(monkeypatch, [daily, daily, daily])
    with pytest.raises(extractor.ExtractionError, match="DAILY quota"):
        extractor.extract_invoice_gemini("text")
    assert models.calls == 1



# ---- Groq (standard-library HTTP; no OpenAI SDK) ------------------------------


class _FakeGroqHTTP:
    """Stands in for extractor._groq_http: returns (status, headers, body) in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.payloads = []

    def __call__(self, payload, api_key):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.outcomes.pop(0)


def _groq_ok(content):
    return (200, {}, {"model": "openai/gpt-oss-120b",
                      "choices": [{"message": {"role": "assistant", "content": content}}],
                      "usage": {"prompt_tokens": 2000, "completion_tokens": 900}})


def _groq_err(status, message, headers=None):
    return (status, headers or {}, {"error": {"message": message}})


@pytest.fixture
def groq_env(monkeypatch):
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_MODEL", "GROQ_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(extractor.time, "sleep", lambda s: None)


def _groq(monkeypatch, outcomes):
    fake = _FakeGroqHTTP(outcomes)
    monkeypatch.setattr(extractor, "_groq_http", fake)
    return fake


def _answer(name="01_happy_path.pdf"):
    doc = _doc(name)
    return doc, build_extraction(_truth(name), doc.text).model_dump_json()


def test_provider_auto_selects_groq(monkeypatch):
    from src import config as cfg
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert cfg.llm_provider() == "groq" and cfg.extraction_model() == "openai/gpt-oss-120b"


def test_groq_request_shape_and_parse(monkeypatch, groq_env):
    doc, answer = _answer()
    fake = _groq(monkeypatch, [_groq_ok(answer)])
    result = extractor.extract_invoice(doc.text)  # through the provider dispatcher
    assert result.extraction.invoice_number == "INV-ABC-2041" and result.input_tokens == 2000
    p = fake.payloads[0]
    assert p["model"] == "openai/gpt-oss-120b" and p["reasoning_effort"] == "low"
    assert p["response_format"]["json_schema"]["strict"] is True
    schema = p["response_format"]["json_schema"]["schema"]
    assert "$ref" not in str(schema) and "anyOf" not in str(schema) and "'enum'" not in str(schema)
    assert schema["properties"]["po_number"]["type"] == ["string", "null"]


def test_groq_rate_limit_is_retried(monkeypatch, groq_env):
    doc, answer = _answer()
    fake = _groq(monkeypatch, [_groq_err(429, "Rate limit reached on tokens per minute (TPM)", {"retry-after": "2"}),
                               _groq_ok(answer)])
    assert extractor.extract_invoice_groq(doc.text).extraction.total_amount == "4,74,950.00"
    assert len(fake.payloads) == 2


def test_groq_daily_limit_not_retried(monkeypatch, groq_env):
    fake = _groq(monkeypatch, [_groq_err(429, "Rate limit reached on tokens per day (TPD): Limit 200000")])
    with pytest.raises(extractor.ExtractionError, match="DAILY limit"):
        extractor.extract_invoice_groq("text")
    assert len(fake.payloads) == 1


def test_groq_drops_unsupported_reasoning_effort(monkeypatch, groq_env):
    doc, answer = _answer()
    fake = _groq(monkeypatch, [_groq_err(400, "`reasoning_effort` is not supported with this model"), _groq_ok(answer)])
    extractor.extract_invoice_groq(doc.text)
    assert "reasoning_effort" not in fake.payloads[1]


def test_groq_bad_key_and_bad_output(monkeypatch, groq_env):
    _groq(monkeypatch, [_groq_err(401, "Invalid API Key")])
    with pytest.raises(extractor.ExtractionError, match="rejected the API key"):
        extractor.extract_invoice_groq("text")
    _groq(monkeypatch, [_groq_ok('{"not": "an invoice"}')])
    with pytest.raises(extractor.ExtractionError, match="does not match the schema"):
        extractor.extract_invoice_groq("text")


def test_groq_over_real_http(monkeypatch, groq_env):
    """Exercise the real urllib code over a socket against a local fake Groq server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    doc, answer = _answer()
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body["model"]))
            if len(seen) == 1:
                out, code, extra = b'{"error":{"message":"Rate limit reached on tokens per minute"}}', 429, {"retry-after": "0"}
            else:
                out, code, extra = json.dumps(_groq_ok(answer)[2]).encode(), 200, {}
            self.send_response(code)
            for k, v in {"Content-Type": "application/json", **extra}.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("GROQ_BASE_URL", f"http://127.0.0.1:{server.server_port}/openai/v1")
        result = extractor.extract_invoice_groq(doc.text)
    finally:
        server.shutdown()
    assert result.extraction.invoice_number == "INV-ABC-2041"
    assert seen[0] == ("/openai/v1/chat/completions", "Bearer gsk_test", "openai/gpt-oss-120b")
    assert len(seen) == 2  # the 429 was retried


def test_groq_path_never_imports_jiter(monkeypatch, groq_env):
    """Windows Smart App Control blocks jiter's compiled file; the Groq path must not need it."""
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "jiter" or name.startswith("jiter."):
            raise ImportError("blocked by Application Control policy (simulated)")
        return real_import(name, *args, **kwargs)

    doc, answer = _answer()
    _groq(monkeypatch, [_groq_ok(answer)])
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert extractor.extract_invoice_groq(doc.text).extraction.invoice_number == "INV-ABC-2041"



def test_unknown_evidence_labels_are_dropped_not_fatal(monkeypatch, groq_env):
    doc, answer = _answer()
    data = json.loads(answer)
    data["evidence"].append({"field": "tax_components", "source_text": "CGST @ 9%"})
    data["evidence"].append({"field": "line_items", "source_text": "Dell"})
    data["ambiguous_fields"] = ["cgst"]
    _groq(monkeypatch, [_groq_ok(json.dumps(data))])
    ex = extractor.extract_invoice_groq(doc.text).extraction
    assert {e.field for e in ex.evidence} <= set(config.EVIDENCE_FIELDS) | {"currency"}
    assert ex.ambiguous_fields == []
    # Still fully verifiable against the PDF
    inv = verify_extraction(ex, doc)
    assert inv.field_checks["total_amount"].status == "VERIFIED"


def test_groq_schema_rejection_is_retried(monkeypatch, groq_env):
    doc, answer = _answer()
    fake = _groq(monkeypatch, [
        _groq_err(400, "Generated JSON does not match the expected schema. Please adjust your prompt."),
        _groq_ok(answer)])
    assert extractor.extract_invoice_groq(doc.text).extraction.invoice_number == "INV-ABC-2041"
    assert len(fake.payloads) == 2


def test_groq_network_blip_is_retried(monkeypatch, groq_env):
    doc, answer = _answer()
    outcomes = [extractor._NetworkError("Could not reach Groq: [Errno 11001] getaddrinfo failed"), _groq_ok(answer)]

    def flaky(payload, api_key):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(extractor, "_groq_http", flaky)
    assert extractor.extract_invoice_groq(doc.text).extraction.invoice_number == "INV-ABC-2041"


def test_network_down_ends_as_error(monkeypatch, groq_env):
    def down(payload, api_key):
        raise extractor._NetworkError("Could not reach Groq: getaddrinfo failed")
    monkeypatch.setattr(extractor, "_groq_http", down)
    with pytest.raises(extractor.ExtractionError, match="check the internet connection"):
        extractor.extract_invoice_groq("text")


def test_prompt_lists_common_invoice_number_labels():
    p = extractor.EXTRACTION_INSTRUCTIONS
    assert "Bill Reference" in p and "Bill No" in p and "GSTIN" in p



def test_ocr_uses_bundled_language_file_by_default(monkeypatch):
    from src.pdf_reader import tessdata_dir
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    assert tessdata_dir() == str(config.BUNDLED_TESSDATA_DIR)
    assert (config.BUNDLED_TESSDATA_DIR / "eng.traineddata").stat().st_size > 1_000_000



@pytest.mark.parametrize("name", ["01_happy_path.pdf", "04_price_within_tolerance.pdf",
                                  "05_price_over_tolerance.pdf", "02_split_billing_part1.pdf"])
def test_currency_determined_by_code_when_extractor_omits_it(name):
    doc = _doc(name)
    ex = build_extraction({**_truth(name), "currency": None}, doc.text)
    inv = verify_extraction(ex, doc)
    assert inv.currency == "INR"
    assert inv.field_checks["currency"].status == "VERIFIED"
    assert "determined by code" in inv.field_checks["currency"].reason


def test_currency_not_assumed_when_several_or_none_printed():
    from src.evidence_validator import currencies_in
    assert set(currencies_in("Total USD 100\nTotal INR 8,300")) == {"USD", "INR"}
    assert currencies_in("Labour: 12 hours. Orders. Repairs.") == {}   # 'rs.' inside words is not Rs.
    assert set(currencies_in("Amount Rs. 4,500.00")) == {"INR"}


@pytest.mark.parametrize("raw", ["INR", "₹", "Rs.", "Indian Rupees (INR)", "Rupees", "inr"])
def test_currency_spellings_normalise_to_inr(raw):
    from src.evidence_validator import normalize_currency
    assert normalize_currency(raw) == "INR"



@pytest.mark.parametrize("junk", ["null", "None", "N/A", "Not specified", "unknown", "-", "Not printed"])
def test_placeholder_currency_text_is_treated_as_missing(junk):
    doc = _doc("01_happy_path.pdf")
    inv = verify_extraction(build_extraction({**_truth("01_happy_path.pdf"), "currency": junk}, doc.text), doc)
    assert inv.currency == "INR" and inv.field_checks["currency"].status == "VERIFIED"


def test_wrong_real_currency_still_needs_review():
    doc = _doc("01_happy_path.pdf")
    inv = verify_extraction(build_extraction({**_truth("01_happy_path.pdf"), "currency": "USD"}, doc.text), doc)
    assert inv.field_checks["currency"].status == "UNVERIFIED"



@pytest.mark.parametrize("name", [m["file"] for m in load_manifest()
                                  if m["ground_truth"] and m["ground_truth"]["invoice_number"]
                                  and m["file"] != "12_scanned_invoice.pdf"])
def test_invoice_number_recovered_from_label_when_ai_omits_it(name):
    doc = _doc(name)
    inv = verify_extraction(build_extraction({**_truth(name), "invoice_number": None}, doc.text), doc)
    assert inv.invoice_number == _truth(name)["invoice_number"]
    assert inv.field_checks["invoice_number"].status == "VERIFIED"
    assert "read by code" in inv.field_checks["invoice_number"].reason


def test_no_invoice_number_is_invented_when_none_is_printed():
    doc = _doc("13_missing_invoice_number.pdf")
    inv = verify_extraction(build_extraction(_truth("13_missing_invoice_number.pdf"), doc.text), doc)
    assert inv.invoice_number is None and inv.field_checks["invoice_number"].status == "MISSING"


def test_ai_flagged_ambiguous_number_is_not_overridden():
    doc = _doc("04_price_within_tolerance.pdf")
    t = {**_truth("04_price_within_tolerance.pdf"), "invoice_number": None}
    ex = build_extraction(t, doc.text)
    ex.ambiguous_fields = ["invoice_number"]
    assert verify_extraction(ex, doc).invoice_number is None


def test_two_different_labelled_numbers_means_no_guess():
    from src.evidence_validator import labelled_invoice_numbers
    text = "Invoice No: INV-100\nBill No. B-2001\nInvoice Date: 05/11/2026"
    assert set(labelled_invoice_numbers(text)) == {"INV-100", "B-2001"}
    assert labelled_invoice_numbers("Invoice Date: 05/11/2026\nTAX INVOICE") == {}



def test_groq_is_preferred_and_placeholders_are_ignored(monkeypatch):
    from src import config as cfg
    for k in ("OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "paste-your-gemini-key-here")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real_looking_key_1234567890")
    assert cfg.llm_provider() == "groq" and cfg.has_llm_key()
    monkeypatch.setenv("GROQ_API_KEY", "paste-your-groq-key-here")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert not cfg.has_llm_key()   # a placeholder is not a key
