"""Run every sample invoice through the REAL pipeline (real OpenAI extraction)
and compare each decision with the expected one. Use this to rehearse the demo
and to measure live extraction reliability.

    python scripts/run_samples.py            # resets demo data first
    python scripts/run_samples.py --repeat 3 # run the whole set 3 times (reset between) to check consistency
    python scripts/run_samples.py --only 08 09 10 11   # just these samples (saves free-tier quota)
"""
import argparse
import time
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, database as db  # noqa: E402
from src.pipeline import load_run_bundle, process_invoice  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="NN",
                        help="Run only samples whose file starts with these numbers, e.g. --only 08 09 10")
    parser.add_argument("--delay", type=float, default=None,
                        help="Seconds to wait between invoices (default 6 Gemini, 30 Groq, 0 OpenAI)")
    args = parser.parse_args()
    engine = db.init_db(db.get_engine())
    manifest = json.loads((config.SAMPLE_INVOICE_DIR / "manifest.json").read_text(encoding="utf-8"))
    if args.only:
        manifest = [m for m in manifest if m["file"][:2] in {o.zfill(2) for o in args.only}]
    # Free tiers: Gemini limits requests/minute; Groq limits tokens/minute (~3k tokens per invoice).
    delay = args.delay if args.delay is not None else {"gemini": 6.0, "groq": 30.0}.get(config.llm_provider(), 0.0)
    mismatches = 0
    for round_no in range(1, args.repeat + 1):
        if not args.no_reset:
            db.reset_demo(engine)
        print(f"\nRound {round_no} — {config.llm_provider()} · {config.extraction_model()}")
        for m in manifest:
            pdf = (config.SAMPLE_INVOICE_DIR / m["file"]).read_bytes()
            r = process_invoice(pdf, m["file"], engine=engine)
            time.sleep(delay if m["ground_truth"] else 0)
            ok = r.decision == m["expected"]
            mismatches += not ok
            print(f"  {'OK ' if ok else 'XX '} {m['file']:<34} {r.decision:<7} (expected {m['expected']})")
            if not ok:
                b = load_run_bundle(r.run_id, engine)
                print(f"      {r.headline}")
                for c in b["checks"]:
                    if c["outcome"] != "PASS":
                        print(f"      {c['rule_number']} {c['outcome']}: {c['reason'][:160]}")
                ver = b.get("verification") or {}
                for name, fc in (ver.get("field_checks") or {}).items():
                    if fc["status"] in ("UNVERIFIED", "INVALID"):
                        print(f"      field {name}: AI returned {fc['raw_value']!r} -> {fc['status']} ({fc['reason']})")
                ex = b.get("extraction") or {}
                if ex:
                    print(f"      AI raw: currency={ex.get('currency')!r} invoice_number={ex.get('invoice_number')!r} "
                          f"total={ex.get('total_amount')!r}")
    print(f"\n{mismatches} mismatch(es)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
