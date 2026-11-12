"""Wipe all processing runs, reservations and reviewer actions and reload the seed data.
Run this before recording the demo video or before the live interview.

    python scripts/reset_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, database as db  # noqa: E402

if __name__ == "__main__":
    db.reset_demo(db.get_engine())
    print(f"Demo data reset on {config.database_url().split('@')[-1]}")
