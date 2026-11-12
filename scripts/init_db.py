"""Create the tables and load the seed data if the database is empty.

    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, database as db  # noqa: E402

if __name__ == "__main__":
    engine = db.init_db(db.get_engine())
    print(f"Database ready: {config.database_url().split('@')[-1]}")
