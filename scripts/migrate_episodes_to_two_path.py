# Author: Bradley R. Kinnard
# Drops the legacy rag_weight column from the episodes table.
# SQLite < 3.35 does not support DROP COLUMN, so we rebuild the table
# in place.
#
# Run from the project root:
#   python3 scripts/migrate_episodes_to_two_path.py
#
# Idempotent: bails out cleanly if the column is already gone.

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "rlfo_cache.db"


def main() -> int:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}; nothing to migrate.")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()]
        if "rag_weight" not in cols:
            print("episodes table is already on the 2-path schema; nothing to do.")
            return 0

        print(f"Migrating episodes table at {DB_PATH}: dropping rag_weight column")
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE episodes_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                reward REAL,
                cag_weight REAL,
                graph_weight REAL,
                fused_context TEXT,
                proactive_suggestions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            INSERT INTO episodes_new
                (id, query, response, reward, cag_weight, graph_weight,
                 fused_context, proactive_suggestions, created_at)
            SELECT id, query, response, reward, cag_weight, graph_weight,
                   fused_context, proactive_suggestions, created_at
            FROM episodes
        """)
        conn.execute("DROP TABLE episodes")
        conn.execute("ALTER TABLE episodes_new RENAME TO episodes")
        conn.commit()
        print("Migration complete.")
        return 0
    except sqlite3.Error as exc:
        conn.rollback()
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
