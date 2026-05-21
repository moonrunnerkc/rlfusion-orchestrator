# Author: Bradley R. Kinnard
# Idempotent migration to the 2026-05-21 episodes schema.
#
# Two responsibilities (both safe to rerun):
#   1. Drop the legacy `rag_weight` column from the 4-path schema if present.
#   2. Add the new 2026-05-21 columns: obs_features, from_cache,
#      policy_weights, effective_weights, had_empty_path, policy_action,
#      schema_version.
#   3. Wipe rows produced by the deleted add_batch_episodes helper so the
#      trainer is not biased by synthetic / hand-crafted PII rows.
#
# Run from the project root:
#   python3 scripts/migrate_episodes_to_two_path.py

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "db" / "rlfo_cache.db"

_NEW_COLUMNS: tuple[tuple[str, str], ...] = (
    ("obs_features", "TEXT"),
    ("from_cache", "INTEGER DEFAULT 0"),
    ("policy_weights", "TEXT"),
    ("effective_weights", "TEXT"),
    ("had_empty_path", "INTEGER DEFAULT 0"),
    ("policy_action", "TEXT"),
    ("schema_version", "INTEGER NOT NULL DEFAULT 2"),
)


def _drop_rag_column(conn: sqlite3.Connection) -> None:
    cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()]
    if "rag_weight" not in cols:
        return
    print("Dropping legacy rag_weight column via table rebuild.")
    conn.execute("BEGIN")
    conn.execute(
        """
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
        """
    )
    conn.execute(
        """
        INSERT INTO episodes_new
            (id, query, response, reward, cag_weight, graph_weight,
             fused_context, proactive_suggestions, created_at)
        SELECT id, query, response, reward, cag_weight, graph_weight,
               fused_context, proactive_suggestions, created_at
        FROM episodes
        """
    )
    conn.execute("DROP TABLE episodes")
    conn.execute("ALTER TABLE episodes_new RENAME TO episodes")
    conn.commit()


def _add_new_columns(conn: sqlite3.Connection) -> int:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
    added = 0
    for name, sql_type in _NEW_COLUMNS:
        if name in cols:
            continue
        # NOT NULL DEFAULT works in ALTER TABLE ADD COLUMN since SQLite 3.7.
        conn.execute(f"ALTER TABLE episodes ADD COLUMN {name} {sql_type}")
        added += 1
        print(f"  + episodes.{name} ({sql_type})")
    if added:
        conn.commit()
    return added


def _purge_synthetic_rows(conn: sqlite3.Connection) -> int:
    """Drop rows seeded by the deleted add_batch_episodes helper.

    The helper used a fake critique scorer and hand-rolled query strings;
    those rows would skew CQL training. Conservative match: response body
    references the synthetic scorer or the seed script by name.
    """
    cur = conn.execute(
        """
        DELETE FROM episodes
        WHERE response LIKE '%simulate_critique_score%'
           OR response LIKE '%add_batch_episodes%'
           OR proactive_suggestions LIKE '%add_batch_episodes%'
        """
    )
    deleted = cur.rowcount or 0
    if deleted:
        conn.commit()
        print(f"Removed {deleted} synthetic episodes (add_batch_episodes seed).")
    return deleted


def main() -> int:
    if not DB_PATH.exists():
        print(f"No DB at {DB_PATH}; nothing to migrate.")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    try:
        _drop_rag_column(conn)
        added = _add_new_columns(conn)
        purged = _purge_synthetic_rows(conn)
        if added == 0 and purged == 0:
            print("Episodes table already on the 2026-05-21 schema.")
        else:
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
