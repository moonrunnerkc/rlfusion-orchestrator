#!/bin/bash
# Author: Bradley R. Kinnard
# Initialize the RLFusion database. Idempotent: rerun any time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_ROOT/db/rlfo_cache.db"

echo "Initializing RLFusion database..."

mkdir -p "$PROJECT_ROOT/db"

sqlite3 "$DB_PATH" << 'EOF'
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    key_hash TEXT,
    value TEXT,
    score REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_key_hash ON cache(key_hash);

-- episodes carries one row per chat turn. Columns added in the 2026-05-21
-- schema bump:
--   obs_features        JSON-serialized 10-feature vector (F1.1)
--   from_cache          1 if served by CAG fast-path (F1.6)
--   policy_weights      JSON [cag, graph] from the policy directly (F1.7)
--   effective_weights   JSON [cag, graph] after rebalancing (F1.7)
--   had_empty_path      1 if policy was overridden by empty-path rebalance
--   policy_action       JSON raw pre-softmax action (F1.12)
--   schema_version      bumped on every breaking change (F6.5)
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    reward REAL,
    cag_weight REAL,
    graph_weight REAL,
    fused_context TEXT,
    proactive_suggestions TEXT,
    obs_features TEXT,
    from_cache INTEGER DEFAULT 0,
    policy_weights TEXT,
    effective_weights TEXT,
    had_empty_path INTEGER DEFAULT 0,
    policy_action TEXT,
    schema_version INTEGER NOT NULL DEFAULT 2,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS replay (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state BLOB,
    action BLOB,
    reward REAL,
    next_state BLOB,
    terminal INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_profile (
    fact_key TEXT PRIMARY KEY,
    fact_value TEXT,
    category TEXT
);
EOF

echo "Database initialized at: $DB_PATH"
echo "   Tables: cache, episodes, replay, conversations, user_profile"
