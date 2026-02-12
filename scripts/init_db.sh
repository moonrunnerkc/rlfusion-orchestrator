#!/bin/bash
# Author: Bradley R. Kinnard
# Initialize the RLFusion database

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_ROOT/db/rlfo_cache.db"

echo "Initializing RLFusion database..."

mkdir -p "$PROJECT_ROOT/db"

sqlite3 "$DB_PATH" << 'EOF'
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT,
    score REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT,
    response TEXT,
    reward REAL,
    rag_weight REAL,
    cag_weight REAL,
    graph_weight REAL,
    fused_context TEXT,
    proactive_suggestions TEXT,
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
EOF

echo "✅ Database initialized at: $DB_PATH"
echo "   Tables: cache, episodes, replay, conversations"
