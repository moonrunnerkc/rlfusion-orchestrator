#!/bin/sh
# Author: Bradley R. Kinnard
# Container entrypoint. Initializes the SQLite database on the bind-mounted
# volume the first time the container runs against a fresh host directory,
# then hands off to the configured CMD (uvicorn by default).

set -eu

DB_DIR="${RLFUSION_DB_DIR:-/app/db}"
DB_PATH="$DB_DIR/rlfo_cache.db"

if [ ! -f "$DB_PATH" ]; then
    echo "[entrypoint] initializing $DB_PATH"
    /bin/sh /app/scripts/init_db.sh
fi

exec "$@"
