#!/usr/bin/env bash
# Restore the database from a backup file:  ./restore.sh backups/evallab_2026-11-06_0200.sql.gz
set -euo pipefail
cd "$(dirname "$0")"
[ -f "${1:-}" ] || { echo "Usage: ./restore.sh backups/<file>.sql.gz"; exit 1; }
read -rp "This REPLACES all current results with $1. Type YES to continue: " ok
[ "$ok" = "YES" ] || exit 1
docker compose stop app
docker compose exec -T db psql -U evallab -d evallab -c "DROP TABLE IF EXISTS eval_progress, eval_submissions;"
gunzip -c "$1" | docker compose exec -T db psql -U evallab -d evallab
docker compose start app
echo "✅ Restored from $1"
