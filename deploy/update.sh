#!/usr/bin/env bash
# Pull the latest code from GitHub and restart with zero data loss (results live in the database).
set -euo pipefail
cd "$(dirname "$0")"
git pull
docker compose up -d --build app
docker image prune -f >/dev/null
echo "✅ Updated. Participants already on the page should refresh."
