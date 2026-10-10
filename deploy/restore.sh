#!/bin/sh
# Restore a backup made by deploy/backup.sh into the running stack (replaces current data).
# Usage: deploy/restore.sh backups/postgres-<stamp>.dump backups/qdrant-<stamp>.snapshot
set -eu
cd "$(dirname "$0")/.."
[ $# -eq 2 ] || { echo "usage: $0 POSTGRES_DUMP QDRANT_SNAPSHOT" >&2; exit 2; }
[ -s "$1" ] && [ -s "$2" ] || { echo "backup file missing or empty" >&2; exit 2; }

docker compose up -d --wait postgres qdrant
docker compose stop api

# --no-owner/--no-privileges: role names may differ between machines; the read-only
# role keeps SELECT through the default privileges set by docker/postgres/init.
docker compose exec -T postgres sh -c \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges' \
  < "$1"
docker compose run --rm --no-deps -T api python -m scripts.qdrant_snapshot restore < "$2"

docker compose up -d
echo "restore ok"
