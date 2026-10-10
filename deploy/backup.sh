#!/bin/sh
# Back up PostgreSQL (jobs, skills, aliases) and the Qdrant job vectors.
# Uses whatever stack `docker compose` resolves (COMPOSE_FILE in .env on the server).
# Usage: deploy/backup.sh            env: BACKUP_DIR (default backups), BACKUP_KEEP_DAYS (7)
set -eu
cd "$(dirname "$0")/.."

dir=${BACKUP_DIR:-backups}
keep=${BACKUP_KEEP_DAYS:-7}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$dir"

# Write to .part files first so a failed run never leaves a truncated "good" backup.
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > "$dir/postgres-$stamp.dump.part"
docker compose exec -T api python -m scripts.qdrant_snapshot save \
  > "$dir/qdrant-$stamp.snapshot.part"
mv "$dir/postgres-$stamp.dump.part" "$dir/postgres-$stamp.dump"
mv "$dir/qdrant-$stamp.snapshot.part" "$dir/qdrant-$stamp.snapshot"

find "$dir" -type f \( -name 'postgres-*.dump' -o -name 'qdrant-*.snapshot' \) \
  -mtime +"$keep" -delete
echo "backup ok: $dir/postgres-$stamp.dump $dir/qdrant-$stamp.snapshot"
