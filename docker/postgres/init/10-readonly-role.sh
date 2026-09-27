#!/bin/sh
set -eu

if [ -z "${POSTGRES_READONLY_USER:-}" ] || [ -z "${POSTGRES_READONLY_PASSWORD:-}" ]; then
  echo "Skipping read-only role setup: POSTGRES_READONLY_USER/PASSWORD are empty"
  exit 0
fi

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=readonly_user="$POSTGRES_READONLY_USER" \
  --set=readonly_password="$POSTGRES_READONLY_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'readonly_user', :'readonly_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'readonly_user') \gexec
SELECT format('ALTER ROLE %I PASSWORD %L', :'readonly_user', :'readonly_password') \gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'readonly_user') \gexec
SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'readonly_user') \gexec
SELECT format('GRANT SELECT ON ALL TABLES IN SCHEMA public TO %I', :'readonly_user') \gexec
SELECT format(
  'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO %I',
  :'readonly_user'
) \gexec
SQL
