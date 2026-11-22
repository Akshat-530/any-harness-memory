#!/bin/sh
set -e
PGHOST="${MEMORY_PG_HOST:-postgres}"
PGPORT="${MEMORY_PG_PORT:-5432}"
echo "[memory] waiting for postgres at ${PGHOST}:${PGPORT} ..."
i=0
until python -c "import socket; socket.create_connection(('${PGHOST}', ${PGPORT}), timeout=3)" 2>/dev/null; do
  i=$((i+1)); if [ "$i" -ge 60 ]; then echo "[memory] postgres not reachable; starting anyway"; break; fi
  sleep 2
done
echo "[memory] ensuring database schema ..."
python - <<'PY'
import os, psycopg
dsn = "postgresql://%s:%s@%s:%s/%s" % (
    os.environ.get("MEMORY_PG_USER", "memory"),
    os.environ.get("MEMORY_PG_PASSWORD", ""),
    os.environ.get("MEMORY_PG_HOST", "postgres"),
    os.environ.get("MEMORY_PG_PORT", "5432"),
    os.environ.get("MEMORY_PG_DB", "memory"),
)
with psycopg.connect(dsn) as c:
    exists = c.execute("SELECT to_regclass('public.mem_canonical')").fetchone()[0]
    if exists:
        print("[memory] schema already present")
    else:
        sql = open("/app/schema.sql", "r", encoding="utf-8-sig").read()
        c.execute(sql)
        c.commit()
        print("[memory] schema applied from schema.sql")
PY
if [ "${MEMORY_MAINTENANCE:-0}" = "1" ]; then
  mkdir -p /data/logs
  echo "[memory] starting maintenance cron (nightly consolidate/dream/decay/promote/backup)"
  MEMORY_EAGER_PREWARM=0 python -u /app/tools/maintenance_cron.py >> /data/logs/cron_stderr.log 2>&1 &
fi
exec python /app/memory_http_server.py
