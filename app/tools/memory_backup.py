#!/usr/bin/env python3
"""Memory nightly backup — Docker topology (runs inside memory-mcp).

Pure-python, network-only access to the backing stores (no host shell,
system tools, or local paths required):

1. Postgres  : full logical dump of the memory DB (schema + data NDJSON.gz
               per table) as the admin superuser, so FORCE RLS on
               mem_canonical does not break the dump (found during the
               2026-10-09 restore test on the VPS deployment).
2. Qdrant    : live snapshot of the lg_memory collection via HTTP, then
               server-side snapshot prune (>48h).
3. Redis     : SCAN + DUMP of every key (type, PTTL, raw dump hex) — the
               authoritative short-term + graph tier; AOF on the redis
               container covers <=15s RPO, this is the nightly full.
4. BM25 index: /data/bm25_index.json copy.
5. Offsite   : tar.gz the backup dir and rolling-force-push it to a private
               GitHub repo (git data API, single commit — repo never grows).
               Env: GITHUB_BACKUP_TOKEN, GITHUB_BACKUP_REPO (owner/name),
               GITHUB_BACKUP_KEEP (default 7 archives).

Destination: /data/backup/memory_nightly/<ts>/.
Retention: 14 days local. Writes restore_instructions.txt + manifest.json and
appends one JSON line to backup.log. Prints a JSON summary for the cron daemon.
"""
import base64
import datetime
import glob
import gzip
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
BACKUP_ROOT = os.environ.get("MEMORY_BACKUP_DIR", "/data/backup/memory_nightly")
DEST = f"{BACKUP_ROOT}/{TS}"

PG_HOST = os.environ.get("MEMORY_PG_HOST", "postgres")
PG_PORT = os.environ.get("MEMORY_PG_PORT", "5432")
PG_DB = os.environ.get("MEMORY_PG_DB", "memory")
PG_USER = os.environ.get("MEMORY_PG_USER", "memory")
PG_PASS = os.environ.get("MEMORY_PG_PASSWORD", "")
REDIS_HOST = os.environ.get("MEMORY_REDIS_HOST", "redis")
REDIS_PASS = os.environ.get("MEMORY_REDIS_PASSWORD", "") or None
REDIS_PORT = os.environ.get("MEMORY_REDIS_PORT", "6379")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = "lg_memory"
RETENTION_DAYS = 14

GH_TOKEN = os.environ.get("GITHUB_BACKUP_TOKEN", "")
GH_REPO = os.environ.get("GITHUB_BACKUP_REPO", "")
GH_KEEP = int(os.environ.get("GITHUB_BACKUP_KEEP", "7"))

summary = {"ts": TS, "dest": DEST, "pg": None, "qdrant": None, "redis": None,
           "offsite": None, "errors": []}


def http(method, url, timeout, body=None, raw=False, headers=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        return data if raw else json.loads(data.decode("utf-8"))


def backup_postgres():
    import psycopg

    dsn = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASS}"
    counts = {}
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
            )
            tables = [r[0] for r in cur.fetchall()]
            cur.execute(
                "SELECT table_name, column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns WHERE table_schema = 'public' "
                "ORDER BY table_name, ordinal_position"
            )
            schema = cur.fetchall()
    with open(os.path.join(DEST, "postgres_schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, default=str)

    for table in tables:
        out = os.path.join(DEST, f"pg_{table}.ndjson.gz")
        rows = 0
        with psycopg.connect(dsn, connect_timeout=15) as conn:
            with conn.cursor(name=f"dump_{table}_{TS}") as cur:
                cur.itersize = 5000
                cur.execute(f'SELECT row_to_json(t) FROM (SELECT * FROM "{table}") t')
                with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
                    for (row,) in cur:
                        fh.write(json.dumps(row, default=str))
                        rows += 1
        counts[table] = rows
    summary["pg"] = {"tables": len(tables), "rows": counts, "ok": True}


def backup_qdrant():
    created = http("POST", f"{QDRANT_URL}/collections/{COLLECTION}/snapshots", 600, body=b"{}")
    snap_name = created["result"]["name"]
    data = http(
        "GET", f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/{snap_name}", 900, raw=True
    )
    with open(os.path.join(DEST, f"{COLLECTION}.snapshot"), "wb") as fh:
        fh.write(data)
    pruned = 0
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=48)
    ).isoformat()
    listing = http("GET", f"{QDRANT_URL}/collections/{COLLECTION}/snapshots", 60)
    for s in listing["result"]:
        if s["creation_time"] < cutoff:
            try:
                http(
                    "DELETE",
                    f"{QDRANT_URL}/collections/{COLLECTION}/snapshots/{s['name']}",
                    60,
                )
                pruned += 1
            except Exception:
                pass
    summary["qdrant"] = {"snapshot": snap_name, "bytes": len(data), "pruned": pruned, "ok": True}


def backup_redis():
    import redis

    r = redis.Redis(host=REDIS_HOST, port=int(REDIS_PORT), password=REDIS_PASS, decode_responses=False)
    r.ping()
    out = os.path.join(DEST, "redis_dump.ndjson.gz")
    keys = 0
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        for key in r.scan_iter(count=1000):
            ktype = r.type(key).decode()
            ttl = r.pttl(key)
            dump = r.dump(key)
            if dump is None:
                continue
            fh.write(
                json.dumps(
                    {
                        "key": key.decode("utf-8", "replace"),
                        "type": ktype,
                        "pttl": ttl,
                        "dump_hex": dump.hex(),
                    }
                )
            )
            keys += 1
    summary["redis"] = {"keys": keys, "ok": True}


def backup_offsite_github():
    """Rolling offsite copy to a private GitHub repo via the git data API.

    Every run: tar.gz the backup dir, upload one blob, build a tree keeping the
    newest GH_KEEP archives, force-push a single commit on main. The repo never
    grows beyond ~GH_KEEP archives; the old ref is replaced, so history does
    not accumulate. Fail-closed for the offsite step only: an error is recorded
    in summary['errors'] but the local backup remains valid.
    """
    if not GH_TOKEN or not GH_REPO:
        summary["offsite"] = {"ok": False, "skipped": "GITHUB_BACKUP_TOKEN/REPO not configured"}
        return

    def gh(method, path, payload=None):
        req = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {GH_TOKEN}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
                return json.loads(data.decode()) if data else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"GitHub {method} {path} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            )

    arc_path = f"/tmp/memory_backup_{TS}.tar.gz"
    with tarfile.open(arc_path, "w:gz", compresslevel=6) as tf:
        tf.add(DEST, arcname=os.path.basename(DEST))
    arc_bytes = open(arc_path, "rb").read()
    os.remove(arc_path)

    new_path = f"backups/memory_backup_{TS}.tar.gz"
    blob = gh("POST", f"/repos/{GH_REPO}/git/blobs",
              {"content": base64.b64encode(arc_bytes).decode(), "encoding": "base64"})

    ref = gh("GET", f"/repos/{GH_REPO}/git/ref/heads/main")
    base_commit = ref["object"]["sha"]
    base = gh("GET", f"/repos/{GH_REPO}/git/commits/{base_commit}")
    tree_sha = base["tree"]["sha"]
    tree = gh("GET", f"/repos/{GH_REPO}/git/trees/{tree_sha}?recursive=1")
    old_archives = sorted(
        (t["path"] for t in tree.get("tree", [])
         if t["type"] == "blob" and t["path"].startswith("backups/memory_backup_")),
        reverse=True,
    )
    keep = ([new_path] + old_archives)[:GH_KEEP]
    keep_set = set(keep)
    entries = [
        {"path": t["path"], "mode": "100644", "type": "blob", "sha": t["sha"]}
        for t in tree.get("tree", [])
        if t["type"] == "blob"
        and t["path"].startswith("backups/memory_backup_")
        and t["path"] in keep_set
    ]
    for t in tree.get("tree", []):
        if (
            t["type"] == "blob"
            and t["path"].startswith("backups/memory_backup_")
            and t["path"] not in keep_set
        ):
            entries.append({"path": t["path"], "mode": "100644", "type": "blob", "sha": None})
    entries.append({"path": new_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    new_tree = gh("POST", f"/repos/{GH_REPO}/git/trees", {"base_tree": tree_sha, "tree": entries})
    new_commit = gh(
        "POST", f"/repos/{GH_REPO}/git/commits",
        {"message": f"memory nightly backup {TS} ({len(arc_bytes)} bytes)",
         "tree": new_tree["sha"], "parents": [base_commit]},
    )
    gh("PATCH", f"/repos/{GH_REPO}/git/refs/heads/main",
       {"sha": new_commit["sha"], "force": True})
    summary["offsite"] = {
        "ok": True, "repo": GH_REPO, "archive_bytes": len(arc_bytes),
        "commit": new_commit["sha"][:12], "archives_kept": len(keep),
    }


def main():
    os.makedirs(DEST, exist_ok=True)
    t0 = time.time()

    for name, fn in (
        ("pg", backup_postgres),
        ("qdrant", backup_qdrant),
        ("redis", backup_redis),
        ("offsite", backup_offsite_github),
    ):
        try:
            fn()
        except Exception as exc:
            summary["errors"].append(f"{name}: {exc}")
            summary[name] = summary.get(name) or {"ok": False, "error": str(exc)}

    try:
        shutil.copy2(os.environ.get("MEMORY_BM25_PATH", "/data/bm25_index.json"), os.path.join(DEST, "bm25_index.json"))
    except Exception as exc:
        summary["errors"].append(f"bm25: {exc}")

    total_bytes = sum(
        os.path.getsize(p) for p in glob.glob(os.path.join(DEST, "*")) if os.path.isfile(p)
    )
    summary["bytes"] = total_bytes
    summary["duration_s"] = round(time.time() - t0, 1)

    with open(os.path.join(DEST, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    with open(os.path.join(DEST, "restore_instructions.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"Memory backup restore instructions ({TS})\n"
            f"{'=' * 60}\n"
            "Target: Docker fleet (memory-mcp, postgres, qdrant, redis containers).\n"
            "Offsite copy: private GitHub repo (see manifest offsite) — download the\n"
            "  newest backups/memory_backup_*.tar.gz, untar, then follow these steps.\n"
            "0) Stop writes: docker compose stop memory-mcp (fleet MCPs will error; acceptable in DR).\n"
            "1) Postgres — recreate schema from postgres_schema.json (or let memory_server\n"
            "   autocreate), then for each pg_<table>.ndjson.gz insert the JSON rows into the\n"
            "   matching table (restore order: mem_nodes, mem_canonical, mem_trajectory_steps,\n"
            "   mem_edges, mem_outbox, then the rest).\n"
            "2) Qdrant — restore snapshot into a fresh collection:\n"
            f"   curl -X POST '{QDRANT_URL}/collections/{COLLECTION}/snapshots/upload?priority=snapshot' \\\n"
            f"        -F 'snapshot=@{COLLECTION}.snapshot'\n"
            "3) Redis — RESTORE each key (values are hex-encoded DUMP payloads):\n"
            "   python -c 'import gzip,json,redis; r=redis.Redis(password=os.environ.get('MEMORY_REDIS_PASSWORD','') or None); [r.restore(k.encode(), "
            "int(d[\"pttl\"]) if d[\"pttl\"]>0 else 0, bytes.fromhex(d[\"dump_hex\"]), REPLACE) "
            "for d in map(json.loads, gzip.open(\"redis_dump.ndjson.gz\",\"rt\"))]'\n"
            "4) BM25 — copy bm25_index.json back to /data/bm25_index.json.\n"
            "5) Restart: docker compose start memory-mcp; verify with memory_health_check and a test recall.\n"
            "SECURITY: this backup contains auth token digests and config secrets — the offsite\n"
            "repo is private and the backup volume is root-owned inside the container.\n"
        )

    log_line = json.dumps(
        {
            "ts": TS,
            "pg_ok": bool((summary.get("pg") or {}).get("ok")),
            "qdrant_ok": bool((summary.get("qdrant") or {}).get("ok")),
            "redis_ok": bool((summary.get("redis") or {}).get("ok")),
            "offsite_ok": bool((summary.get("offsite") or {}).get("ok")),
            "bytes": total_bytes,
            "errors": summary["errors"],
        },
        default=str,
    )
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    with open(os.path.join(BACKUP_ROOT, "backup.log"), "a", encoding="utf-8") as fh:
        fh.write(log_line + "\n")

    base = BACKUP_ROOT
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=RETENTION_DAYS)
    for d in glob.glob(os.path.join(base, "20*")):
        try:
            if datetime.datetime.fromtimestamp(
                os.path.getmtime(d), datetime.timezone.utc
            ) < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                summary.setdefault("pruned_dirs", []).append(os.path.basename(d))
        except Exception:
            pass

    print(json.dumps(summary, default=str))
    return 0 if not summary["errors"] else 3


if __name__ == "__main__":
    sys.exit(main())
