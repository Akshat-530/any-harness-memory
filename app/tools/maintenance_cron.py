#!/usr/bin/env python3
"""Memory maintenance cron daemon (runs inside the memory-mcp container).

Operator-path design: calls the operator-only tools in-process under
system_context() -- the ONLY authorized path (network callers can never enter
system context). This daemon schedules:

- 02:10 UTC daily : memory_maintenance (full auto-learning chain)
- 02:40 UTC daily : memory_backup (PG + Qdrant + Redis + BM25 snapshot)

Behaviour:
- Boot catch-up: any job whose last success is older than 26h runs immediately
  on start, so work is never lost across container restarts -- the daemon is
  (re)started automatically by the compose command wrapper.
- fcntl lock prevents duplicate daemons.
- JSON-lines log with secret redaction; state file survives recreation.

Manual run:  docker exec memory-mcp /bin/sh /app/tools/start_maintenance_cron.sh
(direct, foreground). Background: docker exec -d ... same command.
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_PATH = os.environ.get(
    "MEMORY_CRON_STATE", "/data/maintenance_cron_state.json"
)
LOG_DIR = os.environ.get("MEMORY_CRON_LOGDIR", "/data/logs")
LOCK_PATH = "/tmp/memory_maintenance_cron.lock"
STALE_AFTER_H = 26
BACKUP_TIMEOUT_S = 3600

REDACT = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[REDACTED:key]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "[REDACTED:github]"),
    (re.compile(r"xox[baposr]-[A-Za-z0-9\-]{10,}"), "[REDACTED:slack]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?\S+"), r"\1[REDACTED]"),
    (
        re.compile(r"(?i)(\"(?:password|token|secret|api[_-]?key|pgpassword)\"\s*[:=]\s*)\"[^\"]+\""),
        r"\1\"[REDACTED]\"",
    ),
]


def redact(text):
    for rx, rep in REDACT:
        text = rx.sub(rep, text)
    return text


def log_write(kind, payload):
    os.makedirs(LOG_DIR, exist_ok=True)
    month = datetime.now(timezone.utc).strftime("%Y%m")
    path = os.path.join(LOG_DIR, "maintenance_%s.log" % month)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind}
    rec.update(payload)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(redact(json.dumps(rec, ensure_ascii=False, default=str)) + "\n")


def read_state():
    try:
        with open(STATE_PATH) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_PATH)


def job_maintenance():
    from memory_server import _tool_maintenance, system_context

    t0 = time.time()
    with system_context("maintenance-cron"):
        res = _tool_maintenance({"dry_run": False})
    summary = json.loads(res.content[0].text)
    return summary, round(time.time() - t0, 1)


def job_backup():
    t0 = time.time()
    proc = subprocess.run(
        ["/bin/sh", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightly_backup.sh")],
        capture_output=True,
        timeout=BACKUP_TIMEOUT_S,
    )
    dur = round(time.time() - t0, 1)
    out = proc.stdout.decode("utf-8", "replace").strip().splitlines()
    summary = {}
    for line in reversed(out):
        try:
            summary = json.loads(line)
            break
        except Exception:
            continue
    if proc.returncode != 0:
        summary["returncode"] = proc.returncode
        summary["stderr_tail"] = proc.stderr.decode("utf-8", "replace")[-1500:]
    return summary, dur


JOBS = [
    ("maintenance", "02:10", job_maintenance),
    ("backup", "02:40", job_backup),
]


def next_run_dt(now, hhmm):
    hh, mm = (int(x) for x in hhmm.split(":"))
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= now:
        cand += timedelta(days=1)
    return cand


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    lock = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log_write("start", {"status": "another-daemon-holds-lock-exit"})
        return 0
    log_write("start", {"pid": os.getpid(), "jobs": {n: t for n, t, _ in JOBS}})

    # Boot catch-up: run any job whose last success is missing or stale.
    for name, hhmm, fn in JOBS:
        state = read_state()
        last_ok = (state.get(name) or {}).get("last_success")
        age_h = STALE_AFTER_H + 1
        if last_ok:
            try:
                age_h = (
                    datetime.now(timezone.utc) - datetime.fromisoformat(last_ok)
                ).total_seconds() / 3600.0
            except Exception:
                age_h = STALE_AFTER_H + 1
        if age_h > STALE_AFTER_H:
            reason = (
                "no prior successful run recorded"
                if not last_ok
                else "last_success %s is %.1fh old" % (last_ok, age_h)
            )
            log_write("catchup", {"job": name, "reason": reason})
            try:
                summary, dur = fn()
                log_write("run", {"job": name, "duration_s": dur, "summary": summary})
                state = read_state()
                state.setdefault(name, {})
                state[name]["last_success"] = datetime.now(timezone.utc).isoformat()
                state[name]["last_mode"] = "catchup"
                write_state(state)
            except Exception as exc:
                log_write(
                    "error",
                    {
                        "job": name,
                        "where": "catchup",
                        "error": redact(str(exc)),
                        "trace": redact(traceback.format_exc()[-2000:]),
                    },
                )

    while True:
        now = datetime.now(timezone.utc)
        nxt_job, nxt_dt = min(
            ((name, next_run_dt(now, hhmm)) for name, hhmm, _ in JOBS), key=lambda x: x[1]
        )
        sleep_s = max(15.0, (nxt_dt - now).total_seconds())
        log_write(
            "wait",
            {"next_job": nxt_job, "next_run_utc": nxt_dt.isoformat(), "sleep_s": int(sleep_s)},
        )
        time.sleep(sleep_s)
        name, hhmm, fn = next((j for j in JOBS if j[0] == nxt_job), JOBS[0])
        try:
            summary, dur = fn()
            log_write("run", {"job": name, "duration_s": dur, "summary": summary})
            state = read_state()
            state.setdefault(name, {})
            state[name]["last_success"] = datetime.now(timezone.utc).isoformat()
            state[name]["last_mode"] = "scheduled"
            state[name]["last_cron_utc"] = hhmm
            write_state(state)
        except Exception as exc:
            log_write(
                "error",
                {
                    "job": name,
                    "where": "scheduled",
                    "error": redact(str(exc)),
                    "trace": redact(traceback.format_exc()[-2000:]),
                },
            )
            time.sleep(900)


if __name__ == "__main__":
    sys.exit(main() or 0)
