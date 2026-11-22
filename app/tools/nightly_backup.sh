#!/bin/sh
# Memory nightly backup entrypoint. Scheduled by maintenance_cron.py.
exec /opt/memory-venv/bin/python "$(dirname "$0")/memory_backup.py"
