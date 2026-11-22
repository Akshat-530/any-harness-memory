#!/bin/sh
# Launches the maintenance cron daemon inside the memory-mcp container.
# Usage:  docker compose exec memory-mcp sh /app/tools/start_maintenance_cron.sh
LOG_DIR="${MEMORY_CRON_LOGDIR:-/data/logs}"
mkdir -p "$LOG_DIR"
exec /opt/memory-venv/bin/python -u "$(dirname "$0")/maintenance_cron.py" >> "$LOG_DIR/cron_stderr.log" 2>&1
