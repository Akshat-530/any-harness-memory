# Memory — standalone semantic memory server
# Qdrant (vectors) + PostgreSQL (canonical ledger) + Redis (short-term cache/graph)
#
# Self-contained image: no dependency on the wider MCP fleet.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf \
    TOKENIZERS_PARALLELISM=false \
    PATH="/opt/memory-venv/bin:$PATH"

# libgomp1 is required by the CPU torch wheel; curl is used by healthchecks.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Isolated venv for the memory stack (keeps the fleet/torch bloat out).
COPY requirements-memory.txt /tmp/requirements-memory.txt
RUN python -m venv /opt/memory-venv \
 && /opt/memory-venv/bin/pip install --upgrade pip \
 && /opt/memory-venv/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu torch \
 && /opt/memory-venv/bin/pip install --no-cache-dir \
        -r /tmp/requirements-memory.txt \
 && rm /tmp/requirements-memory.txt

WORKDIR /app
COPY app/ /app/
COPY schema.sql /app/schema.sql
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Non-root runtime user; /data holds the BM25 index, model cache and backups.
RUN groupadd -g 10001 memory \
 && useradd -m -u 10001 -g memory memory \
 && mkdir -p /data/hf /data/backup \
 && chown -R memory:memory /data /app \
 && chmod +x /usr/local/bin/docker-entrypoint.sh

USER memory
ENV MEMORY_BM25_PATH=/data/bm25_index.json

EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 4000), timeout=4)" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
