# Memory

A self-contained, production-ready **semantic memory server** for AI agents.
One Docker Compose stack ships the whole backend:

| Service | Role | Persistence |
|---|---|---|
| **memory-mcp** | FastMCP server (agent profile ~40 tools, or 74 full) + web dashboard + Prometheus `/metrics` | stateless (code in image) |
| **PostgreSQL 17** | Canonical durable ledger — the **source of truth** | `pgdata` volume, `synchronous_commit=on` |
| **Qdrant** | Vector store — dense embeddings, HNSW | `qdrantdata` volume, on-disk WAL |
| **Redis** | Short-term KV, semantic cache, graph edges, version snapshots | `redisdata` volume, AOF `everysec` + RDB |

Everything except `memory-mcp` is internal to a private Docker network. Only
`memory-mcp` is published, bound to `127.0.0.1`.

## Why it is durable (the "long writes" design)

Postgres is the **only** source of truth. A write commits `mem_canonical` **and**
a `mem_outbox` event in one transaction, then background outbox workers apply the
change to Qdrant, BM25 and the graph **idempotently and version-ordered**. A crash
mid-flight loses nothing — the outbox replays on restart. A corrupted vector index
is rebuilt from canonical. Redis is acceleration only; losing it never loses memory.

## Portability

Runs on any machine with Docker + Compose. Nothing is machine-bound: **no host
paths, no baked accounts, no API keys, no cloud dependency**. Every setting is
an environment variable (`.env.example`), all data lives in named Docker volumes,
and the database schema bootstraps itself on first boot. Mount nothing, clone
anywhere, `docker compose up -d --build`.

## Retrieval

Hybrid: Qdrant dense (default `Qwen/Qwen3-Embedding-0.6B`, 1024-d) **+** BM25
sparse, fused with Reciprocal Rank Fusion, then cross-encoder rerank
(`cross-encoder/ms-marco-MiniLM-L6-v2`). A canonical-liveness filter enforces
tenant scoping. Results scored by relevance / importance / recency.

## Lifecycle (long memory)

Decay (Ebbinghaus, 180-day half-life) → consolidation (dedupe / evict / archive) →
trajectory compaction → `dream()` insight synthesis → procedure promotion from
repeated successful trajectories. Plus graph memory and versioned rollback.

## Quick start

```bash
cp .env.example .env          # edit passwords + port
docker compose up -d --build  # first build compiles the CPU-torch venv (~5-10 min)
docker compose logs -f memory-mcp
```

First boot bootstraps the PostgreSQL schema from `schema.sql` (the exact schema the app expects, taken from the proven live DB) and downloads the embedding + reranker
models into the `memdata` volume (one-time, ~1.5 GB).

The MCP endpoint is at `http://127.0.0.1:${MEMORY_HTTP_PORT_HOST}/mcp` (streamable
HTTP). Dashboard: `/ui`. Metrics: `/metrics`.

### Point a client at it

```jsonc
// e.g. Kilo / Cursor / LangChain MCP client
{
  "mcpServers": {
    "memory": { "type": "remote", "url": "http://127.0.0.1:4000/mcp" }
  }
}
```

> With token auth enabled, send `X-Memory-Token: <token>` (or `Authorization: Bearer <token>`).

## Two artifacts, one stack

- **Backend** — `qdrant` + `postgres` + `redis`. Holds the data. Internal only, no host ports.
- **Memory MCP** — the `memory-mcp` service (`app/memory_*.py`). Speaks the Model Context Protocol
  (streamable HTTP **and** stdio), exposes tool profiles: a ~40-tool agent surface, or all 74 for operators, the dashboard and metrics.
  This is what a harness connects to.
- **Memory MCP, standalone** — the server is env-only: point it at *any* compatible
  backends with `MEMORY_PG_HOST/PORT/DB/USER/PASSWORD`, `MEMORY_REDIS_HOST/PORT/PASSWORD`
  and `QDRANT_URL`, then run `python /app/memory_http_server.py`. It does not require the
  bundled compose stack.

They ship in one `compose.yml` on a private `memory` network, fully interconnected:

```
harness ──HTTP──▶ memory-mcp :4000 ──┬─▶ postgres  (source of truth, outbox)
                 /mcp /ui /metrics     ├─▶ qdrant    (vectors + BM25 payloads)
                 /healthz              └─▶ redis     (short-term, cache, graph)
```

## Connect any harness

Endpoint: **`http://127.0.0.1:4000/mcp`** (streamable HTTP). Copy-paste configs in [`examples/`](examples/).

| Harness | Transport | File |
|---|---|---|
| Kilo / Kilo Desktop | remote HTTP | `examples/kilo.jsonc` |
| Cursor | remote HTTP / bridge | `examples/cursor.mcp.json` |
| Claude Desktop | stdio bridge (`mcp-remote`) | `examples/claude_desktop_config.json` |
| Cline / Continue | remote or stdio | `examples/cline.json`, `examples/cline-stdio.json` |
| LangChain / Python | streamable HTTP | `examples/langchain.py` |
| Any stdio-only client | stdio (Docker) | `docker compose run --rm -T --entrypoint python memory-mcp /app/memory_server.py` |

### Transports
- **Streamable HTTP** (recommended): one warm process, every tool, dashboard + metrics. Connect any remote-HTTP MCP client to `/mcp`.
- **stdio**: `docker compose run --rm -T --entrypoint python memory-mcp /app/memory_server.py` (inherits backend env + network; runs as the dev anon principal, or set `MEMORY_STDIO_IDENTITY`). For stdio-only desktop apps, bridge with `npx -y mcp-remote http://127.0.0.1:4000/mcp`.

### Endpoints
| Path | Purpose |
|---|---|
| `/mcp` | MCP endpoint (streamable HTTP) |
| `/ui` | Dashboard — browse, edit, export memories |
| `/metrics` | Prometheus metrics |
| `/healthz` | Liveness (no auth, no data) |

### Auth
Send `Authorization: Bearer <token>` or `X-Memory-Token: <token>`. Identity is resolved server-side from the token; Postgres RLS enforces tenant isolation. Dev default is anonymous in an isolated namespace.

## Core injection (zero-call context)

Memory can behave as a reflex, not a tool you remember to call:

- **Resources** — pull context at boot with no tool call: `memory://context` (assembled session context), `memory://profile` (preferences/behavioral block), `memory://core` (hot memories).
- **Tool profiles** — `MEMORY_TOOL_PROFILE=agent` (default) advertises a curated ~40-tool agent surface; `full` exposes all 74, including the 21 operator-only tools that are refused to network callers.
- **Turn ingest** — `memory_turn_ingest` captures one turn (facts + `error→fix`) in a single call.
- **Bi-temporal / point-in-time** — every memory carries `valid_from`/`valid_until` and an indexed `event_time`; supersession keeps the historical version (never deletes it) and closes incident graph edges. `memory_recall(as_of=<ISO8601>)` (and `memory_get`/`memory_list`/`memory_graph_neighbors`) return what was true then, `as_of_field=event` filters by event time, and `memory_timeline` returns the full version/supersession chain with per-version content previews.
- **Virtual context** — with `VIRTUAL_CONTEXT_ENABLED=1`, recall/session responses carry an injected `_virtual_context` block (token-budgeted).
- **Harness client** — `client/memory_client.py` implements the lifecycle hooks (session-start context → pre-model recall → capture → task_end); `client/memory_protocol.md` is a paste-in system prompt for harnesses you cannot wrap.
- **Maintenance** — `MEMORY_MAINTENANCE=1` runs the nightly consolidate / dream / decay / promote + backup in-container.

## Benchmarks

`benchmarks/memory_bench.py` is a reproducible recall@k / MRR harness (self-test included):

```bash
docker run --rm --network memory-mcp_memory --entrypoint python \
  -v "$PWD/benchmarks/memory_bench.py:/b.py:ro" memory:latest /b.py --self-test --url http://memory-mcp:4000/mcp
```

For LoCoMo/LongMemEval, ingest turns with `memory_turn_ingest`, query with `as_of`, and grade with the benchmark's judge — see `benchmarks/README.md`.

## Authentication

* **Default (dev / single-user):** `MEMORY_ALLOW_ANONYMOUS=1` — callers are
  confined to an isolated namespace and cannot reach any tenant's data.
* **Production / multi-user:** mount an `auth_tokens` file at `/run/auth_tokens`,
  set `MEMORY_AUTH_TOKENS=/run/auth_tokens` and `MEMORY_ALLOW_ANONYMOUS=0`.
  See `auth_tokens.example`. Identity is resolved **only** from the credential —
  never from a client-supplied header. Postgres row-level security enforces the
  tenant boundary underneath every query.

## Configuration

All settings are environment variables (see `.env.example`). Key ones:

| Var | Default | Meaning |
|---|---|---|
| `MEMORY_PG_*` | `memory` / dev pw | Postgres host/db/user/password |
| `MEMORY_REDIS_PASSWORD` | — | Redis AUTH (required) |
| `MEMORY_HTTP_PORT_HOST` | `4000` | Host loopback port |
| `MEMORY_PRINCIPAL` | `global` | Namespace registry / grant ceiling |
| `MEMORY_ALLOW_ANONYMOUS` | `1` | Confine credential-less callers to an isolated namespace |
| `MEMORY_EAGER_PREWARM` | `1` | Warm embedding + BM25 at boot |
| `MEMORY_JUDGE_ENDPOINT` / `_MODEL` / `_API_KEY` | blank | Optional OpenAI-compatible LLM judge |
| `MEMORY_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model (changing it requires a new collection) |

## Operations

```bash
docker compose ps
docker compose exec postgres psql -U memory -d memory -c "select status,count(*) from mem_canonical group by status;"
docker compose exec memory-mcp python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:4000/metrics').status)"
```

Backups: `app/tools/memory_backup.py` dumps canonical + graph to the `memdata`
volume (`/data/backup`). The nightly cron is opt-in; schedule it with your
orchestrator or a sidecar.

## Security notes

* Postgres: `scram-sha-256` on all TCP, no host port published.
* Redis: `requirepass` + `protected-mode yes`, no host port published.
* The container only exposes `127.0.0.1:${MEMORY_HTTP_PORT_HOST}`.
* No secret is baked into the image; all credentials come from `.env`.

## Layout

```
compose.yml            stack definition
Dockerfile             standalone image (python:3.12-slim + CPU torch venv)
docker-entrypoint.sh   wait-for-postgres, migrate, serve
requirements-memory.txt
app/                   memory_server.py (core), memory_http_server.py (HTTP+dashboard)
  tools/               backup / maintenance helpers
config/                postgresql.conf, pg_hba.conf, redis.conf, qdrant.yaml
schema.sql             authoritative PostgreSQL schema (bootstrapped on first boot)
```