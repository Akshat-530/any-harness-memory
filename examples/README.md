# Connect a harness

The Memory MCP endpoint is **streamable HTTP** at:

```
http://127.0.0.1:4000/mcp
```

(The port is `${MEMORY_HTTP_PORT_HOST}` from `.env`; `4000` is the default.)

Drop-in configs for common harnesses live in this folder. Adjust the port if you
changed `MEMORY_HTTP_PORT_HOST`.

| File | Harness |
|---|---|
| `kilo.jsonc` | Kilo / Kilo Desktop |
| `cursor.mcp.json` | Cursor |
| `claude_desktop_config.json` | Claude Desktop (via `mcp-remote` bridge) |
| `cline.json` | Cline / Continue (remote) |
| `cline-stdio.json` | Cline / Continue (stdio via Docker) |
| `langchain.py` | LangChain / Python |

## Transports

**Streamable HTTP (recommended)** — one warm process, all 72 tools, dashboard and
metrics. Any MCP client that supports remote HTTP connects directly to `/mcp`.

**stdio** — for clients that only speak stdio, run the server attached to the
compose network so it inherits the backend env and service DNS:

```bash
docker compose -f <ABSOLUTE_PATH_TO>/memory-mcp/compose.yml \
  run --rm -T --entrypoint python memory-mcp /app/memory_server.py
```

Or bridge HTTP to stdio from any machine:

```bash
npx -y mcp-remote http://127.0.0.1:4000/mcp
```

## Authentication

By default (dev) the server runs anonymous inside an isolated namespace, so no
token is needed. For multi-user, enable token auth (see the main README) and add
the header to every client:

```
Authorization: Bearer <token>
```

or

```
X-Memory-Token: <token>
```

Each harness expresses headers slightly differently — for example a remote Kilo
entry accepts a `headers` block; the `mcp-remote` bridge takes `--header`.

## Verify the connection

```bash
curl -s http://127.0.0.1:4000/healthz          # ok
curl -s http://127.0.0.1:4000/metrics | head    # prometheus (auth may apply)
```