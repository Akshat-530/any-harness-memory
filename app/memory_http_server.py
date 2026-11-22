#!/usr/bin/env python3
"""Shared Memory MCP server over streamable-http.

Reuses the canonical `_server` + tools from memory_server.py, but serves it
over ONE long-lived HTTP endpoint so every Kilo run (main + sub-agents)
connects to a single warm process instead of each spawning its own stdio MCP
and re-loading the embedding model + Qdrant + BM25 index.

Env:
  MEMORY_EAGER_PREWARM=1   force the one-time warm (embedding + BM25) at boot.
  MEMORY_HTTP_HOST         default 127.0.0.1
  MEMORY_HTTP_PORT         default 3901
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pull in the full tool registry + _server from the canonical module.
import memory_server  # noqa: E402

_server = memory_server._server
_TOKEN_HEADER = memory_server._TOKEN_HEADER

log = logging.getLogger("memory-http")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def validate_deployment_security() -> None:
    """PART 20: refuse to serve user traffic in an unsafe configuration.

    A misconfigured deployment must fail LOUDLY rather than silently defaulting
    to a shared namespace. Required:
      * MEMORY_PRINCIPAL        - the namespace registry / grant ceiling.
      * MEMORY_AUTH_TOKENS      - credential store (sha256 digests).
    Escape hatch for a genuinely single-user box: MEMORY_ALLOW_ANONYMOUS=1 binds
    credential-less callers to MEMORY_ANON_IDENTITY (default `anon`), an
    isolated namespace that is NOT any real tenant, so anonymous traffic can
    never read or write tenant data.
    """
    ceiling = memory_server._deployment_grants()
    if ceiling is None:
        raise SystemExit(
            "FATAL: MEMORY_PRINCIPAL is unset or invalid. This deployment has no "
            "namespace registry and would serve nothing safely. Refusing to start."
        )
    tokens = memory_server.token_count()
    for err in memory_server._TOKEN_LOAD_ERRORS:
        raise SystemExit(f"FATAL: malformed credential store: {err}")
    if tokens == 0:
        anon = os.environ.get("MEMORY_ALLOW_ANONYMOUS", "").strip()
        if anon != "1":
            raise SystemExit(
                "FATAL: no authentication credentials configured "
                "(MEMORY_AUTH_TOKENS is empty). Without credentials every caller "
                "would be anonymous, and anonymous is authorized for NOTHING -- "
                "the server would be unusable but safe. Either provision tokens:\n"
                "  printf '%s' \"$(openssl rand -hex 32)\" | sha256sum\n"
                "  # add 'sha256:<digest>|<identity>|<scopes>' to the token file\n"
                "or, for a strictly single-user host, set MEMORY_ALLOW_ANONYMOUS=1 "
                "(traffic is then confined to the isolated 'anon' namespace and "
                "cannot reach any tenant)."
            )
        anon_id = os.environ.get("MEMORY_ANON_IDENTITY", "anon").strip() or "anon"
        try:
            anon_n = memory_server._normalize_scope(anon_id)
        except memory_server.ScopeError as e:
            raise SystemExit(f"FATAL: invalid MEMORY_ANON_IDENTITY: {e}")
        explicit_grants = set()
        for _p in os.environ.get("MEMORY_PRINCIPAL", "").split(","):
            _p = _p.strip()
            if _p:
                try:
                    explicit_grants.add(memory_server._normalize_scope(_p))
                except memory_server.ScopeError:
                    pass
        if anon_n in explicit_grants:
            raise SystemExit(
                f"FATAL: MEMORY_ANON_IDENTITY={anon_n!r} collides with a real "
                f"tenant in MEMORY_PRINCIPAL={sorted(explicit_grants)}. Anonymous traffic "
                "would inherit tenant data. Refusing to start."
            )
        log.warning(
            "SECURITY: anonymous access ENABLED. Credential-less callers are "
            "confined to namespace %r, which is not a tenant. Tenant data is "
            "unreachable, but any anonymous caller shares this namespace.",
            anon_n)
        os.environ["MEMORY_ANON_PRINCIPAL"] = anon_n
    else:
        log.info("security: %d credential(s) loaded; anonymous access denied",
                 tokens)
    log.info("security: namespace ceiling = %s", sorted(ceiling))


def _extract_token(scope) -> str | None:
    """Pull a bearer credential out of the request headers.

    Accepts `X-Memory-Token: <t>` and `Authorization: Bearer <t>`. The
    X-Memory-Identity header is deliberately NOT consulted here: identity used
    to come straight from that header, which meant any client that could reach
    the socket could simply *claim* a tenant and read its memories. Identity is
    now resolved server-side from the credential only.
    """
    for raw_name, raw_value in scope.get("headers") or ():
        name = raw_name.lower()
        if name == _TOKEN_HEADER:
            try:
                return raw_value.decode("latin-1").strip() or None
            except Exception:
                return None
        if name == b"authorization":
            try:
                val = raw_value.decode("latin-1").strip()
            except Exception:
                continue
            if val.lower().startswith("bearer "):
                return val[7:].strip() or None
    return None


def bind_caller_identity(manager):
    """Return an ASGI app that authenticates and binds the caller per request.

    MUST wrap `manager.asgi_app`, NOT `manager.handle_request`. In __init__ the
    manager does

        self.asgi_app = RequestBodyLimitMiddleware(self._handle_request, limit)

    so `asgi_app` is an instance attribute holding a middleware that captured the
    bound handler at construction time. Patching `handle_request` therefore has
    no effect: every request goes straight to the captured handler, the identity
    is never bound, and tenant isolation is silently disabled while every call
    still returns success. This exact bug shipped once, so the wiring lives in
    a named, directly tested function instead of inline in _main().

    The principal is derived from a credential, never from a caller-supplied
    identity. A request with no valid credential is bound to ANONYMOUS, which
    is authorized for NO namespace -- every memory tool then raises AuthError
    instead of falling back to `global`.
    """
    inner = manager.asgi_app

    async def asgi_with_identity(scope, receive, send):
        try:
            token = _extract_token(scope)
            principal = memory_server.authenticate_token(token)
        except Exception as exc:  # never let auth handling take the server down
            log.error("authentication failure: %s", exc)
            principal = memory_server.ANONYMOUS
        memory_server.set_caller_principal(principal)
        try:
            result = await inner(scope, receive, send)
            return result
        except Exception:
            log.exception("ASGI inner exception")
            raise
        finally:
            memory_server.clear_caller_principal()

    return asgi_with_identity


# ===========================================================================
# Part 3 — user dashboard (3A-1) + Prometheus /metrics (3C-3)
# ===========================================================================
# The dashboard is a thin frontend over the SAME tool handlers and the SAME
# authorization choke point the MCP tools use: every request resolves its
# principal from the token, every operation re-authorizes its scope through
# authorize_scope(), and RLS enforces the tenant boundary at Postgres level.
# No new data path exists for the UI to forget to filter.

import json as _json

from starlette.applications import Starlette as _Starlette
from starlette.responses import JSONResponse as _JSONResponse
from starlette.responses import PlainTextResponse as _PlainTextResponse
from starlette.responses import HTMLResponse as _HTMLResponse
from starlette.routing import Route as _Route
from starlette.concurrency import run_in_threadpool as _run_in_threadpool

_MS = memory_server


class _DashUnauthorized(_JSONResponse):
    def __init__(self, msg="authentication required: present X-Memory-Token"):
        super().__init__({"error": msg}, status_code=401)


def _dash_auth(scope):
    """Resolve the dashboard caller from request headers. None when anonymous."""
    token = _extract_token(scope)
    principal = _MS.authenticate_token(token)
    if principal is _MS.ANONYMOUS or not principal.identity:
        return None
    return principal


def _tool_json(result):
    """Unwrap a CallToolResult into (status_ok, parsed_json)."""
    try:
        text = result.content[0].text
        data = _json.loads(text)
    except Exception:
        return False, {"error": "unparseable tool result"}
    ok = not getattr(result, "is_error", False)
    return ok, data


def _dash_guard(request):
    """Auth wrapper for Starlette-style handlers. Returns (principal, None)
    on success or (None, 401 response)."""
    principal = _dash_auth(request.scope)
    if principal is None:
        return None, _DashUnauthorized()
    return principal, None


def _dash_overview(principal):
    scopes = sorted(_MS.authorized_scopes() or [])
    counts = {}
    for s in scopes:
        try:
            _MS.authorize_scope(s)
            n = _MS._pg_query(
                "SELECT status, count(*) AS n FROM mem_canonical "
                "WHERE scope=:s GROUP BY status", {"s": s})
            bucket = {}
            for r in n:
                if isinstance(r, dict):
                    bucket[str(r.get("status"))] = int(r.get("n") or 0)
                else:
                    bucket[str(r[0])] = int(r[1])
            counts[s] = bucket
        except Exception:
            counts[s] = {}
    health = {}
    try:
        ok, health = _tool_json(_MS._tool_health_check())
    except Exception as e:
        health = {"error": str(e)}
    return _JSONResponse({
        "version": _MS.MCP_SERVER_VERSION,
        "identity": principal.identity,
        "authorized_scopes": scopes,
        "canonical_counts": counts,
        "write_gate": {k: v for k, v in _MS._WRITE_GATE_STATS.items() if k != "by_source"},
        "health": {
            "overall": health.get("overall"),
            "stores": {k: (v.get("status") if isinstance(v, dict) else v)
                       for k, v in health.items()
                       if k in ("redis", "qdrant", "postgres")},
            "breakers": health.get("circuit_breakers"),
            "latency": health.get("tool_latency", {}),
        },
    })


def _params_from(scope):
    """Query params from a Starlette request or a raw ASGI scope dict."""
    if hasattr(scope, "query_params"):
        return dict(scope.query_params)
    from urllib.parse import parse_qs
    raw = scope.get("query_string", b"").decode("latin-1") if hasattr(scope, "get") else ""
    q = parse_qs(raw)
    return {k: v[0] for k, v in q.items()}


def _dash_memories_handler(principal, scope):
    p = _params_from(scope)
    sel = p.get("scope") or principal.identity
    limit = min(int(p.get("limit") or 50), 200)
    offset = int(p.get("offset") or 0)
    query = (p.get("q") or "").strip()
    category = p.get("category") or None
    if query:
        ok, data = _tool_json(_MS._tool_recall({
            "query": query, "top_k": limit, "scope": sel,
            "category": category}))
        mems = data.get("memories", []) if ok else []
        return _JSONResponse({"count": len(mems), "memories": mems,
                              "scope": sel, "mode": "search"})
    args = {"limit": limit, "offset": offset, "scope": sel}
    if category:
        args["category"] = category
    ok, data = _tool_json(_MS._tool_list(args))
    if not ok:
        return _JSONResponse(data, status_code=400)
    return _JSONResponse({"count": data.get("count", 0),
                          "memories": data.get("memories", []),
                          "scope": sel, "mode": "list"})


def _dash_tool_call(fn, *args, **kwargs):
    """Run a tool handler for the dashboard and convert auth failures into
    clean 403s instead of 500s (the MCP wrapper normally does this)."""
    try:
        ok, data = _tool_json(fn(*args, **kwargs))
        return _JSONResponse(data, status_code=200 if ok else
                              (403 if ok is False and _is_auth_error(data) else 404))
    except _MS.AuthError as e:
        return _JSONResponse({"error": f"not authorized: {e}"}, status_code=403)
    except _MS.ScopeError as e:
        return _JSONResponse({"error": f"scope rejected: {e}"}, status_code=403)


def _is_auth_error(data):
    msg = str(data).lower()
    return "belongs to sc" in msg or "not authorized" in msg or "ownership" in msg


def _dash_memory_handler(principal, request, memory_id):
    return _dash_tool_call(_MS._tool_get, {"memory_id": memory_id})


def _dash_update_handler(principal, request, memory_id, body):
    args = {"memory_id": memory_id}
    for k in ("content", "category", "importance"):
        if k in body:
            args[k] = body[k]
    if "tags" in body:
        args["tags"] = [t.strip() for t in str(body["tags"]).split(",") if t.strip()]
    return _dash_tool_call(_MS._tool_update, args)


def _dash_delete_handler(principal, request, memory_id):
    return _dash_tool_call(_MS._tool_delete, {"memory_id": memory_id})


def _dash_versions_handler(principal, request, memory_id):
    return _dash_tool_call(_MS._tool_versions, {"memory_id": memory_id, "limit": 20})


def _dash_restore_handler(principal, request, memory_id, body):
    return _dash_tool_call(_MS._tool_version_restore,
                           {"memory_id": memory_id, "version_id": body.get("version_id")})


def _dash_profile_handler(principal, scope):
    sel = _params_from(scope).get("scope") or principal.identity
    _MS.authorize_scope(sel)  # enforce; raises -> 400 path via wrapper
    text, tokens, _ = _MS._vc_get_user_profile(sel, _MS.VIRTUAL_CONTEXT_PROFILE_BUDGET)
    return _JSONResponse({"scope": sel, "tokens": tokens,
                          "profile": text or "(no profile memories yet for this scope)"})


def _dash_audit_handler(principal, scope):
    p = _params_from(scope)
    allowed = set(_MS.authorized_scopes() or [])
    sel = p.get("scope")
    limit = min(int(p.get("limit") or 200), 1000)
    if sel:
        if sel not in allowed:
            return _JSONResponse({"error": "scope not authorized"}, status_code=403)
        sel_filter = [sel]
    else:
        sel_filter = sorted(allowed)
    # SQL-side scope filter (post-filtering a small LIMIT page would drop
    # rows whenever another tenant's traffic dominates recent audit volume).
    rows = _MS._pg_query(
        "SELECT op, memory_id, actor, scope, detail, created_at FROM mem_audit "
        "WHERE scope = ANY(:sels) ORDER BY created_at DESC LIMIT :lim",
        {"sels": sel_filter, "lim": limit})
    out = []
    for r in rows:
        d = r if isinstance(r, dict) else {}
        if not d:
            continue
        out.append({"op": d.get("op"), "memory_id": d.get("memory_id"),
                    "actor": d.get("actor"), "scope": d.get("scope"),
                    "detail": d.get("detail"), "created_at": str(d.get("created_at"))})
    return _JSONResponse({"count": len(out), "rows": out})


def _dash_export_handler(principal, scope):
    """GDPR-grade export: everything the system holds for one scope."""
    sel = _params_from(scope).get("scope") or principal.identity
    _MS.authorize_scope(sel)
    canon = _MS._pg_query(
        "SELECT memory_id, node_type, version, status, created_at, updated_at, "
        "content, payload FROM mem_canonical WHERE scope=:s AND status<>'deleted' "
        "ORDER BY created_at DESC LIMIT 2000", {"s": sel})
    rows = []
    for r in canon:
        d = r if isinstance(r, dict) else {}
        if not d:
            continue
        mid = d.get("memory_id")
        item = {"memory_id": mid, "node_type": d.get("node_type"),
                "version": d.get("version"), "status": d.get("status"),
                "created_at": str(d.get("created_at")),
                "updated_at": str(d.get("updated_at")),
                "content": d.get("content"),
                "payload": d.get("payload")}
        try:
            vok, vdata = _tool_json(_MS._tool_versions({"memory_id": mid, "limit": 20}))
            if vok:
                item["versions"] = [
                    {k: v.get(k) for k in ("version_id", "created_at")}
                    for v in vdata.get("versions", [])]
        except Exception:
            pass
        rows.append(item)
    audit = _MS._pg_query(
        "SELECT op, actor, detail, created_at FROM mem_audit WHERE scope=:s "
        "ORDER BY created_at DESC LIMIT 2000", {"s": sel})
    audit_rows = [{"op": r.get("op"), "actor": r.get("actor"),
                   "detail": r.get("detail"), "created_at": str(r.get("created_at"))}
                  for r in audit if isinstance(r, dict)]
    payload = {"scope": sel, "exported_at": _MS._now_iso(),
               "system": "Memory", "version": _MS.MCP_SERVER_VERSION,
               "memories": rows, "audit_trail": audit_rows}
    body = _json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return _PlainTextResponse(
        body,
        headers={"Content-Disposition":
                 f'attachment; filename="memory_export_{sel.replace(":", "_")}.json"'})


def _dash_refresh_handler(principal, scope, body):
    if not principal.is_system:
        return _JSONResponse({"error": "operator token required"}, status_code=403)
    n = _MS.refresh_identity_registry()
    return _JSONResponse({"status": "reloaded", "records": n})


def _dash_status_handler(principal, scope):
    ok, data = _tool_json(_MS._tool_health_check())
    data["identity"] = principal.identity
    data["authorized_scopes"] = sorted(_MS.authorized_scopes() or [])
    return _JSONResponse(data)


def _prometheus_metrics(principal, scope):
    """3C-3: Prometheus text format from the live telemetry structures."""
    lines = []
    stats = _MS._get_latency_stats()  # {tool: {p50_ms, p95_ms, p99_ms, ...}}
    if isinstance(stats, dict):
        for tool, td in stats.items():
            if not isinstance(td, dict):
                continue
            for q, key in (("0.5", "p50_ms"), ("0.95", "p95_ms"), ("0.99", "p99_ms")):
                if key in td:
                    lines.append(f'memory_latency_ms{{tool="{tool}",quantile="{q}"}} {td[key]:.1f}')
    wg = _MS._WRITE_GATE_STATS
    for k in ("evaluated", "stored", "rejected", "quarantined", "secret_rejected", "pii_masked"):
        v = wg.get(k, 0)
        if isinstance(v, (int, float)):
            lines.append(f"memory_write_gate_{k}_total {v}")
    try:
        ob = _MS._outbox_metrics()
        lines.append(f'memory_outbox_backlog {ob.get("outbox_backlog", 0)}')
        lines.append(f'memory_outbox_dead_letters {ob.get("outbox_dead_letters", 0)}')
    except Exception:
        pass
    try:
        cs = _tool_json(_MS._tool_cache_stats())
        hit_rate = cs[1].get("hit_rate") or cs[1].get("overall", {}).get("hit_rate")
        if hit_rate is not None:
            lines.append(f"memory_cache_hit_rate {hit_rate}")
    except Exception:
        pass
    try:
        ok, h = _tool_json(_MS._tool_health_check())
        for store in ("redis", "qdrant", "postgres"):
            st = h.get(store, {}).get("status", "unknown")
            lines.append(f'memory_store_health{{store="{store}"}} {1 if st == "healthy" else 0}')
        lines.append(f'memory_version_info{{version="{h.get("version","")}"}} 1')
    except Exception:
        pass
    try:
        vcm = _MS._VC_METRICS
        if vcm.get("tokens_injected"):
            import statistics as _st
            lines.append(f"memory_vc_tokens_injected {_st.mean(list(vcm['tokens_injected'])[-200:])}")
    except Exception:
        pass
    return _PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


_DASH_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Memory — Memory Dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--text:#e6edf3;--dim:#8b949e;--acc:#58a6ff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--border);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:17px} .logo{color:var(--acc);font-weight:700}
select,input,button,textarea{background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer} button.primary{background:#1f6feb;border-color:#1f6feb} button.danger{background:#b62324;border-color:#b62324}
input#q{flex:1;min-width:220px}
nav{display:flex;gap:2px;margin-left:auto}
nav button{border:none;background:none;color:var(--dim);padding:8px 14px;border-bottom:2px solid transparent}
nav button.active{color:var(--text);border-bottom-color:var(--acc)}
main{padding:20px;max-width:1200px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;cursor:pointer;transition:border .15s}
.card:hover{border-color:var(--acc)} .card.quarantine{border-color:var(--warn)}
.card h3{font-size:13px;margin-bottom:6px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card p{color:var(--dim);font-size:12px;max-height:52px;overflow:hidden}
.badge{display:inline-block;font-size:10px;padding:1px 7px;border-radius:10px;margin-right:5px;border:1px solid var(--border);color:var(--dim)}
.badge.cat{color:var(--acc)} .badge.imp{color:var(--ok)} .badge.conf{color:#bc8cff} .badge.q{color:var(--warn);border-color:var(--warn)}
.badge.inferred{color:var(--warn)} .badge.stated{color:var(--ok)}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--border);vertical-align:top}
th{color:var(--dim);font-weight:600}
.drawer{position:fixed;right:0;top:0;bottom:0;width:520px;max-width:95vw;background:var(--panel);border-left:1px solid var(--border);padding:20px;overflow-y:auto;z-index:20;transform:translateX(100%);transition:transform .2s}
.drawer.open{transform:none}
.drawer pre{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:12px;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow-y:auto}
.drawer .row{margin:10px 0} .drawer label{display:block;color:var(--dim);font-size:11px;margin-bottom:3px}
.drawer textarea{width:100%;min-height:120px;font-family:ui-monospace,monospace;font-size:12px}
.actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:10}
.modal{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:22px;z-index:30;min-width:340px}
.dim{color:var(--dim)} .muted{color:var(--dim);font-size:11px}
#toast{position:fixed;bottom:18px;right:18px;background:var(--panel);border:1px solid var(--acc);border-radius:8px;padding:10px 16px;display:none;z-index:40}
.statgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
.stat b{font-size:20px;display:block} .stat span{color:var(--dim);font-size:11px}
.profile-box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px}
</style></head><body>
<header>
 <span class="logo">◆ Memory</span>
 <select id="scope"></select>
 <input id="q" placeholder="Search memories… (semantic)">
 <select id="cat"><option value="">All categories</option></select>
 <button class="primary" onclick="loadMemories()">Search</button>
 <button onclick="exportScope()">Export JSON</button>
 <nav>
  <button class="active" data-tab="memories" onclick="tab('memories',this)">Memories</button>
  <button data-tab="profile" onclick="tab('profile',this)">Profile</button>
  <button data-tab="audit" onclick="tab('audit',this)">Audit</button>
  <button data-tab="status" onclick="tab('status',this)">Status</button>
 </nav>
</header>
<main>
 <div id="memories" class="cards"></div>
 <div id="profile" style="display:none"><div class="profile-box" id="profileBox">loading…</div>
   <p class="muted" style="margin-top:8px">Stated preferences (user's own words) rank above inferred habits. Delete or correct entries below and they leave the profile immediately.</p></div>
 <div id="audit" style="display:none"><table id="auditTable"></table></div>
 <div id="status" style="display:none"><div class="statgrid" id="statGrid"></div><pre id="statusRaw" style="margin-top:14px;font-size:11px;color:var(--dim)"></pre></div>
</main>
<div class="drawer" id="drawer"><div id="drawerBody"></div></div>
<div id="toast"></div>
<script>
const $=s=>document.querySelector(s); const tokenKey='memory_dash_token';
let TOKEN=sessionStorage.getItem(tokenKey)||prompt('Dashboard token (X-Memory-Token):')||'';
sessionStorage.setItem(tokenKey,TOKEN);
const api=async(path,opts={})=>{const r=await fetch('/ui/api/'+path,{...opts,
  headers:{'X-Memory-Token':TOKEN,'Content-Type':'application/json',...(opts.headers||{})}});
  if(r.status===401){TOKEN=prompt('Token rejected. Dashboard token:')||'';sessionStorage.setItem(tokenKey,TOKEN);return null}
  const ct=r.headers.get('content-type')||'';
  return ct.includes('json')?r.json():r.text()};
const scope=()=>$('#scope').value;
function toast(m,ms=2600){$('#toast').textContent=m;$('#toast').style.display='block';setTimeout(()=>$('#toast').style.display='none',ms)}
async function overview(){
  const o=await api('overview'); if(!o)return;
  $('#scope').innerHTML=o.authorized_scopes.map(s=>`<option ${s===o.identity?'selected':''}>${s}</option>`).join('');
  const cats=new Set(); (o.memories||[]); // categories loaded with list below
  window.OVERVIEW=o; return o}
async function loadMemories(){
  const q=$('#q').value.trim();
  const cat=$('#cat').value;
  const d=await api(`memories?scope=${encodeURIComponent(scope())}&limit=100&q=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}`);
  if(!d)return; window.LAST=d;
  $('#memories').innerHTML=(d.memories||[]).map(m=>card(m)).join('')||'<p class="dim">No memories found.</p>';
}
function card(m){
  const conf=m.confidence!=null?`<span class="badge conf">conf ${(+m.confidence).toFixed(2)}</span>`:'';
  const prov=m.provenance?`<span class="badge ${m.provenance}">${m.provenance}</span>`:'';
  const q=m.quarantine?`<span class="badge q">⚠ quarantined</span>`:'';
  return `<div class="card ${m.quarantine?'quarantine':''}" onclick="openMem('${m.memory_id}')">
   <h3>${esc((m.content||'').split('\\n')[0].slice(0,90))}</h3>
   <p>${esc((m.content||'').slice(0,140))}</p>
   <div style="margin-top:8px"><span class="badge cat">${m.category||'general'}</span><span class="badge imp">imp ${m.importance??'-'}</span>${conf}${prov}${q}</div>
  </div>`}
async function openMem(id){
  const m=await api('memory/'+id); if(!m)return;
  const body=$('#drawerBody');
  body.innerHTML=`
   <div class="row"><label>Memory ID</label><span class="muted">${m.memory_id}</span></div>
   <div class="row"><label>Meta</label>
    <span class="badge cat">${m.category||'general'}</span><span class="badge imp">importance ${m.importance??'-'}</span>
    ${m.confidence!=null?`<span class="badge conf">confidence ${(+m.confidence).toFixed(2)}</span>`:''}
    ${m.provenance?`<span class="badge ${m.provenance}">${m.provenance}</span>`:''}
    ${m.quarantine?`<span class="badge q">⚠ quarantined (poison ${m.poison_score??'?'})</span>`:''}
    ${m.corroboration_count>1?`<span class="badge">corroborated ×${m.corroboration_count}</span>`:''}
   </div>
   <div class="row"><label>Scope / created</label><span class="muted">${m.scope} · ${m.created_at}</span></div>
   <div class="row"><label>Tags</label><span class="muted">${m.tags||'—'}</span></div>
   <div class="row"><label>Content</label><pre>${esc(m.content||'')}</pre></div>
   <div class="actions">
    <button onclick="editMode('${id}')">Edit</button>
    <button onclick="showVersions('${id}')">Versions (${m.versions_count||0})</button>
    <button class="danger" onclick="del('${id}')">Delete</button>
    <button onclick="closeDrawer()">Close</button>
   </div>
   <div id="editArea"></div><div id="verArea"></div>`;
  $('#drawer').classList.add('open');
}
function closeDrawer(){$('#drawer').classList.remove('open')}
function editMode(id){
  const m=(window.LAST?.memories||[]).find(x=>x.memory_id===id)||{};
  $('#editArea').innerHTML=`<div class="row" style="margin-top:14px"><label>Edit content</label>
   <textarea id="newContent">${esc(m.content||'')}</textarea></div>
   <div class="row"><label>Tags (comma-sep)</label><input id="newTags" value="${esc(m.tags||'')}"></div>
   <div class="row"><label>Importance</label><input id="newImp" type="number" min="1" max="10" value="${m.importance??5}"></div>
   <div class="actions"><button class="primary" onclick="saveEdit('${id}')">Save</button></div>`;
}
async function saveEdit(id){
  const r=await api('memory/'+id+'/update',{method:'POST',body:JSON.stringify({
    content:$('#newContent').value, tags:$('#newTags').value, importance:+$('#newImp').value})});
  if(!r)return; toast(r.status==='updated'?'Updated (new version snapshot kept)':'Update: '+JSON.stringify(r).slice(0,120));
  loadMemories(); openMem(id);
}
async function showVersions(id){
  const v=await api('memory/'+id+'/versions'); if(!v)return;
  $('#verArea').innerHTML=`<div class="row" style="margin-top:14px"><label>Version history</label>
   <table><tr><th>Version</th><th>Created</th><th></th></tr>
   ${(v.versions||[]).map(x=>`<tr><td>${x.version_id?.slice(0,8)||'—'}</td><td>${x.created_at||''}</td>
   <td><button onclick="restore('${id}','${x.version_id}')">Restore</button></td></tr>`).join('')}</table></div>`;
}
async function restore(id,vid){
  const r=await api('memory/'+id+'/restore',{method:'POST',body:JSON.stringify({version_id:vid})});
  if(!r)return; toast('Restored version '+String(vid).slice(0,8)); loadMemories();
}
async function del(id){
  if(!confirm('Delete this memory from ALL stores? This is tombstoned and audited; versions remain for a grace window.'))return;
  const r=await api('memory/'+id+'/delete',{method:'POST'}); if(!r)return;
  toast(r.status==='deleted'?'Deleted across stores':'Delete: '+JSON.stringify(r).slice(0,120));
  closeDrawer(); loadMemories();
}
async function exportScope(){location.href='/ui/api/export?scope='+encodeURIComponent(scope())}
async function loadProfile(){const p=await api('profile?scope='+encodeURIComponent(scope()));if(p)$('#profileBox').textContent=p.profile}
async function loadAudit(){const a=await api('audit?scope='+encodeURIComponent(scope()));if(!a)return;
 $('#auditTable').innerHTML='<tr><th>When</th><th>Actor</th><th>Op</th><th>Memory</th><th>Detail</th></tr>'+
 (a.rows||[]).map(r=>`<tr><td>${r.created_at||''}</td><td>${r.actor||''}</td><td>${r.op||''}</td>
 <td class="muted">${(r.memory_id||'').slice(0,8)}</td><td class="muted">${esc(JSON.stringify(r.detail||{}).slice(0,80))}</td></tr>`).join('')}
async function loadStatus(){
  const o=window.OVERVIEW||await overview(); if(!o)return;
  const h=o.health||{};
  $('#statGrid').innerHTML=[
    ['Overall',h.overall||'?'],['Version',o.version],
    ['Postgres',(h.stores||{}).postgres||'?'],['Qdrant',(h.stores||{}).qdrant||'?'],['Redis',(h.stores||{}).redis||'?'],
    ['Stores p95 (recall)',((h.latency||{})['memory_recall']||{}).p95+' ms'||'?'],
    ['Write gate',`${(o.write_gate||{}).stored||0} stored / ${(o.write_gate||{}).rejected||0} rejected`],
  ].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
  $('#statusRaw').textContent=JSON.stringify(h,null,2);
}
function tab(name,btn){
 document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));btn.classList.add('active');
 ['memories','profile','audit','status'].forEach(t=>$('#'+t).style.display=t===name?'':'none');
 if(name==='memories')loadMemories(); if(name==='profile')loadProfile();
 if(name==='audit')loadAudit(); if(name==='status')loadStatus();
}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'\u0026':'\u0026amp;','\u003c':'\u0026lt;','\u003e':'\u0026gt;','"':'\u0026quot;',"'":'\u0026#39;'}[c]))}
(async()=>{await overview();const cats=['preference','fact','procedure','decision','event','project','conversation','general','anti_pattern','summary'];
 $('#cat').innerHTML='<option value="">All categories</option>'+cats.map(c=>`<option>${c}</option>`).join('');
 loadMemories()})();
</script></body></html>"""


def _build_dashboard_app():
    """Starlette-native dashboard app: every endpoint authenticates the token,
    binds the principal, runs its (sync) data handler in a worker thread so
    the event loop never blocks on PG/Qdrant/embedding work."""

    def _endpoint(sync_fn, needs_body=False):
        async def endpoint(request):
            principal, err = _dash_guard(request)
            if err is not None:
                return err
            body = {}
            if needs_body:
                try:
                    body = await request.json()
                except Exception:
                    body = {}
            _MS.set_caller_principal(principal)
            try:
                return await _run_in_threadpool(sync_fn, principal, request, body)
            except Exception as e:
                log.exception("dashboard handler failed: %s", sync_fn)
                return _JSONResponse({"error": str(e)}, status_code=500)
            finally:
                _MS.clear_caller_principal()
        return endpoint

    async def h_root(request):
        return _HTMLResponse(_DASH_HTML)

    def h_memories(principal, request, body):
        return _dash_memories_handler(principal, request)

    def h_memory(principal, request, body):
        rest = request.path_params.get("rest", "")
        parts = rest.strip("/").split("/")
        memory_id = parts[0] if parts else ""
        action = parts[1] if len(parts) > 1 else None
        if not memory_id:
            return _JSONResponse({"error": "memory_id required"}, status_code=400)
        if action == "update":
            return _dash_update_handler(principal, request, memory_id, body)
        if action == "delete":
            return _dash_delete_handler(principal, request, memory_id)
        if action == "restore":
            return _dash_restore_handler(principal, request, memory_id, body)
        if action == "versions":
            return _dash_versions_handler(principal, request, memory_id)
        return _dash_memory_handler(principal, request, memory_id)

    def h_profile(principal, request, body):
        return _dash_profile_handler(principal, request)
    def h_audit(principal, request, body):
        return _dash_audit_handler(principal, request)
    def h_export(principal, request, body):
        return _dash_export_handler(principal, request)
    def h_refresh(principal, request, body):
        return _dash_refresh_handler(principal, request, body)
    def h_status(principal, request, body):
        return _dash_status_handler(principal, request)
    def h_metrics(principal, request, body):
        return _prometheus_metrics(principal, request)
    def h_overview(principal, request, body):
        return _dash_overview(principal)

    return _Starlette(routes=[
        _Route("/", h_root, methods=["GET"]),
        _Route("/api/overview", _endpoint(h_overview), methods=["GET"]),
        _Route("/api/memories", _endpoint(h_memories), methods=["GET"]),
        _Route("/api/memory/{rest:path}", _endpoint(h_memory, needs_body=True),
               methods=["GET", "POST"]),
        _Route("/api/profile", _endpoint(h_profile), methods=["GET"]),
        _Route("/api/audit", _endpoint(h_audit), methods=["GET"]),
        _Route("/api/export", _endpoint(h_export), methods=["GET"]),
        _Route("/api/identity/refresh", _endpoint(h_refresh, needs_body=True), methods=["POST"]),
        _Route("/api/status", _endpoint(h_status), methods=["GET"]),
        _Route("/api/metrics", _endpoint(h_metrics), methods=["GET"]),
    ])


def _main() -> int:
    # Do the one-time warm (Qdrant + embedding + BM25) up-front in this single
    # long-lived process. Sub-agents that later connect over HTTP inherit a
    # fully-warmed server and pay none of that cost per run.
    # Refuse to serve in an unsafe configuration BEFORE opening the socket.
    validate_deployment_security()

    if os.environ.get("MEMORY_EAGER_PREWARM") == "1":
        # Startup reconciliation touches every namespace by design.
        with memory_server.system_context("startup-prewarm"):
            memory_server._prewarm()

    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Mount
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    host = os.environ.get("MEMORY_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("MEMORY_HTTP_PORT", "3901"))

    manager = StreamableHTTPSessionManager(
        app=_server,
        event_store=None,
        stateless=True,
        security_settings=None,
    )

    # Bind the AUTHENTICATED caller for the duration of each MCP request. The
    # root of trust is the credential presented on the request, resolved
    # server-side against the operator's token store. A caller-supplied
    # identity header can neither establish nor widen authority.
    async def _lifespan(app):
        async with manager.run():
            log.info("Memory streamable-http manager up on %s:%s", host, port)
            # Stage 4: start the outbox worker pool and recover any claims a
            # previous process abandoned (its lease may not have expired yet, so
            # also force-recover them here). Boot recovery must not wait for a
            # lease timeout to make progress.
            with memory_server.system_context("startup-outbox-recovery"):
                memory_server._outbox_recover_abandoned()
                started = memory_server.start_outbox_workers()
            log.info("outbox worker pool started: %d worker(s); backlog: %s",
                     started, memory_server._outbox_metrics().get("outbox_convergence_lag"))
            # 1C-1: automatic session compaction worker (Q62 deliverable)
            if memory_server.start_compaction_worker():
                log.info("compaction worker started (auto session compaction ON)")
            # Sprint-1 1.1: durable op-telemetry write-behind flusher
            memory_server._start_op_telemetry_flusher()
            log.info("op telemetry flusher started")
            # Sprint-1 1.2: async LLM plausibility judge worker
            memory_server._start_judge_worker()
            try:
                yield
            finally:
                memory_server.stop_compaction_worker()
                memory_server.stop_outbox_workers()

    # Part 3: mount the user dashboard (3A-1) and Prometheus /metrics (3C-3).
    # Both authenticate per request exactly like the MCP endpoint: token ->
    # principal -> per-operation authorize_scope(), plus PG RLS underneath.
    dashboard_app = _build_dashboard_app()

    async def _root_metrics(request):
        principal, err = _dash_guard(request)
        if err is not None:
            return err
        _MS.set_caller_principal(principal)
        try:
            return await _run_in_threadpool(_prometheus_metrics, principal, request)
        except Exception as e:
            return _PlainTextResponse(f"# metrics error: {e}\n")
        finally:
            _MS.clear_caller_principal()

    from starlette.routing import Route as _RootRoute

    async def _healthz(request):
        return _PlainTextResponse("ok")

    app = Starlette(
        routes=[
            Mount("/mcp", app=bind_caller_identity(manager)),
            Mount("/ui", app=dashboard_app),
            _RootRoute("/metrics", _root_metrics, methods=["GET"]),
            _RootRoute("/healthz", _healthz, methods=["GET"]),
        ],
        lifespan=_lifespan,
    )
    log.info("Serving shared Memory MCP on %s:%s/mcp | dashboard %s:%s/ui | metrics %s:%s/metrics | healthz %s:%s/healthz",
             host, port, host, port, host, port, host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
