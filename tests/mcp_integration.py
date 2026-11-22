"""Full-stack MCP integration test â€” store/get/recall/list/stats/graph/context/
resources/turn-ingest/health through the live endpoint, as a harness would.

    docker run --rm --network "$(basename $PWD)_memory" --entrypoint python \
      -v "$PWD/tests/mcp_integration.py:/t.py:ro" memory:latest /t.py
"""
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"


async def _json(session, name, args):
    r = await session.call_tool(name, args)
    assert not r.is_error, f"{name}: {r.content[0].text[:200]}"
    return json.loads(r.content[0].text)


async def _eventually(fn, pred, tries=8, delay=2.0):
    last = None
    for _ in range(tries):
        last = await fn()
        if pred(last):
            return last
        await asyncio.sleep(delay)
    return last


async def main():
    async with streamable_http_client(URL) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            assert len(tools) >= 30, len(tools)
            assert "memory_turn_ingest" in tools, "turn ingest not advertised"
            assert "memory_dream" not in tools, "operator tool leaked into agent profile"

            # resources: zero-call bootstrap
            for uri in ("memory://context", "memory://profile", "memory://core"):
                rr = await s.read_resource(uri)
                assert rr.contents and rr.contents[0].text is not None, uri

            # turn ingest (one-call capture)
            ti = await _json(s, "memory_turn_ingest", {
                "user_message": "Integration turn: I prefer concise answers.",
                "assistant_message": "Noted.",
                "tool_events": [{"tool": "memory_store", "status": "ok"}]})
            assert ti["status"] == "ingested", ti

            a = await _json(s, "memory_store", {
                "content": "Integration A: the quick brown fox jumps over the lazy dog.",
                "category": "fact", "importance": 6})
            b = await _json(s, "memory_store", {
                "content": "Integration B: Qdrant and Postgres back the memory stack.",
                "category": "fact", "importance": 6})
            aid, bid = a["memory_id"], b["memory_id"]

            got = await _eventually(
                lambda: _json(s, "memory_get", {"memory_id": aid}),
                lambda d: d.get("memory_id") == aid)
            assert got.get("memory_id") == aid, "store not retrievable"

            rc = await _eventually(
                lambda: _json(s, "memory_recall", {"query": "quick brown fox", "top_k": 5}),
                lambda d: d.get("count", 0) >= 1)
            assert rc["count"] >= 1, "recall empty"

            ls = await _json(s, "memory_list", {"limit": 20})
            assert ls["count"] >= 2, ls.get("count")
            st = await _json(s, "memory_stats", {})
            assert st.get("total_memories", 0) >= 2, st

            await _json(s, "memory_graph_link",
                        {"memory_id": aid, "related_id": bid, "relation": "related_to"})
            nb = await _json(s, "memory_graph_neighbors", {"memory_id": aid})
            assert isinstance(nb, dict), nb

            cp = await _json(s, "memory_context_preview", {"query": "memory stack"})
            assert isinstance(cp, dict), cp

            h = await s.call_tool("memory_health_check", {})
            assert not h.is_error and "healthy" in h.content[0].text

            print("INTEGRATION_OK tools=%d ingested=%s stored=%s,%s list=%d"
                  % (len(tools), len(ti.get("stored", [])), aid[:8], bid[:8], ls["count"]))


asyncio.run(main())
