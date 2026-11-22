"""Memory -> RAG reconciliation test: a confirmed memory marks stale doc chunks
and appends an update chunk (stale excluded from search by default)."""
import asyncio
import json
import os
import uuid

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"


async def _j(s, name, args):
    r = await s.call_tool(name, args)
    if getattr(r, "is_error", False):
        raise RuntimeError(f"{name}: {r.content[0].text[:200]}")
    return json.loads(r.content[0].text)


async def main():
    tok = os.environ.get("MTOK")
    hc = httpx2.AsyncClient(headers={"Authorization": "Bearer " + tok}, timeout=60.0) if tok else None
    async with streamable_http_client(URL, http_client=hc) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()

            tag = uuid.uuid4().hex[:6].upper()
            src = "kbsync-%s.md" % tag
            await _j(s, "memory_kb_ingest", {
                "text": "The API rate limit is 100 requests per minute.", "source": src, "title": "API limits"})

            before = await _j(s, "memory_kb_search", {"query": "API rate limit requests per minute", "top_k": 5})
            assert any("100" in (x.get("content") or "") for x in before["results"]), before

            rec = await _j(s, "memory_kb_reconcile", {
                "content": "The API rate limit changed to 500 requests per minute."})
            assert rec["marked"], ("nothing marked stale", rec)
            assert rec["update_chunk"] is True, rec

            after = await _j(s, "memory_kb_search", {"query": "API rate limit requests per minute", "top_k": 5})
            assert any("500" in (x.get("content") or "") for x in after["results"]), ("update not surfaced", after)
            assert not any("is 100" in (x.get("content") or "") for x in after["results"]), ("stale leaked", after)

            stale = await _j(s, "memory_kb_search", {
                "query": "API rate limit requests per minute", "top_k": 5, "include_stale": True})
            assert any("is 100" in (x.get("content") or "") for x in stale["results"]), ("stale not retrievable", stale)

            await _j(s, "memory_kb_delete", {"source": src})
            await _j(s, "memory_kb_delete", {"source": src + "#update"})

            print("KB_SYNC_OK marked=%d update=%s" % (len(rec["marked"]), rec["update_chunk"]))


asyncio.run(main())
