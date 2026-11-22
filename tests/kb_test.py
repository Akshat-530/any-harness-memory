"""RAG lane test: ingest -> search (with citations) -> sources -> delete."""
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
            tools = [t.name for t in (await s.list_tools()).tools]
            assert "memory_kb_ingest" in tools and "memory_kb_search" in tools, "kb tools not advertised"

            tag = uuid.uuid4().hex[:6].upper()
            src = "zephyr-%s.md" % tag
            text = ("Zephyr %s deployment runs on Kubernetes in eu-west-1 and uses PostgreSQL 17. "
                    "The backup window is 02:00 UTC and the retention policy is 30 days." % tag)

            ing = await _j(s, "memory_kb_ingest", {"text": text, "source": src, "title": "Zephyr Runbook"})
            assert ing["chunks"] >= 1, ing

            sch = await _j(s, "memory_kb_search", {"query": "Zephyr %s backup window retention" % tag, "top_k": 5})
            assert sch["count"] >= 1, sch
            top = sch["results"][0]
            assert top.get("citation"), top
            low = (top.get("content") or "").lower()
            assert ("backup window" in low) or ("retention" in low), top
            assert top.get("source") == src, top

            srcs = await _j(s, "memory_kb_sources", {})
            assert any(x["source"] == src for x in srcs["sources"]), srcs

            de = await _j(s, "memory_kb_delete", {"source": src})
            assert de["deleted"]["documents"] >= 1, de

            after = await _j(s, "memory_kb_search", {"query": "Zephyr %s backup window" % tag, "top_k": 5, "source": src})
            assert after["count"] == 0, ("delete did not remove chunks", after)

            print("KB_OK chunks=%d hits=%d citation=%s tools=%d"
                  % (ing["chunks"], sch["count"], top["citation"], len(tools)))


asyncio.run(main())

