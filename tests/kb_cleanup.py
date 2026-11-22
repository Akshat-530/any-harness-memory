"""Purge all knowledge-base documents in the caller's scope (maintenance helper)."""
import asyncio
import json
import os

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"


async def main():
    tok = os.environ.get("MTOK")
    hc = httpx2.AsyncClient(headers={"Authorization": "Bearer " + tok}, timeout=60.0) if tok else None
    n = 0
    async with streamable_http_client(URL, http_client=hc) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()
            srcs = json.loads((await s.call_tool("memory_kb_sources", {})).content[0].text)
            for x in srcs.get("sources", []):
                await s.call_tool("memory_kb_delete", {"source": x["source"]})
                n += 1
    if hc is not None:
        await hc.aclose()
    print("KB_CLEANUP removed_sources=%d" % n)


asyncio.run(main())