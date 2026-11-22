import asyncio
import json
import os

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"


async def main():
    tok = os.environ.get("MTOK")
    hc = httpx2.AsyncClient(headers={"Authorization": "Bearer " + tok}, timeout=60.0)
    async with streamable_http_client(URL, http_client=hc) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()
            r1 = await s.call_tool("memory_recall", {"query": "widget", "top_k": 3})
            print("RECALL is_error=", getattr(r1, "is_error", None))
            print(r1.content[0].text[:400])
    await hc.aclose()


asyncio.run(main())