import asyncio
import json
import os

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main():
    tok = os.environ["MTOK"]
    hc = httpx2.AsyncClient(headers={"Authorization": "Bearer " + tok})
    async with streamable_http_client("http://memory-mcp:4000/mcp", http_client=hc) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            st = await s.call_tool("memory_stats", {})
            txt = st.content[0].text
            try:
                d = json.loads(txt)
                keys = {k: d.get(k) for k in ("total_memories", "total", "count", "by_category", "categories") if k in d}
                if not keys:
                    keys = {"raw": txt[:300]}
            except Exception:
                keys = {"raw": txt[:300]}
            print("TOOLS=%d all_memory_prefix=%s" % (len(tools), all(n.startswith("memory_") for n in tools)))
            print("STATS=", json.dumps(keys, ensure_ascii=False)[:400])
    await hc.aclose()


asyncio.run(main())