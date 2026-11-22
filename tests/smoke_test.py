"""End-to-end smoke test for the Memory container (profile-agnostic).

    docker run --rm --network "$(basename $PWD)_memory" --entrypoint python \
      -v "$PWD/tests/smoke_test.py:/smoke.py:ro" memory:latest /smoke.py
"""
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"
TOKEN = "smoke-" + "0123456789abcdef"


async def main():
    async with streamable_http_client(URL) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            assert len(tools) >= 30, f"too few tools: {len(tools)}"
            assert "memory_store" in tools and "memory_recall" in tools
            assert not any(n.startswith("memeory") for n in tools), "old prefix leaked"

            st = await s.call_tool("memory_store", {
                "content": f"Smoke test {TOKEN}: durable store + immediate recall.",
                "category": "fact", "importance": 7})
            assert not st.is_error, st.content[0].text
            stored = json.loads(st.content[0].text)
            assert stored.get("status") == "stored", stored

            rc = await s.call_tool("memory_recall", {
                "query": f"Smoke test {TOKEN} durable", "top_k": 3})
            assert not rc.is_error, rc.content[0].text
            assert json.loads(rc.content[0].text)["count"] >= 1, "store not visible to recall"

            # resources are the zero-call bootstrap path
            ctx = await s.read_resource("memory://context")
            assert ctx.contents and ctx.contents[0].text is not None

            hc = await s.call_tool("memory_health_check", {})
            assert not hc.is_error and "healthy" in hc.content[0].text

            print("SMOKE_OK tools=%d stored=%s" % (len(tools), stored.get("memory_id")))


asyncio.run(main())
