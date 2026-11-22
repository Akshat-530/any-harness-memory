"""Verify the stdio transport (for stdio-only harnesses).

Runs inside the memory-mcp container so the spawned server inherits the backend
env + network:

    docker cp tests/stdio_test.py <container>:/tmp/stdio_test.py
    docker compose exec -T memory-mcp python /tmp/stdio_test.py
"""
import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command="/opt/memory-venv/bin/python",
        args=["/app/memory_server.py"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            assert len(tools.tools) >= 72, len(tools.tools)
            st = await s.call_tool("memory_health_check", {})
            assert not st.is_error
            print("STDIO_OK tools=%d" % len(tools.tools))


asyncio.run(main())