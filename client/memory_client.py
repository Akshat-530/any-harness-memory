"""MemoryClient — thin harness-side client for the Memory MCP.

Gives any Python harness the lifecycle hooks needed to make memory "core-injected":

    client = MemoryClient("http://127.0.0.1:4000/mcp")
    ctx    = await client.context()            # memory://context  (bootstrap)
    hits   = await client.recall("user question", top_k=3)
    await client.capture(user_msg, asst_msg)   # memory_turn_ingest
    await client.task_end("task", "success")

Requires: mcp  (already a dependency of the memory image).
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MemoryClient:
    def __init__(self, url: str = "http://127.0.0.1:4000/mcp", token: str | None = None):
        self.url = url
        self.token = token

    def _http(self):
        if not self.token:
            return None
        try:
            import httpx2 as _h  # mcp 2.x transport client
        except Exception:  # pragma: no cover
            import httpx as _h  # type: ignore
        return _h.AsyncClient(headers={"Authorization": f"Bearer {self.token}"})

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        """Open one warmed session; batch many calls inside it."""
        hc = self._http()
        try:
            async with streamable_http_client(self.url, http_client=hc) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    yield s
        finally:
            if hc is not None:
                await hc.aclose()

    # -- low level ---------------------------------------------------------
    async def call(self, name: str, args: dict | None = None) -> Any:
        async with self.session() as s:
            r = await s.call_tool(name, args or {})
            if getattr(r, "is_error", False):
                raise RuntimeError(f"{name}: {r.content[0].text[:300]}")
            text = r.content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text

    async def _resource(self, uri: str) -> str:
        async with self.session() as s:
            rr = await s.read_resource(uri)
            try:
                return rr.contents[0].text
            except Exception:
                return ""

    # -- lifecycle hooks ---------------------------------------------------
    async def context(self) -> str:
        """Assembled session context to prepend to the system prompt (once)."""
        return await self._resource("memory://context")

    async def profile(self) -> str:
        return await self._resource("memory://profile")

    async def core(self) -> Any:
        txt = await self._resource("memory://core")
        try:
            return json.loads(txt)
        except Exception:
            return txt

    async def session_init(self, task: str, task_type: str | None = None, top_k: int = 5) -> Any:
        args: dict[str, Any] = {"task": task, "top_k": top_k}
        if task_type:
            args["task_type"] = task_type
        return await self.call("memory_session_init", args)

    async def recall(self, query: str, top_k: int = 3, scope: str | None = None) -> Any:
        args: dict[str, Any] = {"query": query, "top_k": top_k}
        if scope:
            args["scope"] = scope
        return await self.call("memory_recall", args)

    async def store(self, content: str, **kw) -> Any:
        return await self.call("memory_store", {"content": content, **kw})

    async def capture(self, user_message: str, assistant_message: str = "",
                      tool_events: list | None = None, scope: str | None = None) -> Any:
        """End-of-turn capture in ONE call (facts + error->fix)."""
        args: dict[str, Any] = {"user_message": user_message,
                                "assistant_message": assistant_message}
        if tool_events:
            args["tool_events"] = tool_events
        if scope:
            args["scope"] = scope
        return await self.call("memory_turn_ingest", args)

    async def task_end(self, task: str, outcome: str, **kw) -> Any:
        return await self.call("memory_task_end", {"task": task, "outcome": outcome, **kw})

    async def feedback(self, memory_id: str, query: str, rating: str) -> Any:
        return await self.call("memory_feedback",
                               {"memory_id": memory_id, "query": query, "rating": rating})