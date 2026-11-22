"""Interconnect test: docs route to RAG (not memory), and recall also searches RAG."""
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

            # 1) "save a documentation" -> goes to RAG, NOT memory
            doc = ("# Policy %s\n\nThis document describes the deployment policy.\n\n"
                   "- Data residency: eu-west-1\n- Backup window: 02:00 UTC\n"
                   "```\nretention: 30d\n```\n" % tag)
            routed = await _j(s, "memory_store", {"content": doc, "category": "documentation",
                                                  "source": "policy-%s.md" % tag})
            assert routed.get("routed_to") == "rag", ("doc not routed to RAG", routed)
            assert routed.get("chunks", 0) >= 1, routed
            assert "memory_id" not in routed, ("doc leaked into memory", routed)

            kbs = await _j(s, "kb_search" if False else "memory_kb_search",
                           {"query": "Policy %s data residency backup window" % tag, "top_k": 5})
            assert any(x.get("source") == "policy-%s.md" % tag for x in kbs["results"]), kbs

            # 2) recall also searches RAG
            await _j(s, "memory_kb_ingest", {"text": "Widget %s color is teal." % tag,
                                             "source": "widget-%s.md" % tag})
            await _j(s, "memory_store", {"content": "Widget %s size is large." % tag,
                                         "category": "fact", "importance": 6})
            await asyncio.sleep(2)
            rec = await _j(s, "memory_recall", {"query": "Widget %s" % tag, "top_k": 5})
            assert rec.get("knowledge_count", 0) >= 1, ("recall did not search RAG", rec)
            assert any("teal" in (x.get("content") or "") for x in rec["knowledge"]), rec["knowledge"]
            assert any("large" in (m.get("content") or "") for m in rec["memories"]), rec["memories"]

            # cleanup
            await _j(s, "memory_kb_delete", {"source": "policy-%s.md" % tag})
            await _j(s, "memory_kb_delete", {"source": "widget-%s.md" % tag})

            print("INTERCONNECT_OK routed_docs=%d kb_hits=%d recall_knowledge=%d memories=%d"
                  % (routed["chunks"], kbs["count"], rec["knowledge_count"], rec["count"]))


asyncio.run(main())
