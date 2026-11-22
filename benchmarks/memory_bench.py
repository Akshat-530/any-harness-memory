"""Runnable accuracy harness for Memory (recall@k / MRR / hit@1).

Stores a dataset of memories, then answers queries and scores whether the
expected evidence is retrieved. Works against the live MCP endpoint.

Usage:
    python memory_bench.py dataset.json --url http://127.0.0.1:4000/mcp --k 10
    python memory_bench.py --self-test

Dataset format:
    {
      "memories": [{"content": "...", "category": "fact", "importance": 6}, ...],
      "queries":  [{"query": "...", "expect": ["substring", "..."]}, ...]
    }
A query is a hit if any `expect` substring appears in a retrieved memory's
content. This is a transparent, reproducible proxy — not the official LoCoMo
judge (which needs the LoCoMo dataset + LLM grading); use it to track recall
quality across changes.
"""
import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SELF_TEST = {
    "memories": [
        {"content": "The staging cluster runs on port 8443.", "category": "fact"},
        {"content": "Alice prefers dark mode interfaces.", "category": "preference"},
        {"content": "Deploys happen on Fridays at 17:00 UTC.", "category": "fact"},
    ],
    "queries": [
        {"query": "what port does the staging cluster use", "expect": ["8443"]},
        {"query": "what theme does alice like", "expect": ["dark mode"]},
        {"query": "when are deploys", "expect": ["friday"]},
    ],
}


async def run(url: str, data: dict, k: int) -> dict:
    async with streamable_http_client(url) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for m in data.get("memories", []):
                args = {"content": m["content"]}
                for key in ("category", "importance", "tags"):
                    if key in m:
                        args[key] = m[key]
                await s.call_tool("memory_store", args)
            await asyncio.sleep(2)

            n = hit1 = hitk = 0
            rr = 0.0
            for q in data.get("queries", []):
                n += 1
                res = await s.call_tool("memory_recall", {"query": q["query"], "top_k": k})
                mems = json.loads(res.content[0].text).get("memories", [])
                expect = [e.lower() for e in q.get("expect", [])]
                rank = None
                for i, mm in enumerate(mems, 1):
                    c = (mm.get("content") or "").lower()
                    if expect and any(e in c for e in expect):
                        rank = i
                        break
                if rank == 1:
                    hit1 += 1
                if rank:
                    hitk += 1
                    rr += 1.0 / rank
            return {
                "queries": n, "k": k,
                "hit@1": round(hit1 / n, 4) if n else 0.0,
                "hit@k": round(hitk / n, 4) if n else 0.0,
                "mrr": round(rr / n, 4) if n else 0.0,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", help="path to dataset JSON, or --self-test")
    ap.add_argument("--url", default="http://127.0.0.1:4000/mcp")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test or a.dataset == "--self-test":
        data = SELF_TEST
    else:
        with open(a.dataset, encoding="utf-8") as fh:
            data = json.load(fh)

    print(json.dumps(asyncio.run(run(a.url, data, a.k)), indent=2))


if __name__ == "__main__":
    main()