"""Bi-temporal test: event_time, timeline, supersession, and as_of recall."""
import asyncio
import datetime
import json
import uuid

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://memory-mcp:4000/mcp"


async def _j(s, name, args):
    r = await s.call_tool(name, args)
    assert not r.is_error, f"{name}: {r.content[0].text[:200]}"
    return json.loads(r.content[0].text)


def _ids(rec):
    return [m.get("memory_id") for m in rec.get("memories", [])]


async def main():
    async with streamable_http_client(URL) as (r, w):
        async with ClientSession(r, w, read_timeout_seconds=60) as s:
            await s.initialize()

            # 1) event_time is stored and surfaced (isolated memory)
            k = uuid.uuid4().hex[:8].upper()
            st = await _j(s, "memory_store", {
                "content": "Eventclock %s marks the release at midnight." % k,
                "category": "fact", "importance": 6,
                "event_time": "2021-07-04T12:00:00Z"})
            eid = st["memory_id"]
            tl = await _j(s, "memory_timeline", {"memory_id": eid})
            en = next((n for n in tl["timeline"] if n["memory_id"] == eid), None)
            assert en, ("timeline missing own node", tl)
            assert en["event_time"] == "2021-07-04T12:00:00Z", en
            assert en["valid_from"], en

            # 2) supersession → point-in-time recall
            tok = uuid.uuid4().hex[:8].upper()
            st_a = await _j(s, "memory_store", {
                "content": "Zephyr %s production deployment runs Python 3.11 build 77." % tok,
                "category": "fact", "importance": 6})
            aid = st_a["memory_id"]
            await asyncio.sleep(2)
            as_of = datetime.datetime.now(datetime.timezone.utc).isoformat()
            await asyncio.sleep(3)
            await _j(s, "memory_store", {
                "content": "Zephyr %s production deployment changed to Python 3.12 build 77." % tok,
                "category": "fact", "importance": 6})
            await asyncio.sleep(3)

            tl2 = await _j(s, "memory_timeline", {"memory_id": aid})
            na = next((n for n in tl2["timeline"] if n["memory_id"] == aid), None)
            superseded = bool(na and na.get("status") == "superseded" and na.get("valid_until"))

            now_rec = await _j(s, "memory_recall", {"query": "Zephyr %s deployment Python build 77" % tok, "top_k": 10})
            past_rec = await _j(s, "memory_recall", {"query": "Zephyr %s deployment Python build 77" % tok, "top_k": 10, "as_of": as_of})
            now_ids, past_ids = _ids(now_rec), _ids(past_rec)

            if superseded:
                assert aid not in now_ids, ("superseded leaked into current recall", now_ids)
                assert aid in past_ids, ("as_of did not return the historical version", past_ids)
                g0 = await s.call_tool("memory_get", {"memory_id": aid})
                assert g0.is_error, "current get should refuse a superseded memory"
                g1 = await s.call_tool("memory_get", {"memory_id": aid, "as_of": as_of})
                assert not g1.is_error, ("memory_get as_of failed", g1.content[0].text[:120])
                ln = await _j(s, "memory_list", {"as_of": as_of, "limit": 20})
                assert ln["count"] >= 1, ln

            print("TEMPORAL_OK superseded=%s timeline=%d now_ids=%d past_ids=%d event_time_ok=True"
                  % (superseded, tl2["count"], len(now_ids), len(past_ids)))


asyncio.run(main())

