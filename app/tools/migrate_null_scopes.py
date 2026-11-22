#!/usr/bin/env python3
"""Migrate legacy NULL-scope Qdrant payloads to an explicit scope.

`_build_filter()` used to admit legacy NULL-scope rows via an
`OR scope IS NULL` branch. That branch is what made every scoped read a
cross-tenant read, so it was removed. The 7 legacy NULL-scope rows must
therefore be given a real scope before they become unreachable.

Idempotent: only touches points whose payload has no `scope` key or a null one.
Usage:  python3 tools/migrate_null_scopes.py [--dry-run] [--scope global]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import memory_server as M  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scope", default="global",
                    help="target scope for legacy rows (default: global)")
    args = ap.parse_args()

    target = M.authorize_scope(args.scope)
    fixed = scanned = 0
    offset = None
    while True:
        body = {"collection_name": M.COLLECTION_NAME, "limit": 1000,
                "with_payload": True}
        if offset is not None:
            body["offset"] = offset
        pts, offset = M._client.scroll(**body)
        if not pts:
            break
        for p in pts:
            scanned += 1
            pld = p.payload or {}
            if pld.get("scope") is not None:
                continue
            fixed += 1
            if args.dry_run:
                print(f"  would set scope={target} on {p.id}")
                continue
            M._client.set_payload(
                collection_name=M.COLLECTION_NAME,
                points=[p.id],
                payload={"scope": target},
            )
            print(f"  set scope={target} on {p.id}")
        if offset is None:
            break

    print(f"scanned={scanned} null_scope_rows={fixed} "
          f"{'(dry-run)' if args.dry_run else '(applied)'}")
    if fixed and not args.dry_run:
        M._cached_recall.cache_clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())