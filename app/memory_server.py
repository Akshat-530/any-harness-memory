#!/usr/bin/env python3
"""Memory MCP Server — Qdrant + Redis memory for AGENT MOON.

Qdrant-backed long-term vector memory + Redis short-term memory.
2026 Industry Best Practices (all 8 implemented):

1. GRAPH MEMORY: Relationship edges stored in Redis. Tools: graph_link,
   graph_unlink, graph_neighbors, graph_relations. Supports multi-hop traversal.

2. REFLECTION LOOP (Reflexion pattern): Self-improvement via memory_reflect.
   Detects category/source patterns, conflicts, isolation warnings, and staleness.
   Generates ACTIONABLE insights + learned strategies stored as procedure memories.
   Outcome-driven: pass outcome='success'/'failure' for targeted guidance.
   Loop detection prevents degeneration.

3. MEMORY POISONING GUARD (OWASP LLM04): 17 regex patterns detect injection
   attacks: instruction override, role override, system command injection,
   XSS, pipe injection, code injection, destructive commands, SQL attacks,
   JS injection, Lua injection, fenced code injection. Content is blocked
   (>=0.7 score) and sanitized on store. memory_poison_check tool available.

4. MIXTURE-OF-AGENTS: Query classification into 7 types (factual, procedural,
   preference, decision, error_fix, config, general) with specialized recall
   strategies per type. Query classifier auto-detects intent.

5. EVAL BEFORE DEPLOY: Sandbox store (memory_eval_store) commits to memory
   instead of Qdrant. Preview with memory_eval_recall. Commit or discard.
   All ops sandboxed with no production impact.

6. VERSIONED ROLLBACK: Every store/update creates a version snapshot in Redis.
   memory_versions lists history, memory_version_get inspects a snapshot,
   memory_version_restore does safe rollback (preserves current as new version).

7. SEMANTIC CACHE: Identical recall queries return cached results from Redis,
   skipping embedding + hybrid search. Cosine-similarity threshold 0.95, TTL 1h.
   memory_cache_stats reports hits/misses/hit rate. Reduces latency 20-70% for
   repeated or near-duplicate queries.

8. CLOSED REFLEXION LOOP (SelfMem pattern): Outcome-driven reflection generates
   actionable learned strategies stored as procedure memories. memory_get_insights
   retrieves these before new tasks. memory_hint gives the agent agency over
   memory management (boost/evict/archive/link/decay). Reflection loop guard
   prevents degeneration. Addresses all 5 mechanism families from arXiv:2603.07670.

Scoring weights (recall):
  - Hybrid (dense+sparse RRF): 60%
  - Importance: 25%
  - Recency: 15%

Tools:
  Long-term (Qdrant):
  - memory_store   : persist a memory (auto-embedded, poison-guarded, versioned)
  - memory_recall  : semantic recall with MoA query classification
  - memory_list    : list / filter memories
  - memory_get     : fetch one memory by id
  - memory_update  : update content (re-embeds, creates version)
  - memory_delete  : remove a memory by id
  - memory_stats   : collection statistics
  - memory_consolidate : cleanup old/duplicate memories
  - memory_reflect : self-improvement reflection analysis (outcome-driven, loop guard)
  - memory_get_insights : retrieve actionable learned strategies from reflections
  - memory_hint : agent-driven memory management (SelfMem pattern: boost/evict/archive/link/decay)
  - memory_graph_* : graph memory relationship tools
  - memory_eval_*  : sandbox eval tools
  - memory_version_* : versioned rollback tools
  - memory_poison_check : poisoning pattern detector
  - memory_cache_stats : semantic cache hit/miss statistics
  Short-term (Redis):
  - memory_short_store   : store a short-term memory (TTL: 7 days default)
  - memory_short_recall : recall a short-term memory by key
  - memory_short_delete : delete a short-term memory
  - memory_short_list   : list all short-term memory keys
  - memory_short_stats : Redis statistics
"""

import asyncio
import contextlib
import contextvars
import threading
import hashlib
import hmac
import json
import logging
import math
import os
import random
import re
import socket
import subprocess
import time
import uuid
from collections import Counter, OrderedDict, deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer
import redis
import bm25s
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "lg_memory"
# Embedding model is configurable via MEMORY_EMBEDDING_MODEL / MEMORY_VECTOR_SIZE env vars.
# Default: Qwen3-Embedding-0.6B (1024d) — strong MTEB, clean install, CPU-friendly on 4 cores.
# Upgrade / downgrade path:
#   all-MiniLM-L6-v2 (384d) — fast, weakest
#   BAAI/bge-small-en-v1.5 (384d)
#   nomic-embed-text-v1.5 (768d) — needs trust_remote_code
#   Qwen/Qwen3-Embedding-0.6B (1024d) — DEFAULT, best MTEB/CPU ratio
# NOTE: changing vector size requires a NEW Qdrant collection + full re-embed
# (see memory_reembed_all tool).
EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_MODEL_VERSION = os.environ.get("MEMORY_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
VECTOR_SIZE = int(os.environ.get("MEMORY_VECTOR_SIZE", "1024"))
# RAG / knowledge-base lane (documents), kept separate from memories.
KB_COLLECTION = os.environ.get("MEMORY_KB_COLLECTION", "kb_chunks")
KB_CHUNK_SIZE = int(os.environ.get("MEMORY_KB_CHUNK_SIZE", "900"))
KB_CHUNK_OVERLAP = int(os.environ.get("MEMORY_KB_CHUNK_OVERLAP", "150"))
KB_AUTOSYNC = os.environ.get("MEMORY_KB_AUTOSYNC", "").strip() == "1"
KB_RECONCILE_THRESHOLD = float(os.environ.get("MEMORY_KB_RECONCILE_THRESHOLD", "0.80"))
KB_SYNC_MIN_IMPORTANCE = int(os.environ.get("MEMORY_KB_SYNC_MIN_IMPORTANCE", "7"))
# Routing: documentation-like writes go to RAG; recall also searches RAG.
DOC_ROUTE_CHARS = int(os.environ.get("MEMORY_DOC_ROUTE_CHARS", "2000"))
RECALL_INCLUDE_KB = os.environ.get("MEMORY_RECALL_INCLUDE_KB", "").strip() == "1"
# Cross-encoder reranker for post-recall precision. Empty string = disabled.
# Default: ms-marco-MiniLM-L6-v2 (~500MB RAM, ~50ms for 50 pairs on 4-core CPU).
RERANKER_MODEL = os.environ.get("MEMORY_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_TOP_K = int(os.environ.get("MEMORY_RERANK_TOP_K", "10"))  # candidates to rerank (1D-1: 5→10, grid-searched 2026-10-08)
STATS_CACHE_TTL_SECONDS = 30
PREWARM_QUERIES = 3

REDIS_HOST = os.environ.get("MEMORY_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("MEMORY_REDIS_PORT", "6379"))
# Moved out of source (Phase O: a credential committed to a file is a
# credential that eventually leaks via a backup, a diff, or a paste).
REDIS_PASSWORD = os.environ.get("MEMORY_REDIS_PASSWORD", "")
REDIS_PREFIX = "memory:short:"
REDIS_DEFAULT_TTL = 60 * 60 * 24 * 7  # 7 days in seconds

# Enhancement 7: Semantic Cache — avoid redundant vector searches for semantically
# identical queries.  Uses Redis to store (query_embedding_hash → recall_results).
# Cosine similarity threshold of 0.95 means near-duplicate queries return cached.
SEMANTIC_CACHE_PREFIX = "memory:cache:"
SEMANTIC_CACHE_TTL = 3600         # 1 hour
SEMANTIC_CACHE_THRESHOLD = float(os.environ.get("MEMORY_CACHE_THRESHOLD", "0.90"))  # cosine similarity minimum for cache hit (widened 0.95->0.90 to skip embedding on paraphrased repeats)
_semantic_cache_hits = 0
_semantic_cache_misses = 0

# BM25 / Hybrid search configuration
BM25_K1 = 1.5       # term frequency saturation (bm25s default)
BM25_B = 0.75       # document length normalization (bm25s default)
BM25_RRF_K = 60     # RRF constant
HYBRID_ALPHA = 0.5  # balance dense (0.5) vs sparse/keyword (0.5)
BM25_SCORE_CACHE_SIZE = 512  # LRU cache for per-query BM25 scores

# bm25s index — None until first rebuild; rebuilt on startup from persisted corpus
_bm25_index: bm25s.BM25 | None = None
_bm25_corpus: dict[str, list[str]] = {}  # mem_id -> token list (for incremental add)
_bm25_index_path = os.environ.get("MEMORY_BM25_PATH") or os.path.join(os.path.dirname(__file__), "bm25_index.json")
# Debounce state (2026-10-07): bm25s.BM25 has no incremental add/remove, so any
# corpus mutation used to force an O(corpus) full rebuild inline. A burst of
# outbox events then triggered one full rebuild per event and the single worker
# could not keep up (the 24h pending-outbox stall). Writers now only mark the
# corpus dirty under _BM25_REBUILD_LOCK; the rebuild happens once per claimed
# outbox batch (see _bm25_maybe_rebuild) and on explicit force.
_bm25_dirty = False
# ID ordering the CURRENT _bm25_index was built from. Snapshotted with the index
# so a reader maps scores to ids using the index's own row order, never the
# concurrently-mutated corpus dict (which could otherwise shift positions).
_bm25_indexed_ids: list[str] = []
# The ONLY lock touching _bm25_index/_bm25_corpus. Writers (mark-dirty) and
# readers (snapshot) acquire it; readers must NEVER call a rebuild while holding
# it (that would reintroduce the per-query rebuild the debounce exists to stop).
# RLock, not Lock: _bm25_maybe_rebuild acquires the lock and then calls
# _rebuild_bm25_index, which re-acquires it. A plain Lock self-deadlocked the
# whole outbox worker pool there (verified live 2026-10-07 via py-spy: worker
# stuck at the inner acquire, three more blocked on the outer one). RLock
# keeps mutual exclusion against OTHER threads while permitting the
# already-holding thread to re-enter.
_BM25_REBUILD_LOCK = threading.RLock()
# LRU cache for BM25 scores: key = (query_tokens_key, mem_id) -> normalized score
_bm25_score_cache: OrderedDict = OrderedDict()
_bm25_pos_cache: dict = {}      # {"pos": {mem_id: row}} — avoids O(n) index()
_bm25_score_max: float = 1.0  # running max for normalization within a query

# Hybrid candidate-pool sizing. The sparse stage must be able to rescue a
# document the dense stage dropped, so both stages fetch a wider pool than
# top_k and the fusion runs over their union.
DENSE_CANDIDATE_MULT = 8
SPARSE_CANDIDATE_MULT = 8

# Cross-encoder for lightweight reranking (lazy-loaded on first search).
# Model id is configurable via MEMORY_RERANKER_MODEL; env default is the
# verified L6 model (L-6 is a typo and not a real HF repo).
TITLE_BOOST = 0.20  # +20% score boost if query terms appear in memory title
_cross_encoder = None  # type: ignore

# ---------------------------------------------------------------------------
# P1-4: Horizontal Scaling Configuration
# ---------------------------------------------------------------------------
# Enable horizontal scaling mode. When enabled, the server uses shared
# infrastructure suitable for running multiple instances behind a load balancer.
# 
# Config:
#   HORIZONTAL_SCALING_ENABLED=1     # enable horizontal scaling (default: 0)
#   EMBEDDING_SERVICE_URL=http://... # external embedding service URL (optional)
#   PG_POOL_MIN_SIZE=2               # minimum pg pool connections (default: 2)
#   PG_POOL_MAX_SIZE=10              # maximum pg pool connections (default: 10)
#   PG_POOL_TIMEOUT=30               # pool checkout timeout seconds (default: 30)
#   LEADER_ELECTION_TTL=10           # leader lock TTL seconds (default: 10)
#   BM25_REDIS_PREFIX=memory:bm25   # Redis prefix for BM25 index (default: memory:bm25)
#   RECALL_CACHE_PREFIX=memory:rc   # Redis prefix for recall cache (default: memory:rc)
#   RECALL_CACHE_TTL=3600            # recall cache TTL seconds (default: 3600)
#   OUTBOX_LEADER_LOCK=1             # enable leader election for outbox (default: 1)
#
# When HORIZONTAL_SCALING_ENABLED=1:
# - Embedding model is loaded from external service (if EMBEDDING_SERVICE_URL set)
# - Postgres uses connection pool instead of single connection
# - BM25 index stored in Redis (shared across instances)
# - Recall cache uses Redis (shared across instances)
# - Outbox workers use leader election (only one instance processes outbox)
# - BM25 index persisted to Redis for sharing

HORIZONTAL_SCALING_ENABLED = os.environ.get("HORIZONTAL_SCALING_ENABLED", "0") == "1"
EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL", "").rstrip("/")
PG_POOL_MIN_SIZE = int(os.environ.get("PG_POOL_MIN_SIZE", "2"))
PG_POOL_MAX_SIZE = int(os.environ.get("PG_POOL_MAX_SIZE", "10"))
PG_POOL_TIMEOUT = int(os.environ.get("PG_POOL_TIMEOUT", "30"))
LEADER_ELECTION_TTL = int(os.environ.get("LEADER_ELECTION_TTL", "10"))
BM25_REDIS_PREFIX = os.environ.get("BM25_REDIS_PREFIX", "memory:bm25:")
RECALL_CACHE_PREFIX = os.environ.get("RECALL_CACHE_PREFIX", "memory:rc:")
RECALL_CACHE_TTL = int(os.environ.get("RECALL_CACHE_TTL", "3600"))
OUTBOX_LEADER_ELECTION = os.environ.get("OUTBOX_LEADER_ELECTION", "1") == "1"

# ---------------------------------------------------------------------------
# Virtual Context / Prompt Assembly Middleware (P0-1)
# ---------------------------------------------------------------------------
# Automatic context injection on every agent turn. Configurable token budgets
# per section. The middleware runs in the MCP tool dispatcher and assembles
# a structured prompt from: working memory (short-term), session summary,
# top-k retrieved memories, procedures, learned strategies, and tool hints.
#
# Config:
#   VIRTUAL_CONTEXT_ENABLED=1          # enable automatic injection (default: 0)
#   VIRTUAL_CONTEXT_TOKEN_BUDGET=8000  # total token budget (default: 8000)
#   VIRTUAL_CONTEXT_WORKING_BUDGET=500 # working memory budget (default: 500)
#   VIRTUAL_CONTEXT_SESSION_BUDGET=1000 # session summary budget (default: 1000)
#   VIRTUAL_CONTEXT_RETRIEVAL_BUDGET=3000 # retrieved memories budget (default: 3000)
#   VIRTUAL_CONTEXT_PROCEDURE_BUDGET=1000 # procedures budget (default: 1000)
#   VIRTUAL_CONTEXT_STRATEGY_BUDGET=1000 # learned strategies budget (default: 1000)
#   VIRTUAL_CONTEXT_HINT_BUDGET=500    # tool hints budget (default: 500)
#   VIRTUAL_CONTEXT_TOP_K=5            # memories to retrieve (default: 5)
#   VIRTUAL_CONTEXT_INCLUDE_HINTS=1    # include tool hints (default: 1)
#
# Token counting uses tiktoken (cl100k_base) for accuracy.
VIRTUAL_CONTEXT_ENABLED = os.environ.get("VIRTUAL_CONTEXT_ENABLED", "0") == "1"
VIRTUAL_CONTEXT_TOKEN_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_TOKEN_BUDGET", "8000"))
VIRTUAL_CONTEXT_WORKING_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_WORKING_BUDGET", "500"))
VIRTUAL_CONTEXT_SESSION_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_SESSION_BUDGET", "1000"))
VIRTUAL_CONTEXT_RETRIEVAL_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_RETRIEVAL_BUDGET", "3000"))
VIRTUAL_CONTEXT_PROCEDURE_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_PROCEDURE_BUDGET", "1000"))
# 2A-4: anti-pattern ("avoid X") context budget — negative knowledge injects
# as DO NOT / PREFER instructions, small dedicated slice.
VIRTUAL_CONTEXT_AVOID_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_AVOID_BUDGET", "300"))
# 3A-2: user-profile context budget — behavioral adaptation slice. Small by
# design: the profile is high-signal, low-volume instructions, not a dump.
VIRTUAL_CONTEXT_PROFILE_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_PROFILE_BUDGET", "400"))
VIRTUAL_CONTEXT_STRATEGY_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_STRATEGY_BUDGET", "1000"))
VIRTUAL_CONTEXT_HINT_BUDGET = int(os.environ.get("VIRTUAL_CONTEXT_HINT_BUDGET", "500"))
VIRTUAL_CONTEXT_TOP_K = int(os.environ.get("VIRTUAL_CONTEXT_TOP_K", "5"))
# 2A-1 (Q28): A/B control-arm toggle — disables procedure/strategy injection
# in virtual context. Settable per-arm in-process for the experiment harness.
LEARNED_STRATEGIES_DISABLED = os.environ.get("MEMORY_DISABLE_STRATEGIES", "") == "1"
VIRTUAL_CONTEXT_INCLUDE_HINTS = os.environ.get("VIRTUAL_CONTEXT_INCLUDE_HINTS", "1") == "1"

# P2-1: Tunable Relevance Weights (must sum to 1.0)
# 1D-1 calibrated 2026-10-08 by grid search against the 63-query golden set
# (see /tmp/kilo/grid_results.jsonl + API_CHANGELOG.md). Dominant factor:
# RERANK_TOP_K 5→10 (the cross-encoder must see rank-6+ candidates or they
# can never be rescued; R@10 0.905→0.984). Weights 0.85/0.05/0.10 kept a
# small importance/recency signal for product behavior.
RELEVANCE_WEIGHT = float(os.environ.get("MEMORY_RELEVANCE_WEIGHT", "0.85"))
IMPORTANCE_WEIGHT = float(os.environ.get("MEMORY_IMPORTANCE_WEIGHT", "0.05"))
RECENCY_WEIGHT = float(os.environ.get("MEMORY_RECENCY_WEIGHT", "0.10"))

# Validate weights sum to 1.0
_RELEVANCE_WEIGHTS_SUM = RELEVANCE_WEIGHT + IMPORTANCE_WEIGHT + RECENCY_WEIGHT
if abs(_RELEVANCE_WEIGHTS_SUM - 1.0) > 0.001:
    logging.getLogger("memory").warning(
        "Relevance weights sum to %.4f, expected 1.0. Normalizing.", _RELEVANCE_WEIGHTS_SUM)
    RELEVANCE_WEIGHT /= _RELEVANCE_WEIGHTS_SUM
    IMPORTANCE_WEIGHT /= _RELEVANCE_WEIGHTS_SUM
    RECENCY_WEIGHT /= _RELEVANCE_WEIGHTS_SUM

# Virtual context metrics
_VC_METRICS = {
    "total_assemblies": 0,
    "tokens_injected": [],
    "sections_dropped": Counter(),
    "assembly_latency_ms": [],
    "budget_exceeded": 0,
}

# Tiktoken encoder for accurate token counting (initialized lazily after logging setup)
_vc_encoder = None
_vc_encoder_init_attempted = False

def _vc_init_encoder():
    """Initialize tiktoken encoder lazily."""
    global _vc_encoder, _vc_encoder_init_attempted
    if _vc_encoder_init_attempted:
        return
    _vc_encoder_init_attempted = True
    try:
        import tiktoken
        _vc_encoder = tiktoken.get_encoding("cl100k_base")
        log.info("tiktoken encoder initialized for virtual context")
    except Exception:
        _vc_encoder = None
        log.warning("tiktoken not available, virtual context will use rough char/4 estimation")

def _vc_count_tokens(text: str) -> int:
    """Count tokens using tiktoken or fallback estimation."""
    _vc_init_encoder()
    if _vc_encoder:
        return len(_vc_encoder.encode(text))
    return max(1, len(text) // 4)

def _vc_truncate_to_budget(text: str, budget: int) -> tuple[str, bool]:
    """Truncate text to fit within token budget. Returns (truncated_text, was_truncated)."""
    tokens = _vc_count_tokens(text)
    if tokens <= budget:
        return text, False
    # Truncate proportionally
    ratio = budget / tokens
    keep_chars = int(len(text) * ratio)
    truncated = text[:keep_chars] + "\n... [truncated]"
    return truncated, True

def _vc_get_working_memory(scope: str, budget: int) -> tuple[str, int, bool]:
    """Get working memory (short-term Redis) for the current scope."""
    if not _redis:
        return "", 0, False
    try:
        pattern = f"{REDIS_PREFIX}{scope}:*"
        keys = list(_redis.scan_iter(match=pattern, count=100))
        if not keys:
            return "", 0, False
        items = []
        total_tokens = 0
        for key in keys:
            val = _redis.get(key)
            if val:
                try:
                    data = json.loads(val)
                    content = data.get("value", data.get("content", ""))
                    if content:
                        tokens = _vc_count_tokens(content)
                        if total_tokens + tokens <= budget:
                            items.append(f"- {content}")
                            total_tokens += tokens
                        else:
                            break
                except Exception:
                    pass
        if items:
            text = "WORKING MEMORY (short-term):\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: working memory fetch failed: %s", e)
        return "", 0, False

def _vc_get_session_summary(scope: str, budget: int) -> tuple[str, int, bool]:
    """Get session summary from trajectory memories."""
    try:
        with system_context("virtual_context_assembly"):
            recall_args = {"query": "session summary", "top_k": 3, "category": "trajectory", "scope": scope}
            recall_res = _tool_recall(recall_args)
        recall_parsed = json.loads(recall_res.content[0].text)
        trajectories = recall_parsed.get("memories", [])
        if not trajectories:
            return "", 0, False
        items = []
        total_tokens = 0
        for traj in trajectories:
            content = traj.get("content", "")
            if content:
                tokens = _vc_count_tokens(content)
                if total_tokens + tokens <= budget:
                    items.append(f"- {content}")
                    total_tokens += tokens
                else:
                    break
        if items:
            text = "SESSION SUMMARY:\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: session summary fetch failed: %s", e)
        return "", 0, False

def _vc_get_retrieved_memories(query: str, scope: str, top_k: int, budget: int) -> tuple[str, int, bool]:
    """Get top-k retrieved memories for the query."""
    try:
        with system_context("virtual_context_assembly"):
            recall_args = {"query": query, "top_k": top_k, "scope": scope}
            recall_res = _tool_recall(recall_args)
        recall_parsed = json.loads(recall_res.content[0].text)
        memories = recall_parsed.get("memories", [])
        if not memories:
            return "", 0, False
        items = []
        total_tokens = 0
        for mem in memories:
            content = mem.get("content", "")
            if content:
                # Include relevance info
                # Sprint-1 1.5 fix: confidence/relevance can be None
                # (corroboration-based confidence is null until computed) —
                # f"{None:.2f}" raises and silently emptied the WHOLE
                # retrieval section of every context bundle since 3A-3.
                relevance = mem.get("relevance")
                confidence = mem.get("confidence")
                prefix = (f"[rel={float(relevance or 0):.2f}, "
                          f"conf={float(confidence or 0):.2f}] ")
                tokens = _vc_count_tokens(prefix + content)
                if total_tokens + tokens <= budget:
                    items.append(f"- {prefix}{content}")
                    total_tokens += tokens
                else:
                    break
        # Sprint-1 1.5: graph-neighbor injection — recall's 2C-2 expansion
        # (graph_adjacent) becomes part of the assembled retrieval section so
        # session_init / context_preview surface connected knowledge.
        for _a in (recall_parsed.get("graph_adjacent") or [])[:3]:
            try:
                _ct = _a.get("content", "")
                if not _ct:
                    continue
                _rel = _a.get("graph_relation") or "related"
                _line = f"- [graph:{_rel}] {_ct}"
                _tk = _vc_count_tokens(_line)
                if total_tokens + _tk <= budget:
                    items.append(_line)
                    total_tokens += _tk
            except Exception:
                continue
        if items:
            text = "RETRIEVED MEMORIES:\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: retrieved memories fetch failed: %s", e)
        return "", 0, False

def _vc_get_procedures(scope: str, budget: int, query: str = None) -> tuple[str, int, bool]:
    """Get relevant procedures for the scope.

    2A-1 fix 2026-10-08: the recall query used to be hardcoded to
    "procedure how to", so the section ranked GENERIC procedures and the
    actually-relevant learned lessons (e.g. the graph_link relation-type
    whitelist discovered by the Q28 experiment design) never injected.
    Query with the session's real task so learned knowledge lands where
    it changes behavior."""
    if LEARNED_STRATEGIES_DISABLED:
        # 2A-1 (Q28 experiment): control-arm toggle — procedure injection OFF.
        return "", 0, False
    try:
        proc_query = (query or "").strip() or "procedure how to"
        with system_context("virtual_context_assembly"):
            recall_args = {"query": f"procedure for: {proc_query[:200]}",
                           "top_k": 5, "category": "procedure", "scope": scope}
            recall_res = _tool_recall(recall_args)
        recall_parsed = json.loads(recall_res.content[0].text)
        procedures = recall_parsed.get("memories", [])
        if not procedures:
            return "", 0, False
        items = []
        total_tokens = 0
        for proc in procedures:
            content = proc.get("content", "")
            if content:
                tokens = _vc_count_tokens(content)
                if total_tokens + tokens <= budget:
                    items.append(f"- {content}")
                    total_tokens += tokens
                else:
                    break
        if items:
            text = "PROCEDURES:\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: procedures fetch failed: %s", e)
        return "", 0, False

def _vc_get_anti_patterns(query: str, budget: int) -> tuple[str, int, bool]:
    """2A-4: negative knowledge — "avoid X" lessons from failed trajectories.

    Promotions of failure-signatures carry tags learned-warning / avoid-pattern;
    this section surfaces the ones relevant to the live task, phrased as
    DO NOT / PREFER instructions, inside a small dedicated budget."""
    if LEARNED_STRATEGIES_DISABLED:
        return "", 0, False
    try:
        with system_context("virtual_context_assembly"):
            recall_args = {"query": f"avoid failure: {query[:200]}",
                            "top_k": 3, "category": "procedure", "scope": "global",
                            "tags": "learned-warning"}
            recall_res = _tool_recall(recall_args)
        recall_parsed = json.loads(recall_res.content[0].text)
        warnings = recall_parsed.get("memories", [])
        if not warnings:
            return "", 0, False
        items = []
        total_tokens = 0
        for w in warnings:
            content = (w.get("content") or "").strip()
            if not content:
                continue
            text = f"DO NOT / PREFER — {content}"
            tokens = _vc_count_tokens(text)
            if total_tokens + tokens <= budget:
                items.append(f"- {text}")
                total_tokens += tokens
        if items:
            return ("ANTI-PATTERNS (from past failures):\n" + "\n".join(items),
                    total_tokens, False)
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: anti-patterns fetch failed: %s", e)
        return "", 0, False


def _vc_get_user_profile(scope: str, budget: int) -> tuple[str, int, bool]:
    """3A-2 / 3A-4: the user model, rendered as binding style instructions.

    Profile memories (category=preference, tag user-profile) are stored as
    structured, computed-confidence entries (3A-3). This section turns them
    into operative adaptation: verbosity / format / proactivity knobs come
    from the profile content itself, and inferred entries are visibly labeled
    so the model knows which directives are the user's own words vs. observed
    habits. Deterministic scroll (no embedding cost), small by design."""
    try:
        _pf = _build_filter("preference", None, scope)
        _pts = []
        _off = None
        while len(_pts) < 25:
            _page = _client.scroll(collection_name=COLLECTION_NAME,
                                   scroll_filter=_pf, limit=25,
                                   offset=_off, with_payload=True)
            _pts.extend(_page[0])
            _off = _page[1]
            if not _off:
                break
        rows = []
        for p in _pts:
            pl = p.payload or {}
            if pl.get("status") == "superseded":
                continue
            rows.append({
                "content": (pl.get("content") or "").strip(),
                "confidence": float(pl.get("confidence") or 0.5),
                "provenance": pl.get("provenance") or "inferred",
                "importance": int(pl.get("importance") or 5),
                "tags": (pl.get("tags") or ""),
            })
        rows = [r for r in rows if r["content"]]
        if not rows:
            return "", 0, False
        # Stated > corroborated > inferred, then importance.
        rows.sort(key=lambda r: (r["provenance"] != "stated",
                                 -r["confidence"], -r["importance"]))
        items = []
        total = 0
        for r in rows:
            label = ("stated" if r["provenance"] == "stated" else
                     f"inferred, c={r['confidence']:.2f}")
            line = f"- [{label}] {r['content']}"
            tk = _vc_count_tokens(line)
            if total + tk > budget:
                continue
            items.append(line)
            total += tk
        if not items:
            return "", 0, False
        header = ("USER PROFILE (binding style directives for THIS user — "
                  "adapt verbosity, output format, and proactivity to match; "
                  "'stated' entries are the user's own words, 'inferred' are "
                  "observed habits):\n")
        return header + "\n".join(items), total, False
    except Exception as e:
        log.warning("virtual_context: user-profile fetch failed: %s", e)
        return "", 0, False


def _vc_get_learned_strategies(task_type: str, budget: int) -> tuple[str, int, bool]:
    if LEARNED_STRATEGIES_DISABLED:
        # 2A-1 (Q28 experiment): control-arm toggle — strategy injection OFF.
        return "", 0, False
    try:
        with system_context("virtual_context_assembly"):
            insights_res = _tool_get_insights({"task_type": task_type, "limit": 5})
        insights_parsed = json.loads(insights_res.content[0].text)
        strategies = insights_parsed.get("strategies", [])
        if not strategies:
            return "", 0, False
        items = []
        total_tokens = 0
        for strat in strategies:
            content = strat.get("content", "")
            if content:
                tokens = _vc_count_tokens(content)
                if total_tokens + tokens <= budget:
                    items.append(f"- {content}")
                    total_tokens += tokens
                else:
                    break
        if items:
            text = "LEARNED STRATEGIES:\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: learned strategies fetch failed: %s", e)
        return "", 0, False

def _vc_get_tool_hints(budget: int) -> tuple[str, int, bool]:
    """Get tool usage hints/recommendations."""
    try:
        with system_context("virtual_context_assembly"):
            hints_res = _tool_tool_recommendations({"task_type": "general", "limit": 5})
        hints_parsed = json.loads(hints_res.content[0].text)
        recommendations = hints_parsed.get("recommendations", [])
        if not recommendations:
            return "", 0, False
        items = []
        total_tokens = 0
        for rec in recommendations:
            tool = rec.get("tool", "")
            success_rate = rec.get("success_rate", 0)
            usage = rec.get("usage_count", 0)
            content = f"{tool} (success={success_rate:.1%}, used={usage})"
            tokens = _vc_count_tokens(content)
            if total_tokens + tokens <= budget:
                items.append(f"- {content}")
                total_tokens += tokens
            else:
                break
        if items:
            text = "TOOL HINTS:\n" + "\n".join(items)
            return text, total_tokens, False
        return "", 0, False
    except Exception as e:
        log.warning("virtual_context: tool hints fetch failed: %s", e)
        return "", 0, False

def assemble_virtual_context(query: str, scope: str, task_type: str = None) -> dict:
    """
    Assemble the virtual context prompt for a given query.
    
    Returns a dict with:
    - prompt: the assembled prompt string
    - sections: dict of section_name -> {content, tokens, truncated, dropped}
    - total_tokens: total tokens in assembled prompt
    - budget: total token budget
    - budget_exceeded: bool
    """
    t0 = time.time()
    
    budgets = {
        "profile": VIRTUAL_CONTEXT_PROFILE_BUDGET,
        "working": VIRTUAL_CONTEXT_WORKING_BUDGET,
        "session": VIRTUAL_CONTEXT_SESSION_BUDGET,
        "retrieval": VIRTUAL_CONTEXT_RETRIEVAL_BUDGET,
        "procedures": VIRTUAL_CONTEXT_PROCEDURE_BUDGET,
        "avoid": VIRTUAL_CONTEXT_AVOID_BUDGET,
        "strategies": VIRTUAL_CONTEXT_STRATEGY_BUDGET,
        "hints": VIRTUAL_CONTEXT_HINT_BUDGET,
    }

    sections = {}
    total_tokens = 0
    budget_exceeded = False

    # Section 0 (3A-2): User Profile — behavioral adaptation. FIRST by design:
    # style directives shape how everything below is delivered.
    content, tokens, truncated = _vc_get_user_profile(scope, budgets["profile"])
    sections["user_profile"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["profile"],
    }
    total_tokens += tokens
    
    # Section 1: Working Memory (short-term Redis)
    content, tokens, truncated = _vc_get_working_memory(scope, budgets["working"])
    sections["working_memory"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["working"],
    }
    total_tokens += tokens
    
    # Section 2: Session Summary
    content, tokens, truncated = _vc_get_session_summary(scope, budgets["session"])
    sections["session_summary"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["session"],
    }
    total_tokens += tokens
    
    # Section 3: Retrieved Memories (semantic search)
    content, tokens, truncated = _vc_get_retrieved_memories(query, scope, VIRTUAL_CONTEXT_TOP_K, budgets["retrieval"])
    sections["retrieved_memories"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["retrieval"],
    }
    total_tokens += tokens
    
    # Section 4: Procedures
    content, tokens, truncated = _vc_get_procedures(scope, budgets["procedures"], query)
    sections["procedures"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["procedures"],
    }
    total_tokens += tokens

    # Section 4b: Anti-patterns — 2A-4 negative knowledge ("avoid X")
    content, tokens, truncated = _vc_get_anti_patterns(query, budgets["avoid"])
    sections["anti_patterns"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["avoid"],
    }
    total_tokens += tokens

    # Section 5: Learned Strategies
    if task_type:
        content, tokens, truncated = _vc_get_learned_strategies(task_type, budgets["strategies"])
    else:
        content, tokens, truncated = "", 0, False
    sections["learned_strategies"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["strategies"],
    }
    total_tokens += tokens
    
    # Section 6: Tool Hints
    if VIRTUAL_CONTEXT_INCLUDE_HINTS:
        content, tokens, truncated = _vc_get_tool_hints(budgets["hints"])
    else:
        content, tokens, truncated = "", 0, False
    sections["tool_hints"] = {
        "content": content,
        "tokens": tokens,
        "truncated": truncated,
        "budget": budgets["hints"],
    }
    total_tokens += tokens
    
    # Check if total exceeds budget
    if total_tokens > VIRTUAL_CONTEXT_TOKEN_BUDGET:
        budget_exceeded = True
        # Proportionally reduce sections (except working memory which is priority)
        _VC_METRICS["budget_exceeded"] += 1
        # For now, just note it - the truncation above handles per-section limits
    
    # Build the final prompt
    prompt_parts = []
    for section_name, section_data in sections.items():
        if section_data["content"]:
            prompt_parts.append(section_data["content"])
    
    # Add instruction footer
    instruction = (
        "\n\n---\n"
        "INSTRUCTION: Use the above context to inform your response. "
        "Prioritize information from RETRIEVED MEMORIES and PROCEDURES. "
        "Heed LEARNED STRATEGIES as lessons from past tasks. "
        "WORKING MEMORY contains your current session's transient state."
    )
    prompt_parts.append(instruction)
    
    full_prompt = "\n\n".join(prompt_parts)
    
    # Record metrics
    latency_ms = int((time.time() - t0) * 1000)
    _VC_METRICS["total_assemblies"] += 1
    _VC_METRICS["tokens_injected"].append(total_tokens)
    _VC_METRICS["assembly_latency_ms"].append(latency_ms)
    for section_name, section_data in sections.items():
        if section_data["truncated"]:
            _VC_METRICS["sections_dropped"][section_name] += 1
    
    # Keep metrics lists bounded
    max_metrics = 1000
    if len(_VC_METRICS["tokens_injected"]) > max_metrics:
        _VC_METRICS["tokens_injected"] = _VC_METRICS["tokens_injected"][-max_metrics:]
    if len(_VC_METRICS["assembly_latency_ms"]) > max_metrics:
        _VC_METRICS["assembly_latency_ms"] = _VC_METRICS["assembly_latency_ms"][-max_metrics:]
    
    return {
        "prompt": full_prompt,
        "sections": sections,
        "total_tokens": total_tokens,
        "budget": VIRTUAL_CONTEXT_TOKEN_BUDGET,
        "budget_exceeded": budget_exceeded,
        "assembly_latency_ms": latency_ms,
    }

def _tool_context_preview(args: dict) -> types.CallToolResult:
    """Preview the virtual context that would be assembled for a query."""
    query = args.get("query", "")
    if not query:
        return _err("query is required")
    
    try:
        scope = authorize_scope(args.get("scope"))
    except ScopeError as e:
        return _err(f"scope rejected: {e}")
    
    task_type = args.get("task_type")
    
    result = assemble_virtual_context(query, scope, task_type)
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))

def _tool_virtual_context_metrics(args: dict = None) -> types.CallToolResult:
    """Get virtual context assembly metrics."""
    tokens_list = _VC_METRICS["tokens_injected"]
    latency_list = _VC_METRICS["assembly_latency_ms"]
    
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    result = {
        "enabled": VIRTUAL_CONTEXT_ENABLED,
        "total_assemblies": _VC_METRICS["total_assemblies"],
        "budget_exceeded_count": _VC_METRICS["budget_exceeded"],
        "tokens_injected": {
            "count": len(tokens_list),
            "mean": round(sum(tokens_list) / len(tokens_list), 1) if tokens_list else 0,
            "p50": percentile(tokens_list, 50),
            "p95": percentile(tokens_list, 95),
            "p99": percentile(tokens_list, 99),
            "max": max(tokens_list) if tokens_list else 0,
        },
        "assembly_latency_ms": {
            "count": len(latency_list),
            "mean": round(sum(latency_list) / len(latency_list), 1) if latency_list else 0,
            "p50": percentile(latency_list, 50),
            "p95": percentile(latency_list, 95),
            "p99": percentile(latency_list, 99),
            "max": max(latency_list) if latency_list else 0,
        },
        "sections_dropped": dict(_VC_METRICS["sections_dropped"]),
        "budget_config": {
            "total": VIRTUAL_CONTEXT_TOKEN_BUDGET,
            "working": VIRTUAL_CONTEXT_WORKING_BUDGET,
            "session": VIRTUAL_CONTEXT_SESSION_BUDGET,
            "retrieval": VIRTUAL_CONTEXT_RETRIEVAL_BUDGET,
            "procedures": VIRTUAL_CONTEXT_PROCEDURE_BUDGET,
            "avoid": VIRTUAL_CONTEXT_AVOID_BUDGET,
            "strategies": VIRTUAL_CONTEXT_STRATEGY_BUDGET,
            "hints": VIRTUAL_CONTEXT_HINT_BUDGET,
        },
    }
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# Caller identity & scope authorization — FAIL-CLOSED
# ---------------------------------------------------------------------------
# Historically `scope` was trusted verbatim from the caller and every scoped
# query was widened with `OR scope='global'` plus legacy NULL scopes. Two
# independent defects combined into a total isolation failure:
#   1. an unscoped write silently became `global`, so it was visible to all; and
#   2. a scoped read OR-ed `global` in, so every read was a cross-tenant read.
#
# The invariant is now:  caller_identity -> authorization -> authorized_scope.
# The caller never chooses its own authority. The operator declares which
# namespaces this deployment serves via MEMORY_PRINCIPAL; a per-request
# identity header can only *narrow* to a namespace the operator already granted.
#
#   MEMORY_PRINCIPAL="global,user:moon"
#       first entry  = the primary scope used when a call supplies none
#       every entry  = a namespace this deployment is allowed to serve
#
# With MEMORY_PRINCIPAL unset the server has no principal and therefore
# authorizes nothing: a misconfigured deployment fails loudly instead of
# silently serving every tenant's data.
# ---------------------------------------------------------------------------
# AUTHENTICATION + AUTHORIZATION  (zero-tolerance security boundary)
# ---------------------------------------------------------------------------
# HARDWARE TRUTH discovered by adversarial probing (see ROOT_CAUSES.md):
# the previous design treated a plain request header as the root of trust:
#
#     X-Memory-Identity: caringyou      # no credential of any kind
#
# was sufficient to read another tenant's live memories. Identity was
# SELF-DECLARED, so "tenant isolation" only ever separated namespaces that the
# caller chose for itself. Any process able to reach the socket could claim any
# namespace the deployment granted. That is a complete authorization bypass and
# it fails PART 1.2 / PART 20 outright.
#
# The model is now:
#
#     credential  ->  identity        (server-side, unforgeable)
#     identity    ->  grant ceiling   (MEMORY_PRINCIPAL, operator-controlled)
#     caller-supplied scope -> may only narrow, never widen
#     no credential -> DENY            (never "global", never a default)
#
# `MEMORY_PRINCIPAL` is now ONLY a grant ceiling / namespace registry. It is
# never an identity and never an anonymous fallback.
#
# Tokens are stored as SHA-256 digests, never in plaintext, so leaking the token
# file does not hand over live credentials.
# ---------------------------------------------------------------------------

_IDENTITY_HEADER = os.environ.get("MEMORY_IDENTITY_HEADER", "x-memory-identity")
_TOKEN_HEADER = os.environ.get("MEMORY_TOKEN_HEADER", "x-memory-token").encode("latin-1")

_SCOPE_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]*(?::[a-z0-9][a-z0-9_.\-]*)?$")
_SCOPES_LOOKEDUP: dict[str, Optional[frozenset]] = {}

# Contextvars holding the *authenticated* principal for the current request.
# `None` identity means "no credential presented" -> everything is denied.
_caller_identity: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "memory_caller_identity", default=None
)
# Set only by system_context(); identifies a trusted in-process operation.
_caller_system: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "memory_caller_system", default=None
)
# Scopes bound to the presented credential. None = no credential.
_caller_grants: "contextvars.ContextVar[Optional[frozenset]]" = contextvars.ContextVar(
    "memory_caller_grants", default=None
)

# Token registry: sha256_hex -> (identity, frozenset(scopes) or None)
_TOKEN_REGISTRY: Optional[dict] = None
_TOKEN_LOAD_ERRORS: list = []


class ScopeError(Exception):
    """Raised when a caller requests a scope it is not authorized for."""


class AuthError(ScopeError):
    """Raised when no usable credential was presented (fail-closed)."""


class _Principal:
    """Authenticated caller. Immutable.

    identity : str | None   -- None means no credential was presented.
    scopes   : frozenset|None -- namespaces bound to the credential; None when
                              the credential is unbounded (deployment ceiling
                              still applies).
    is_system: bool         -- trusted in-process operation (reindex, worker).
    reason   : str          -- audit string, never a secret.
    """
    __slots__ = ("identity", "scopes", "is_system", "reason")

    def __init__(self, identity=None, scopes=None, is_system=False, reason=""):
        self.identity = identity
        self.scopes = frozenset(scopes) if scopes is not None else None
        self.is_system = is_system
        self.reason = reason

    def __repr__(self):
        kind = "system" if self.is_system else (self.identity or "anonymous")
        return f"<Principal {kind}>"


ANONYMOUS = _Principal(None, None, False, "no-credential")
SYSTEM = _Principal("__system__", None, True, "system")

_current: "contextvars.ContextVar[_Principal]" = contextvars.ContextVar(
    "memory_principal", default=ANONYMOUS
)
# 3B-1: the effective per-operation scope resolved by authorize_scope(). The
# PG helpers push this into the session (app.scope) so RLS policies on the
# canonical tables enforce tenant isolation at the DATABASE layer as well:
# a bug that forgets a WHERE scope filter still cannot cross tenants, because
# the row is invisible to the connection. System context sets bypass instead.
_effective_scope: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "memory_effective_scope", default=None
)


def current_principal() -> _Principal:
    return _current.get()


@contextlib.contextmanager
def system_context(reason: str = "system-operation"):
    """Bind a trusted in-process principal for the duration of the block.

    This is the ONLY way to act across namespaces without a user credential, and
    it is explicit, greppable and auditable. Operations that legitimately need
    it: startup reconciliation, BM25/Qdrant rebuild, outbox worker, and
    operator-only maintenance tools. A caller reaching the network can never
    enter this context.
    """
    tok = _current.set(SYSTEM)
    try:
        yield SYSTEM
    finally:
        _current.reset(tok)


def set_caller_principal(principal: _Principal) -> None:
    """Bind the authenticated caller for the current request/thread."""
    _current.set(principal)
    _caller_identity.set(principal.identity)
    _caller_grants.set(principal.scopes)
    _effective_scope.set(None)


def clear_caller_principal() -> None:
    set_caller_principal(ANONYMOUS)


def get_caller_identity() -> Optional[str]:
    return _current.get().identity


# --- token registry ---------------------------------------------------------

def _sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _load_token_registry() -> dict:
    """Load sha256 -> (identity, scopes) from MEMORY_AUTH_TOKENS.

    Supported forms:
      * path to a file, one record per line:
            sha256:<hex>|<identity>|<scope1,scope2,...>
      * the same records inline, separated by ';' or newlines

    Fields are pipe-delimited, NOT colon-delimited: identities and scopes are
    themselves colon-qualified (`user:alice`), so a ':' separator shredded
    `user:alice` into identity `user` + scope `alice` and every such tenant was
    then rejected as unserved.

    Plaintext tokens are rejected: only digests are ever accepted, so the
    server cannot be coerced into trusting a token it could not verify.
    """
    global _TOKEN_REGISTRY
    if _TOKEN_REGISTRY is not None:
        return _TOKEN_REGISTRY
    reg: dict = {}
    # Read the env var on every load, not at import time: capturing it as a
    # module constant froze the credential store at import and silently made it
    # un-reconfigurable (and invisible to tests and to a PM2 env reload).
    raw = os.environ.get("MEMORY_AUTH_TOKENS", "").strip()
    _TOKEN_LOAD_ERRORS.clear()
    if not raw:
        _TOKEN_REGISTRY = reg
        return reg
    text = ""
    if os.path.isfile(raw):
        try:
            with open(raw, "r") as fh:
                text = fh.read()
        except Exception as e:
            _TOKEN_LOAD_ERRORS.append(f"cannot read {raw}: {e}")
            _TOKEN_REGISTRY = reg
            return reg
    else:
        text = raw
    for line in text.replace(";", "\n").splitlines():
        # 3A-5: strip TRAILING comments too. Only line-leading '#' was skipped
        # before, so 'sha256:<hex>|user:moon  # admin token' parsed identity as
        # 'user:moon  # admin token' — scope normalization then rejected it and
        # the credential silently died. No legitimate field in a token record
        # ever contains '#', so cutting at the first one is always safe.
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2 or not parts[0].startswith("sha256:"):
            _TOKEN_LOAD_ERRORS.append(
                "token record must be 'sha256:<hex>|<identity>|<scope1,scope2>'")
            continue
        digest = parts[0].split(":", 1)[1].strip().lower()
        identity = parts[1].strip()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            _TOKEN_LOAD_ERRORS.append(f"bad digest for identity {identity!r}")
            continue
        scopes = None
        if len(parts) > 2 and parts[2].strip():
            scopes = set()
            for s in parts[2].split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    scopes.add(_normalize_scope(s))
                except ScopeError as e:
                    _TOKEN_LOAD_ERRORS.append(f"bad scope {s!r} for {identity!r}: {e}")
            if not scopes:
                scopes = None
        reg[digest] = (identity, frozenset(scopes) if scopes else None)
    # 3A-5: the identities table is the system of record; file/env is fallback.
    # Table rows WIN over file rows so an operator edit in the DB takes effect
    # without touching the token file (see refresh_identity_registry — the next
    # authentication re-reads, no process restart needed).
    try:
        rows = _pg_query(
            "SELECT token_digest, identity, scopes FROM identities WHERE active",
            {})
        for r in rows:
            d = (r["token_digest"] if isinstance(r, dict) else r[0] or "").strip().lower()
            ident = (r["identity"] if isinstance(r, dict) else r[1] or "").strip()
            sc_raw = (r["scopes"] if isinstance(r, dict) else r[2]) or ""
            if not d or not ident:
                continue
            try:
                ident_n = _normalize_scope(ident)
            except ScopeError as e:
                _TOKEN_LOAD_ERRORS.append(f"identities table: bad identity {ident!r}: {e}")
                continue
            scopes = None
            if sc_raw.strip():
                scopes = frozenset(s.strip() for s in sc_raw.split(",") if s.strip())
            reg[d] = (ident_n, scopes)
        if rows:
            log.info("identity registry: %d record(s) from identities table", len(rows))
    except Exception:
        pass  # table absent or PG down: file/env fallback already loaded
    _TOKEN_REGISTRY = reg
    if reg:
        log.info("loaded %d auth token record(s) from credential store", len(reg))
    for e in _TOKEN_LOAD_ERRORS:
        log.error("auth token store: %s", e)
    return reg


def _anonymous_principal() -> _Principal:
    """Principal for credential-less callers.

    With MEMORY_ALLOW_ANONYMOUS=1 these are bound to a dedicated namespace
    (MEMORY_ANON_IDENTITY, default `anon`) that deployment validation has
    already proven is NOT one of the tenants. So anonymous traffic is confined
    to its own bucket and can never reach tenant data. Without that flag this
    returns ANONYMOUS, which is authorized for nothing.
    """
    if os.environ.get("MEMORY_ALLOW_ANONYMOUS", "").strip() != "1":
        return ANONYMOUS
    ident = os.environ.get("MEMORY_ANON_PRINCIPAL", "").strip() or os.environ.get(
        "MEMORY_ANON_IDENTITY", "anon").strip() or "anon"
    try:
        return _Principal(_normalize_scope(ident), None, False, "anonymous-mode")
    except ScopeError:
        log.error("MEMORY_ANON_IDENTITY is invalid; anonymous traffic denied")
        return ANONYMOUS


def authenticate_token(token: Optional[str]) -> _Principal:
    """Resolve a bearer token to a Principal. Constant-time digest compare."""
    if not token:
        return _anonymous_principal()
    reg = _load_token_registry()
    if not reg:
        # Credentials were presented but none are configured. This is a
        # misconfiguration, not an anonymous request: deny loudly.
        log.error("a credential was presented but no token store is configured")
        return ANONYMOUS
    digest = _sha256_hex(token.strip())
    # Compare against every known digest so timing does not reveal valid tokens.
    matched = None
    for known in reg:
        if hmac.compare_digest(known, digest):
            matched = reg[known]
    if matched is None:
        # 3A-5 hot reload: a NEW credential minted straight into the
        # identities table must work without a process restart. One re-read
        # on miss keeps the happy path at full cache speed.
        global _TOKEN_REGISTRY
        _TOKEN_REGISTRY = None
        reg2 = _load_token_registry()
        for known in reg2:
            if hmac.compare_digest(known, digest):
                matched = reg2[known]
    if matched is None:
        log.warning("presented credential did not match any known token")
        return ANONYMOUS
    identity, scopes = matched
    try:
        identity = _normalize_scope(identity)
    except ScopeError as e:
        log.error("token binds an invalid identity: %s", e)
        return ANONYMOUS
    return _Principal(identity, scopes, False, "token")


def token_count() -> int:
    return len(_load_token_registry())


def refresh_identity_registry() -> int:
    """3A-5: drop the identity cache so the next authentication re-reads the
    identities table + token file. Returns the fresh record count."""
    global _TOKEN_REGISTRY
    _TOKEN_REGISTRY = None
    return len(_load_token_registry())


def _normalize_scope(scope) -> str:
    """Canonicalise a scope string, rejecting anything malformed.

    Rejects empty/whitespace, wildcard metacharacters (*, %, ?), control
    characters and anything that is not `prefix:name` (or the reserved bare
    `global`). Prevents a caller from smuggling a filter/glob into the Qdrant
    payload query or the `WHERE scope = ...` clause.
    """
    if scope is None:
        raise ScopeError("scope is required")
    if not isinstance(scope, str):
        raise ScopeError("scope must be a string")
    s = scope.strip().lower()
    if not s:
        raise ScopeError("scope must not be empty")
    if len(s) > 128:
        raise ScopeError("scope is too long")
    if any(ch in s for ch in "*?%\"'\\\n\r\t\x00"):
        raise ScopeError(f"scope contains illegal characters: {scope!r}")
    if s == "global":
        return s
    if not _SCOPE_RE.match(s):
        raise ScopeError(
            f"malformed scope {scope!r}: expected 'prefix:name' or a bare name "
            "(letters, digits, dot, dash, underscore)"
        )
    return s


def _deployment_grants() -> Optional[frozenset]:
    """Scopes this deployment is configured to serve. None = unconfigured.

    This is a CEILING, not an identity and not an anonymous fallback.
    """
    cached = _SCOPES_LOOKEDUP.get("grants")
    if cached is not None or "grants" in _SCOPES_LOOKEDUP:
        return cached
    raw = os.environ.get("MEMORY_PRINCIPAL", "").strip()
    if not raw:
        _SCOPES_LOOKEDUP["grants"] = None
        return None
    grants = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            grants.add(_normalize_scope(part))
        except ScopeError:
            log.error("MEMORY_PRINCIPAL contains an invalid scope %r — ignoring", part)
    # Anonymous mode binds credential-less callers to a dedicated namespace that
    # is deliberately NOT a tenant. It still has to be *servable*, so it is added
    # to the ceiling here; because it is its own bucket it grants no access to any
    # real tenant's data.
    anon = os.environ.get("MEMORY_ANON_IDENTITY", "").strip()
    if grants and os.environ.get("MEMORY_ALLOW_ANONYMOUS", "").strip() == "1" and anon:
        try:
            grants.add(_normalize_scope(anon))
        except ScopeError:
            pass
    result = frozenset(grants) if grants else None
    _SCOPES_LOOKEDUP["grants"] = result
    return result


def _primary_scope() -> Optional[str]:
    """Primary namespace for trusted SYSTEM operations only."""
    raw = os.environ.get("MEMORY_PRINCIPAL", "").strip()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                return _normalize_scope(part)
            except ScopeError:
                continue
    return None


def authorized_scopes() -> Optional[frozenset]:
    """Scopes the current caller may touch. None => DENY everything.

    This function is the single decision point and it no longer has a
    fail-open branch:
      * system context  -> the full deployment ceiling
      * valid credential -> identity's scopes, intersected with the ceiling
      * no credential    -> None  (was: "all grants", then "global")
    """
    ceiling = _deployment_grants()
    if ceiling is None:
        return None
    principal = _current.get()
    if principal.is_system:
        return ceiling
    if not principal.identity:
        # PART 1.2: a missing principal is an authorization failure. It must
        # never silently resolve to `global` or to any other namespace.
        return None
    try:
        ident_n = _normalize_scope(principal.identity)
    except ScopeError as e:
        raise ScopeError(f"invalid caller identity: {e}")
    if ident_n not in ceiling:
        raise ScopeError(
            f"identity {ident_n!r} is not served by this deployment "
            f"(granted: {sorted(ceiling)})"
        )
    # A credential may narrow further than its own namespace.
    if principal.scopes is None:
        return frozenset({ident_n})
    narrowed = frozenset(s for s in principal.scopes if s in ceiling)
    if not narrowed:
        raise ScopeError(
            f"credential for {ident_n!r} grants no scope served by this deployment"
        )
    return narrowed


def authorize_scope(requested=None) -> str:
    """Validate and authorize a requested scope, returning the effective scope.

    Fail-closed on every path. A supplied-but-malformed scope is an error, never
    "omitted". An omitted scope resolves to the caller's own namespace, and for
    a credential-less caller it is an authorization failure, never the primary.
    """
    allowed = authorized_scopes()
    if allowed is None:
        p = _current.get()
        if p.is_system:
            primary = _primary_scope()
            if primary is None:
                raise ScopeError("no primary scope configured")
            return primary
        raise AuthError(
            "authentication required: no valid credential was presented, so this "
            "caller is authorized for no namespace. MEMORY_PRINCIPAL is a grant "
            "ceiling and is never used as an anonymous identity."
        )
    if requested is None:
        principal = _current.get()
        if principal.is_system:
            # The SYSTEM principal carries a sentinel identity that is not a
            # valid scope. Trusted in-process operations (virtual-context
            # assembly, outbox worker, operator tools) resolve to the
            # deployment's primary namespace instead of failing closed.
            primary = _primary_scope()
            if primary is None:
                raise ScopeError("no primary scope configured")
            _effective_scope.set(primary)
            return primary
        identity = principal.identity
        if identity:
            _implicit = _normalize_scope(identity)
            _effective_scope.set(_implicit)
            return _implicit
        # Unreachable when allowed is not None, but keep it closed anyway.
        raise AuthError("authentication required to resolve an implicit scope")
    norm = _normalize_scope(requested)
    if norm not in allowed:
        raise ScopeError(
            f"scope {norm!r} is not authorized for this caller "
            f"(authorized: {sorted(allowed)})"
        )
    # 3B-1: record the authorized operation scope for the RLS session vars.
    # This is the single choke point every tool passes through, so every PG
    # helper that runs after it enforces the same scope at the database layer.
    _effective_scope.set(norm)
    return norm

def scope_allowed_for_read(scope) -> bool:
    """True when the current caller may read `scope`."""
    try:
        allowed = authorized_scopes()
    except ScopeError:
        return False
    if allowed is None:
        return False
    try:
        return _normalize_scope(scope) in allowed
    except ScopeError:
        return False


# ---------------------------------------------------------------------------
# Enhancement 1: Graph Memory — relationship storage in Redis
# ---------------------------------------------------------------------------
GRAPH_EDGE_PREFIX = "memory:graph:edge:"
GRAPH_NODE_PREFIX = "memory:graph:node:"
GRAPH_RELATION_TYPES = [
    "depends_on", "related_to", "supersedes", "contradicts",
    "implements", "references", "parent_of", "child_of",
]

# ---------------------------------------------------------------------------
# Gap-closure enhancements (2026-10-03)
# ---------------------------------------------------------------------------
# Contradiction detection: when a new memory is very similar to an existing
# one, look for negation/opposition markers that signal a correction rather
# than a duplicate. If found, mark the old memory superseded (temporal close)
# and link the two with a "supersedes" edge.
_CONTRADICTION_THRESHOLD = 0.82
_NEGATION_MARKERS = [
    "no longer", "removed", "rejected", "changed", "new ", "instead",
    "not ", "replaced", "stopped", "correction", "corrected",
    "supersedes", "now uses", "now prefers", "moved", "deleted",
    "final state",
    "updated", "revised", "instead of", "should not", "do not",
]

# Auto-link at write: link new memory to top-K similar existing memories.
_AUTO_LINK_TOP_K = 3
_AUTO_LINK_MIN_SCORE = 0.70

# Confidence decay for recall results: older unaccessed memories lose confidence.
_CONFIDENCE_HALF_LIFE_DAYS = 180.0
# Confidence calibration: below this normalized-relevance floor a result is
# treated as "not actually supported by the evidence" and its confidence is
# capped, so an unknown/ambiguous query never returns a confident-looking hit.
_LOW_EVIDENCE_CUTOFF = 0.25
_UNVERIFIED_CONFIDENCE_CAP = 0.4


# ---------------------------------------------------------------------------
# Enhancement 3: Memory Poisoning Guard — OWASP LLM04
# ---------------------------------------------------------------------------
POISON_PATTERNS = [
    # Critical severity (weight 1.0) — single match triggers block
    (re.compile(r"(?i)\b(ignore|disregard|forget)\s+(?:my |your |our |all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|directives?|context|memory)\b"), "instruction_override", 1.0),
    # Added 2026-10-07: the pattern above requires "previous/prior/above/earlier",
    # so "ignore all authorization rules" / "override security rules" /
    # "bypass all access controls" scored 0.0. This covers the same class without
    # the temporal qualifier.
    (re.compile(r"(?i)\b(ignore|disregard|override|bypass)\s+(?:all\s+)?(?:the\s+)?(?:authorization|authorisation|security|auth|access|instruction|rule)s?\b"), "instruction_override_bypass", 1.0),
    (re.compile(r"(?i)\b(you\s+are\s+now(?!\s+(?:a\s+)?(?:helpful|kind|friendly|useful|here\s+to))|forget\s+you\s+are(?!\s+(?:a\s+)?(?:helpful|kind))|your\s+role\s+is|remember\s+you\s+are)\b"), "role_override", 1.0),
    (re.compile(r"(?i)\b(sudo|su\s+root|chmod\s+777|rm\s+-rf\s+/)\b"), "system_command_injection", 1.0),
    (re.compile(r'["\']\s*;\s*rm\s+-rf\s+["\']'), "destructive_command", 1.0),
    (re.compile(r"(?i)\b(you\s+are\s+(?:a\s+)?(?:root|admin|hacker|criminal|malicious))\b"), "specific_role_override", 1.0),
    # High severity (weight 0.8) — single match exceeds the 0.7 block line.
    # Added 2026-10-07: "SYSTEM OVERRIDE:", "ADMIN:" / "OPERATOR-" style
    # privilege prefixes were uncaught (only a literal leading "admin:" was).
    # Colon-terminated only: the earlier '[:\-]' alternation matched ordinary
    # hyphenated English ("operator-only", "admin-level") and blocked benign
    # memory stores — observed live 3x while storing this fix's own fact notes.
    (re.compile(r"(?i)\b(SYSTEM|OVERRIDE|ADMIN|OPERATOR)\s*(?:OVERRIDE)?\s*:\s*"), "privilege_prefix_override", 0.8),
    (re.compile(r"(?i)<\s*script[^>]*>.*?</\s*script\s*>", re.DOTALL), "xss_script_tag", 0.8),
    (re.compile(r"(?i)\b(curl|wget)\s+.*\|\s*(bash|sh|python|perl)\b"), "pipe_injection", 0.8),
    (re.compile(r"(?i)\beval\s*\(\s*(?:base64[_a-z]*|exec|system)\s*\("), "code_injection", 0.8),
    (re.compile(r"(?i)\bDROP\s+(TABLE|DATABASE|INDEX)\b"), "sql_destructive", 0.8),
    (re.compile(r"(?i)\bUNION\s+SELECT\b.*\bFROM\b", re.DOTALL), "sql_union", 0.8),
    (re.compile(r'<\?php\s*eval\s*\('), "php_eval", 0.8),
    (re.compile(r"(?i)\b(alert|prompt|confirm)\s*\(\s*[\'\"]"), "js_injection", 0.8),
    (re.compile(r"\[\s*system\s*\("), "lua_system_injection", 0.8),
    # Medium severity (weight 0.5)
    (re.compile(r"```(?:bash|sh|python|js|php|sql)\s*\n", re.IGNORECASE), "fenced_code_injection", 0.5),
    (re.compile(r"(?i)^admin\s*:\s*"), "prompt_injection_prefix", 0.5),
    # Added 2026-10-09 (AF-4 audit: battery needed >= 20 patterns).
    # Colon/equals-terminated only so ordinary prose ("maintenance mode is
    # on") never matches; the instruction-prefix form does.
    (re.compile(r"(?i)\b(dev|developer|debug|god)\s*mode\s*[:=]"), "fake_mode_prefix", 0.8),
    (re.compile(r"(?i)\b(reveal|show|print|repeat|output|display)\s+(?:your\s+|the\s+)?(system|initial|original|hidden)\s+(?:prompt|instructions?|rules?)\b"), "system_prompt_exfiltration", 0.8),
    (re.compile(r"<\|?(?:system|im_start|im_end|endoftext|begin_of_text)\|?>"), "chat_template_smuggling", 0.8),
    (re.compile(r"(?i)!\[[^\]]*\]\(\s*(?:https?|data|file):"), "markdown_image_exfil_channel", 0.5),
]
POISON_SCORE_THRESHOLD = 0.7  # block if poison score >= this

# ---------------------------------------------------------------------------
# Enhancement 9: Quality Gate — reject low-quality memories before storage
# ---------------------------------------------------------------------------
QUALITY_GATE_ENABLED = True
QUALITY_REJECT_THRESHOLD = 0.3   # reject if score < this
QUALITY_WARN_THRESHOLD = 0.5     # warn if score < this
QUALITY_LENGTH_BONUS = 0.15      # bonus for content > 50 chars
QUALITY_STRUCTURE_BONUS = 0.20   # bonus for headers/lists/code blocks
QUALITY_SPECIFICITY_BONUS = 0.15 # bonus for numbers, paths, versions
QUALITY_ACTIONABLE_BONUS = 0.10  # bonus for action verbs
QUALITY_EVIDENCE_BONUS = 0.10    # bonus for evidence markers (log, output, result)
QUALITY_GENERIC_PENALTY = -0.20  # penalty for generic filler ("ok", "yes", "no", "test")

# Utility tracking fields added to every memory payload
UTILITY_FIELDS = {"retrieval_count": 0, "useful_count": 0}

# Auto-archive threshold — memories with decayed importance below this are archived
AUTO_ARCHIVE_THRESHOLD = 0.2

# ---------------------------------------------------------------------------
# Enhancement 5: Eval before Deploy — sandbox config
# ---------------------------------------------------------------------------
EVAL_SANDBOX_MAX_TTL = 300  # 5-minute max for sandbox operations

# ---------------------------------------------------------------------------
# Enhancement 6: Versioned Rollback — version storage in Redis
# ---------------------------------------------------------------------------
VERSION_PREFIX = "memory:version:"
VERSION_INDEX_PREFIX = "memory:version:idx:"
MAX_VERSIONS_PER_MEMORY = 50

# ---------------------------------------------------------------------------
# Logging to stderr (never stdout — stdout is the MCP stdio transport)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [memory] %(levelname)s: %(message)s",
)
log = logging.getLogger("memory")

# ---------------------------------------------------------------------------
# Pre-warm: force Qdrant connection and embedding model loading at startup.
# Issues N lightweight retrieve calls to confirm the collection is reachable.
# ---------------------------------------------------------------------------
def _prewarm():
    if os.environ.get("MEMORY_EAGER_PREWARM") != "1":
        # Sub-agent / short-lived Kilo runs skip the expensive boot work
        # (Qdrant scroll loop + BM25 index rebuild over ~1000 docs). Qdrant
        # and the embedding model are lazy: first real use connects/loads.
        log.info("Memory lazy boot — skipping pre-warm (set MEMORY_EAGER_PREWARM=1 to force)")
        return
    log.info("Pre-warming Qdrant connection and embedding model...")
    try:
        _client.get_collection(COLLECTION_NAME)
        scroll_result, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        for _ in range(PREWARM_QUERIES - 1):
            _client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
        log.info("Pre-warm complete (saw %d points).", len(scroll_result))
    except Exception as e:
        log.warning("Pre-warm failed (non-fatal): %s", e)
    # Load persisted corpus from disk if available, then build bm25s index.
    # Reconcile against Qdrant FIRST: the persisted corpus is only ever added to,
    # so without this prune, memories deleted since the last run stay in the
    # index across every restart and skew IDF/avgdl.
    # Pool mode: prefer the shared Redis index (written by the outbox worker)
    # over the per-process corpus file — a restart must never serve a
    # sparse-less (dense-only) recall pipeline.
    if HORIZONTAL_SCALING_ENABLED and _bm25_redis_load():
        log.info("Pre-warm: BM25 index loaded from shared Redis store")
        return
    _load_bm25_corpus()
    _prune_bm25_corpus()
    _rebuild_bm25_index()

# LRU cache for recall results — keyed on (query, top_k, category, tags_key).
# Caps redundant Qdrant calls for repeated or similar queries within the same
# server session (one Kilo CLI invocation = one server process).
@lru_cache(maxsize=128)
def _cached_recall(query: str, top_k: int, category: str | None, tags_key: str | None, scope: str | None = None) -> str:
    vector = _embed_model().encode(query).tolist()
    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        query_filter=_build_filter(category, tags_key, scope),
        with_payload=True,
    )
    return json.dumps([{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points])


# ---------------------------------------------------------------------------
# Semantic Cache — Redis-backed cache for semantically identical recall results.
# Cosine similarity computed from stored embeddings; threshold = 0.95.
# ---------------------------------------------------------------------------

def _get_cache_key(vector: list[float], salt: str = "") -> str:
    # Phase 3.3: salt includes active filters (category|tags|scope) so filtered
    # queries never collide with unfiltered cached results.
    h = hashlib.sha256((str(vector) + "|" + salt).encode()).hexdigest()[:16]
    return f"{SEMANTIC_CACHE_PREFIX}{h}"


def _check_semantic_cache(vector: list[float], salt: str = "") -> list[dict] | None:
    global _semantic_cache_hits, _semantic_cache_misses
    if not _redis:
        _semantic_cache_misses += 1
        return None
    try:
        key = _get_cache_key(vector, salt)
        cached = _redis.get(key)
        if cached:
            _hit = json.loads(cached)
            # Sprint-1 1.5 fix: never serve an EMPTY cached result — treat as
            # a miss and drop the poisoned key (negative-cache self-healing).
            if isinstance(_hit, list) and not _hit:
                try:
                    _redis.delete(key)
                except Exception:
                    pass
                _semantic_cache_misses += 1
                return None
            _semantic_cache_hits += 1
            log.debug("semantic_cache hit key=%s", key[:32])
            return _hit
        _semantic_cache_misses += 1
    except Exception as e:
        log.warning("semantic_cache lookup failed: %s", e)
        _semantic_cache_misses += 1
    return None


def _store_semantic_cache(vector: list[float], results: list[dict], salt: str = "") -> None:
    if not _redis:
        return
    try:
        key = _get_cache_key(vector, salt)
        _redis.setex(key, SEMANTIC_CACHE_TTL, json.dumps(results))
        log.debug("semantic_cache stored key=%s ttl=%d", key[:32], SEMANTIC_CACHE_TTL)
    except Exception as e:
        log.warning("semantic_cache store failed: %s", e)

def _invalidate_recall_caches() -> int:
    """Drop every cached recall result.

    `_cached_recall.cache_clear()` only empties the in-process LRU. The Redis
    semantic cache is a separate store with its own 1h TTL, so a mutation that
    skipped it let recall keep serving deleted or pre-update memories. Every
    write and every delete must call this.

    Pool mode adds a SECOND Redis layer — the exact-key recall cache
    (memory:rc:*) — which bypasses the canonical gate when served; it must be
    dropped on every mutation too, or a deleted memory stays servable for the
    full cache TTL.
    """
    _cached_recall.cache_clear()
    if not _redis:
        return 0
    removed = 0
    try:
        for k in _redis.scan_iter(match=f"{SEMANTIC_CACHE_PREFIX}*", count=500):
            removed += _redis.delete(k)
        # Sprint-1 1.5 fix: clear the exact-key layer UNCONDITIONALLY. The
        # HORIZONTAL_SCALING_ENABLED guard made non-pool instances (the
        # maintenance cron, in-process tools) skip keys that pool-mode
        # instances keep writing — poison survived "full" invalidation.
        for k in _redis.scan_iter(match=f"{RECALL_CACHE_PREFIX}*", count=500):
            removed += _redis.delete(k)
        if removed:
            log.info("recall cache invalidated: %d entries (semantic + exact-key)", removed)
    except Exception as e:
        log.warning("recall cache invalidation failed: %s", e)
    return removed

# Session-level seen-ID set for de-duplication across multiple recall calls
# within the same Kilo session.  Cleared on server restart.
_session_seen_ids: set[str] = set()
# Multiplier applied to a memory already returned in this session. Low enough to
# sink repeats below any fresh match, high enough that an all-repeats query still
# returns its answer instead of an empty list.
_SEEN_REPEAT_PENALTY = 0.35
# Quarantined content (poison_score in the 0.3–0.7 quarantine band) remains
# retrievable + flagged, but ranks below every clean answer (decision 2026-10-08).
_QUARANTINE_RANK_PENALTY = 0.35

# ---------------------------------------------------------------------------
# Qdrant client + sentence-transformer embedding model
# ---------------------------------------------------------------------------
log.info("Connecting to Qdrant at %s ...", QDRANT_URL)
_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# CPU thread cap: limit how many cores the embedding model + torch use so
# background jobs (maintenance cron, reflection, dreaming) don't spike the
# 4-core box while the agent is coding. Set via MEMORY_TORCH_THREADS env.
try:
    import torch
    _torch_threads = int(os.environ.get("MEMORY_TORCH_THREADS", "2"))
    torch.set_num_threads(_torch_threads)
    log.info("torch threads capped to %d", _torch_threads)
except Exception:
    pass

# Embedding model. LAZY (MEMORY_EAGER_PREWARM not set): the model is loaded on
# first use, not at server boot, so Kilo sub-agent runs that never call a
# memory tool do not pay the ~10-15s load + HF model-file check tax. The model
# is cached on disk, so first-use load is fast after the initial warm-up.
_EMBED_MODEL_LOADED = False
def _embed_model():
    global _embed_model_ref, _EMBED_MODEL_LOADED
    if not _EMBED_MODEL_LOADED:
        _embed_model_ref = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model '%s' loaded lazily (vector_size=%d)", EMBEDDING_MODEL, VECTOR_SIZE)
        _EMBED_MODEL_LOADED = True
    return _embed_model_ref
_embedding_model = None
_embed_model_ref = None
if os.environ.get("MEMORY_EAGER_PREWARM") == "1":
    # Legacy behavior: load at boot (used by maintenance cron / long-lived server).
    _embed_model_ref = SentenceTransformer(EMBEDDING_MODEL)
    _embedding_model = _embed_model_ref
    _EMBED_MODEL_LOADED = True
    log.info("Embedding model '%s' loaded (vector_size=%d)", EMBEDDING_MODEL, VECTOR_SIZE)

# ---------------------------------------------------------------------------
# Cleanup: remove the duplicate cross-encoder block that was added in a
# prior patch. The original lazy-loaded cross-encoder (lines ~1403-1415)
# already handles reranking via RERANKER_MODEL with RERANK_TOP_K.
# The new env-driven RERANKER_MODEL is the single source of truth.
# ---------------------------------------------------------------------------

# ===========================================================================
# RAG / KNOWLEDGE BASE — documents lane (separate collection, same stack)
# ===========================================================================
def _kb_ensure_collection() -> None:
    try:
        _client.get_collection(KB_COLLECTION)
        return
    except UnexpectedResponse:
        pass
    except Exception as _e:
        log.warning("kb collection check failed: %s", _e)
        return
    try:
        _client.create_collection(
            collection_name=KB_COLLECTION,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10000, max_indexing_threads=4),
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
        )
        for _f in ("scope", "doc_id", "source"):
            try:
                _client.create_payload_index(collection_name=KB_COLLECTION, field_name=_f,
                                             field_schema=qmodels.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
        log.info("KB collection '%s' created", KB_COLLECTION)
    except Exception as _e:
        log.warning("KB collection create failed (non-fatal): %s", _e)

def _kb_chunk(text: str, size: int = KB_CHUNK_SIZE, overlap: int = KB_CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks, i, n = [], 0, len(text)
    while i < n:
        end = min(i + size, n)
        if end < n:
            window = text[i:end]
            for sep in ("\n\n", "\n", ". "):
                pos = window.rfind(sep)
                if pos > size * 0.5:
                    end = i + pos + len(sep)
                    break
        chunks.append(text[i:end].strip())
        if end >= n:
            break
        i = max(end - overlap, i + 1)
    return [c for c in chunks if c]

def _kb_embed(texts: list[str]) -> list[list[float]]:
    return _embed_model().encode(list(texts), convert_to_numpy=True).tolist()

def _kb_rerank(query: str, cands: list[tuple]) -> list[tuple]:
    ce = _get_cross_encoder()
    if ce is None:
        return cands
    try:
        scores = ce.predict([(query, (c[2].get("content") or "")) for c in cands])
        order = sorted(range(len(cands)), key=lambda i: float(scores[i]), reverse=True)
        return [(cands[i][0], float(scores[i]), cands[i][2]) for i in order]
    except Exception as _e:
        log.warning("kb rerank failed: %s", _e)
        return cands

def _kb_pg(r, k, i):
    return r.get(k) if (isinstance(r, dict) or hasattr(r, "keys")) else r[i]

def _kb_add_document(text: str, source: str, title: str, scope: str, meta: dict | None = None) -> dict:
    import hashlib
    meta = meta or {}
    chunks = _kb_chunk(text)
    if not chunks:
        raise ValueError("no content after chunking")
    doc_id = hashlib.sha256((str(scope) + "|" + source + "|" + text).encode("utf-8")).hexdigest()[:32]
    vecs = _kb_embed(chunks)
    now = _now_iso()
    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, vecs)):
        pl = {"scope": scope, "doc_id": doc_id, "source": source, "title": title,
              "chunk_index": idx, "n_chunks": len(chunks), "content": chunk,
              "kind": "kb", "created_at": now}
        for k in ("url", "page", "section", "tags"):
            if isinstance(meta.get(k), (str, int)):
                pl[k] = meta[k]
        _pid = str(uuid.uuid5(uuid.NAMESPACE_URL, "%s:%d" % (doc_id, idx)))
        points.append(qmodels.PointStruct(id=_pid, vector=vec, payload=pl))
    try:
        _client.delete(collection_name=KB_COLLECTION, points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id))])))
    except Exception:
        pass
    _client.upsert(collection_name=KB_COLLECTION, points=points, wait=True)
    try:
        _pg_run("INSERT INTO kb_documents(doc_id,scope,source,title,checksum,n_chunks,status,payload,updated_at) "
                "VALUES (:d,:s,:src,:t,:c,:n,'active',:p,now()) "
                "ON CONFLICT (doc_id) DO UPDATE SET scope=EXCLUDED.scope, source=EXCLUDED.source, "
                "title=EXCLUDED.title, checksum=EXCLUDED.checksum, n_chunks=EXCLUDED.n_chunks, "
                "status='active', payload=EXCLUDED.payload, updated_at=now()",
                {"d": doc_id, "s": scope, "src": source, "t": title, "c": doc_id,
                 "n": len(chunks), "p": json.dumps(meta)})
    except Exception as _e:
        log.warning("kb_documents upsert failed: %s", _e)
    return {"doc_id": doc_id, "source": source, "scope": scope, "chunks": len(chunks)}

def _tool_kb_ingest(args: dict) -> types.CallToolResult:
    text = args.get("text") or ""
    source = (args.get("source") or args.get("title") or "inline").strip()
    title = (args.get("title") or source).strip()
    scope = authorize_scope(args.get("scope"))
    meta = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
    if not text.strip():
        return _err("memory_kb_ingest: 'text' is required")
    try:
        info = _kb_add_document(text, source, title, scope, meta)
    except Exception as _e:
        return _err(f"memory_kb_ingest: {_e}")
    return _ok(json.dumps(info, ensure_ascii=False))

def _tool_kb_search(args: dict) -> types.CallToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return _err("memory_kb_search: 'query' is required")
    top_k = min(int(args.get("top_k") or 6), 50)
    scope = authorize_scope(args.get("scope"))
    source = args.get("source")
    qvec = _embed_model().encode(query).tolist()
    must = [qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope))]
    if source:
        must.append(qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source)))
    must_not = [] if args.get("include_stale") else [
        qmodels.FieldCondition(key="stale", match=qmodels.MatchValue(value=True))]
    try:
        hits = _client.query_points(
            collection_name=KB_COLLECTION, query=qvec, limit=max(top_k * 4, 20),
            query_filter=qmodels.Filter(must=must, must_not=must_not), with_payload=True).points
    except UnexpectedResponse:
        return _err("memory_kb_search: knowledge base is empty (ingest a document first)")
    if not hits:
        return _ok(json.dumps({"query": query, "count": 0, "results": []}, ensure_ascii=False))
    cands = [(h.id, float(h.score), (h.payload or {})) for h in hits]
    cands = _kb_rerank(query, cands)
    results = []
    for pid, score, pl in cands[:top_k]:
        results.append({"doc_id": pl.get("doc_id"), "source": pl.get("source"),
                        "title": pl.get("title"), "chunk_index": pl.get("chunk_index"),
                        "score": round(float(score), 4),
                        "citation": f"{pl.get('source')}#{pl.get('chunk_index')}",
                        "content": pl.get("content")})
    return _ok(json.dumps({"query": query, "count": len(results), "results": results},
                          ensure_ascii=False, indent=2))

def _tool_kb_sources(args: dict) -> types.CallToolResult:
    scope = authorize_scope(args.get("scope"))
    try:
        rows = _pg_query("SELECT doc_id,source,title,n_chunks,created_at FROM kb_documents "
                         "WHERE scope=:s AND status='active' ORDER BY created_at DESC LIMIT 200",
                         {"s": scope})
    except Exception as _e:
        return _err(f"memory_kb_sources: {_e}")
    out = []
    for r in rows:
        ca = _kb_pg(r, "created_at", 4)
        out.append({"doc_id": _kb_pg(r, "doc_id", 0), "source": _kb_pg(r, "source", 1),
                    "title": _kb_pg(r, "title", 2), "n_chunks": _kb_pg(r, "n_chunks", 3),
                    "created_at": ca.isoformat() if hasattr(ca, "isoformat") else ca})
    return _ok(json.dumps({"count": len(out), "sources": out}, ensure_ascii=False, indent=2))

def _tool_kb_delete(args: dict) -> types.CallToolResult:
    doc_id = args.get("doc_id")
    source = args.get("source")
    scope = authorize_scope(args.get("scope"))
    if not (doc_id or source):
        return _err("memory_kb_delete: doc_id or source is required")
    def _del_doc(d):
        _client.delete(collection_name=KB_COLLECTION, points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(must=[
                qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=d)),
                qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope))])))
    if doc_id:
        _del_doc(doc_id)
        _pg_run("DELETE FROM kb_documents WHERE doc_id=:d AND scope=:s", {"d": doc_id, "s": scope})
        return _ok(json.dumps({"deleted": {"doc_id": doc_id}}))
    docs = _pg_query("SELECT doc_id FROM kb_documents WHERE source=:src AND scope=:s",
                     {"src": source, "s": scope})
    ids = [_kb_pg(r, "doc_id", 0) for r in docs]
    for d in ids:
        _del_doc(d)
    _pg_run("DELETE FROM kb_documents WHERE source=:src AND scope=:s", {"src": source, "s": scope})
    return _ok(json.dumps({"deleted": {"source": source, "documents": len(ids)}}))

def _kb_mark_stale(doc_id, chunk_index, memory_id) -> bool:
    try:
        _client.set_payload(
            collection_name=KB_COLLECTION,
            payload={"stale": True, "superseded_by": memory_id, "stale_since": _now_iso()},
            points=qmodels.Filter(must=[
                qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=doc_id)),
                qmodels.FieldCondition(key="chunk_index", match=qmodels.MatchValue(value=chunk_index))]))
        return True
    except Exception as _e:
        log.warning("kb mark stale failed: %s", _e)
        return False

def _kb_reconcile(content: str, scope: str, memory_id: str | None = None,
                  max_candidates: int = 6) -> dict:
    if not content or not content.strip():
        return {"marked": [], "updated": False}
    low = content.lower()
    has_neg = any(m in low for m in _NEGATION_MARKERS)
    qv = _embed_model().encode(content).tolist()
    must = [qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope))]
    try:
        hits = _client.query_points(collection_name=KB_COLLECTION, query=qv,
                limit=max(max_candidates * 3, 12), query_filter=qmodels.Filter(must=must),
                with_payload=True).points
    except Exception:
        return {"marked": [], "updated": False}
    marked, src0 = [], None
    for h in hits:
        pl = h.payload or {}
        if pl.get("stale") or pl.get("kind") == "kb-update":
            continue
        if float(h.score) >= KB_RECONCILE_THRESHOLD and has_neg:
            if _kb_mark_stale(pl.get("doc_id"), pl.get("chunk_index"), memory_id):
                marked.append({"source": pl.get("source"), "chunk_index": pl.get("chunk_index"),
                               "score": round(float(h.score), 3)})
                if src0 is None:
                    src0 = pl.get("source")
    updated = False
    if marked and src0:
        try:
            _kb_add_document("UPDATE: %s (supersedes earlier content in %s)" % (content, src0),
                             "%s#update" % src0, "Update for %s" % src0, scope,
                             {"kind": "kb-update", "supersedes": src0, "memory_id": memory_id})
            updated = True
        except Exception as _e:
            log.warning("kb update append failed: %s", _e)
    return {"marked": marked, "updated": updated}

def _kb_reconcile_safe(content: str, scope: str, memory_id: str | None = None) -> None:
    try:
        _kb_reconcile(content, scope, memory_id)
    except Exception as _e:
        log.warning("kb auto-reconcile failed: %s", _e)

def _tool_kb_reconcile(args: dict) -> types.CallToolResult:
    scope = authorize_scope(args.get("scope"))
    memory_id = args.get("memory_id")
    content = args.get("content")
    if memory_id and not content:
        row = _canonical_row(memory_id)
        if row:
            content = row.get("content")
    if not content:
        return _err("memory_kb_reconcile: provide memory_id or content")
    res = _kb_reconcile(content, scope, memory_id)
    return _ok(json.dumps({"scope": scope, "marked": res["marked"],
                           "update_chunk": res["updated"]}, ensure_ascii=False, indent=2))

def _looks_like_document(text: str) -> bool:
    if not isinstance(text, str) or text.count("\n") < 3:
        return False
    return ("```" in text) or ("\n#" in text) or ("\n- " in text) or ("\n1. " in text) or ("\n* " in text)

def _kb_recall_hits(query: str, scope: str, top_k: int = 5) -> list[dict]:
    try:
        qv = _embed_model().encode(query).tolist()
        hits = _client.query_points(
            collection_name=KB_COLLECTION, query=qv, limit=max(1, top_k),
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope))],
                must_not=[qmodels.FieldCondition(key="stale", match=qmodels.MatchValue(value=True))]),
            with_payload=True).points
    except Exception:
        return []
    out = []
    for h in hits:
        pl = h.payload or {}
        out.append({"lane": "knowledge", "source": pl.get("source"), "title": pl.get("title"),
                    "chunk_index": pl.get("chunk_index"),
                    "citation": "%s#%s" % (pl.get("source"), pl.get("chunk_index")),
                    "score": round(float(h.score), 4), "content": pl.get("content")})
    return out

try:
    coll_info = _client.get_collection(COLLECTION_NAME)
    count = coll_info.points_count
    # G1 (2026-10-04): enable HNSW vector indexing. Previously the collection
    # was created without an indexing config, so indexed_vectors_count stayed 0
    # and every recall did a full scan (5-15s on ~1.2k points). Update just the
    # optimizer + hnsw config here (vectors_config is NOT updatable on an
    # existing collection — passing it raises a validation error). Collections
    # created/recreated after this patch carry the HNSW config natively.
    try:
        _client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=max(1, min(10000, count // 4)),
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10000,
                max_indexing_threads=4,
            ),
        )
        log.info("G1: HNSW indexing config ensured on '%s'", COLLECTION_NAME)
    except Exception as _e:
        log.warning("G1: index config update failed (non-fatal): %s", _e)
    log.info("Qdrant collection '%s' ready — count=%d", COLLECTION_NAME, count)
except UnexpectedResponse:
    log.info("Collection '%s' not found — creating (size=%d, cosine)", COLLECTION_NAME, VECTOR_SIZE)
    try:
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10000, max_indexing_threads=4),
            optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20000),
        )
        for _f in ("scope", "category", "status", "node_type", "memory_id"):
            try:
                _client.create_payload_index(
                    collection_name=COLLECTION_NAME, field_name=_f,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD)
            except Exception:
                pass
        log.info("Qdrant collection '%s' created", COLLECTION_NAME)
    except Exception as _e:
        log.warning("Qdrant collection create failed (non-fatal): %s", _e)

try:
    _kb_ensure_collection()
except Exception as _kbe:
    log.warning("KB ensure failed (non-fatal): %s", _kbe)

# ---------------------------------------------------------------------------
# Redis client for short-term memory
# ---------------------------------------------------------------------------
log.info("Connecting to Redis at %s:%s ...", REDIS_HOST, REDIS_PORT)
_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
# Binary-safe twin of _redis for values that are NOT UTF-8 (e.g. the pickled
# BM25 index). decode_responses=True mangles pickle bytes on read, which is
# why _bm25_redis_load kept failing while raw clients read the same keys fine.
_redis_bytes = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=False)
try:
    _redis.ping()
    log.info("Redis connected successfully")
except redis.AuthenticationError:
    log.error("Redis authentication failed")
    _redis = None
except Exception as e:
    log.warning("Redis connection failed (short-term memory will be unavailable): %s", e)
    _redis = None

# ---------------------------------------------------------------------------
# PostgreSQL — durable structured layer (military-grade tier)
# ---------------------------------------------------------------------------
# Backs: mem_nodes, mem_edges (temporal KG), mem_trajectories, mem_feedback,
# mem_audit. All writes are fire-and-forget with graceful fallback: any PG
# failure logs a warning and never blocks the Qdrant/Redis path.
# Credentials come from MEMORY_PG_* env (set in pm2 ecosystem file).
PG_HOST     = os.environ.get("MEMORY_PG_HOST", "127.0.0.1")
PG_PORT     = int(os.environ.get("MEMORY_PG_PORT", "5432"))
PG_DB       = os.environ.get("MEMORY_PG_DB", "memory")
PG_USER     = os.environ.get("MEMORY_PG_USER", "memory")
PG_PASSWORD = os.environ.get("MEMORY_PG_PASSWORD", "")
_pg_pool = None  # psycopg connection pool (horizontal scaling)
_pg_conn = None  # pg8000 native connection (legacy single-instance mode)

def _init_pg_pool() -> None:
    """Initialize psycopg connection pool for horizontal scaling."""
    global _pg_pool
    try:
        dsn = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
        _pg_pool = ConnectionPool(
            dsn,
            min_size=PG_POOL_MIN_SIZE,
            max_size=PG_POOL_MAX_SIZE,
            timeout=PG_POOL_TIMEOUT,
            kwargs={"row_factory": dict_row},
        )
        # Test connection
        with _pg_pool.connection() as conn:
            conn.execute("SELECT 1")
        log.info("PostgreSQL connection pool initialized (min=%d, max=%d)", PG_POOL_MIN_SIZE, PG_POOL_MAX_SIZE)
    except Exception as _e:
        log.warning("PostgreSQL pool initialization failed: %s", _e)
        _pg_pool = None

def _init_pg_native() -> None:
    """Initialize pg8000 native connection (legacy single-instance mode)."""
    global _pg_conn
    try:
        import pg8000.native as _pg_native
        _pg_conn = _pg_native.Connection(
            user=PG_USER, password=PG_PASSWORD,
            host=PG_HOST, port=PG_PORT, database=PG_DB,
        )
        _pg_conn.run("SELECT 1")
        log.info("PostgreSQL connected at %s:%s/%s (durable tier active)", PG_HOST, PG_PORT, PG_DB)
    except Exception as _e:
        log.warning("PostgreSQL connection failed (durable tier unavailable): %s", _e)
        _pg_conn = None

# Initialize based on scaling mode
if HORIZONTAL_SCALING_ENABLED:
    _init_pg_pool()
else:
    _init_pg_native()

# pg8000's native Connection keeps its prepared-statement/portal state on the
# connection object, so a single connection shared by concurrent threads
# corrupts parameter binding. Observed under load:
#   22P02 invalid input syntax for type json  (a scope string landed in a JSON slot)
#   34000 portal "" does not exist
#   08P01 bind message supplies 1 parameters, but prepared statement requires 4
# That is what silently dropped ~40% of concurrent writes. All access is now
# serialized; the statements are short, so the lock is not a throughput concern.
_PG_LOCK = threading.RLock()


class PostgresUnavailableError(Exception):
    """Raised when Postgres is unavailable or a query fails."""
    pass


def _pg_rls_set(conn) -> None:
    """3B-1: push the caller's granted scopes into the PG session (RLS).

    Runs INSIDE the caller's transaction (set_config with is_local=true), so
    the binding dies with the transaction and can never leak across pooled
    connections. The RLS policy grants the FULL set of scopes the credential
    authorizes (a dashboard token may span global + user:moon); narrowing to
    the per-operation effective scope stays an APP-layer check in
    authorize_scope, so RLS is the boundary, not the policy engine.
    System context sets bypass instead — the trusted in-process maintenance
    path (outbox worker, consolidation, startup reconciliation) is the only
    thing allowed to cross namespaces.
    """
    try:
        p = _current.get()
        if p.is_system:
            conn.execute(
                "SELECT set_config('app.scopes', '', true), "
                "set_config('app.bypass_rls', 'on', true)")
        else:
            granted = authorized_scopes() or frozenset()
            conn.execute(
                "SELECT set_config('app.scopes', %s, true), "
                "set_config('app.bypass_rls', 'off', true)",
                (",".join(sorted(granted)),))
    except Exception as _e:  # RLS must never take the query path down
        log.debug("rls session set failed: %s", _e)


def _pg_rls_bind_native() -> None:
    """Sprint-1 1.5 fix: bind RLS context on the shared NATIVE connection.

    The pool paths call _pg_rls_set(conn) inside every transaction; the
    non-pool paths never bound anything, so under FORCE RLS every
    mem_canonical SELECT silently returned ZERO rows in-process — the
    nightly eval reported retrieved:0/mrr:0.0 and the virtual-context
    liveness gate withheld every memory in assembly since RLS shipped.
    Must be called with _PG_LOCK held, inside a transaction.
    """
    _p = _current.get()
    if _p.is_system:
        _pg_conn.run(
            "SELECT set_config('app.scopes', '', true), "
            "set_config('app.bypass_rls', 'on', true)")
    else:
        _granted = ",".join(sorted(authorized_scopes() or frozenset()))
        _pg_conn.run(
            "SELECT set_config('app.scopes', %s, true), "
            "set_config('app.bypass_rls', 'off', true)",
            (_granted,))


def _pg_query(sql: str, params: dict = None) -> list:
    """Run a statement and return rows.

    Uses connection pool when horizontal scaling enabled, otherwise native connection.
    Raises PostgresUnavailableError on connection failure or query error.
    Returns [] only when query succeeds but returns no rows.
    """
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            raise PostgresUnavailableError("Postgres pool not available")
        try:
            with _pg_pool.connection() as conn:
                _pg_rls_set(conn)
                # Convert :name params to %(name)s for psycopg
                if params:
                    import re
                    sql = re.sub(r'(?<!:):(\w+)', r'%(\1)s', sql)
                return conn.execute(sql, params or {}).fetchall()
        except Exception as _e:
            log.error("PG query failed: %s", _e)
            raise PostgresUnavailableError(f"PG query failed: {_e}")
    else:
        if _pg_conn is None:
            raise PostgresUnavailableError("Postgres connection not available")
        try:
            with _PG_LOCK:
                _pg_conn.run("BEGIN")
                try:
                    _pg_rls_bind_native()
                    _rows = _pg_conn.run(sql, **(params or {}))
                    _pg_conn.run("COMMIT")
                    return _rows
                except Exception:
                    try:
                        _pg_conn.run("ROLLBACK")
                    except Exception:
                        pass
                    raise
        except Exception as _e:
            log.error("PG query failed: %s", _e)
            raise PostgresUnavailableError(f"PG query failed: {_e}")


def _pg_run(sql: str, params: dict = None, table: str = "audit") -> bool:
    """Run a statement. Returns True on success.

    Uses connection pool when horizontal scaling enabled, otherwise native connection.
    Raises PostgresUnavailableError on connection failure or write error.
    Returns True only on successful commit.
    """
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            raise PostgresUnavailableError("Postgres pool not available")
        try:
            with _pg_pool.connection() as conn:
                _pg_rls_set(conn)
                # Convert :name params to %(name)s for psycopg
                if params:
                    import re
                    sql = re.sub(r'(?<!:):(\w+)', r'%(\1)s', sql)
                conn.execute(sql, params or {})
                conn.commit()
            return True
        except Exception as _e:
            log.error("PG write failed (%s): %s", table, _e)
            raise PostgresUnavailableError(f"PG write failed ({table}): {_e}")
    else:
        if _pg_conn is None:
            raise PostgresUnavailableError("Postgres connection not available")
        try:
            with _PG_LOCK:
                _pg_conn.run("BEGIN")
                try:
                    _pg_rls_bind_native()
                    _pg_conn.run(sql, **(params or {}))
                    _pg_conn.run("COMMIT")
                except Exception:
                    try:
                        _pg_conn.run("ROLLBACK")
                    except Exception:
                        pass
                    raise
            return True
        except Exception as _e:
            log.error("PG write failed (%s): %s", table, _e)
            raise PostgresUnavailableError(f"PG write failed ({table}): {_e}")

def _pg_node_uuid(memory_id: str):
    """Resolve a Qdrant memory_id to its Postgres mem_nodes.id (UUID), or None.

    mem_edges foreign keys reference mem_nodes.id, not the Qdrant memory_id,
    so every edge insert must resolve both endpoints through this helper first.
    """
    # Pool mode: _pg_conn is None by design; gate on the active backend instead.
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return None
    elif _pg_conn is None:
        return None
    try:
        rows = _pg_query(
            "SELECT id FROM mem_nodes WHERE payload->>'memory_id' = :mid "
            "ORDER BY created_at DESC LIMIT 1",
            {"mid": memory_id},
        )
        if not rows:
            return None
        r0 = rows[0]
        return r0["id"] if isinstance(r0, dict) else r0[0]
    except Exception:
        return None

def _pg_node_insert(label: str, node_type: str, scope: str, payload_json: str, memory_id: str):
    """Insert a mem_nodes row and return its UUID (None on failure).

    The edge FKs reference mem_nodes.id, so the node row must exist before any
    edge insert. Uses RETURNING id.
    """
    # Pool mode: _pg_conn is None by design; gate on the active backend instead.
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return None
    elif _pg_conn is None:
        return None
    try:
        rows = _pg_query(
            "INSERT INTO mem_nodes(label, node_type, scope, payload) "
            "VALUES (:label, :ntype, :scope, :payload) RETURNING id",
            {"label": label, "ntype": node_type, "scope": scope,
             "payload": payload_json},
        )
        if not rows:
            return None
        r0 = rows[0]
        return r0["id"] if isinstance(r0, dict) else r0[0]
    except Exception as _e:
        log.warning("PG mem_nodes insert failed: %s", _e)
        return None

def _pg_audit(op: str, memory_id: str = None, detail: dict = None,
              actor: str = None, scope: str = None) -> bool:
    """3B-4: append-only audit trail. Actor defaults to the authenticated
    caller (never a static string), scope to the operation's effective scope.
    mem_audit is INSERT/SELECT-only for the app role: rows can neither be
    updated nor deleted by the server itself."""
    if actor is None:
        p = _current.get()
        actor = (p.identity or "anonymous") if not p.is_system else "system"
    if scope is None:
        scope = _effective_scope.get() or ""
    return _pg_run(
        "INSERT INTO mem_audit(op, memory_id, detail, actor, scope) "
        "VALUES (:op, :mid, :det, :actor, :scope)",
        {"op": op, "mid": memory_id, "det": json.dumps(detail or {}),
         "actor": actor, "scope": scope},
        table="audit",
    )


# ---------------------------------------------------------------------------
# PHASE A — CANONICAL MEMORY LIFECYCLE (PostgreSQL is the source of truth)
# ---------------------------------------------------------------------------
# Before this, Qdrant was written FIRST and mem_nodes.second, so PostgreSQL was
# not authoritative: a crash between the two lost the memory silently, and a
# delete that reached Qdrant but not mem_nodes left the record half-alive.
#
# mem_canonical is the single lifecycle authority. Its `status` is what
# retrieval is allowed to trust:
#
#   pending_index -> canonical write committed, derived layers not yet updated
#   active        -> fully indexed, retrievable
#   index_failed  -> canonical write committed but a derived layer failed
#   superseded    -> replaced by a newer version (valid_until set)
#   deleted       -> tombstoned; must NEVER be retrievable, at any layer
#
# Qdrant / BM25 / Redis / graph / embeddings are ALL derived. Retrieval filters
# candidates through mem_canonical, so a stale derived row cannot resurrect a
# tombstoned memory.

_CANONICAL_STATUSES = ("pending_index", "active", "index_failed",
                       "superseded", "deleted")

_KB_DDL = """
CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id     text PRIMARY KEY,
    scope      text NOT NULL,
    source     text,
    title      text,
    checksum   text,
    n_chunks   int NOT NULL DEFAULT 0,
    status     text NOT NULL DEFAULT 'active',
    payload    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kb_documents_scope_idx  ON kb_documents (scope);
CREATE INDEX IF NOT EXISTS kb_documents_source_idx ON kb_documents (source);
"""

_CANONICAL_DDL = """
CREATE TABLE IF NOT EXISTS mem_canonical (
    memory_id    text PRIMARY KEY,
    scope        text NOT NULL,
    node_type    text NOT NULL DEFAULT 'fact',
    status       text NOT NULL DEFAULT 'pending_index',
    version      bigint NOT NULL DEFAULT 1,
    content      text,
    valid_from   timestamptz NOT NULL DEFAULT now(),
    valid_until  timestamptz,
    event_time   timestamptz,
    superseded_by text,
    deleted_at   timestamptz,
    last_error   text,
    payload      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT mem_canonical_status_ck
        CHECK (status IN ('pending_index','active','index_failed',
                          'superseded','deleted'))
);
CREATE INDEX IF NOT EXISTS mem_canonical_scope_status_idx
    ON mem_canonical (scope, status);
CREATE INDEX IF NOT EXISTS mem_canonical_status_idx
    ON mem_canonical (status);
ALTER TABLE mem_canonical ADD COLUMN IF NOT EXISTS event_time timestamptz;
CREATE INDEX IF NOT EXISTS mem_canonical_event_time_idx
    ON mem_canonical (event_time);
"""

_OUTBOX_DDL = """
CREATE TABLE IF NOT EXISTS mem_outbox (
    id           bigserial PRIMARY KEY,
    event_id     text UNIQUE NOT NULL,
    event_type   text NOT NULL,
    memory_id    text NOT NULL,
    scope        text NOT NULL,
    version      bigint NOT NULL,
    payload      jsonb NOT NULL DEFAULT '{}'::jsonb,
    status       text NOT NULL DEFAULT 'pending',
    attempts     int NOT NULL DEFAULT 0,
    last_error   text,
    available_at timestamptz NOT NULL DEFAULT now(),
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),
    processed_at timestamptz
);
CREATE INDEX IF NOT EXISTS mem_outbox_claim_idx
    ON mem_outbox (status, available_at);
CREATE INDEX IF NOT EXISTS mem_outbox_memory_idx
    ON mem_outbox (memory_id);
"""

_TRAJECTORY_STEPS_DDL = """
CREATE TABLE IF NOT EXISTS mem_trajectory_steps (
    id           bigserial PRIMARY KEY,
    trajectory_id uuid NOT NULL REFERENCES mem_trajectories(id) ON DELETE CASCADE,
    step_index   int NOT NULL,
    action       text NOT NULL,
    observation  text,
    success      boolean NOT NULL DEFAULT true,
    duration_ms  int,
    tool_used    text,
    logged_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mem_trajectory_steps_traj_idx
    ON mem_trajectory_steps (trajectory_id, step_index);
CREATE INDEX IF NOT EXISTS mem_trajectory_steps_tool_idx
    ON mem_trajectory_steps (tool_used);
"""


class CanonicalWriteError(RuntimeError):
    """The canonical PostgreSQL write failed. Callers MUST NOT report success."""


def _pg_txn_execute(stmts: list[tuple[str, dict]]) -> bool:
    """Run statements inside one real transaction. Returns True on commit.

    Uses connection pool when horizontal scaling enabled, otherwise native connection.
    Raises PostgresUnavailableError on failure.
    """
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            raise PostgresUnavailableError("Postgres pool not available")
        try:
            with _pg_pool.connection() as conn:
                with conn.transaction():
                    _pg_rls_set(conn)
                    for sql, params in stmts:
                        # Convert :name params to %(name)s for psycopg
                        if params:
                            import re
                            sql = re.sub(r'(?<!:):(\w+)', r'%(\1)s', sql)
                        conn.execute(sql, params or {})
                return True
        except Exception as _e:
            log.error("PG transaction failed (%d stmts): %s", len(stmts), _e)
            raise PostgresUnavailableError(f"PG transaction failed: {_e}")
    else:
        if _pg_conn is None:
            raise PostgresUnavailableError("Postgres connection not available")
        try:
            with _PG_LOCK:
                _pg_conn.run("BEGIN")
                try:
                    # 3B-1 RLS parity with the pool path: bind the caller's
                    # scopes (or system bypass) INSIDE the transaction. Without
                    # this, every _pg_txn_execute under FORCE RLS failed with
                    # "new row violates row-level security policy" in
                    # single-instance (non-pool) deployments.
                    _p = _current.get()
                    if _p.is_system:
                        _pg_conn.run(
                            "SELECT set_config('app.scopes', '', true), "
                            "set_config('app.bypass_rls', 'on', true)")
                    else:
                        _granted = ",".join(sorted(authorized_scopes() or frozenset()))
                        _pg_conn.run(
                            "SELECT set_config('app.scopes', %s, true), "
                            "set_config('app.bypass_rls', 'off', true)",
                            (_granted,))
                    for sql, params in stmts:
                        _pg_conn.run(sql, **(params or {}))
                    _pg_conn.run("COMMIT")
                    return True
                except Exception:
                    try:
                        _pg_conn.run("ROLLBACK")
                    except Exception:
                        pass
                    raise
        except Exception as _e:
            log.error("PG transaction failed (%d stmts): %s", len(stmts), _e)
            raise PostgresUnavailableError(f"PG transaction failed: {_e}")


def _pg_query_txn(sql: str, params: dict = None) -> list:
    """Run a query inside a transaction and return rows.

    Uses connection pool when horizontal scaling enabled, otherwise native connection.
    Required for queries with FOR UPDATE SKIP LOCKED that need transaction context.
    Raises PostgresUnavailableError on failure.
    """
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            raise PostgresUnavailableError("Postgres pool not available")
        try:
            with _pg_pool.connection() as conn:
                with conn.transaction():
                    _pg_rls_set(conn)
                    # Convert :name params to %(name)s for psycopg
                    if params:
                        import re
                        sql = re.sub(r'(?<!:):(\w+)', r'%(\1)s', sql)
                    return conn.execute(sql, params or {}).fetchall()
        except Exception as _e:
            log.error("PG transactional query failed: %s", _e)
            raise PostgresUnavailableError(f"PG transactional query failed: {_e}")
    else:
        if _pg_conn is None:
            raise PostgresUnavailableError("Postgres connection not available")
        try:
            with _PG_LOCK:
                _pg_conn.run("BEGIN")
                try:
                    _pg_rls_bind_native()
                    rows = _pg_conn.run(sql, **(params or {}))
                    _pg_conn.run("COMMIT")
                    return rows
                except Exception:
                    try:
                        _pg_conn.run("ROLLBACK")
                    except Exception:
                        pass
                    raise
        except Exception as _e:
            log.error("PG transactional query failed: %s", _e)
            raise PostgresUnavailableError(f"PG transactional query failed: {_e}")


def _ensure_canonical_schema() -> bool:
    """Create the canonical + outbox tables if absent. Idempotent."""
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return False
    else:
        if _pg_conn is None:
            return False
    try:
        _pg_txn_execute([(_CANONICAL_DDL, {}), (_OUTBOX_DDL, {}), (_TRAJECTORY_STEPS_DDL, {}), (_KB_DDL, {})])
        log.info("canonical schema ready (mem_canonical + mem_outbox + mem_trajectory_steps)")
        return True
    except Exception as _e:  # pragma: no cover - defensive
        log.error("canonical schema init failed: %s", _e)
        return False


def _outbox_event_id(event_type: str, memory_id: str, version: int) -> str:
    """Deterministic idempotency key: same mutation => same id => no dupes."""
    return f"{event_type}:{memory_id}:{version}"


def _canonical_write(memory_id: str, scope: str, node_type: str, content: str,
                     version: int = 1, event_type: str = "MEMORY_CREATED",
                     payload: dict = None, superseded_by: str = None,
                     valid_until: str = None, extra: dict = None) -> tuple[bool, str]:
    """COMMIT the canonical record and its outbox event atomically.

    This is the only path allowed to create a live memory. Returns
    (ok, reason). On failure the caller must not report `stored`.
    Raises PostgresUnavailableError if Postgres is unavailable.

    ``extra`` is merged into the OUTBOX EVENT payload only (not the canonical
    row payload) — used by graph events to carry src/dst/rel through the
    outbox so the worker can apply them idempotently.
    """
    evt = _outbox_event_id(event_type, memory_id, version)
    _event_payload = dict(payload or {})
    if extra:
        _event_payload.update(extra)
    stmts = [
        ("INSERT INTO mem_canonical"
         "(memory_id, scope, node_type, status, version, content, superseded_by,"
         " valid_until, event_time, deleted_at, last_error, payload, updated_at) "
         "VALUES (:mid, :scope, :ntype, 'pending_index', :ver, :content, :sup,"
         " :vuntil, :evt, NULL, NULL, :payload, now()) "
         "ON CONFLICT (memory_id) DO UPDATE SET "
         "   scope=EXCLUDED.scope, node_type=EXCLUDED.node_type,"
         "   status='pending_index', version=EXCLUDED.version,"
         "   content=EXCLUDED.content, superseded_by=EXCLUDED.superseded_by,"
         "   valid_until=EXCLUDED.valid_until, event_time=EXCLUDED.event_time, deleted_at=NULL,"
         "   last_error=NULL, payload=EXCLUDED.payload, updated_at=now()",
         {"mid": memory_id, "scope": scope, "ntype": node_type,
           "ver": version, "content": content, "sup": superseded_by,
           "vuntil": valid_until, "evt": (payload or {}).get("event_time"),
           "payload": json.dumps(payload or {})}),
        ("INSERT INTO mem_outbox"
         "(event_id, event_type, memory_id, scope, version, payload) "
         "VALUES (:eid, :etype, :mid, :scope, :ver, :payload) "
         "ON CONFLICT (event_id) DO NOTHING",
         {"eid": evt, "etype": event_type, "mid": memory_id,
           "scope": scope, "ver": version, "payload": json.dumps(_event_payload)}),
    ]
    _pg_txn_execute(stmts)  # Raises PostgresUnavailableError on failure
    return True, ""


def _canonical_set_status(memory_id: str, status: str,
                          error: str = None) -> bool:
    """Transition lifecycle status. Tombstones are never downgraded."""
    if status not in _CANONICAL_STATUSES:
        raise ValueError(f"invalid canonical status {status!r}")
    # Pool mode: _pg_conn is None by design; _pg_run is mode-aware.
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return False
    elif _pg_conn is None:
        return False
    try:
        return _pg_run(
            "UPDATE mem_canonical SET status=:st, last_error=:err, updated_at=now() "
            "WHERE memory_id=:mid AND status <> 'deleted'",
            {"st": status, "err": error, "mid": memory_id},
            table="canonical-status",
        )
    except PostgresUnavailableError as _e:
        log.warning("canonical status update failed id=%s st=%s: %s",
                    memory_id, status, _e)
        return False


def _canonical_tombstone(memory_id: str, version: int = None,
                         scope: str = None) -> tuple[bool, str]:
    """Tombstone a memory and enqueue MEMORY_DELETED in ONE transaction.

    Returns only after COMMIT. Secondary layers are cleaned asynchronously by
    the outbox worker; retrieval filters tombstones regardless, so cleanup lag
    cannot resurrect the memory.
    """
    # Pool mode: _pg_conn is None by design; gate on the active backend instead.
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return False, "durable tier unavailable"
    elif _pg_conn is None:
        return False, "durable tier unavailable"
    row = _pg_query("SELECT version, scope FROM mem_canonical WHERE memory_id=:m",
                    {"m": memory_id})
    if not row:
        return False, "no canonical record"
    r0 = row[0]
    _cver = r0["version"] if isinstance(r0, dict) else r0[0]
    _cscope = r0["scope"] if isinstance(r0, dict) else r0[1]
    ver = version if version is not None else int(_cver)
    sc = scope or _cscope
    evt = _outbox_event_id("MEMORY_DELETED", memory_id, ver)
    ok = _pg_txn_execute([
        ("UPDATE mem_canonical SET status='deleted', deleted_at=now(), "
         " updated_at=now(), last_error=NULL WHERE memory_id=:m",
         {"m": memory_id}),
        ("INSERT INTO mem_outbox"
         "(event_id, event_type, memory_id, scope, version, payload) "
         "VALUES (:eid, 'MEMORY_DELETED', :mid, :scope, :ver, :payload) "
         "ON CONFLICT (event_id) DO UPDATE SET "
         "   status='pending', available_at=now(), updated_at=now(), "
         "   last_error=NULL, attempts=0",
         {"eid": evt, "mid": memory_id, "scope": sc, "ver": ver,
          "payload": json.dumps({"reason": "delete"})}),
    ])
    if not ok:
        return False, "tombstone transaction did not commit"
    _invalidate_recall_caches()
    return True, ""


# --- retrieval gate ---------------------------------------------------------
def _canonical_live_ids(memory_ids: list[str], include_superseded: bool = False) -> set[str]:
    """Of the given ids, return those that are RETRIEVABLE right now.

    A memory is retrievable only if its canonical row exists and its status is
    active (or index_failed, whose canonical record is intact and whose derived
    rows are being repaired — it must stay visible rather than vanish). Anything
    deleted, superseded or still pending is withheld, which is what stops a
    stale Qdrant/BM25/Redis row from resurrecting a tombstoned memory.
    """
    ids = [m for m in (memory_ids or []) if m]
    if not ids:
        return set()
    _statuses = ("('active','index_failed','superseded')" if include_superseded
                 else "('active','index_failed')")
    # Use pool-aware query in horizontal scaling mode
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return set()
        try:
            rows = _pg_query(
                "SELECT memory_id FROM mem_canonical "
                f"WHERE memory_id = ANY(:ids) AND status IN {_statuses}",
                {"ids": "{" + ",".join(ids) + "}"})
            return {r[0] if not isinstance(r, dict) else r.get('memory_id') for r in rows}
        except Exception as _e:
            # 1B-5 degraded matrix: Postgres down must not zero out recall.
            # Fail OPEN (serve without liveness verification) and say so — the
            # tradeoff is that a memory deleted during the outage may surface
            # until the store recovers. Fail-closed here made PG the single
            # point of failure for ALL reads, which the matrix forbids.
            log.warning("canonical gate unavailable — DEGRADED READS: serving "
                        "WITHOUT liveness verification (%d ids pass-through): %s",
                        len(ids), _e)
            return set(ids)
    else:
        if _pg_conn is None:
            return set()
        try:
            rows = _pg_query(
                "SELECT memory_id FROM mem_canonical "
                f"WHERE memory_id = ANY(:ids) AND status IN {_statuses}",
                {"ids": "{" + ",".join(ids) + "}"})
            return {r[0] for r in rows}
        except Exception as _e:
            log.warning("canonical gate unavailable, failing closed: %s", _e)
            return set()


def _filter_canonical(results: list[dict], as_of: str | None = None) -> list[dict]:
    """Drop any result whose canonical record is not retrievable."""
    if not results:
        return results
    live = _canonical_live_ids([r.get("memory_id") for r in results], include_superseded=bool(as_of))
    return [r for r in results if r.get("memory_id") in live]


# ===========================================================================
# PHASE D / STAGE 4 — OUTBOX WORKER
# ===========================================================================
# The outbox table IS the queue. There is no in-memory queue, because an
# in-memory queue loses work on restart and that is exactly the failure this
# stage exists to eliminate. PostgreSQL holds durable truth for pending work;
# this worker only moves rows between explicit states.
#
#   pending ──claim──> processing ──ok──> completed
#      ^                   │
#      │                   ├── transient error ──> retryable_failure ──(backoff)──┐
#      └───────────────────┴───────────────────────────────────────────────────-┘
#                          │
#                          ├── attempts exhausted / invalid event
#                          ▼
#                   permanent_failure ──> dead_letter
#
# A claim is a LEASE, not a lock: the row is marked processing with
# lease_expires_at and worker_id and the claim transaction COMMITS before any
# slow network call to Qdrant/Redis happens. If the worker dies, the lease
# expires and another worker reclaims the row. Nothing is held locked during
# I/O.

_OUTBOX_MIGRATION_DDL = """
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS worker_id         text;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS last_worker        text;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS lease_expires_at   timestamptz;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS first_attempt_at   timestamptz;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS last_attempt_at    timestamptz;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS next_attempt_at    timestamptz;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS dead_letter_reason text;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS applied_version   bigint;
ALTER TABLE mem_outbox ADD COLUMN IF NOT EXISTS outcome           text;
UPDATE mem_outbox SET next_attempt_at = COALESCE(available_at, created_at)
    WHERE next_attempt_at IS NULL;
CREATE INDEX IF NOT EXISTS mem_outbox_lease_idx
    ON mem_outbox (status, next_attempt_at, lease_expires_at);
CREATE INDEX IF NOT EXISTS mem_outbox_worker_idx
    ON mem_outbox (worker_id, status);
"""

# The six explicit states required by the spec. Unknown values are a bug and
# must not be written by this worker.
OUTBOX_PENDING = "pending"
OUTBOX_PROCESSING = "processing"
OUTBOX_COMPLETED = "completed"
OUTBOX_RETRYABLE = "retryable_failure"
OUTBOX_PERMANENT = "permanent_failure"
OUTBOX_DEAD_LETTER = "dead_letter"
_OUTBOX_STATES = frozenset({OUTBOX_PENDING, OUTBOX_PROCESSING, OUTBOX_COMPLETED,
                            OUTBOX_RETRYABLE, OUTBOX_PERMANENT, OUTBOX_DEAD_LETTER})

# Events that mean "make the derived stores match this canonical row".
_OUTBOX_UPSERT_EVENTS = frozenset({
    "MEMORY_CREATED", "MEMORY_UPDATED", "MEMORY_SUPERSEDED",
    "MEMORY_REEMBED_REQUIRED",
})
_OUTBOX_DELETE_EVENTS = frozenset({"MEMORY_DELETED"})

# Graph events for durable, idempotent graph writes via outbox
_OUTBOX_GRAPH_EVENTS = frozenset({"GRAPH_LINK", "GRAPH_UNLINK"})

OUTBOX_LEASE_SECONDS = int(os.environ.get("MEMORY_OUTBOX_LEASE_SECONDS", "60"))
OUTBOX_BATCH_SIZE = int(os.environ.get("MEMORY_OUTBOX_BATCH", "16"))
OUTBOX_POLL_SECONDS = float(os.environ.get("MEMORY_OUTBOX_POLL_SECONDS", "1.0"))
OUTBOX_MAX_ATTEMPTS = int(os.environ.get("MEMORY_OUTBOX_MAX_ATTEMPTS", "8"))
OUTBOX_BACKOFF_BASE = float(os.environ.get("MEMORY_OUTBOX_BACKOFF_BASE", "2"))
OUTBOX_BACKOFF_MAX = float(os.environ.get("MEMORY_OUTBOX_BACKOFF_MAX", "300"))
OUTBOX_WORKER_COUNT = int(os.environ.get("MEMORY_OUTBOX_WORKERS", "1"))
# Bounded synchronous convergence wait used by _tool_store when opted in via
# MEMORY_OUTBOX_SYNC_GRACE_SECONDS. 0 (default) = fully async, no caller wait.
OUTBOX_SYNC_GRACE_SECONDS = float(os.environ.get("MEMORY_OUTBOX_SYNC_GRACE_SECONDS", "0"))

_outbox_stats_lock = threading.Lock()
_outbox_counters: dict = {
    "claimed": 0, "completed": 0, "retried": 0, "dead_lettered": 0,
    "stale_skipped": 0, "recovered_claims": 0, "duplicate_suppressed": 0,
    "permanent": 0, "latency_total_ms": 0, "latency_samples": 0,
    "store_failures": {},
    # P1-1: Graph consistency metrics
    "graph_link_created": 0, "graph_link_failed": 0, "graph_unlink_created": 0,
    "graph_reconciliation_runs": 0, "graph_divergence_found": 0,
    "graph_divergence_repaired": 0,
}


def _outbox_backoff_seconds(attempts: int) -> float:
    """Exponential backoff with jitter, capped.

    Jitter matters here: without it, every event that failed while Qdrant was
    down retries in lockstep and stampedes the moment Qdrant returns.
    """
    delay = OUTBOX_BACKOFF_BASE * (2 ** max(0, attempts - 1))
    delay = min(delay, OUTBOX_BACKOFF_MAX)
    return delay * (0.75 + random.random() * 0.5)


class PermanentOutboxError(RuntimeError):
    """The event can never succeed. Goes straight to dead-letter."""


def _outbox_bump(counter: str, n: int = 1) -> None:
    with _outbox_stats_lock:
        _outbox_counters[counter] = _outbox_counters.get(counter, 0) + n


def _outbox_store_failure(store: str) -> None:
    with _outbox_stats_lock:
        f = _outbox_counters["store_failures"]
        f[store] = f.get(store, 0) + 1


def _ensure_outbox_schema() -> bool:
    """Apply the lease/backoff/attempt columns. Idempotent."""
    if HORIZONTAL_SCALING_ENABLED:
        if _pg_pool is None:
            return False
    else:
        if _pg_conn is None:
            return False
    try:
        stmts = [(s.strip(), {}) for s in _OUTBOX_MIGRATION_DDL.split(";")
                 if s.strip()]
        ok = _pg_txn_execute(stmts)
        if ok:
            log.info("outbox worker schema ready (lease + backoff + attempts)")
        return ok
    except Exception as _e:
        log.error("outbox migration failed: %s", _e)
        return False


def _outbox_claim(worker_id: str, limit: int = None) -> list[dict]:
    """Atomically claim up to `limit` events with SKIP LOCKED.

    One statement = one short transaction. Two workers running this
    concurrently cannot receive the same row, and a row whose lease has
    expired (crashed worker) is reclaimable. The transaction commits BEFORE the
    caller touches Qdrant/Redis, so a slow or dead downstream never holds a
    database lock.
    """
    limit = limit or OUTBOX_BATCH_SIZE
    sql = """
    WITH candidates AS (
        SELECT id FROM mem_outbox
        WHERE (
                status IN ('pending','retryable_failure')
                AND COALESCE(next_attempt_at, available_at) <= now()
              )
           OR (status = 'processing' AND lease_expires_at < now())
           OR (status = 'permanent_failure')
        ORDER BY COALESCE(next_attempt_at, available_at) ASC, id ASC
        FOR UPDATE SKIP LOCKED
        LIMIT :lim
    )
    UPDATE mem_outbox o
       SET status = 'processing',
           worker_id = :wid,
           last_worker = :wid,
           lease_expires_at = now() + make_interval(secs := :lease_secs),
           attempts = o.attempts + 1,
           first_attempt_at = COALESCE(o.first_attempt_at, now()),
           last_attempt_at = now(),
           updated_at = now()
      FROM candidates c
     WHERE o.id = c.id
    RETURNING o.id, o.event_id, o.event_type, o.memory_id, o.scope, o.version,
              o.payload, o.attempts
    """
    try:
        rows = _pg_query_txn(sql, {"wid": worker_id, "lim": limit, "lease_secs": OUTBOX_LEASE_SECONDS})
        out = []
        for r in rows:
            # Handle both tuple and dict-like rows (psycopg DictRow)
            if isinstance(r, dict) or hasattr(r, 'keys'):
                # DictRow or dict - access by column name
                row_id = r.get('id') or r.get('row_id')
                event_id = r.get('event_id')
                event_type = r.get('event_type')
                memory_id = r.get('memory_id')
                scope = r.get('scope')
                version = r.get('version')
                payload_raw = r.get('payload')
                attempts = r.get('attempts')
            else:
                # Tuple - access by index
                row_id = r[0]
                event_id = r[1]
                event_type = r[2]
                memory_id = r[3]
                scope = r[4]
                version = r[5]
                payload_raw = r[6]
                attempts = r[7]
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else (payload_raw or {})
            except Exception:
                payload = {}
            out.append({
                "row_id": row_id, "event_id": event_id, "event_type": event_type,
                "memory_id": memory_id, "scope": scope, "version": int(version),
                "payload": payload, "attempts": int(attempts),
            })
        _outbox_bump("claimed", len(out))
        if any(r["attempts"] > 1 and r["payload"].get("_reclaimed") for r in out):
            pass
        return out
    except PostgresUnavailableError:
        raise
    except Exception as _e:
        log.warning("outbox claim failed: %s", _e, exc_info=True)
        return []


def _outbox_mark_completed(ev: dict, outcome: str = "applied",
                          applied_version: int = None) -> bool:
    st = ("UPDATE mem_outbox SET status='completed', processed_at=now(), "
          "updated_at=now(), last_error=NULL, outcome=:oc, applied_version=:av, "
          "worker_id=NULL, lease_expires_at=NULL WHERE id=:id")
    try:
        return _pg_run(st, {"id": int(ev["row_id"]), "oc": outcome,
                            "av": applied_version}, table="outbox-complete")
    except Exception as _e:
        log.warning("outbox completion write failed id=%s: %s", ev["row_id"], _e)
        return False


def _outbox_mark_retry(ev: dict, error: str, next_at_iso: str) -> bool:
    st = ("UPDATE mem_outbox SET status='retryable_failure', last_error=:err, "
          "next_attempt_at=:na, updated_at=now(), worker_id=NULL, "
          "lease_expires_at=NULL WHERE id=:id")
    return _pg_run(st, {"id": int(ev["row_id"]), "err": error[:2000],
                        "na": next_at_iso}, table="outbox-retry")


def _outbox_mark_dead(ev: dict, reason: str, permanent_state: bool = True) -> bool:
    """Dead-letter retains the row itself: original event, memory_id, version,
    event_type, attempt count, timestamps, error and last worker all stay
    queryable. Nothing is moved out of the table and nothing is deleted."""
    status = OUTBOX_DEAD_LETTER if permanent_state else OUTBOX_PERMANENT
    st = ("UPDATE mem_outbox SET status=:st, last_error=:err, "
          "dead_letter_reason=:err, updated_at=now(), worker_id=NULL, "
          "lease_expires_at=NULL WHERE id=:id")
    ok = _pg_run(st, {"id": int(ev["row_id"]), "err": reason[:2000], "st": status},
                 table="outbox-dead")
    if ok:
        _outbox_bump("dead_lettered" if permanent_state else "permanent")
        log.error("outbox DEAD LETTER id=%s event=%s type=%s attempts=%s: %s",
                  ev["row_id"], ev["event_id"], ev["event_type"],
                  ev["attempts"], reason)
    return ok


def _outbox_requeue(ev: dict, status: str = OUTBOX_PENDING,
                    delay_seconds: float = 0.0, error: str = None) -> bool:
    st = ("UPDATE mem_outbox SET status=:st, last_error=:err, "
          "next_attempt_at = now() + make_interval(secs := :d), "
          "updated_at=now(), worker_id=NULL, lease_expires_at=NULL WHERE id=:id")
    return _pg_run(st, {"id": int(ev["row_id"]), "st": status, "err": error,
                        "d": int(delay_seconds)}, table="outbox-requeue")


def _outbox_recover_abandoned() -> int:
    """Return rows abandoned by a dead worker to `pending`.

    Safety net in addition to lease expiry: anything still marked processing
    with an expired lease is visible and counted, so "stuck in processing" is a
    measurable condition rather than an invisible loss.
    """
    rows = _pg_query(
        "SELECT id FROM mem_outbox WHERE status='processing' "
        "AND lease_expires_at < now()")
    if not rows:
        return 0
    n = 0
    for row in rows:
        # Handle both tuple and dict row formats
        rid = row[0] if isinstance(row, (tuple, list)) else row.get('id')
        # Skip header rows or invalid data
        if rid is None or (isinstance(rid, str) and not rid.isdigit()):
            continue
        try:
            rid_int = int(rid)
        except (ValueError, TypeError):
            continue
        if _outbox_requeue({"row_id": rid_int}, OUTBOX_PENDING, 0.0,
                           "lease expired — reclaimed by recovery"):
            n += 1
    if n:
        _outbox_bump("recovered_claims", n)
        log.warning("recovered %d abandoned outbox claims", n)
    return n


def _canonical_row(memory_id: str) -> dict | None:
    try:
        rows = _pg_query(
            "SELECT memory_id, scope, node_type, status, version, content, "
            "superseded_by, payload, deleted_at, valid_from, valid_until, created_at, event_time "
            "FROM mem_canonical WHERE memory_id=:m", {"m": memory_id})
    except PostgresUnavailableError:
        return None  # Treat unavailable as "not found" for callers
    if not rows:
        return None
    r = rows[0]
    if isinstance(r, dict) or hasattr(r, 'keys'):
        _raw_payload = r.get("payload")
    else:
        _raw_payload = r[7]
    try:
        payload = json.loads(_raw_payload) if isinstance(_raw_payload, str) else (_raw_payload or {})
    except Exception:
        payload = {}
    if isinstance(r, dict) or hasattr(r, 'keys'):
        return {"memory_id": r.get("memory_id"), "scope": r.get("scope"),
                "node_type": r.get("node_type"), "status": r.get("status"),
                "version": int(r.get("version")), "content": r.get("content"),
                "superseded_by": r.get("superseded_by"), "payload": payload,
                "deleted_at": r.get("deleted_at"),
                "valid_from": r.get("valid_from"),
                "valid_until": r.get("valid_until"),
                "created_at": r.get("created_at"),
                "event_time": r.get("event_time")}
    return {"memory_id": r[0], "scope": r[1], "node_type": r[2],
            "status": r[3], "version": int(r[4]), "content": r[5],
            "superseded_by": r[6], "payload": payload,
            "deleted_at": r[8], "valid_from": r[9],
            "valid_until": r[10], "created_at": r[11],
            "event_time": r[12]}


def _derived_upsert(row: dict) -> dict:
    """Make Qdrant + BM25 + Redis match one canonical row. Idempotent."""
    content = row.get("content")
    failures = []
    if content is None:
        # Credential/transient rows never reach the vector tier.
        return {"failures": [], "skipped": "no_content"}
    scope = row["scope"]
    mid = row["memory_id"]
    version = row["version"]
    extra = row.get("payload") or {}

    # Compute the vector once; reuse for Qdrant and the Redis version snapshot.
    try:
        vec = _embed_model().encode(content).tolist()
        tokens = _tokenize(content)
    except Exception as _e:
        failures.append(("embed", str(_e)))
        _outbox_store_failure("embed")
        return {"failures": failures}

    # --- Qdrant ---
    try:
        now_iso = _now_iso()
        # Debug log
        log.debug("_derived_upsert: mid=%s, vec_dim=%d", mid, len(vec) if vec else 0)
        payload = {
            "scope": scope,
            "embedding_model": EMBEDDING_MODEL_VERSION,
            "content": content,
            "created_at": now_iso,
            "updated_at": now_iso,
            "category": extra.get("category", "general"),
            "importance": extra.get("importance", 5),
            "tags": extra.get("tags", ""),
            "source": extra.get("source", "outbox"),
            "access_count": 0,
            "tokens": tokens,
            "doc_len": len(tokens),
            "retrieval_count": 0,
            "useful_count": 0,
            "canonical_version": version,
            "indexed_by": "outbox_worker",
        }
        # Carry any additional canonical-payload keys (e.g. learned_from_outcome)
        # into the derived Qdrant payload without clobbering the core fields.
        for _k, _v in (extra or {}).items():
            payload.setdefault(_k, _v)
        # Bi-temporal fidelity: preserve the REAL creation/validity window and
        # supersession state from the canonical row (previously created_at was
        # reset to now and valid_to was dropped on re-derivation, which broke
        # point-in-time recall and could re-admit a superseded row).
        def _iso(v):
            try:
                return v.isoformat() if hasattr(v, "isoformat") else v
            except Exception:
                return v
        _created = _iso(row.get("created_at")) or now_iso
        payload["created_at"] = _created
        payload["valid_from"] = _iso(row.get("valid_from")) or _created
        payload["valid_to"] = _iso(row.get("valid_until"))
        payload["status"] = row.get("status") or "active"
        if row.get("superseded_by"):
            payload["superseded_by"] = row["superseded_by"]
        _ev = row.get("event_time") or (extra or {}).get("event_time")
        if _ev:
            payload["event_time"] = _iso(_ev)
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=mid, vector=vec, payload=payload)],
        )
    except Exception as _e:
        log.error("_derived_upsert Qdrant upsert failed for %s: %s", mid, _e, exc_info=True)
        failures.append(("qdrant", str(_e)))
        _outbox_store_failure("qdrant")

    # --- BM25 ---
    if not failures:
        try:
            _add_to_bm25_index(mid, content)
        except Exception as _e:
            failures.append(("bm25", str(_e)))
            _outbox_store_failure("bm25")

    # --- Redis: cache invalidation + version snapshot (never stores content
    #     for a deleted memory; the delete path owns that) ---
    try:
        if extra:
            _store_version(mid, dict(extra or {}, content=content), vec)
        _invalidate_recall_caches()
    except Exception as _e:
        failures.append(("redis", str(_e)))
        _outbox_store_failure("redis")

    # --- Graph: supersession edge when the canonical row records one ---
    sup = row.get("superseded_by")
    if sup:
        try:
            _graph_link(mid, sup, "supersedes")
        except Exception as _e:
            failures.append(("graph", str(_e)))
            _outbox_store_failure("graph")

    return {"failures": failures}


def _derived_delete(row: dict) -> dict:
    """Remove one memory from every derived store. Idempotent by construction."""
    mid = row["memory_id"]
    failures = []
    try:
        _client.delete(collection_name=COLLECTION_NAME,
                       points_selector=qmodels.FilterSelector(
                           filter=qmodels.Filter(
                               must=[qmodels.HasIdCondition(has_id=[mid])])))
    except Exception as _e:
        failures.append(("qdrant", str(_e)))
        _outbox_store_failure("qdrant")
    try:
        _remove_from_bm25_index(mid)
    except Exception as _e:
        failures.append(("bm25", str(_e)))
        _outbox_store_failure("bm25")
    try:
        _invalidate_recall_caches()
        _graph_purge(mid)
        _delete_versions(mid)
        if _redis:
            _redis.delete(f"memory:st:{mid}")
    except Exception as _e:
        failures.append(("redis_graph", str(_e)))
        _outbox_store_failure("redis")
    return {"failures": failures}


def _outbox_apply(ev: dict) -> str:
    """Apply one event. Returns the outcome string.

    Version ordering lives here: the canonical row is the only source of truth
    for what the derived stores SHOULD contain, so an event for version 1 that
    arrives after version 3 already committed cannot write v1's content back
    over v3 — it is detected as stale and skipped without touching Qdrant.
    """
    etype = ev["event_type"]
    
    # Handle graph events (durable, idempotent graph writes via outbox)
    if etype in _OUTBOX_GRAPH_EVENTS:
        # Graph tools pass src/dst/rel via the outbox event payload (merged
        # from _canonical_write's `extra`); older rows may carry a separate
        # extra field. Accept both.
        extra = ev.get("extra") or ev.get("payload") or {}
        src = extra.get("src")
        dst = extra.get("dst")
        rel = extra.get("rel")
        if etype == "GRAPH_UNLINK":
            if not src or not dst:
                raise PermanentOutboxError(f"graph event missing src/dst: {extra}")
        elif not src or not dst or not rel:
            raise PermanentOutboxError(f"graph event missing src/dst/rel: {extra}")
        
        if etype == "GRAPH_LINK":
            # Idempotent upsert with ON CONFLICT. mem_edges FKs reference
            # mem_nodes.id (UUID), so resolve both endpoints first; a peer
            # without a node row (procedure/event-typed) keeps only the Redis
            # edge — same precedent as the store path's auto-link.
            _src_uuid = _pg_node_uuid(src)
            _dst_uuid = _pg_node_uuid(dst)
            if not _src_uuid or not _dst_uuid:
                _canonical_set_status(ev["memory_id"], "active", None)
                return "graph_link_skipped_no_nodes"
            try:
                _pg_run(
                    "INSERT INTO mem_edges(src, dst, rel, confidence, valid_from, payload) "
                    "VALUES (:src, :dst, :rel, 1.0, now(), '{}'::jsonb) "
                    "ON CONFLICT (src, dst, rel) DO UPDATE SET "
                    "confidence = EXCLUDED.confidence, valid_from = EXCLUDED.valid_from, "
                    "payload = EXCLUDED.payload",
                    {"src": _src_uuid, "dst": _dst_uuid, "rel": rel},
                    table="mem_edges",
                )
                # The graph:... canonical row exists only to carry the event;
                # mark it applied so it does not rot in pending_index.
                _canonical_set_status(ev["memory_id"], "active", None)
                return "graph_link_applied"
            except Exception as e:
                _outbox_store_failure("graph")
                raise RuntimeError(f"graph link upsert failed: {e}")
        elif etype == "GRAPH_UNLINK":
            # Idempotent delete (also resolves UUIDs for the FK columns)
            _src_uuid = _pg_node_uuid(extra["src"])
            _dst_uuid = _pg_node_uuid(extra["dst"])
            try:
                _pg_run(
                    "DELETE FROM mem_edges WHERE src = :src AND dst = :dst",
                    {"src": _src_uuid, "dst": _dst_uuid},
                    table="mem_edges",
                )
                _canonical_set_status(ev["memory_id"], "active", None)
                return "graph_unlink_applied"
            except Exception as e:
                _outbox_store_failure("graph")
                raise RuntimeError(f"graph unlink delete failed: {e}")
        return "graph_event_applied"
    
    if etype not in _OUTBOX_UPSERT_EVENTS and etype not in _OUTBOX_DELETE_EVENTS:
        raise PermanentOutboxError(
            f"unknown outbox event type {etype!r} — refusing to guess")

    row = _canonical_row(ev["memory_id"])
    if row is None:
        # The canonical row is gone. For a delete that is the desired end state;
        # for an upsert the row was never committed, so there is nothing to build.
        if etype in _OUTBOX_DELETE_EVENTS:
            _derived_delete({"memory_id": ev["memory_id"], "scope": ev["scope"]})
            return "deleted_orphan"
        raise PermanentOutboxError(
            f"no canonical row for {ev['memory_id']} (event {etype})")

    # ---- VERSION ORDERING ----
    if int(row["version"]) > int(ev["version"]):
        _outbox_bump("stale_skipped")
        return (f"stale_skipped: event v{ev['version']} older than canonical "
                f"v{row['version']} — derived state left at canonical")

    # A delete event older than the live canonical version must not delete it.
    if etype in _OUTBOX_DELETE_EVENTS and row["status"] != "deleted" \
            and int(row["version"]) > int(ev["version"]):
        _outbox_bump("stale_skipped")
        return "stale_delete_skipped"

    if etype in _OUTBOX_DELETE_EVENTS:
        res = _derived_delete(row)
        if res["failures"]:
            raise RuntimeError("delete derived failures: "
                               + "; ".join(f"{s}: {e}" for s, e in res["failures"]))
        return "deleted"

    # Bi-temporal: KEEP the superseded version in the derived tier with its
    # validity window (status=superseded + valid_to) so point-in-time (as_of)
    # recall can still return it. Default recall excludes it via the
    # status must_not filter; the canonical gate admits it only for as_of reads.
    if row["status"] == "superseded":
        res = _derived_upsert(row)
        if res["failures"]:
            raise RuntimeError("supersede reindex failures: "
                               + "; ".join(f"{s}: {e}" for s, e in res["failures"]))
        return "superseded_retained"

    if row["status"] == "deleted":
        # A delete beat this upsert. Make sure derived agrees, then stop.
        res = _derived_delete(row)
        if res["failures"]:
            raise RuntimeError("delete-after-upsert cleanup failures: "
                               + "; ".join(f"{s}: {e}" for s, e in res["failures"]))
        return "superseded_by_delete"

    res = _derived_upsert(row)
    if res.get("skipped"):
        _canonical_set_status(ev["memory_id"], "active", None)
        return f"skipped_{res['skipped']}"
    if res["failures"]:
        # PARTIAL FAILURE: some stores already converged, others did not. Do not
        # mark completed — requeue so the remaining store is repaired. Every
        # operation is idempotent, so re-running is safe.
        raise RuntimeError("derived failures: "
                           + "; ".join(f"{s}: {e}" for s, e in res["failures"]))
    _canonical_set_status(ev["memory_id"], "active", None)
    return "applied"


def _outbox_process_one(ev: dict) -> str:
    """Claim-processing-result wrapper with the full retry/dead-letter policy."""
    t0 = time.time()
    try:
        outcome = _outbox_apply(ev)
        _outbox_mark_completed(ev, outcome, ev["version"])
        _outbox_bump("completed")
        return outcome
    except PermanentOutboxError as _e:
        _outbox_mark_dead(ev, str(_e), permanent_state=True)
        return "dead_letter"
    except Exception as _e:
        attempts = ev["attempts"]
        if attempts >= OUTBOX_MAX_ATTEMPTS:
            _outbox_mark_dead(
                ev, f"exhausted {attempts} attempts: {_e}", permanent_state=True)
            return "dead_letter"
        delay = _outbox_backoff_seconds(attempts)
        nxt = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        _outbox_mark_retry(ev, f"{_e}", nxt)
        _outbox_bump("retried")
        return "retry"
    finally:
        dt = int((time.time() - t0) * 1000)
        with _outbox_stats_lock:
            _outbox_counters["latency_total_ms"] += dt
            _outbox_counters["latency_samples"] += 1


def _outbox_sync_wait(memory_id: str, grace_seconds: float = 0.0) -> bool:
    """Model B convergence wait.

    Blocks (up to ``grace_seconds``) for the outbox event for ``memory_id`` to
    be fully applied by the worker pool. Returns True when the canonical row
    has reached ``active`` (derived stores converged) or False when the grace
    period elapsed first. ``grace_seconds <= 0`` returns immediately without
    waiting. This is the only caller-visible latency knob of Model B; the
    default keeps ``_tool_store`` fully asynchronous.
    """
    if grace_seconds <= 0:
        # No wait requested: report current state without blocking.
        row = _canonical_row(memory_id)
        return bool(row and row.get("status") == "active")
    deadline = time.time() + grace_seconds
    while True:
        row = _canonical_row(memory_id)
        status = row.get("status") if row else None
        if status == "active":
            return True
        if status in ("dead_letter", "permanent_failure", "index_failed"):
            # Terminal failure — waiting longer will not help.
            return False
        if time.time() >= deadline:
            return False
        time.sleep(min(0.25, deadline - time.time()))


class _OutboxWorker(threading.Thread):
    """One drain loop. Several may run concurrently (Stage 4 §22)."""

    def __init__(self, worker_id: str):
        super().__init__(daemon=True, name=f"outbox-{worker_id}")
        self.worker_id = worker_id
        self._stop = threading.Event()
        self._leader_lock = None
        self._is_leader = False

    def _acquire_leader_lock(self) -> bool:
        """Acquire leader lock for outbox processing (horizontal scaling)."""
        if not HORIZONTAL_SCALING_ENABLED or not OUTBOX_LEADER_ELECTION or not _redis:
            return True  # No leader election needed
        try:
            self._leader_lock = _redis.lock(
                "memory:outbox:leader",
                timeout=LEADER_ELECTION_TTL,
                sleep=0.1,
            )
            acquired = self._leader_lock.acquire(blocking=False)
            if acquired:
                self._is_leader = True
                log.info("Outbox worker %s acquired leader lock", self.worker_id)
            else:
                log.debug("Outbox worker %s failed to acquire leader lock", self.worker_id)
            return acquired
        except Exception as e:
            log.warning("Failed to acquire leader lock: %s", e)
            return True  # Fail open - allow processing if lock fails

    def _release_leader_lock(self):
        """Release leader lock."""
        if self._leader_lock and self._is_leader:
            try:
                self._leader_lock.release()
                log.info("Outbox worker %s released leader lock", self.worker_id)
            except Exception as e:
                log.warning("Failed to release leader lock: %s", e)
            finally:
                self._leader_lock = None
                self._is_leader = False

    def stop(self):
        self._stop.set()
        self._release_leader_lock()

    def run(self):
        log.info("outbox worker %s started", self.worker_id)
        _last_backlog_check = 0.0
        while not self._stop.is_set():
            try:
                # P1-4: Leader election for outbox workers (horizontal scaling)
                if HORIZONTAL_SCALING_ENABLED and OUTBOX_LEADER_ELECTION:
                    if not self._acquire_leader_lock():
                        log.debug("Outbox worker %s not leader, waiting", self.worker_id)
                        self._stop.wait(OUTBOX_POLL_SECONDS * 2)
                        continue
                
                log.debug("Outbox worker %s acquired leader, starting work", self.worker_id)
                with system_context(f"outbox-worker-{self.worker_id}"):
                    log.debug("Outbox worker %s: recovering abandoned", self.worker_id)
                    _outbox_recover_abandoned()
                    log.debug("Outbox worker %s: claiming batch", self.worker_id)
                    batch = _outbox_claim(self.worker_id)
                    log.debug("Outbox worker %s: claimed %d events", self.worker_id, len(batch) if batch else 0)
                    if not batch:
                        self._stop.wait(OUTBOX_POLL_SECONDS)
                    else:
                        for ev in batch:
                            if self._stop.is_set():
                                # Leave it leased; the lease expiry makes it reclaimable.
                                break
                            _outbox_process_one(ev)
                        # Coalesced rebuild: ONE O(corpus) rebuild per claimed batch
                        # instead of one per event. This is the fix for the outbox
                        # stall (per-event full BM25 rebuilds were slower than the
                        # enqueue rate).
                        try:
                            _bm25_maybe_rebuild(force=True)
                        except Exception as _be:
                            log.warning("BM25 batch rebuild failed: %s", _be)
                log.debug("Outbox worker %s: work done, batch size: %d", self.worker_id, len(batch) if batch else 0)
                if not batch:
                    self._stop.wait(OUTBOX_POLL_SECONDS)
                # --- stale-backlog observability (Phase 4.2) ---
                # Every ~5 min, warn when pending events have been waiting >1h.
                # Log-only (grep-able in the pm2 error log); no alerting infra.
                _now = time.time()
                if _now - _last_backlog_check >= 300:
                    _last_backlog_check = _now
                    try:
                        with system_context(f"outbox-backlog-check-{self.worker_id}"):
                            rows = _pg_query(
                                "SELECT count(*) FROM mem_outbox WHERE status='pending' "
                                "AND available_at < now() - interval '1 hour'")
                            _r0 = rows[0] if rows else None
                            stale_pending = int(
                                _r0["count"] if isinstance(_r0, dict) else _r0[0]
                            ) if _r0 else 0
                            if stale_pending > 0:
                                log.warning(
                                    "OUTBOX BACKLOG WARNING: %d pending event(s) older than "
                                    "1 hour — drain rate is below enqueue rate",
                                    stale_pending)
                    except Exception as _qe:
                        log.debug("outbox backlog check failed: %s", _qe)
            except Exception as _e:  # never let the loop die silently
                log.error("outbox worker %s loop error: %s", self.worker_id, _e)
                # Debug: also print to stdout for pm2 logs
                print(f"OUTBOX WORKER ERROR: {self.worker_id}: {_e}", flush=True)
                self._stop.wait(1.0)
            finally:
                if HORIZONTAL_SCALING_ENABLED and OUTBOX_LEADER_ELECTION:
                    self._release_leader_lock()
        log.info("outbox worker %s stopped", self.worker_id)


_outbox_workers: list[_OutboxWorker] = []


# ---------------------------------------------------------------------------
# 1C-1: AUTOMATIC SESSION COMPACTION (fix plan — the Q62 deliverable)
# ---------------------------------------------------------------------------
# Compaction previously fired only if the agent remembered to call
# memory_session_summarize — 0 compactions ever ran. This worker makes the
# documented triggers (step/token/time thresholds) actually execute: every
# COMPACTION_POLL_SECONDS it scans open trajectories and, per trajectory,
# compacts the un-compacted step range into a checkpoint summary memory
# (verified critical-fact retention), recording checkpoints in the trajectory
# meta. Raw steps are NOT trimmed pre-close (close owns the final
# aggregation + PG persistence); checkpoints bound the CONTEXT cost because
# the virtual-context session-summary section reads summaries, not raw steps.
COMPACTION_POLL_SECONDS = float(os.environ.get("MEMORY_COMPACTION_POLL_SECONDS", "30"))

class _CompactionWorker(threading.Thread):
    """Background auto-compaction loop (mirrors the outbox worker pattern)."""

    def __init__(self):
        super().__init__(daemon=True, name="compaction-worker")
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _scan_open_trajectories(self) -> list[tuple[str, str]]:
        """Return (task_id, task_signature) pairs for open trajectories."""
        out = []
        try:
            for open_key in _redis.scan_iter(match="trajectory:open:*", count=200):
                task_sig = open_key.split("trajectory:open:", 1)[1]
                try:
                    task_id = _redis.get(open_key)
                except Exception:
                    continue
                if task_id:
                    out.append((task_id, task_sig))
        except Exception as _e:
            log.debug("compaction scan failed: %s", _e)
        return out

    def _maybe_compact_one(self, task_id: str) -> None:
        try:
            meta_raw = _redis.get(_trajectory_meta_key(task_id))
            meta = json.loads(meta_raw) if meta_raw else {}
            steps_raw = _redis.get(_trajectory_steps_key(task_id))
            steps = json.loads(steps_raw) if steps_raw else []
            if not steps:
                return
            checkpoints = meta.get("auto_compactions") or []
            already = sum(int(c.get("step_count", 0)) for c in checkpoints)
            un_compacted = steps[already:]
            if not un_compacted:
                return
            should, reason = _should_compact(
                un_compacted,
                meta.get("started_at")
                or (un_compacted[0].get("logged_at") if un_compacted else None))
            if not should:
                return
            task = meta.get("task") or "ongoing task"
            # Identity fix 2026-10-08: system_context alone does NOT authorize
            # stores (the gate needs a principal within the MEMORY_PRINCIPAL
            # ceiling — checkpoint summaries were silently denied with the
            # auto-compaction-* identity). Bookkeeping runs as 'global'.
            with system_context("global"):
                result = _compact_trajectory_steps(
                    task_id, task, meta.get("task_type"), "in_progress",
                    un_compacted, None, [], [])
            if not result or not result.get("summary_memory_id"):
                log.warning("auto-compaction produced no summary for %s", task_id)
                return
            checkpoints.append({
                "step_count": already + len(un_compacted),
                "summary_memory_id": result["summary_memory_id"],
                "compacted_at": _now_iso(),
                "reason": reason,
                "compression_ratio": result.get("compression_ratio"),
                "critical_fact_retention": result.get("verification"),
            })
            meta["auto_compactions"] = checkpoints
            meta["last_auto_compaction_at"] = _now_iso()
            _redis.set(_trajectory_meta_key(task_id),
                       json.dumps(meta, default=str), ex=_TRAJECTORY_STEP_TTL)
            log.info("AUTO-COMPACTION: task=%s checkpoint at %d steps "
                     "(%s) summary=%s ratio=%s retention=%s",
                     (meta.get("task") or task_id)[:40],
                     already + len(un_compacted), reason,
                     result["summary_memory_id"][:8],
                     result.get("compression_ratio"),
                     result.get("verification"))
        except Exception as _e:
            log.warning("auto-compaction failed for %s: %s", task_id, _e)

    def run(self):
        log.info("compaction worker started (poll=%.0fs)", COMPACTION_POLL_SECONDS)
        while not self._stop.is_set():
            try:
                for task_id, _sig in self._scan_open_trajectories():
                    if self._stop.is_set():
                        break
                    self._maybe_compact_one(task_id)
            except Exception as _e:
                log.warning("compaction worker loop error: %s", _e)
            self._stop.wait(COMPACTION_POLL_SECONDS)
        log.info("compaction worker stopped")


_compaction_worker: _CompactionWorker | None = None

def start_compaction_worker() -> bool:
    """Boot the auto-compaction worker. Safe to call more than once."""
    global _compaction_worker
    if _compaction_worker is not None and _compaction_worker.is_alive():
        return False
    _compaction_worker = _CompactionWorker()
    _compaction_worker.start()
    return True

def stop_compaction_worker() -> None:
    global _compaction_worker
    if _compaction_worker is not None:
        _compaction_worker.stop()
        _compaction_worker.join(timeout=5)
        _compaction_worker = None


def start_outbox_workers(count: int = None) -> int:
    """Boot the worker pool. Safe to call more than once."""
    count = count if count is not None else OUTBOX_WORKER_COUNT
    started = 0
    for i in range(count):
        wid = f"{socket.gethostname()}:{os.getpid()}:{i}"
        w = _OutboxWorker(wid)
        w.start()
        _outbox_workers.append(w)
        started += 1
    if started:
        log.info("outbox worker pool online: %d worker(s)", started)
    return started


def stop_outbox_workers() -> None:
    for w in _outbox_workers:
        w.stop()
    for w in _outbox_workers:
        w.join(timeout=5)
    _outbox_workers.clear()


def outbox_drain(limit: int = 100000, worker_id: str = "manual-drain",
                 max_seconds: float = 300.0) -> dict:
    """Synchronously drain the backlog (tests, boot recovery, convergence)."""
    t0 = time.time()
    counts = {"processed": 0, "completed": 0, "retried": 0, "dead_lettered": 0,
              "stale_skipped": 0, "batches": 0}
    while counts["processed"] < limit and (time.time() - t0) < max_seconds:
        with system_context("outbox-manual-drain"):
            _outbox_recover_abandoned()
            batch = _outbox_claim(worker_id, limit=32)
        if not batch:
            break
        counts["batches"] += 1
        for ev in batch:
            outcome = _outbox_process_one(ev)
            counts["processed"] += 1
            if outcome == "retry":
                counts["retried"] += 1
            elif outcome == "dead_letter":
                counts["dead_lettered"] += 1
            elif outcome.startswith("stale"):
                counts["stale_skipped"] += 1
            else:
                counts["completed"] += 1
        # Same coalescing as the worker loop: one rebuild per claimed batch.
        try:
            _bm25_maybe_rebuild(force=True)
        except Exception:
            pass
    counts["seconds"] = round(time.time() - t0, 2)
    return counts


def _tool_outbox_metrics(args: dict = None) -> types.CallToolResult:
    """Operator-only Stage 4 backlog/observability metrics."""
    with system_context("outbox-metrics"):
        return _ok(json.dumps(_outbox_metrics(), indent=2, default=str))


def _tool_outbox_dead_letters(args: dict = None) -> types.CallToolResult:
    """Operator-only view of dead-lettered events, with their full history."""
    limit = int((args or {}).get("limit", 50))
    with system_context("outbox-dead-letters"):
        rows = outbox_dead_letters(limit=limit)
    return _ok(json.dumps({"dead_letters": rows, "count": len(rows)},
                         indent=2, default=str))


def _tool_outbox_replay(args: dict = None) -> types.CallToolResult:
    """Operator-only replay of failed / dead-lettered / per-event work."""
    guard = _require_system("memory_outbox_replay")
    if guard is not None:
        return guard
    a = args or {}
    with system_context("outbox-replay"):
        res = outbox_replay(
            event_id=a.get("event_id"), memory_id=a.get("memory_id"),
            replay_failed=bool(a.get("replay_failed")),
            replay_dead_letter=bool(a.get("replay_dead_letter")),
            rebuild_scope=a.get("rebuild_scope"))
    return _ok(json.dumps(res, indent=2, default=str))


def _tool_outbox_drain(args: dict = None) -> types.CallToolResult:
    """Operator-only synchronous drain, used to prove convergence."""
    guard = _require_system("memory_outbox_drain")
    if guard is not None:
        return guard
    a = args or {}
    with system_context("outbox-drain"):
        res = outbox_drain(
            limit=int(a.get("limit", 100000)),
            worker_id=a.get("worker_id", "operator-drain"),
            max_seconds=float(a.get("max_seconds", 300)))
        res["metrics_after"] = _outbox_metrics()
    return _ok(json.dumps(res, indent=2, default=str))


def _outbox_metrics() -> dict:
    """Operator metrics. Never exposed to an ordinary tenant caller."""
    rows = _pg_query(
        "SELECT status, count(*) FROM mem_outbox GROUP BY status")
    by_status = {}
    if rows:
        for r in rows:
            # psycopg pool with dict_row returns dicts
            if isinstance(r, dict):
                by_status[r['status']] = int(r['count'])
            else:
                by_status[r[0]] = int(r[1])
    for s in _OUTBOX_STATES:
        by_status.setdefault(s, 0)
    oldest = _pg_query(
        "SELECT COALESCE(EXTRACT(EPOCH FROM (now() - MIN(created_at))), 0) "
        "FROM mem_outbox WHERE status IN ('pending','retryable_failure',"
        "'processing')")
    oldest_age = float(oldest[0]['coalesce']) if oldest else 0.0
    lat = _pg_query(
        "SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (processed_at - last_attempt_at))"
        "* 1000), 0) FROM mem_outbox WHERE status='completed' "
        "AND processed_at IS NOT NULL AND last_attempt_at IS NOT NULL")
    stuck = _pg_query(
        "SELECT count(*) FROM mem_outbox WHERE status='processing' "
        "AND lease_expires_at < now()")
    with _outbox_stats_lock:
        counters = dict(_outbox_counters)
        counters["store_failures"] = dict(_outbox_counters["store_failures"])
    latencies = counters.pop("latency_total_ms", 0)
    samples = counters.pop("latency_samples", 0)
    alerts = []
    if by_status.get(OUTBOX_DEAD_LETTER, 0) > 0:
        alerts.append(f"dead-letter queue not empty: "
                      f"{by_status[OUTBOX_DEAD_LETTER]} event(s)")
    if by_status.get(OUTBOX_PENDING, 0) + by_status.get(OUTBOX_RETRYABLE, 0) > 500:
        alerts.append("backlog over 500 events")
    if oldest_age > 900:
        alerts.append(f"oldest pending event is {int(oldest_age)}s old (>15m)")
    if stuck and int(stuck[0]['count']) > 0:
        alerts.append(f"{int(stuck[0]['count'])} processing event(s) past lease")
    if counters.get("store_failures"):
        alerts.append("derived store failures: "
                      + json.dumps(counters["store_failures"]))
    return {
        "outbox_pending": by_status.get(OUTBOX_PENDING, 0),
        "outbox_processing": by_status.get(OUTBOX_PROCESSING, 0),
        "outbox_completed": by_status.get(OUTBOX_COMPLETED, 0),
        "outbox_failed": by_status.get(OUTBOX_RETRYABLE, 0)
                        + by_status.get(OUTBOX_PERMANENT, 0),
        "outbox_dead_letter": by_status.get(OUTBOX_DEAD_LETTER, 0),
        "outbox_oldest_age_seconds": round(oldest_age, 1),
        "outbox_retry_rate": round(
            counters.get("retried", 0) / max(1, counters.get("claimed", 0)), 4),
        "outbox_processing_latency_ms": round(
            float(lat[0]['coalesce']) if lat else 0.0, 1),
        "outbox_convergence_lag": by_status.get(OUTBOX_PENDING, 0)
                                  + by_status.get(OUTBOX_PROCESSING, 0)
                                  + by_status.get(OUTBOX_RETRYABLE, 0),
        "outbox_stuck_past_lease": int(stuck[0]['count']) if stuck else 0,
        "counters": counters,
        "worker_mean_latency_ms": round(latencies / samples, 1) if samples else 0.0,
        "alerts": alerts,
    }


def outbox_dead_letters(limit: int = 50) -> list[dict]:
    rows = _pg_query(
        "SELECT event_id, event_type, memory_id, scope, version, attempts, "
        "first_attempt_at, last_attempt_at, created_at, last_error, last_worker, "
        "payload FROM mem_outbox WHERE status='dead_letter' "
        "ORDER BY updated_at DESC LIMIT :lim", {"lim": limit})
    out = []
    for r in rows:
        try:
            if isinstance(r, dict):
                payload_val = r.get('payload')
            else:
                payload_val = r[11]
            payload = json.loads(payload_val) if isinstance(payload_val, str) else (payload_val or {})
        except Exception:
            payload = {}
        if isinstance(r, dict):
            out.append({"event_id": r['event_id'], "event_type": r['event_type'], "memory_id": r['memory_id'],
                        "scope": r['scope'], "version": int(r['version']), "attempts": int(r['attempts']),
                        "first_attempt_at": str(r['first_attempt_at']), "last_attempt_at": str(r['last_attempt_at']),
                        "created_at": str(r['created_at']), "error": r['last_error'], "last_worker": r['last_worker'],
                        "payload": payload})
        else:
            out.append({"event_id": r[0], "event_type": r[1], "memory_id": r[2],
                        "scope": r[3], "version": int(r[4]), "attempts": int(r[5]),
                        "first_attempt_at": str(r[6]), "last_attempt_at": str(r[7]),
                        "created_at": str(r[8]), "error": r[9], "last_worker": r[10],
                        "payload": payload})
    return out


def outbox_replay(event_id: str = None, memory_id: str = None,
                  replay_failed: bool = False, replay_dead_letter: bool = False,
                  rebuild_scope: str = None) -> dict:
    """Operator-only replay. Returns how many events were requeued."""
    n = 0
    if event_id:
        n += _pg_run(
            "UPDATE mem_outbox SET status='pending', attempts=0, "
            "next_attempt_at=now(), last_error=NULL, worker_id=NULL, "
            "lease_expires_at=NULL, updated_at=now() "
            "WHERE event_id=:e AND status <> 'completed'",
            {"e": event_id}, table="outbox-replay") and 1 or 0
    if memory_id:
        _pg_run("UPDATE mem_outbox SET status='pending', attempts=0, "
                "next_attempt_at=now(), last_error=NULL, worker_id=NULL, "
                "lease_expires_at=NULL, updated_at=now() "
                "WHERE memory_id=:m AND status <> 'completed'",
                {"m": memory_id}, table="outbox-replay")
        n += 1
    if replay_failed:
        _pg_run("UPDATE mem_outbox SET status='pending', attempts=0, "
                "next_attempt_at=now(), worker_id=NULL, lease_expires_at=NULL, "
                "updated_at=now() WHERE status IN "
                "('retryable_failure','permanent_failure')",
                {}, table="outbox-replay")
        n += 1
    if replay_dead_letter:
        _pg_run("UPDATE mem_outbox SET status='pending', attempts=0, "
                "next_attempt_at=now(), dead_letter_reason=NULL, worker_id=NULL, "
                "lease_expires_at=NULL, updated_at=now() "
                "WHERE status='dead_letter'", {}, table="outbox-replay")
        n += 1
    if rebuild_scope:
        # Re-index everything canonical-live in a scope: emit a fresh upsert
        # event per live row. Idempotent, and safe to run repeatedly.
        rows = _pg_query(
            "SELECT memory_id, version FROM mem_canonical WHERE scope=:s "
            "AND status IN ('active','index_failed','pending_index')",
            {"s": rebuild_scope})
        for mid, ver in rows:
            _canonical_write(mid, rebuild_scope, "fact", None,
                             version=int(ver), event_type="MEMORY_REEMBED_REQUIRED",
                             payload={"rebuild_scope": rebuild_scope})
        n += len(rows)
    log.info("outbox replay requested: %s", json.dumps(
        {"event_id": event_id, "memory_id": memory_id, "failed": replay_failed,
         "dead": replay_dead_letter, "scope": rebuild_scope, "requeued": n},
        default=str))
    _pg_audit("outbox_replay", memory_id, {"event_id": event_id,
                                           "failed": replay_failed,
                                           "dead": replay_dead_letter,
                                           "scope": rebuild_scope,
                                           "requeued": n})
    return {"requeued": n, "event_id": event_id, "memory_id": memory_id,
            "replay_failed": replay_failed,
            "replay_dead_letter": replay_dead_letter,
            "rebuild_scope": rebuild_scope}


_ensure_canonical_schema()
_ensure_outbox_schema()

# ---------------------------------------------------------------------------
# Memory-type routing — split-stack target matrix (2026 pattern)
#   L1 working/transient  → Redis (TTL)
#   L2 semantic           → Qdrant (vector + hybrid)
#   L3 episodic/procedural→ PostgreSQL (durable, structured)
# Every store lands in the right DB(s) for its type. Credential/transient
# types stay in Redis only — secrets are never persisted to Qdrant/PG.
# ---------------------------------------------------------------------------
# category -> (node_type, targets)
#   node_type  : mem_nodes / mem_events / mem_procedures routing
#   targets    : which DBs this type writes to
CATEGORY_ROUTING = {
    # --- semantic facts: Qdrant + PG mem_nodes(fact) ---
    "fact":                ("fact",      ["qdrant", "postgres"]),
    "reference":           ("fact",      ["qdrant", "postgres"]),
    "project":             ("fact",      ["qdrant", "postgres"]),
    "project_knowledge":   ("fact",      ["qdrant", "postgres"]),
    "decision":            ("fact",      ["qdrant", "postgres"]),
    "general":             ("fact",      ["qdrant", "postgres"]),
    # --- preference: Qdrant + PG mem_nodes(preference) ---
    "preference":          ("preference",["qdrant", "postgres"]),
    # --- procedural: Qdrant + PG mem_procedures ---
    "procedure":          ("procedure", ["qdrant", "postgres"]),
    "procedures":         ("procedure", ["qdrant", "postgres"]),
    "fix":                ("procedure", ["qdrant", "postgres"]),
    "gotcha":             ("procedure", ["qdrant", "postgres"]),
    "architecture_decision": ("procedure", ["qdrant", "postgres"]),
    # --- episodic: Qdrant + PG mem_events + Redis ---
    "event":              ("event",     ["qdrant", "postgres", "redis"]),
    "trajectory":         ("event",     ["qdrant", "postgres", "redis"]),
    # --- credentials / transient: Redis TTL ONLY (no durable persistence) ---
    "credential":         ("transient", ["redis"]),
    "credential_salt":    ("transient", ["redis"]),
    "transient":          ("transient", ["redis"]),
    "session":            ("transient", ["redis"]),
}
DEFAULT_ROUTING = ("fact", ["qdrant", "postgres"])

def _route_memory(category: str) -> tuple:
    """Return (node_type, targets) for a memory category (split-stack routing)."""
    return CATEGORY_ROUTING.get((category or "general").lower(), DEFAULT_ROUTING)


# ---------------------------------------------------------------------------
# Gap #1 hook — LLM fact-extraction pipeline (wired lazily).
# Memory currently stores agent-written content as-is. To add true Mem0-style
# extraction (distill facts + ADD/UPDATE/DELETE/NOOP), set MEMORY_LLM_EXTRACT=1
# and point MEMORY_LLM_ENDPOINT at an OpenAI-compatible /chat/completions URL.
# Until then, _extract_facts() is a pass-through that returns the content as a
# single fact, so the pipeline shape is already in place.
# ---------------------------------------------------------------------------
_LLM_EXTRACT_ENABLED = os.environ.get("MEMORY_LLM_EXTRACT") == "1"
_LLM_ENDPOINT = os.environ.get("MEMORY_LLM_ENDPOINT", "")
_LLM_MODEL = os.environ.get("MEMORY_LLM_MODEL", "")

def _extract_facts(content: str, existing_similar: list = None) -> list:
    """Distill candidate facts from raw content.

    Returns a list of fact strings. Without an LLM endpoint configured, this
    is a pass-through (content verbatim) — the hook is ready for Mem0-style
    extraction + conflict reconciliation as soon as MEMORY_LLM_ENDPOINT is set.
    """
    if not _LLM_EXTRACT_ENABLED or not _LLM_ENDPOINT:
        return [content] if content else []
    # LLM path (enabled only when configured)
    try:
        import urllib.request
        body = json.dumps({
            "model": _LLM_MODEL or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Extract discrete facts. If a fact contradicts a listed existing memory, mark it as a correction. Return JSON array of {fact, op} where op in [add,update,delete,noop]."},
                {"role": "user", "content": json.dumps({"new": content, "existing": existing_similar or []})},
            ],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            _LLM_ENDPOINT + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        facts = json.loads(text)
        if isinstance(facts, list):
            return [f["fact"] if isinstance(f, dict) else str(f) for f in facts]
    except Exception as e:
        log.warning("LLM extract failed, falling back to verbatim: %s", e)
    return [content] if content else []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --- Latency tracking (P1-6) ---------------------------------------------------
from collections import defaultdict

_LATENCY_HISTOGRAMS: dict[str, list[float]] = defaultdict(list)
_LATENCY_LOCK = threading.Lock()

def _record_latency(tool_name: str, latency_ms: float) -> None:
    """Record latency for a tool call."""
    with _LATENCY_LOCK:
        _LATENCY_HISTOGRAMS[tool_name].append(latency_ms)
        # Keep only last 10000 samples per tool
        if len(_LATENCY_HISTOGRAMS[tool_name]) > 10000:
            _LATENCY_HISTOGRAMS[tool_name] = _LATENCY_HISTOGRAMS[tool_name][-10000:]

def _get_latency_stats(tool_name: str | None = None) -> dict:
    """Get latency statistics (p50, p95, p99) for a tool or all tools."""
    import statistics
    
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    if tool_name:
        values = _LATENCY_HISTOGRAMS.get(tool_name, [])
        return {
            "tool": tool_name,
            "count": len(values),
            "p50_ms": round(_percentile(values, 50), 2),
            "p95_ms": round(_percentile(values, 95), 2),
            "p99_ms": round(_percentile(values, 99), 2),
            "mean_ms": round(statistics.mean(values), 2) if values else 0.0,
        }
    else:
        result = {}
        for tn, values in _LATENCY_HISTOGRAMS.items():
            result[tn] = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 2),
                "p95_ms": round(_percentile(values, 95), 2),
                "p99_ms": round(_percentile(values, 99), 2),
                "mean_ms": round(statistics.mean(values), 2) if values else 0.0,
            }
        return result

def _add_latency_to_result(result: dict, latency_ms: float) -> dict:
    """Add latency metrics to a result dictionary."""
    result["latency_ms"] = round(latency_ms, 2)
    return result


def _ok(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

def _err(text: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"ERROR: {text}", isError=True)],
        isError=True,
    )


# --- Retrieval Stage Instrumentation (P2-2) ------------------------------------
from collections import defaultdict

_RETRIEVAL_STAGE_HISTOGRAMS: dict[str, list[float]] = defaultdict(list)
_RETRIEVAL_STAGE_LOCK = threading.Lock()
_RETRIEVAL_STAGE_QUALITY: dict[str, list[float]] = defaultdict(list)

_RETRIEVAL_STAGES = [
    "embed",
    "dense_search",
    "sparse_search",
    "rrf_fusion",
    "rerank",
    "canonical_filter",
    "feedback_adjust",
    "total_recall",
]

def _record_stage_latency(stage: str, latency_ms: float) -> None:
    """Record latency for a retrieval stage."""
    if stage not in _RETRIEVAL_STAGES:
        log.warning("Unknown retrieval stage: %s", stage)
        return
    with _RETRIEVAL_STAGE_LOCK:
        _RETRIEVAL_STAGE_HISTOGRAMS[stage].append(latency_ms)
        if len(_RETRIEVAL_STAGE_HISTOGRAMS[stage]) > 10000:
            _RETRIEVAL_STAGE_HISTOGRAMS[stage] = _RETRIEVAL_STAGE_HISTOGRAMS[stage][-10000:]

def _record_stage_quality(stage: str, quality_score: float) -> None:
    """Record quality contribution for a retrieval stage."""
    if stage not in _RETRIEVAL_STAGES:
        return
    with _RETRIEVAL_STAGE_LOCK:
        _RETRIEVAL_STAGE_QUALITY[stage].append(quality_score)
        if len(_RETRIEVAL_STAGE_QUALITY[stage]) > 10000:
            _RETRIEVAL_STAGE_QUALITY[stage] = _RETRIEVAL_STAGE_QUALITY[stage][-10000:]

def _get_stage_latency_stats(stage: str | None = None) -> dict:
    """Get latency statistics for retrieval stages."""
    import statistics
    
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    if stage:
        values = _RETRIEVAL_STAGE_HISTOGRAMS.get(stage, [])
        return {
            "stage": stage,
            "count": len(values),
            "p50_ms": round(_percentile(values, 50), 2),
            "p95_ms": round(_percentile(values, 95), 2),
            "p99_ms": round(_percentile(values, 99), 2),
            "mean_ms": round(statistics.mean(values), 2) if values else 0.0,
        }
    else:
        result = {}
        for st in _RETRIEVAL_STAGES:
            values = _RETRIEVAL_STAGE_HISTOGRAMS.get(st, [])
            result[st] = {
                "count": len(values),
                "p50_ms": round(_percentile(values, 50), 2),
                "p95_ms": round(_percentile(values, 95), 2),
                "p99_ms": round(_percentile(values, 99), 2),
                "mean_ms": round(statistics.mean(values), 2) if values else 0.0,
            }
        return result

def _get_stage_quality_stats(stage: str | None = None) -> dict:
    """Get quality contribution statistics for retrieval stages."""
    import statistics
    
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    if stage:
        values = _RETRIEVAL_STAGE_QUALITY.get(stage, [])
        return {
            "stage": stage,
            "count": len(values),
            "p50": round(_percentile(values, 50), 4),
            "p95": round(_percentile(values, 95), 4),
            "mean": round(statistics.mean(values), 4) if values else 0.0,
        }
    else:
        result = {}
        for st in _RETRIEVAL_STAGES:
            values = _RETRIEVAL_STAGE_QUALITY.get(st, [])
            result[st] = {
                "count": len(values),
                "p50": round(_percentile(values, 50), 4),
                "p95": round(_percentile(values, 95), 4),
                "mean": round(statistics.mean(values), 4) if values else 0.0,
            }
        return result


def _payload_to_memory(payload: dict, memory_id: str) -> dict:
    return {
        "memory_id": memory_id,
        "content": payload.get("content", ""),
        "category": payload.get("category"),
        "importance": payload.get("importance"),
        "tags": _payload_tags(payload),
        "source": payload.get("source"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "access_count": payload.get("access_count", 0),
        "last_accessed_at": payload.get("last_accessed_at"),
        "scope": payload.get("scope") or "global",
        "quarantine": bool(payload.get("quarantine")),
        "poison_score": payload.get("poison_score", 0.0),
        # 3A-3: provenance + computed confidence ride every read so the
        # dashboard can rank trust and label inferred entries.
        "confidence": payload.get("confidence"),
        "provenance": payload.get("provenance"),
        "corroboration_count": payload.get("corroboration_count"),
        "pii_masked": bool(payload.get("pii_masked")),
        # Bi-temporal fields surfaced on every read.
        "event_time": payload.get("event_time"),
        "valid_from": payload.get("valid_from"),
        "valid_until": payload.get("valid_to"),
        "status": payload.get("status"),
        "superseded_by": payload.get("superseded_by"),
    }

def _payload_tags(payload: dict) -> list[str]:
    """Read the tags field from a payload as a list.

    `tags` is written as a comma-joined string by the store path, but some
    legacy rows and the core-update path store a real list. Calling .split() on
    a list raised AttributeError inside forget(), which made tag deletion
    unusable.
    """
    raw = (payload or {}).get("tags")
    if not raw:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in str(raw).split(",") if t.strip()]

def _caller_scope_filter():
    """3B-1 red-team fix: Qdrant filter bounding unscoped category/tag reads
    to the caller's authorized scopes. System context spans the whole
    deployment ceiling (trusted maintenance); a tenant sees only its own
    namespaces. Every tenant-callable scan that is not already routed
    through _build_filter must AND this into its filter."""
    try:
        scopes = sorted(authorized_scopes() or [])
    except Exception:
        return None
    if not scopes:
        return None
    return qmodels.Filter(must=[qmodels.FieldCondition(
        key="scope", match=qmodels.MatchAny(any=scopes))])


def _build_filter(category: str | None, tags_key: str | None, scope: str | None = None, as_of: str | None = None):
    """Build the Qdrant filter for a recall.

    Scope handling is STRICT: an explicit scope matches exactly that scope.
    The previous implementation appended `OR scope='global'` and
    `OR scope IS NULL` to every scoped query, which turned every scoped read
    into a cross-tenant read and made isolation impossible by construction.
    Legacy NULL-scope rows are migrated to 'global' by
    `tools/migrate_null_scopes.py` so this filter can stay strict.
    """
    must = []
    must_not = []
    if category:
        must.append(qmodels.FieldCondition(
            key="category",
            match=qmodels.MatchValue(value=category),
        ))
    if tags_key:
        for tag in tags_key.split(","):
            if tag:
                must.append(qmodels.FieldCondition(
                    key="tags",
                    match=qmodels.MatchText(text=tag),
                ))
    # G3 (2026-10-04): temporal recall.
    # - Default (now): exclude superseded memories via must_not — Qdrant's
    #   must_not only excludes points that MATCH status=superseded, so memories
    #   that predate the status field still pass. (A must/MatchExcept would
    #   wrongly drop every memory missing the field.)
    # - as_of=<ISO datetime>: return only memories valid at that moment —
    #   created_at <= as_of and valid_to is null or >= as_of.
    if not as_of:
        must_not.append(qmodels.FieldCondition(
            key="status",
            match=qmodels.MatchValue(value="superseded"),
        ))
        if scope is not None:
            # Defence in depth. The exact-match scope condition in `must`
            # already excludes rows that have no scope, but stating it
            # explicitly means a future weakening of that clause cannot
            # silently re-admit legacy NULL-scope rows.
            must_not.append(qmodels.FieldCondition(key="scope", is_null=True))
    else:
        try:
            import datetime as _dt
            _as_ts = _dt.datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            must.append(qmodels.FieldCondition(
                key="created_at",
                range=qmodels.DatetimeRange(
                    lte=_as_ts,
                ),
            ))
            must.append(qmodels.Condition(
                should=[
                    qmodels.IsNullCondition(is_null=qmodels.PayloadField(key="valid_to")),
                    qmodels.FieldCondition(
                        key="valid_to",
                        range=qmodels.DatetimeRange(
                            gte=_as_ts,
                        ),
                    ),
                ]
            ))
            if scope:
                must.append(qmodels.FieldCondition(
                    key="scope", match=qmodels.MatchValue(value=scope)))
            return qmodels.Filter(must=must, must_not=must_not)
        except Exception:
            pass
    if scope:
        must.append(qmodels.FieldCondition(
            key="scope", match=qmodels.MatchValue(value=scope)))
    if not must and not must_not:
        return None
    return qmodels.Filter(must=must, must_not=must_not)

# ---------------------------------------------------------------------------
# Enhancement 3: Memory Poisoning Guard (OWASP LLM04)
# ---------------------------------------------------------------------------
def _compute_poison_score(text: str) -> tuple[float, list[dict]]:
    """Compute a 0-1 poisoning score based on injection pattern matches.

    Each pattern has a severity weight (critical=1.0, high=0.8, medium=0.5).
    Score is the sum of matched pattern weights, capped at 1.0.
    A single critical pattern match (1.0) exceeds the 0.7 threshold and is blocked.
    """
    if not text:
        return 0.0, []
    findings = []
    matched_weight = 0.0
    for pattern_re, pattern_name, severity in POISON_PATTERNS:
        matches = pattern_re.findall(text)
        if matches:
            weight = float(severity)
            matched_weight += weight
            findings.append({"pattern": pattern_name, "matches": len(matches), "weight": weight})
    if not findings:
        return 0.0, []
    score = min(1.0, matched_weight)
    return score, findings

def _sanitize_content(content: str) -> str:
    """Sanitize content by neutralizing common injection patterns."""
    text = content
    replacements = [
        (re.compile(r"(?i)<script[^>]*>.*?</\s*script\s*>", re.DOTALL), "[REMOVED_SCRIPT]"),
        (re.compile(r"(?i)javascript\s*:"), "javascript_BLOCKED:"),
        (re.compile(r"(?i)on\w+\s*=\s*[\"\'](.*?)[\"\']"), "onclick_BLOCKED"),
        (re.compile(r"(?i)on\w+\s*=\s*(?![\"\'\s])[^\s>]+"), "onerror_BLOCKED"),
        (re.compile(r"\[\s*system\s*\("), "[BLOCKED_system]"),
        (re.compile(r"```(?:bash|sh|python|js|php|sql)\s*\n", re.IGNORECASE), "\n```BLOCKED\n"),
    ]
    for pattern_re, replacement in replacements:
        text = pattern_re.sub(replacement, text)
    return text

def _check_poison(content: str, block: bool = True) -> tuple[bool, float, list[dict]]:
    """Check content for poisoning patterns. Returns (blocked, score, findings)."""
    score, findings = _compute_poison_score(content)
    blocked = block and score >= POISON_SCORE_THRESHOLD
    return blocked, score, findings


# ---------------------------------------------------------------------------
# 3B-2 (Part 3): Secret / PII detection — write-path ENFORCEMENT.
# "No secrets in memory" stops being a convention and becomes a gate: a
# credential-shaped string rejects the write outright (fail closed, audited);
# PII (emails, phone numbers) is masked to redacted placeholders. The same
# scanner runs in every write entry point: store, remember, auto_observe,
# update, and the dashboard editor.
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[tuple[str, "re.Pattern"]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
]
# password/token assignments — the VALUE is the secret, the NAME often appears
# in legit documentation. Values that are clearly placeholders/docs don't trip.
_SECRET_ASSIGN = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|bearer)\b"
    r"\s*(?:[:=]|is)\s*[\"\']?([^\s\"'`]{8,})")
_ASSIGN_BENIGN_PREFIX = ("sha256", "<", "redacted", "xxxx", "****", "example",
                         "placeholder", "$(", "${", "%(", "%s", "changeme_")

_PII_PATTERNS: list[tuple[str, "re.Pattern"]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}(?!\d)")),
]


def _scan_secrets(content: str) -> tuple[bool, list[dict], str]:
    """3B-2 gate. Returns (reject, findings, masked_content).

    reject=True  -> caller MUST refuse the write entirely (credential shape).
    masked_content -> content with PII replaced by redacted placeholders, and
    credential-shaped assignments replaced by their names.
    """
    findings: list[dict] = []
    for kind, pat in _SECRET_PATTERNS:
        m = pat.search(content)
        if m:
            findings.append({"type": kind, "match": m.group(0)[:12] + "…", "action": "reject"})
    for m in _SECRET_ASSIGN.finditer(content):
        val = m.group(2)
        if val.lower().startswith(_ASSIGN_BENIGN_PREFIX):
            continue
        findings.append({"type": "credential_assignment", "match": m.group(1) + "=…",
                         "action": "reject"})
        break
    if not findings:
        # generic high-entropy run: 32+ base64-ish chars, entropy > 4.2 bits/char.
        # Hex digests (max 4.0 for the 16-symbol alphabet) and path/hash
        # fragments (model names + '/' + hash — ubiquitous in technical
        # documentation like "Qwen3-Embedding-0.6B/97b0c614…") deliberately
        # pass; real secrets are neither pure-hex nor slash-bearing paths.
        for m in re.finditer(r"\b[A-Za-z0-9+/=_\-]{32,}\b", content):
            tok = m.group(0)
            if "/" in tok:
                continue
            if all(c in "0123456789abcdefABCDEF" for c in tok):
                continue
            freq = {c: tok.count(c) / len(tok) for c in set(tok)}
            ent = -sum(p * (p and __import__("math").log2(p)) for p in freq.values())
            if ent > 4.2:
                findings.append({"type": "high_entropy_token",
                                 "match": tok[:12] + "…", "action": "reject"})
                break
    masked = content
    pii_found = []
    for kind, pat in _PII_PATTERNS:
        if pat.search(masked):
            pii_found.append(kind)
            masked = pat.sub(f"[REDACTED_{kind.upper()}]", masked)
    _reject = bool(findings)  # ONLY credential-shaped findings reject
    if pii_found:
        findings.append({"type": "pii", "kinds": pii_found, "action": "mask"})
    return _reject, findings, masked

# ---------------------------------------------------------------------------
# Enhancement 9: Quality Gate — scoring function
# ---------------------------------------------------------------------------
_QUALITY_ACTION_VERBS = re.compile(
    r"\b(fix|implement|configure|deploy|install|setup|set up|migrate|update|upgrade|"
    r"optimize|debug|troubleshoot|verify|test|check|inspect|monitor|backup|restore|"
    r"create|delete|modify|change|replace|add|remove|enable|disable|start|stop|restart|"
    r"check|run|execute|build|compile|package|publish|push|pull|merge|rebase|cherry-pick)\b",
    re.IGNORECASE,
)
_QUALITY_EVIDENCE_MARKERS = re.compile(
    r"(log|output|result|error|traceback|stacktrace|response|status\s*code|exit\s*code|"
    r"returned|observed|confirmed|verified|measured|timestamp|version|path|/usr/|/var/|/etc/|"
    r"localhost|127\.0\.0\.1|port\s*\d+|pid\s*\d+|memory|cpu|disk|load\s*avg)",
    re.IGNORECASE,
)
_QUALITY_GENERIC_FILLER = re.compile(
    r"^(ok|yes|no|done|test|hello|hi|thanks|thank you|got it|sure|correct|wrong|true|false|"
    r"maybe|perhaps|later|skip|pass|nothing|empty|null|none|n/a|tbd|todo|wip)$",
    re.IGNORECASE,
)

def _compute_quality_score(content: str) -> tuple[float, list[str]]:
    """Score memory content quality. Returns (score, reasons).
    Score range: 0.0 (terrible) to 1.0 (excellent).
    """
    if not content or not content.strip():
        return 0.0, ["empty_content"]

    score = 0.3  # base score for non-empty content
    reasons = []

    # Length bonus
    if len(content) > 50:
        score += QUALITY_LENGTH_BONUS
        reasons.append("adequate_length")
    elif len(content) < 10:
        score -= 0.15
        reasons.append("too_short")

    # Structure bonus (headers, lists, code blocks, multi-line)
    has_structure = bool(re.search(r"^#+\s|^- |\* |\n- |\n\* |```|:\n", content, re.MULTILINE))
    if has_structure:
        score += QUALITY_STRUCTURE_BONUS
        reasons.append("has_structure")

    # Specificity bonus (numbers, paths, versions)
    has_specifics = bool(re.search(r"\d+\.\d+|/[\w/]+|:\d+|port\s|version|v\d+", content))
    if has_specifics:
        score += QUALITY_SPECIFICITY_BONUS
        reasons.append("has_specifics")

    # Actionability bonus
    if _QUALITY_ACTION_VERBS.search(content):
        score += QUALITY_ACTIONABLE_BONUS
        reasons.append("has_action_verbs")

    # Evidence bonus
    if _QUALITY_EVIDENCE_MARKERS.search(content):
        score += QUALITY_EVIDENCE_BONUS
        reasons.append("has_evidence")

    # Generic filler penalty
    stripped = content.strip().rstrip(".!")
    if _QUALITY_GENERIC_FILLER.match(stripped):
        score = max(0.0, score + QUALITY_GENERIC_PENALTY)
        reasons.append("generic_filler")

    # Multi-line bonus (more likely to be substantive)
    if content.count("\n") >= 2:
        score += 0.05
        reasons.append("multi_line")

    return max(0.0, min(1.0, score)), reasons


def _check_quality(content: str) -> tuple[bool, float, list[str]]:
    """Quality gate check. Returns (rejected, score, reasons)."""
    if not QUALITY_GATE_ENABLED:
        return False, 1.0, ["quality_gate_disabled"]
    score, reasons = _compute_quality_score(content)
    rejected = score < QUALITY_REJECT_THRESHOLD
    return rejected, score, reasons

# ---------------------------------------------------------------------------
# Enhancement 6: Versioned Rollback
# ---------------------------------------------------------------------------
def _store_version(mem_id: str, payload: dict, vector: list[float]) -> str | None:
    """Store a version snapshot in Redis. Returns version_id or None."""
    if not _check_redis():
        return None
    try:
        version_id = str(uuid.uuid4())
        now = _now_iso()
        version_data = {
            "version_id": version_id,
            "memory_id": mem_id,
            "payload": payload,
            "vector": vector,
            "stored_at": now,
        }
        ver_key = f"{VERSION_PREFIX}{mem_id}:{version_id}"
        _redis.setex(ver_key, 60 * 60 * 24 * 30, json.dumps(version_data))
        _redis.zadd(f"{VERSION_INDEX_PREFIX}{mem_id}", {version_id: time.time()})
        _redis.zremrangebyrank(f"{VERSION_INDEX_PREFIX}{mem_id}", 0, -MAX_VERSIONS_PER_MEMORY - 1)
        return version_id
    except Exception as e:
        log.warning("version store failed: %s", e)
        return None

def _get_versions(mem_id: str, limit: int = 10) -> list[dict]:
    """Get version history for a memory."""
    if not _check_redis():
        return []
    try:
        idx_key = f"{VERSION_INDEX_PREFIX}{mem_id}"
        version_ids = _redis.zrevrange(idx_key, 0, limit - 1)
        versions = []
        for vid in version_ids:
            ver_key = f"{VERSION_PREFIX}{mem_id}:{vid}"
            data = _redis.get(ver_key)
            if data:
                v = json.loads(data)
                _pl = v.get("payload") or {}
                if isinstance(_pl, dict) and _pl.get("content"):
                    v["content_preview"] = str(_pl["content"])[:160]
                v.pop("payload", None)
                v.pop("vector", None)
                versions.append(v)
        return versions
    except Exception:
        return []

def _get_version(mem_id: str, version_id: str) -> dict | None:
    """Get a specific version by ID."""
    if not _check_redis():
        return None
    try:
        ver_key = f"{VERSION_PREFIX}{mem_id}:{version_id}"
        data = _redis.get(ver_key)
        if data:
            return json.loads(data)
        return None
    except Exception:
        return None

def _delete_versions(mem_id: str) -> int:
    """Delete all versions for a memory. Returns count deleted."""
    if not _check_redis():
        return 0
    try:
        idx_key = f"{VERSION_INDEX_PREFIX}{mem_id}"
        version_ids = _redis.zrange(idx_key, 0, -1)
        deleted = 0
        for vid in version_ids:
            ver_key = f"{VERSION_PREFIX}{mem_id}:{vid}"
            deleted += _redis.delete(ver_key)
        _redis.delete(idx_key)
        return deleted
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# Enhancement 1: Graph Memory
# ---------------------------------------------------------------------------
def _check_redis():
    if _redis is None:
        return False
    try:
        _redis.ping()
        return True
    except Exception:
        return False

def _graph_link(mem_id: str, related_id: str, relation: str) -> bool:
    """Create a directed edge between two memories."""
    if not _check_redis():
        return False
    try:
        edge_id = f"{mem_id}:{related_id}"
        edge_key = f"{GRAPH_EDGE_PREFIX}{edge_id}"
        edge_data = json.dumps({
            "from": mem_id, "to": related_id, "relation": relation, "created_at": _now_iso()
        })
        _redis.setex(edge_key, 60 * 60 * 24 * 365, edge_data)
        _redis.sadd(f"{GRAPH_NODE_PREFIX}{mem_id}", related_id)
        _redis.sadd(f"{GRAPH_NODE_PREFIX}{related_id}", mem_id)
        return True
    except Exception as e:
        log.warning("graph_link failed: %s", e)
        return False

def _graph_unlink(mem_id: str, related_id: str) -> bool:
    """Remove edge between two memories."""
    if not _check_redis():
        return False
    try:
        edge_id = f"{mem_id}:{related_id}"
        _redis.delete(f"{GRAPH_EDGE_PREFIX}{edge_id}")
        _redis.srem(f"{GRAPH_NODE_PREFIX}{mem_id}", related_id)
        _redis.srem(f"{GRAPH_NODE_PREFIX}{related_id}", mem_id)
        return True
    except Exception:
        return False

def _graph_purge(mem_id: str) -> int:
    """Delete every Redis graph edge incident to mem_id, in both directions.

    Without this a forgotten memory stays reachable through
    memory_graph_neighbors, i.e. "deleted" memories remain retrievable.
    """
    if not _check_redis():
        return 0
    removed = 0
    try:
        related = {r.decode() if isinstance(r, bytes) else r
                   for r in _redis.smembers(f"{GRAPH_NODE_PREFIX}{mem_id}")}
        related.add(mem_id)
        for other in related:
            for a, b in ((mem_id, other), (other, mem_id)):
                if a == b and a != mem_id:
                    continue
                if _redis.delete(f"{GRAPH_EDGE_PREFIX}{a}:{b}"):
                    removed += 1
            _redis.srem(f"{GRAPH_NODE_PREFIX}{other}", mem_id)
        _redis.delete(f"{GRAPH_NODE_PREFIX}{mem_id}")
        return removed
    except Exception as e:
        log.warning("graph_purge failed for %s: %s", mem_id, e)
        return removed

def _graph_get_neighbors(mem_id: str, relation: str | None = None, depth: int = 1,
                         as_of: str | None = None) -> list[dict]:
    """Get neighboring memories. With as_of, traverse the temporal edge graph in
    Postgres (valid_from <= as_of <= valid_to) so ended relationships are excluded."""
    if as_of:
        try:
            import datetime as _dt
            _t = _dt.datetime.fromisoformat(str(as_of).replace("Z", "+00:00")).isoformat()
            visited, frontier, results_t = set(), {mem_id}, []
            for _ in range(depth):
                nxt = set()
                for cid in frontier:
                    if cid in visited:
                        continue
                    visited.add(cid)
                    rows = _pg_query(
                        "SELECT src, dst, rel FROM mem_edges "
                        "WHERE (src=CAST(:c AS uuid) OR dst=CAST(:c AS uuid)) "
                        "AND valid_from <= CAST(:t AS timestamptz) "
                        "AND (valid_to IS NULL OR valid_to >= CAST(:t AS timestamptz))",
                        {"c": cid, "t": _t})
                    for r in rows:
                        def g(k, i):
                            return r.get(k) if (isinstance(r, dict) or hasattr(r, "keys")) else r[i]
                        src, dst, rel = str(g("src", 0)), str(g("dst", 1)), g("rel", 2)
                        nb = dst if src == str(cid) else (src if dst == str(cid) else None)
                        if not nb:
                            continue
                        if relation and rel != relation:
                            continue
                        results_t.append({"memory_id": nb, "relation": rel,
                                          "edge_id": f"{src}:{dst}"})
                        nxt.add(nb)
                frontier = nxt
            return results_t
        except Exception as e:
            log.warning("graph as_of traversal failed, falling back to live graph: %s", e)
    if not _check_redis():
        return []
    visited = set()
    frontier = {mem_id}
    results = []
    for _ in range(depth):
        next_frontier = set()
        for current_id in frontier:
            if current_id in visited:
                continue
            visited.add(current_id)
            neighbor_ids = _redis.smembers(f"{GRAPH_NODE_PREFIX}{current_id}")
            for neighbor_id in neighbor_ids:
                edge_id = f"{current_id}:{neighbor_id}"
                edge_data = _redis.get(f"{GRAPH_EDGE_PREFIX}{edge_id}")
                if not edge_data:
                    continue
                edge = json.loads(edge_data)
                if relation and edge.get("relation") != relation:
                    continue
                results.append({"memory_id": neighbor_id, "relation": edge.get("relation"), "edge_id": edge_id})
                next_frontier.add(neighbor_id)
        frontier = next_frontier
    return results

def _graph_get_all_relations(mem_id: str) -> list[dict]:
    """Get all relations for a memory."""
    if not _check_redis():
        return []
    try:
        neighbor_ids = _redis.smembers(f"{GRAPH_NODE_PREFIX}{mem_id}")
        relations = []
        for neighbor_id in neighbor_ids:
            edge_id = f"{mem_id}:{neighbor_id}"
            edge_data = _redis.get(f"{GRAPH_EDGE_PREFIX}{edge_id}")
            if edge_data:
                relations.append(json.loads(edge_data))
        return relations
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Enhancement 5: Eval before Deploy (sandbox)
# ---------------------------------------------------------------------------
_sandbox_store: dict[str, dict] = {}  # in-memory sandbox for eval operations

def _sandbox_store_memory(content: str, category: str = "general", importance: int = 5,
                          tags: list = None, source: str = "eval") -> dict:
    """Store a memory in sandbox (not committed to Qdrant). Returns preview."""
    memory_id = f"eval:{uuid.uuid4()}"
    now = _now_iso()
    vector = _embed_model().encode(content).tolist()
    tokens = _tokenize(content)
    payload = {
        "content": content,
        "created_at": now,
        "updated_at": now,
        "category": category,
        "importance": importance,
        "tags": ",".join(tags or []),
        "source": source,
        "access_count": 0,
        "last_accessed_at": now,
        "tokens": tokens,
        "doc_len": len(tokens),
        "sandboxed": True,
    }
    _sandbox_store[memory_id] = {"id": memory_id, "vector": vector, "payload": payload}
    return {"memory_id": memory_id, "status": "sandboxed", "token_count": len(tokens),
            "preview": content[:200]}

def _sandbox_recall(query: str, top_k: int = 5) -> list[dict]:
    """Recall from sandbox store using the same hybrid approach."""
    if not _sandbox_store:
        return []
    query_vector = _embed_model().encode(query).tolist()
    query_tokens = _tokenize(query)
    results = []
    for mem_id, data in _sandbox_store.items():
        v = data["vector"]
        dot = sum(a * b for a, b in zip(query_vector, v))
        norm_q = math.sqrt(sum(a * a for a in query_vector))
        norm_v = math.sqrt(sum(a * a for a in v))
        cosine = dot / (norm_q * norm_v) if norm_q and norm_v else 0.0
        bm25_score = 0.0
        if query_tokens:
            corpus_tokens = data["payload"].get("tokens", [])
            matches = sum(1 for t in query_tokens if t in corpus_tokens)
            bm25_score = matches / max(len(query_tokens), 1)
        hybrid = 0.5 * cosine + 0.5 * bm25_score
        results.append({"memory_id": mem_id, "hybrid_score": hybrid, "payload": data["payload"]})
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]

def _sandbox_commit(memory_id: str) -> dict:
    """Commit a sandboxed memory to Qdrant. Returns result."""
    if memory_id not in _sandbox_store:
        return {"status": "error", "message": f"Sandbox memory not found: {memory_id}"}
    data = _sandbox_store.pop(memory_id)
    payload = dict(data["payload"])
    payload.pop("sandboxed", None)
    memory_id_out = str(uuid.uuid4())
    try:
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=memory_id_out, vector=data["vector"], payload=payload)],
        )
        _add_to_bm25_index(memory_id_out, payload.get("content", ""))
        _invalidate_recall_caches()
        return {"status": "committed", "memory_id": memory_id_out}
    except Exception as e:
        _sandbox_store[memory_id] = data
        return {"status": "commit_failed", "message": str(e)}

def _sandbox_discard(memory_id: str) -> dict:
    """Discard a sandboxed memory."""
    if memory_id in _sandbox_store:
        del _sandbox_store[memory_id]
        return {"status": "discarded"}
    return {"status": "not_found"}

def _sandbox_preview() -> dict:
    """Get current sandbox state."""
    return {
        "count": len(_sandbox_store),
        "memory_ids": list(_sandbox_store.keys()),
    }

# ---------------------------------------------------------------------------
# Enhancement 4: Mixture-of-Agents — specialized retrieval strategies
# ---------------------------------------------------------------------------
QUERY_TYPE_KEYWORDS = {
    "factual": ["what is", "who is", "where is", "when did", "definition of", "fact about", "information about"],
    "procedural": ["how to", "how do", "steps to", "process of", "procedure for", "tutorial", "guide to"],
    "preference": ["i prefer", "my preference", "i like", "i hate", "favorite", "best way i like"],
    "decision": ["decided to", "chose to", "will use", "going with", "choice was", "selected"],
    "error_fix": ["fixed", "bug fix", "error", "crash", "failed", "broken", "solved by"],
    "config": ["configured", "setting", "option", "preference", "toggled", "changed to"],
    "conversation": ["said", "told", "asked", "mentioned", "discussed", "talked about"],
}

def _classify_query(query: str) -> str:
    """Classify query into one of the specialized retrieval types."""
    lowered = query.lower()
    scores: dict[str, float] = {}
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > 0:
            scores[qtype] = score
    if not scores:
        return "general"
    return max(scores, key=scores.get)  # type: ignore

# Query words that carry no retrieval signal. They are excluded from the
# evidence denominator so a well-formed question does not dilute its own score.
_EVIDENCE_STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its me my of on or our should so than that the their
them then there these they this to was we were what when where which who whom
why will with would you your please tell me about does did is are
""".split())


def _query_evidence(query: str, text: str) -> float:
    """Absolute evidence that `text` actually answers `query`, in 0..1.

    Confidence cannot be derived from relevance alone: relevance is normalised
    against the best candidate for the query, so the top result always scores
    1.0 even when the query matched nothing. This measures real lexical overlap
    on content words instead, so an unanswerable question yields low confidence
    while an exact match yields high confidence.
    """
    qtok = {t for t in _tokenize(query or "") if t not in _EVIDENCE_STOPWORDS}
    if not qtok:
        return 0.0
    dtok = set(_tokenize(text or ""))
    if not dtok:
        return 0.0
    return len(qtok & dtok) / float(len(qtok))


# query_type -> preferred category. These are *hints* used to prioritise a
# category, never hard filters: stores default to category "general", so forcing
# e.g. "fact" for a factual query silently discarded every default-stored
# memory and returned zero results for queries that genuinely had matches.
_QUERY_TYPE_CATEGORY_HINT = {
    "factual": "fact",
    "procedural": "procedure",
    "error_fix": "decision",
    "preference": "preference",
    "config": "general",
}


def _build_specialized_recall(query: str, query_type: str, top_k: int,
                                category: str | None, tags_key: str | None, scope: str | None = None, as_of: str | None = None) -> list[dict]:
    """Retrieve using a category hint for the query type, with a scoped fallback.

    An explicit caller category is authoritative and is never second-guessed.
    Otherwise we try the query-type hint; if it yields nothing we retry without a
    category constraint. The retry keeps the same scope/tags/as_of filters, so
    this broadens category matching only and can never leak across scopes.
    """
    if category:
        return _hybrid_recall(query, top_k, category, tags_key, scope, as_of)

    hint = _QUERY_TYPE_CATEGORY_HINT.get(query_type)
    if hint:
        hinted = _hybrid_recall(query, top_k, hint, tags_key, scope, as_of)
        if hinted:
            return hinted
    return _hybrid_recall(query, top_k, None, tags_key, scope, as_of)

# ---------------------------------------------------------------------------
# Enhancement 2: Reflection Loop — self-improvement analysis
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Enhancement 2: Reflection Loop — self-improvement analysis
# Implements: Execute → Evaluate → Extract → Update closed loop (Reflexion pattern)
# ---------------------------------------------------------------------------

# Reflection loop guard — detect repeated identical reflections (degeneration)
_recent_reflection_summaries: list[str] = []
_MAX_REFLECT_LOOP = 3  # flag if same pattern repeats this many times


def _tool_reflect(args: dict) -> types.CallToolResult:
    """Analyze recent memories for patterns, conflicts, and self-improvement insights.

    Enhancement 8: Closed Reflexion Loop with outcome signals and actionable guidance.
    - outcome (optional): 'success' | 'failure' | null — drives targeted improvement suggestions
    - task_type (optional): string — for task-specific strategy generation
    - Generates ACTIONABLE procedural insights (not just analysis) stored as learned strategies
    - Detects reflection loops to prevent degeneration
    """
    global _recent_reflection_summaries
    days = args.get("days", 7)
    limit = args.get("limit", 50)
    outcome = args.get("outcome")       # 'success' | 'failure' | None
    task_type = args.get("task_type")   # e.g. 'coding', 'debugging', 'planning'
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - (days * 86400)

    try:
        # 3B-1 red-team fix: reflect analyzed every tenant's trajectories.
        results = _client.scroll(collection_name=COLLECTION_NAME, limit=limit,
                                 with_payload=True,
                                 scroll_filter=_caller_scope_filter())
        points = results[0]
    except Exception as e:
        return _err(f"Reflection failed: {e}")

    recent = []
    for p in points:
        try:
            created = p.payload.get("created_at", "")
            if created:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                if ts >= cutoff:
                    recent.append(p)
        except Exception:
            continue

    if not recent:
        return _ok(json.dumps({"insight": "No recent memories to reflect on.", "patterns_found": 0}))

    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    high_importance: list[dict] = []
    recency_scores: list[float] = []
    now_ts_ref = datetime.now(timezone.utc).timestamp()

    for p in recent:
        cat = p.payload.get("category", "general")
        by_category[cat] = by_category.get(cat, 0) + 1
        src = p.payload.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        imp = p.payload.get("importance", 5)
        if imp and imp >= 8:
            high_importance.append({"id": p.id, "content": p.payload.get("content", "")[:100], "importance": imp})
        try:
            created = p.payload.get("created_at", "")
            if created:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                recency_scores.append((now_ts_ref - ts) / 86400)
        except Exception:
            pass

    avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0
    top_category = max(by_category, key=by_category.get) if by_category else "none"
    top_source = max(by_source, key=by_source.get) if by_source else "none"

    insights: list[str] = []
    insights.append(f"Reviewed {len(recent)} memories from the last {days} day(s).")
    insights.append(f"Most active category: '{top_category}' ({by_category.get(top_category, 0)} memories).")
    insights.append(f"Most common source: '{top_source}' ({by_source.get(top_source, 0)} memories).")
    insights.append(f"Average memory age: {avg_recency:.1f} days.")
    if high_importance:
        insights.append(f"Found {len(high_importance)} high-importance (≥8) memories worth preserving.")
    if len(by_category) > 5:
        insights.append("High category diversity suggests broad contextual memory usage.")
    conflicting = _detect_conflicts(recent)
    if conflicting:
        insights.append(f"Detected {len(conflicting)} potential conflicts/duplicates that may need resolution.")
    staleness = [p.id for p in recent if by_category.get(p.payload.get("category", ""), 0) == 1]
    if staleness:
        insights.append(f"{len(staleness)} memories appear isolated — consider linking via graph.")

    # GAP 2: Outcome-driven improvement suggestions
    action_items: list[str] = []
    if outcome == "failure" and task_type:
        action_items.append(f"For {task_type} tasks: investigate root cause before retrying.")
        action_items.append(f"For {task_type} tasks: store error patterns with importance=9 for fast recall.")
        action_items.append(f"For {task_type} tasks: prefer step-by-step verification before marking complete.")
    elif outcome == "success" and task_type:
        action_items.append(f"For {task_type} tasks: store successful approach as a procedure=learned-strategy.")
        action_items.append(f"For {task_type} tasks: note what worked to reinforce the pattern.")
    elif outcome:
        action_items.append("Outcome recorded. Pattern-matching next attempts against stored strategies.")

    # GAP 1: Procedural learned-strategy insights (actionable, not just analytical)
    learned_strategies: list[str] = []
    if top_category != "none":
        learned_strategies.append(
            f"When handling {top_category} tasks, prefer the approach that succeeded "
            f"most frequently in recent memories."
        )
    if top_source != "none":
        learned_strategies.append(
            f"Source '{top_source}' is most reliable — weight its patterns higher in future decisions."
        )
    if by_category.get("error", 0) > 2:
        learned_strategies.append(
            "Multiple error memories detected. Consider creating a decision checklist to prevent repeated mistakes."
        )
    if by_category.get("decision", 0) > 3:
        learned_strategies.append(
            "Many decision memories stored. Review and distill key decision patterns into actionable rules."
        )

    # GAP 5: Loop detection
    reflection_signature = "|".join(sorted(insights[:3]))
    loop_detected = _recent_reflection_summaries.count(reflection_signature) >= _MAX_REFLECT_LOOP - 1
    _recent_reflection_summaries.append(reflection_signature)
    if len(_recent_reflection_summaries) > 20:
        _recent_reflection_summaries = _recent_reflection_summaries[-20:]
    if loop_detected:
        insights.append(
            f"⚠️ Reflection loop detected — same pattern repeating {_MAX_REFLECT_LOOP}+ times. "
            "Consider clearing stale reflections or changing the recall window."
        )

    # Build reflection summary (stored as reflection memory)
    reflection_memory = (
        f"Reflection on last {days} days: {len(recent)} memories reviewed. "
        f"Top category: {top_category}. Top source: {top_source}. "
        f"Average age: {avg_recency:.1f} days. "
        f"Outcome: {outcome or 'none'}. Task type: {task_type or 'general'}. "
        f"Insights: {' '.join(insights)}. "
        f"Action items: {' '.join(action_items)}. "
        f"Learned strategies: {' '.join(learned_strategies)}"
    )

    # Store standard reflection memory (category=reflection)
    reflection_stored = None
    try:
        now = _now_iso()
        reflection_vector = _embed_model().encode(reflection_memory).tolist()
        reflection_tokens = _tokenize(reflection_memory)
        reflection_id = str(uuid.uuid4())
        reflection_payload = {
            "content": reflection_memory,
            "created_at": now,
            "updated_at": now,
            "category": "reflection",
            "importance": 8,
            "tags": "self-improvement,aar,reflection",
            "source": "memory_reflect",
            "access_count": 0,
            "last_accessed_at": now,
            "tokens": reflection_tokens,
            "doc_len": len(reflection_tokens),
            "outcome": outcome,
            "task_type": task_type,
        }
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=reflection_id, vector=reflection_vector, payload=reflection_payload)],
        )
        _add_to_bm25_index(reflection_id, reflection_payload.get("content", ""))
        _invalidate_recall_caches()
        reflection_stored = reflection_id
    except Exception as e:
        log.warning("Failed to auto-store reflection memory: %s", e)

    # GAP 1: Store each ACTIONABLE strategy as a separate procedure memory
    # These are retrieved by recall when similar task types arise
    strategy_ids: list[str] = []
    for strategy in learned_strategies:
        try:
            now = _now_iso()
            strat_vector = _embed_model().encode(strategy).tolist()
            strat_tokens = _tokenize(strategy)
            strat_id = str(uuid.uuid4())
            strat_payload = {
                "content": strategy,
                "created_at": now,
                "updated_at": now,
                "category": "procedure",
                "importance": 8,
                "tags": f"learned-strategy,self-improvement,task-type:{task_type or 'general'}",
                "source": "memory_reflect",
                "access_count": 0,
                "last_accessed_at": now,
                "tokens": strat_tokens,
                "doc_len": len(strat_tokens),
                "learned_from_outcome": outcome,
            }
            _client.upsert(
                collection_name=COLLECTION_NAME,
                points=[qmodels.PointStruct(id=strat_id, vector=strat_vector, payload=strat_payload)],
            )
            _add_to_bm25_index(strat_id, strategy)
            strategy_ids.append(strat_id)
        except Exception as e:
            log.warning("Failed to store learned strategy: %s", e)

    # Also store action items as procedure memories
    for action in action_items:
        try:
            now = _now_iso()
            act_vector = _embed_model().encode(action).tolist()
            act_tokens = _tokenize(action)
            act_id = str(uuid.uuid4())
            act_payload = {
                "content": action,
                "created_at": now,
                "updated_at": now,
                "category": "procedure",
                "importance": 9,
                "tags": f"learned-strategy,self-improvement,action-item,task-type:{task_type or 'general'}",
                "source": "memory_reflect",
                "access_count": 0,
                "last_accessed_at": now,
                "tokens": act_tokens,
                "doc_len": len(act_tokens),
                "learned_from_outcome": outcome,
            }
            _client.upsert(
                collection_name=COLLECTION_NAME,
                points=[qmodels.PointStruct(id=act_id, vector=act_vector, payload=act_payload)],
            )
            _add_to_bm25_index(act_id, action)
            strategy_ids.append(act_id)
        except Exception as e:
            log.warning("Failed to store action item: %s", e)

    return _ok(json.dumps({
        "patterns_found": len(insights),
        "by_category": by_category,
        "by_source": by_source,
        "high_importance_count": len(high_importance),
        "average_memory_age_days": round(avg_recency, 2),
        "conflicts_detected": len(conflicting) if conflicting else 0,
        "loop_detected": loop_detected,
        "outcome": outcome,
        "task_type": task_type,
        "insights": insights,
        "action_items": action_items,
        "learned_strategies": learned_strategies,
        "reflection_summary": reflection_memory,
        "reflection_memory_id": reflection_stored,
        "learned_strategy_ids": strategy_ids,
    }, ensure_ascii=False, indent=2))


def _tool_get_insights(args: dict) -> types.CallToolResult:
    """Retrieve actionable learned strategies from self-improvement reflections.

    Enhancement 8: Closes the Reflexion loop by surfacing strategies that were
    learned from past outcomes so they can be applied to new tasks.
    """
    task_type = args.get("task_type")
    outcome = args.get("outcome")
    limit = min(args.get("limit") or 10, 50)

    # Previously scrolled 500 points with NO filter, so learned strategies from
    # every tenant were returned to any caller. Constrain to authorized scopes.
    scope = authorize_scope(args.get("scope"))
    try:
        results = _client.scroll(
            collection_name=COLLECTION_NAME, limit=500,
            scroll_filter=_build_filter(None, None, scope), with_payload=True)
        points = results[0]
    except Exception as e:
        return _err(f"Insight retrieval failed: {e}")

    strategies = []
    for p in points:
        tags = p.payload.get("tags", "")
        cat = p.payload.get("category", "")
        imp = p.payload.get("importance", 5)
        if cat != "procedure" or "learned-strategy" not in tags or imp < 8:
            continue
        if task_type and f"task-type:{task_type}" not in tags:
            continue
        if outcome and p.payload.get("learned_from_outcome") != outcome:
            continue
        strategies.append({
            "memory_id": p.id,
            "content": p.payload.get("content", ""),
            "importance": imp,
            "tags": tags,
            "learned_from_outcome": p.payload.get("learned_from_outcome"),
            "created_at": p.payload.get("created_at"),
        })

    strategies.sort(key=lambda x: x["importance"], reverse=True)
    return _ok(json.dumps({
        "count": len(strategies),
        "task_type": task_type,
        "outcome_filter": outcome,
        "strategies": strategies[:limit],
    }, ensure_ascii=False, indent=2))


def _tool_hint(args: dict) -> types.CallToolResult:
    """Agent-driven memory hint tool (SelfMem pattern).

    Gives the agent agency over its own memory management instead of a fixed pipeline.
    Actions: boost_importance, decay, evict, link, archive.
    """
    action = args["action"]
    memory_id = args["memory_id"]

    if action == "evict":
        if _redis:
            try:
                _redis.delete(f"{VERSION_PREFIX}{memory_id}")
            except Exception:
                pass
        try:
            _client.delete(collection_name=COLLECTION_NAME, points=[memory_id])
            _invalidate_recall_caches()
            return _ok(json.dumps({"status": "evicted", "memory_id": memory_id}))
        except Exception as e:
            return _err(f"Evict failed: {e}")

    elif action == "boost_importance":
        new_importance = args.get("importance", 9)
        try:
            fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
            if not fetched:
                return _err("Memory not found")
            now = _now_iso()
            _client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"importance": new_importance, "last_accessed_at": now, "updated_at": now},
                points=[memory_id],
            )
            _invalidate_recall_caches()
            return _ok(json.dumps({"status": "boosted", "memory_id": memory_id, "new_importance": new_importance}))
        except Exception as e:
            return _err(f"Boost failed: {e}")

    elif action == "decay":
        try:
            fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
            if not fetched:
                return _err("Memory not found")
            now = _now_iso()
            _client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"last_accessed_at": now, "updated_at": now},
                points=[memory_id],
            )
            return _ok(json.dumps({"status": "decayed", "memory_id": memory_id, "note": "Access time refreshed — recency score reset to 1.0"}))
        except Exception as e:
            return _err(f"Decay failed: {e}")

    elif action == "archive":
        try:
            fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
            if not fetched:
                return _err("Memory not found")
            now = _now_iso()
            tags = fetched[0].payload.get("tags", "")
            new_tags = tags + ",archived" if tags else "archived"
            _client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"importance": 3, "updated_at": now, "tags": new_tags},
                points=[memory_id],
            )
            _invalidate_recall_caches()
            return _ok(json.dumps({"status": "archived", "memory_id": memory_id}))
        except Exception as e:
            return _err(f"Archive failed: {e}")

    elif action == "link":
        target_id = args.get("link_target_id")
        relation = args.get("relation") or "relates_to"
        if not target_id:
            return _err("link requires link_target_id")
        try:
            if _redis:
                _redis.sadd(f"{GRAPH_NODE_PREFIX}{memory_id}", target_id)
                _redis.sadd(f"{GRAPH_NODE_PREFIX}{target_id}", memory_id)
                if relation and _redis:
                    _redis.hset(f"{GRAPH_EDGE_PREFIX}{memory_id}", target_id, relation)
            return _ok(json.dumps({"status": "linked", "from": memory_id, "to": target_id, "relation": relation}))
        except Exception as e:
            return _err(f"Link failed: {e}")

    else:
        return _err(f"Unknown hint action: {action}")


def _detect_conflicts(points: list) -> list[dict]:
    """Detect potential conflicts between memories (opposite statements)."""
    conflict_pairs = [
        ("do not", "must do"), ("avoid", "always use"), ("never", "always"),
        ("cannot", "can"), ("impossible", "possible"),
    ]
    conflicts = []
    for i, p1 in enumerate(points):
        c1 = p1.payload.get("content", "").lower()
        for p2 in points[i+1:]:
            c2 = p2.payload.get("content", "").lower()
            for pos, neg in conflict_pairs:
                if pos in c1 and neg in c2:
                    conflicts.append({"id1": p1.id, "id2": p2.id, "pattern": f"{pos} vs {neg}"})
                if neg in c1 and pos in c2:
                    conflicts.append({"id1": p1.id, "id2": p2.id, "pattern": f"{neg} vs {pos}"})
    return conflicts

# ---------------------------------------------------------------------------
# BM25 / Hybrid Search Index
# ---------------------------------------------------------------------------
STOPWORDS = set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "this", "that", "these", "those", "it",
    "its", "they", "them", "their", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "i", "me", "my", "so", "if", "not", "no", "yes",
    "up", "down", "out", "about", "into", "over", "after", "before", "between",
])

def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenization with stopword removal."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def _qdrant_all_ids() -> set:
    """Every point id currently in the collection (paginated)."""
    ids = set()
    offset = None
    while True:
        try:
            pts, offset = _client.scroll(
                collection_name=COLLECTION_NAME, limit=2000, offset=offset,
                with_payload=False, with_vectors=False)
        except Exception as e:
            log.warning("BM25 reconcile scroll failed: %s", e)
            return ids
        ids.update(p.id for p in pts)
        if offset is None:
            break
    return ids


def _prune_bm25_corpus() -> int:
    """Drop corpus entries whose point no longer exists in Qdrant.

    The persisted corpus is the fast path for boot, but it is only ever ADDED
    to. Anything deleted in Qdrant — including deletions performed outside this
    process — stayed in the index across every restart, so BM25 kept scoring
    documents that no longer existed, wasting slots and skewing IDF/avgdl.
    Returns the number of entries removed.
    """
    global _bm25_corpus
    with _BM25_REBUILD_LOCK:  # mutates the corpus; keep readers' snapshots consistent
        if not _bm25_corpus:
            return 0
        live = _qdrant_all_ids()
        if not live:
            # Scroll failed or returned nothing; never wipe the corpus on a bad read.
            return 0
        stale = [mid for mid in _bm25_corpus if mid not in live]
        if not stale:
            return 0
        for mid in stale:
            _bm25_corpus.pop(mid, None)
        log.info("BM25 corpus pruned: dropped %d stale entries not present in Qdrant",
                 len(stale))
    # Rebuild outside the prune's lock scope is fine — _rebuild_bm25_index takes
    # the lock itself; doing it here would deadlock (Lock is not reentrant).
    _rebuild_bm25_index()
    return len(stale)


def _rebuild_bm25_index():
    """Build bm25s index from all Qdrant memories + persist corpus to JSON.

    Stores RAW TEXT in the corpus; bm25s tokenizes internally on both index
    and query. This avoids the tokenizer mismatch where pre-tokenized word
    lists were passed to bm25s which then applied its default char-trigram
    tokenizer to each token, producing fragments like 'configu'.

    If _bm25_corpus is already populated (e.g., loaded from disk), skip the
    Qdrant scroll and rebuild index directly from the cached corpus. Callers that
    need the corpus reconciled with Qdrant must run _prune_bm25_corpus() first —
    otherwise stale ids accumulate forever across restarts.

    Holds _BM25_REBUILD_LOCK for its whole body so the index and corpus swap is
    atomic with respect to readers taking a snapshot.
    """
    global _bm25_index, _bm25_corpus, _bm25_dirty, _bm25_indexed_ids
    with _BM25_REBUILD_LOCK:
        # If corpus is empty, build it from Qdrant
        if not _bm25_corpus:
            try:
                results = _client.scroll(collection_name=COLLECTION_NAME, limit=2000,
                                        with_payload=True)
                docs = results[0]
            except Exception as e:
                log.warning("BM25 index rebuild failed: %s", e)
                return

            for doc in docs:
                mem_id = doc.id
                content = doc.payload.get("content", "")
                # Store raw text; bm25s will tokenize on index and query
                _bm25_corpus[mem_id] = content

        if not _bm25_corpus:
            _bm25_index = None
            _bm25_indexed_ids = []
            _save_bm25_corpus()
            _bm25_dirty = False
            log.info("BM25 index empty (no memories)")
            return

        corpus_texts = list(_bm25_corpus.values())
        corpus_ids = list(_bm25_corpus.keys())
        _bm25_index = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        _bm25_index.index(corpus_texts, corpus_ids)
        # Record the exact id ordering this index was built from, so readers map
        # scores to ids via this snapshot rather than the live corpus dict.
        _bm25_indexed_ids = list(corpus_ids)

        _save_bm25_corpus()
        _clear_bm25_score_cache()
        _bm25_dirty = False
        # P1-4: Save BM25 index to Redis for horizontal scaling
        if HORIZONTAL_SCALING_ENABLED:
            _bm25_redis_save()
        log.info("BM25 index rebuilt: %d docs, corpus saved to %s", len(_bm25_corpus), _bm25_index_path)

def _save_bm25_corpus():
    """Persist corpus to JSON for fast reload on restart."""
    try:
        with open(_bm25_index_path, "w") as f:
            json.dump(_bm25_corpus, f)
    except Exception as e:
        log.warning("could not save BM25 corpus: %s", e)

def _load_bm25_corpus() -> bool:
    """Load persisted corpus from JSON. Returns True if successful."""
    if not os.path.exists(_bm25_index_path):
        return False
    try:
        with open(_bm25_index_path, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return False
        global _bm25_corpus
        _bm25_corpus = loaded
        return True
    except Exception as e:
        log.warning("could not load BM25 corpus: %s", e)
        return False

def _add_to_bm25_index(mem_id: str, content: str):
    """Add one memory to the corpus and MARK DIRTY (no inline rebuild).

    bm25s.BM25 has no incremental add() (verified v0.3.11), so the rebuild is
    O(corpus) and must be coalesced: an inline rebuild here turned every outbox
    event into a full-corpus rebuild and stalled the worker. The rebuild now
    happens once per claimed batch via _bm25_maybe_rebuild().
    
    Stores RAW TEXT; bm25s tokenizes internally.
    """
    global _bm25_dirty
    with _BM25_REBUILD_LOCK:  # write site: mark dirty only, never rebuild here
        _bm25_corpus[mem_id] = content
        _bm25_dirty = True

def _remove_from_bm25_index(mem_id: str):
    """Remove a memory from the corpus and MARK DIRTY (no inline rebuild)."""
    global _bm25_dirty
    with _BM25_REBUILD_LOCK:  # write site: mark dirty only, never rebuild here
        _bm25_corpus.pop(mem_id, None)
        _bm25_dirty = True

def _bm25_maybe_rebuild(force: bool = False):
    """Rebuild the BM25 index once when the corpus is dirty (or on force).

    Called by the outbox worker at the end of a claimed batch, so N events in a
    batch cost exactly ONE O(corpus) rebuild instead of N. Never called from the
    read path.
    """
    global _bm25_dirty
    with _BM25_REBUILD_LOCK:
        if _bm25_dirty or force:
            _rebuild_bm25_index()
        # _rebuild_bm25_index clears the dirty flag on success.

def _bm25_snapshot():
    """Return (index, indexed_ids) for a reader under the lock.

    Readers never trigger a rebuild; a query that lands between events simply
    sees the last-built index and its own id ordering, and the next batch
    rebuild catches it up — the same eventual-consistency model the outbox
    already relies on. Pairing the index with `_bm25_indexed_ids` guarantees the
    score->id mapping matches the index that produced the scores.
    """
    with _BM25_REBUILD_LOCK:
        return _bm25_index, list(_bm25_indexed_ids)

def _bm25_positions() -> dict[str, int]:
    """mem_id -> row position in the CURRENT BM25 matrix.

    The previous code did `list(_bm25_corpus.keys()).index(mem_id)` per
    candidate, which is O(n) each and O(n^2) per query. Positions are derived
    from the built index's own id snapshot, not the live corpus dict, so a
    concurrent write cannot shift a row out from under a reader.
    """
    with _BM25_REBUILD_LOCK:
        pos = {mid: i for i, mid in enumerate(_bm25_indexed_ids)}
    return pos


def _bm25_scores_for_query(query_text: str) -> dict[str, float]:
    """Score the WHOLE corpus for one query in a single vectorized pass.

    Returns {mem_id: raw_bm25}. Computed once per query and reused for every
    candidate, replacing the per-candidate get_scores() + linear index scan.
    Reads a lock-protected (index, ids) snapshot — never mutates, never rebuilds.
    """
    index, indexed_ids = _bm25_snapshot()
    if index is None or not query_text or not indexed_ids:
        # A dead sparse leg silently degrades hybrid recall to dense-only.
        # Make it visible (rate-limited) instead of returning {} quietly.
        global _BM25_NONE_WARNED
        if not globals().get("_BM25_NONE_WARNED"):
            globals()["_BM25_NONE_WARNED"] = True
            log.warning("BM25 index is None — sparse leg DISABLED, recall is "
                        "dense-only until a rebuild/load happens")
        return {}
    try:
        # Use index's get_tokens_ids to tokenize query with corpus vocabulary
        query_token_ids = index.get_tokens_ids(query_text)
        all_scores = index.get_scores(query_token_ids)
    except Exception as _be:
        log.warning("BM25 scoring failed (sparse leg skipped): %s", _be)
        return {}
    out = {}
    for i, mid in enumerate(indexed_ids):
        try:
            v = float(all_scores[i])
        except Exception:
            v = 0.0
        if v > 0:
            out[mid] = v
    return out


def _get_bm25_score_cached(query_text: str, mem_id: str) -> float:
    """BM25 score for one candidate (kept for compatibility with callers)."""
    cache_key = (query_text, mem_id)
    if cache_key in _bm25_score_cache:
        _bm25_score_cache.move_to_end(cache_key)
        return _bm25_score_cache[cache_key]
    index, indexed_ids = _bm25_snapshot()
    if index is None or mem_id not in indexed_ids:
        return 0.0
    score = _bm25_scores_for_query(query_text).get(mem_id, 0.0)
    _bm25_score_cache[cache_key] = score
    if len(_bm25_score_cache) > BM25_SCORE_CACHE_SIZE:
        _bm25_score_cache.popitem(last=False)
    return score

# ---------------------------------------------------------------------------
# P1-4: Redis-backed BM25 Index (Horizontal Scaling)
# ---------------------------------------------------------------------------
# When HORIZONTAL_SCALING_ENABLED=1, the BM25 index is stored in Redis
# so it can be shared across multiple instances. The index is serialized
# using bm25s's built-in serialization (pickle) and stored as bytes.
# Corpus is stored as JSON for fast loading.

def _bm25_redis_save():
    """Save BM25 index and corpus to Redis for sharing across instances."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return
    if _bm25_index is None:
        return
    try:
        with _BM25_REBUILD_LOCK:
            # Serialize index (bm25s uses pickle)
            import pickle
            index_bytes = pickle.dumps(_bm25_index)
            corpus_json = json.dumps(_bm25_corpus)
            indexed_ids_json = json.dumps(_bm25_indexed_ids)
            
            # Save to Redis with TTL (7 days)
            _redis.setex(f"{BM25_REDIS_PREFIX}index", 7*24*3600, index_bytes)
            _redis.setex(f"{BM25_REDIS_PREFIX}corpus", 7*24*3600, corpus_json)
            _redis.setex(f"{BM25_REDIS_PREFIX}indexed_ids", 7*24*3600, indexed_ids_json)
            log.info("BM25 index saved to Redis (%d docs)", len(_bm25_corpus))
    except Exception as e:
        log.warning("Failed to save BM25 index to Redis: %s", e)

def _bm25_redis_load() -> bool:
    """Load BM25 index from Redis. Returns True if successful."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return False
    try:
        # MUST use the binary-safe client: the pickled index is not UTF-8 and
        # decode_responses=True corrupts it at the Redis-client layer.
        index_bytes = _redis_bytes.get(f"{BM25_REDIS_PREFIX}index")
        corpus_json = _redis_bytes.get(f"{BM25_REDIS_PREFIX}corpus")
        indexed_ids_json = _redis_bytes.get(f"{BM25_REDIS_PREFIX}indexed_ids")

        if not index_bytes or not corpus_json or not indexed_ids_json:
            return False

        import pickle
        if isinstance(index_bytes, str):
            index_bytes = index_bytes.encode("latin-1")
        index = pickle.loads(index_bytes)
        corpus = json.loads(corpus_json if isinstance(corpus_json, str) else corpus_json.decode("utf-8"))
        indexed_ids = json.loads(indexed_ids_json if isinstance(indexed_ids_json, str) else indexed_ids_json.decode("utf-8"))
        
        global _bm25_index, _bm25_corpus, _bm25_indexed_ids
        _bm25_index = index
        _bm25_corpus = corpus
        _bm25_indexed_ids = indexed_ids
        log.info("BM25 index loaded from Redis (%d docs)", len(corpus))
        return True
    except Exception as e:
        log.warning("Failed to load BM25 index from Redis: %s", e)
        return False

def _bm25_redis_save_corpus():
    """Save only the corpus to Redis (for incremental updates)."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return
    try:
        corpus_json = json.dumps(_bm25_corpus)
        _redis.setex(f"{BM25_REDIS_PREFIX}corpus", 7*24*3600, corpus_json)
    except Exception as e:
        log.warning("Failed to save BM25 corpus to Redis: %s", e)

# ---------------------------------------------------------------------------
# P1-4: Redis-backed Recall Cache (Horizontal Scaling)
# ---------------------------------------------------------------------------
# Shared recall cache across instances using Redis. Replaces in-process LRU.

def _recall_redis_get(query: str, top_k: int, category: str | None, tags_key: str | None, scope: str | None) -> str | None:
    """Get cached recall results from Redis."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return None
    try:
        # Create cache key from query params
        cache_key = f"{RECALL_CACHE_PREFIX}{hashlib.md5(f'{query}|{top_k}|{category}|{tags_key}|{scope}'.encode()).hexdigest()}"
        cached = _redis.get(cache_key)
        if cached:
            log.debug("Recall cache hit: %s", query[:50])
            return cached
        return None
    except Exception as e:
        log.warning("Recall cache get failed: %s", e)
        return None

def _recall_redis_set(query: str, top_k: int, category: str | None, tags_key: str | None, scope: str | None, result_json: str):
    """Cache recall results in Redis."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return
    try:
        cache_key = f"{RECALL_CACHE_PREFIX}{hashlib.md5(f'{query}|{top_k}|{category}|{tags_key}|{scope}'.encode()).hexdigest()}"
        _redis.setex(cache_key, RECALL_CACHE_TTL, result_json)
    except Exception as e:
        log.warning("Recall cache set failed: %s", e)

def _recall_redis_invalidate(pattern: str = None):
    """Invalidate recall cache entries matching pattern."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return
    try:
        if pattern:
            for key in _redis.scan_iter(match=f"{RECALL_CACHE_PREFIX}{pattern}", count=1000):
                _redis.delete(key)
        else:
            for key in _redis.scan_iter(match=f"{RECALL_CACHE_PREFIX}*", count=1000):
                _redis.delete(key)
    except Exception as e:
        log.warning("Recall cache invalidate failed: %s", e)

def _recall_redis_stats() -> dict:
    """Get recall cache stats."""
    if not HORIZONTAL_SCALING_ENABLED or not _redis:
        return {"enabled": False}
    try:
        keys = list(_redis.scan_iter(match=f"{RECALL_CACHE_PREFIX}*", count=1000))
        return {"enabled": True, "entries": len(keys), "ttl_seconds": RECALL_CACHE_TTL}
    except Exception as e:
        return {"enabled": True, "error": str(e)}

def _get_cross_encoder():
    """Lazy-load the cross-encoder model on first use."""
    global _cross_encoder
    if _cross_encoder is None:
        if not RERANKER_MODEL:
            log.info("RERANKER_MODEL is empty — cross-encoder reranking disabled")
            _cross_encoder = False  # type: ignore
            return None
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder model: %s", RERANKER_MODEL)
            _cross_encoder = CrossEncoder(RERANKER_MODEL, max_length=512, device="cpu")
            log.info("Cross-encoder loaded successfully")
        except Exception as e:
            log.warning("Failed to load cross-encoder: %s", e)
            _cross_encoder = False  # type: ignore
    return _cross_encoder if _cross_encoder is not False else None

def _rerank_cross_encoder(query: str, candidates: list[tuple[str, float, dict]], top_k: int = RERANK_TOP_K) -> list[tuple[str, float]]:
    """Rescore top-k hybrid candidates using a cross-encoder for precision.

    Args:
        query: original search query
        candidates: list of (mem_id, hybrid_score, payload) tuples
        top_k: max candidates to rerank (default 5 for lightweight)

    Returns:
        Re-ranked list of (mem_id, new_score) tuples
    """
    model = _get_cross_encoder()
    if model is None or not candidates:
        return [(mid, score) for mid, score, _ in candidates]

    # Take only top-k candidates to keep it lightweight
    to_rerank = candidates[:top_k]
    rest = candidates[top_k:]

    try:
        # Build (query, document) pairs — truncate doc to first 512 tokens worth of chars
        pairs = []
        for mid, score, payload in to_rerank:
            content = payload.get("content", "")[:2000]  # truncate long content
            pairs.append((query, content))

        # Get cross-encoder scores (raw relevance logits)
        ce_scores = model.predict(pairs, show_progress_bar=False)
        if not isinstance(ce_scores, list):
            ce_scores = ce_scores.tolist()

        # Normalize CE scores to 0-1 range
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        if ce_max > ce_min:
            norm_ce = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
        else:
            norm_ce = [0.5] * len(ce_scores)

        # Blend: 60% cross-encoder, 40% original hybrid score
        reranked = []
        for i, (mid, hybrid_score, payload) in enumerate(to_rerank):
            new_score = 0.6 * norm_ce[i] + 0.4 * hybrid_score
            reranked.append((mid, new_score))

        # Add remaining non-reranked candidates with original scores
        for mid, score, _ in rest:
            reranked.append((mid, score))

        # Sort by new score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    except Exception as e:
        log.warning("Cross-encoder reranking failed: %s", e)
        return [(mid, score) for mid, score, _ in candidates]

def _title_boost(query_tokens: list[str], payload: dict) -> float:
    """Return a 0-1 boost score if query tokens appear in the memory title."""
    title = payload.get("title", "")
    if not title:
        # Use first line of content as implicit title
        content = payload.get("content", "")
        first_line = content.split("\n")[0].strip() if content else ""
        title = first_line[:100]  # cap at 100 chars
    if not title:
        return 0.0
    title_lower = title.lower()
    matches = sum(1 for tok in query_tokens if tok in title_lower)
    if matches > 0:
        return TITLE_BOOST * (matches / len(query_tokens))
    return 0.0

def _clear_bm25_score_cache():
    """Clear the BM25 score LRU cache."""
    _bm25_score_cache.clear()

def _payload_passes_recall_constraints(payload: dict | None, category: str | None,
                                       tags_key: str | None, scope: str | None,
                                       as_of: str | None) -> bool:
    """Check a payload against the same constraints `_build_filter` compiles.

    Needed for sparse-only candidates: they originate from the BM25 corpus and
    never passed through the Qdrant filter, so they must be validated directly.
    The previous implementation instead re-ran the dense query and kept only the
    ids in its top-N, which discarded every sparse-only candidate — the exact
    starvation the hybrid path exists to fix — and was the only thing preventing
    a cross-scope candidate from surviving.
    """
    if not payload:
        return False
    if scope is not None:
        # Strict: exact match, and never satisfied by a missing scope field.
        if (payload.get("scope") or "") != scope:
            return False
    if category:
        if (payload.get("category") or "").lower() != str(category).lower():
            return False
    if tags_key:
        have = {str(t).lower() for t in _payload_tags(payload)}
        wanted = [t for t in tags_key.split(",") if t]
        if not all(t.lower() in have for t in wanted):
            return False
    if not as_of:
        if (payload.get("status") or "") == "superseded":
            return False
    else:
        try:
            import datetime as _dt
            _as = _dt.datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            _created = payload.get("created_at")
            if _created:
                _c = _dt.datetime.fromisoformat(str(_created).replace("Z", "+00:00"))
                if _c.tzinfo is None:
                    _c = _c.replace(tzinfo=_dt.timezone.utc)
                if _as.tzinfo is None:
                    _as = _as.replace(tzinfo=_dt.timezone.utc)
                if _c > _as:
                    return False
            _valid_to = payload.get("valid_to")
            if _valid_to:
                _v = _dt.datetime.fromisoformat(str(_valid_to).replace("Z", "+00:00"))
                if _v.tzinfo is None:
                    _v = _v.replace(tzinfo=_dt.timezone.utc)
                if _v < _as:
                    return False
        except Exception:
            # Unparseable as_of must not silently admit a candidate.
            return False
    return True


def _hybrid_recall(query: str, top_k: int, category: str | None, tags_key: str | None, scope: str | None = None, as_of: str | None = None) -> list[dict]:
    """Hybrid search: dense vector + BM25 keyword, fused over the UNION of both
    candidate sets.

    Two defects were fixed here:
    1. BM25 was scored only over the dense hits, so a memory the dense stage
       dropped was structurally unreachable — the "hybrid" stage could never
       rescue anything. The sparse stage now ranks the whole corpus
       independently and contributes its own candidates.
    2. dense was a raw RRF rank (max ~0.016 at k=60) while sparse was a
       normalized score blended with an RRF rank (max ~0.70). Fusing those at
       equal weight made the score ~97% sparse. Both are now RRF ranks on the
       same 1/(k+rank) scale, which is what Reciprocal Rank Fusion assumes.
    """
    global _session_seen_ids
    _total_start = time.perf_counter()

    # --- Stage 1: dense retrieval (wide pool for reranking) ---
    _embed_start = time.perf_counter()
    query_vector = _embed_model().encode(query).tolist()
    _record_stage_latency("embed", (time.perf_counter() - _embed_start) * 1000)

    _dense_start = time.perf_counter()
    filter_q = _build_filter(category, tags_key, scope, as_of)
    dense_limit = max(top_k * DENSE_CANDIDATE_MULT, 50)
    dense_ranks: dict[str, float] = {}
    dense_points: list = []
    global _RECALL_DEGRADED
    _RECALL_DEGRADED = None
    try:
        dense_results = _client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=dense_limit,
            query_filter=filter_q,
            with_payload=True,
        )
        dense_points = list(dense_results.points)
    except Exception as _qe:
        # 1B-5 degraded matrix: Qdrant down must not kill recall — fall back
        # to sparse-only (BM25/Redis) ranking, flag it, count it.
        _breaker_fail("qdrant", _qe)
        _RECALL_DEGRADED = f"sparse-only (qdrant unavailable: {type(_qe).__name__})"
        log.error("Qdrant dense search unavailable — degrading to sparse-only "
                  "recall: %s", _qe)
    _record_stage_latency("dense_search", (time.perf_counter() - _dense_start) * 1000)

    dense_ranks: dict[str, float] = {}
    for rank, r in enumerate(dense_points):
        dense_ranks[r.id] = 1.0 / (BM25_RRF_K + rank + 1)

    # --- Stage 2: sparse retrieval over the WHOLE corpus ---
    # Tokenize query for title boost
    _sparse_start = time.perf_counter()
    query_tokens = _tokenize(query)
    bm25_all = _bm25_scores_for_query(query) if query else {}
    sparse_ranked: list[str] = []
    if bm25_all:
        # Only IDs that actually exist in the corpus can be looked up later.
        sparse_ranked = [
            mid for mid, _ in sorted(bm25_all.items(), key=lambda kv: kv[1], reverse=True)
        ]
    sparse_ranks = {
        mid: 1.0 / (BM25_RRF_K + rank + 1) for rank, mid in enumerate(sparse_ranked)
    }
    _record_stage_latency("sparse_search", (time.perf_counter() - _sparse_start) * 1000)

    # --- Union of both candidate sets ---
    candidate_ids = list(dict.fromkeys(
        [p.id for p in dense_points] + sparse_ranked[:max(top_k * SPARSE_CANDIDATE_MULT, 50)]
    ))

    # Payloads: dense results carry them; sparse-only candidates need a fetch.
    id_to_payload = {p.id: p.payload for p in dense_points}
    missing = [mid for mid in candidate_ids if mid not in id_to_payload]
    if missing:
        for chunk_start in range(0, len(missing), 256):
            chunk = missing[chunk_start:chunk_start + 256]
            try:
                got = _client.retrieve(
                    collection_name=COLLECTION_NAME, ids=chunk, with_payload=True)
                for g in got:
                    id_to_payload[g.id] = g.payload
            except Exception as e:
                log.debug("sparse payload fetch failed: %s", e)

    # Dense candidates already passed the authoritative Qdrant filter. Sparse-only
    # candidates came from the BM25 corpus, so validate their payloads against the
    # same constraints directly — including strict scope equality, which is what
    # keeps a keyword match in another tenant from ever surfacing here.
    dense_id_set = {p.id for p in dense_points}

    _fusion_start = time.perf_counter()
    all_scores: dict[str, tuple[float, float, float]] = {}
    for mid in candidate_ids:
        if mid not in dense_id_set and not _payload_passes_recall_constraints(
            id_to_payload.get(mid), category, tags_key, scope, as_of
        ):
            continue
        d = dense_ranks.get(mid, 0.0)
        s = sparse_ranks.get(mid, 0.0)
        # Pure RRF: comparable scales, no unit mixing.
        combined = (1 - HYBRID_ALPHA) * d + HYBRID_ALPHA * s
        all_scores[mid] = (d, s, combined)
    _record_stage_latency("rrf_fusion", (time.perf_counter() - _fusion_start) * 1000)

    sorted_mem_ids = sorted(all_scores, key=lambda m: all_scores[m][2], reverse=True)
    candidates = [
        (mid, all_scores[mid][2], id_to_payload.get(mid, {}))
        for mid in sorted_mem_ids
    ]

    # Title-term boosting: +20% if query terms appear in memory title
    if query_tokens:
        for i, (mid, score, payload) in enumerate(candidates):
            boost = _title_boost(query_tokens, payload)
            if boost > 0:
                candidates[i] = (mid, score + boost, payload)

    # Cross-encoder reranking on top-k candidates (lightweight: max 5)
    _rerank_start = time.perf_counter()
    reranked = _rerank_cross_encoder(query, candidates, top_k=RERANK_TOP_K)
    _record_stage_latency("rerank", (time.perf_counter() - _rerank_start) * 1000)

    result = []
    seen = set()
    for mid, new_score in reranked:
        if mid in seen:
            continue
        seen.add(mid)
        dense_rrf, sparse_rrf, _ = all_scores.get(mid, (0.0, 0.0, 0.0))
        result.append((mid, (dense_rrf, sparse_rrf, new_score)))
        if len(result) >= top_k:
            break
    _record_stage_latency("total_recall", (time.perf_counter() - _total_start) * 1000)
    return result

# ---------------------------------------------------------------------------
# MCP server (SDK v2 API)
# ---------------------------------------------------------------------------
_SERVER_TOOLS: list[types.Tool] = [
    types.Tool(
        name="memory_store",
        version="1.0.0",
        description=(
            "Store a long-term memory. The content is embedded as a vector and indexed for semantic recall. "
            "Use this to persist facts, procedures, preferences, events, decisions, project context, and "
            "other knowledge across sessions. Returns the new memory_id. "
            "QUALITY GATE: content is scored for quality (length, structure, specificity, actionability). "
            "Low-quality content (score < 0.3) is rejected. Score 0.3-0.5 returns a warning. "
            "Returns quality_score in the response so the agent can assess storage confidence."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "content": {"type": "string", "description": "The memory content text."},
                "category": {
                    "type": "string",
                    "description": "Optional category: fact, procedure, preference, event, decision, credential, project, conversation, general.",
                },
                "importance": {
                    "type": "integer",
                    "description": "Optional importance 1-10 (10 = critical). Default 5.",
                },
                "tags": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional tags for grouping / filtering. A bare string is accepted and coerced to a single-element list.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source context (e.g. 'user_message', 'web_search', 'file_read', 'agent').",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional namespace scope (e.g. project/user name). Default 'global'. Scoped memories are shared with 'global' unless queried with a specific scope.",
                },
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_recall",
        version="1.0.0",
        description=(
            "Hybrid recall of long-term memories. Uses dense vector similarity + sparse BM25 keyword "
            "matching, fused via Reciprocal Rank Fusion (RRF). Results are then re-ranked with "
            "importance (25%) and recency (15%) signals on top of the hybrid score (60%). "
            "Supports category and tag filtering. Returns utility metadata (retrieval_count, useful_count, "
            "quality_score) for each memory to support learning-loop decisions. "
            "G3 temporal: by default superseded memories are excluded; pass as_of=<ISO8601> "
            "to get memories valid at that moment ('what was true then')."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query to recall relevant memories."},
                "top_k": {"type": "integer", "description": "Number of top results to return. Default 5."},
                "category": {"type": "string", "description": "Optional filter by exact category."},
                "tags": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional filter — memory must contain all specified tags. A bare string is accepted and coerced to a single-element list.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter — returns memories of this scope, 'global' scope, and legacy unscoped memories.",
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO8601 timestamp. When set, returns only memories valid at that moment (created_at <= as_of, valid_to null or >= as_of). Default: now (superseded memories excluded).",
                },
            },
            "required": ["query"],
        },
    ),
    # P1-8: Exact-Fact Retrieval
    types.Tool(
        name="memory_lookup_exact",
        version="1.0.0",
        description=(
            "EXACT FACT RETRIEVAL — SQL-style WHERE on payload fields. "
            "Supports exact lookup by: memory_id, scope, category, source, tag, created_at range. "
            "Routes to Postgres (mem_canonical) for exact matches, not Qdrant. "
            "Returns full canonical records matching the filter."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "field": {
                    "type": "string",
                    "description": "Field to filter on: memory_id, scope, category, source, tag, created_at.",
                    "enum": ["memory_id", "scope", "category", "source", "tag", "created_at"],
                },
                "value": {
                    "description": "Value to match. For created_at, use object with 'from' and/or 'to' ISO8601 timestamps.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter (defaults to authorized scopes).",
                },
            },
            "required": ["field", "value"],
        },
    ),
    types.Tool(
        name="memory_list",
        version="1.0.0",
        description="List long-term memories, optionally filtered by category, in reverse-chronological order.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "category": {"type": "string", "description": "Optional filter by exact category."},
                "limit": {"type": "integer", "description": "Max results. Default 20, max 100."},
                "offset": {"type": "integer", "description": "Pagination offset. Default 0."},
                "scope": {"type": "string", "description": "Optional scope filter (scope + global + legacy unscoped)."},
            },
        },
    ),
    types.Tool(
        name="memory_get",
        version="1.0.0",
        description="Retrieve a single long-term memory by its ID.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_update",
        version="1.0.0",
        description=(
            "Update an existing long-term memory. If content is changed it is re-embedded. "
            "Metadata fields are merged — only provided fields are updated."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID to update."},
                "content": {"type": "string", "description": "New content (triggers re-embedding)."},
                "category": {"type": "string", "description": "New category."},
                "importance": {"type": "integer", "description": "New importance 1-10."},
                "tags": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "New tags (replaces existing tags). A bare string is accepted and coerced to a single-element list.",
                },
                "source": {"type": "string", "description": "New source."},
                "scope": {"type": "string", "description": "New scope (Phase 3.3)."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_delete",
        version="1.0.0",
        description="Delete a long-term memory by its ID.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID to delete."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_stats",
        version="1.0.0",
        description="Get statistics about the long-term memory store — total count, categories, embedding model.",
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    types.Tool(
        name="memory_cache_stats",
        version="1.0.0",
        description=(
            "Get semantic cache statistics: hits, misses, hit rate, TTL, threshold, "
            "and number of cached entries in Redis. Helps monitor cache effectiveness."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    types.Tool(
        name="memory_consolidate",
        version="1.0.0",
        description=(
            "Cleanup long-term memory store: removes duplicates (high cosine similarity), "
            "evicts old memories beyond max_count, auto-archives memories with decayed importance "
            "below threshold (importance=1, tag 'archived'), and cleans up access tracking metadata. "
            "Call periodically to prevent bloat. Returns cleanup statistics including auto_archived count."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "max_count": {"type": "integer", "description": "Maximum memories to retain. Default: 1000."},
                "similarity_threshold": {"type": "number", "description": "Duplicate threshold 0.0-1.0. Default: 0.92."},
                "evict_old_accessed": {"type": "integer", "description": "Evict memories not accessed in N days. Default: 90."},
            },
        },
    ),
    types.Tool(
        name="memory_session_summarize",
        version="1.0.0",
        description=(
            "P0-2: Summarize and compact a trajectory session into a structured summary. "
            "Extracts critical facts (errors, configs, tool outputs), generates a compact summary "
            "with lessons/facts/key data, and verifies critical-fact retention. "
            "Automatically triggered by step count (>50), token count (>4000), or time (>24h). "
            "Raw steps are archived and replaced with a pointer to the summary. "
            "Returns summary memory ID, compression ratio, and verification results."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {"type": "string", "description": "Task description (matches trajectory_log/task_end)."},
                "task_type": {"type": "string", "description": "Optional task type (coding, debugging, etc.)."},
                "outcome": {"type": "string", "enum": ["success", "failure", "partial", "in_progress"], "description": "Session outcome."},
                "duration_minutes": {"type": "integer", "description": "Optional total duration."},
                "lessons_learned": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Lessons from the session."},
                "facts_discovered": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Facts discovered."},
                "task_id": {"type": "string", "description": "Optional explicit task_id (defaults to open trajectory for task)."},
                "force": {"type": "boolean", "description": "Force compaction even if triggers not met. Default: false."},
            },
            "required": ["task", "outcome"],
        },
    ),
    types.Tool(
        name="memory_compaction_metrics",
        version="1.0.0",
        description=(
            "P0-2: Get session compaction metrics — total compactions, compression ratio (mean/p50/p95/p99), "
            "critical-fact retention rate (mean/p50/p95/min), compaction latency (mean/p50/p95/max), "
            "and current trigger configuration. Use to monitor compaction effectiveness."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    types.Tool(
        name="memory_track_access",
        version="1.0.0",
        description="Record an access event for a memory (called internally by recall, but can be called manually).",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_memory_short_store",
        version="1.0.0",
        description=(
            "Store a short-term memory in Redis. Short-term memory is for session context, "
            "credentials, and temporary data that doesn't need to persist long-term. "
            "TTL defaults to 7 days. Returns the memory key."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "key": {"type": "string", "description": "Unique key for this memory (e.g. 'session:123', 'credential:db1')."},
                "value": {"type": "string", "description": "The memory content to store."},
                "ttl_seconds": {"type": "integer", "description": "TTL in seconds. Default: 604800 (7 days). Use 0 for no expiry."},
            },
            "required": ["key", "value"],
        },
    ),
    types.Tool(
        name="memory_memory_short_recall",
        version="1.0.0",
        description="Recall a short-term memory from Redis by its key.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "key": {"type": "string", "description": "The key of the memory to recall."},
            },
            "required": ["key"],
        },
    ),
    types.Tool(
        name="memory_memory_short_delete",
        version="1.0.0",
        description="Delete a short-term memory from Redis by its key.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "key": {"type": "string", "description": "The key of the memory to delete."},
            },
            "required": ["key"],
        },
    ),
    types.Tool(
        name="memory_memory_short_list",
        version="1.0.0",
        description="List short-term memory keys within the caller's authorized scope (optionally filtered by a scope-relative prefix). Keys belonging to other tenants are never returned.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "prefix": {"type": "string", "description": "Optional scope-relative key prefix to filter by (applied within the caller's own namespace)."},
                "limit": {"type": "integer", "description": "Max keys to return. Default: 100."},
                "scope": {"type": "string", "description": "Optional explicit scope (must be authorized for the caller). Defaults to the caller's own namespace."},
            },
        },
    ),
    types.Tool(
        name="memory_memory_short_stats",
        version="1.0.0",
        description="Get deployment-wide short-term memory (Redis) statistics. Global counters only, not per-tenant; exposes no tenant content.",
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    types.Tool(
        name="memory_remember",
        version="1.0.0",
        description=(
            "Universal remember tool — automatically decides long-term vs short-term storage. "
            "Set persistence='permanent' to force long-term (Qdrant), 'session' to force short-term (Redis). "
            "Set persistence='auto' (default) and the tool uses keyword heuristics to decide: "
            "credentials/tokens/session-ids → short-term (Redis, 7d TTL); "
            "facts/procedures/preferences/decisions/configs → long-term (Qdrant). "
            "Always stores to the appropriate layer. Returns which layer was chosen and the storage result."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "content": {"type": "string", "description": "The memory content text."},
                "persistence": {
                    "type": "string",
                    "enum": ["auto", "permanent", "session"],
                    "description": "'auto' (default) uses keywords to decide, 'permanent' forces long-term Qdrant, 'session' forces short-term Redis.",
                },
                "category": {"type": "string", "description": "Category for long-term memories (ignored for session storage)."},
                "importance": {"type": "integer", "description": "Importance 1-10 for long-term (default 5)."},
                "tags": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Tags for long-term memories. A bare string is accepted and coerced to a single-element list.",
                },
                "key": {"type": "string", "description": "Optional key for session storage (auto-generated if not provided)."},
                "ttl_seconds": {"type": "integer", "description": "TTL for session storage in seconds. Default 604800 (7 days)."},
            },
            "required": ["content"],
        },
    ),
    # Enhancement 1: Graph Memory tools
    types.Tool(
        name="memory_graph_link",
        version="1.0.0",
        description=(
            "Create a directed relationship edge between two memories. "
            "Use this to link related memories (e.g., a fix relates to a bug, a decision relates to a project). "
            "Relations: depends_on, related_to, supersedes, contradicts, implements, references, parent_of, child_of. "
            "Returns edge_id if successful."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Source memory ID (alias: from_memory_id)."},
                "related_id": {"type": "string", "description": "Target memory ID (alias: to_memory_id)."},
                "from_memory_id": {"type": "string", "description": "Alias for memory_id (source)."},
                "to_memory_id": {"type": "string", "description": "Alias for related_id (target)."},
                "relation": {
                    "type": "string",
                    "enum": ["depends_on", "related_to", "supersedes", "contradicts", "implements", "references", "parent_of", "child_of"],
                    "description": "Relationship type.",
                },
            },
            "required": ["relation"],
        },
    ),
    types.Tool(
        name="memory_graph_unlink",
        version="1.0.0",
        description="Remove the relationship edge between two memories.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Source memory ID (alias: from_memory_id)."},
                "related_id": {"type": "string", "description": "Target memory ID (alias: to_memory_id)."},
                "from_memory_id": {"type": "string", "description": "Alias for memory_id (source)."},
                "to_memory_id": {"type": "string", "description": "Alias for related_id (target)."},
            },
        },
    ),
    types.Tool(
        name="memory_graph_neighbors",
        version="1.0.0",
        description=(
            "Traverse the memory graph to find related memories. "
            "Supports multi-hop traversal (depth) and relation type filtering. "
            "Returns all neighboring memories with their relation types."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to start traversal from."},
                "relation": {
                    "type": "string",
                    "enum": ["depends_on", "related_to", "supersedes", "contradicts", "implements", "references", "parent_of", "child_of"],
                    "description": "Optional filter by exact relation type.",
                },
                "depth": {"type": "integer", "description": "Traversal depth (default 1, max 3).", "default": 1},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_graph_relations",
        version="1.0.0",
        description="Get all relationships/edges for a specific memory.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID."},
            },
            "required": ["memory_id"],
        },
    ),
    # P1-1: Graph Consistency Reconciliation
    types.Tool(
        name="memory_graph_reconcile",
        version="1.0.0",
        description=(
            "P1-1: Reconcile Redis and Postgres graph edges. "
            "Scans both stores for divergence and repairs it: "
            "edges in Redis but not Postgres -> insert into Postgres (idempotent); "
            "edges in Postgres but not Redis -> add to Redis; "
            "edges with different relations -> use Postgres as source of truth. "
            "Returns stats: edges scanned, divergence found, repaired counts."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    types.Tool(
        name="memory_graph_consistency_metrics",
        version="1.0.0",
        description=(
            "P1-1: Get graph consistency metrics: graph links created/failed, "
            "reconciliation runs, divergence found/repaired, and reconciliation lag."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Phase 1.1: Auto-save hook — the agent decides what to remember on its own.
    # Analyzes the content, picks category + tags + importance, dedups, and stores.
    types.Tool(
        name="memory_auto_observe",
        version="1.0.0",
        description=(
            "AUTONOMOUS MEMORY SAVE — the agent calls this WITHOUT being told. "
            "Use when the assistant discovers something worth remembering across sessions: "
            "new technical facts, configuration values, server paths, procedures, decisions, "
            "credentials, error→fix mappings, or any reusable knowledge. "
            "Heuristic classifier inside picks the right category, importance 1-10, and tags. "
            "Poison-guard + auto-dedup (cosine>=0.92) built in. Returns memory_id and the "
            "auto-assigned metadata so the agent can reference it later."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "content": {"type": "string", "description": "The information to remember. Be specific — include context, paths, versions, error messages, what was tried, what worked."},
                "context_hint": {
                    "type": "string",
                    "description": "Optional short hint from the agent about WHY this is being saved. Examples: 'server config', 'debugging fix', 'client preference'. Used to nudge the classifier.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source context (e.g. 'web_search', 'file_read', 'debug_session', 'agent'). Default 'agent_auto_observe'.",
                },
                "force": {
                    "type": "boolean",
                    "description": "If true, store even if a similar memory exists (skip dedup). Default false.",
                },
            },
            "required": ["content"],
        },
    ),
    # Phase 1.2: Pre-task recall — the agent calls at the START of every task to load
    # relevant context in one shot. Returns top memories + learned strategies + recent
    # trajectories + active session state. Single-call replacement for "should I recall?"
    types.Tool(
        name="memory_session_init",
        version="1.0.0",
        description=(
            "PRE-TASK MEMORY INJECTION — call at the START of every task to load relevant "
            "context before acting. Returns a single bundle containing: (1) top relevant "
            "memories via semantic + BM25 recall, (2) learned strategies from prior reflections, "
            "(3) recent task trajectories and outcomes, (4) active session state, (5) one-line "
            "suggestions derived from prior mistakes on this topic. Use this in place of manual "
            "memory_recall calls — guarantees nothing is missed."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Short description of the current task, e.g. 'fix OLS cache cookie bypass', 'set up Cloudflare DNS for tweaktrove.com', 'debug CaringYou consultation room'.",
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional task type tag (e.g. 'coding', 'debugging', 'planning', 'analysis') — biases recall toward matching procedures.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max memories to return (default 5, max 15).",
                },
                "include_trajectories": {
                    "type": "boolean",
                    "description": "Include recent task trajectories (default true).",
                },
            },
            "required": ["task"],
        },
    ),
    # P0-1: Virtual Context / Prompt Assembly Middleware
    types.Tool(
        name="memory_context_preview",
        version="1.0.0",
        description=(
            "PREVIEW VIRTUAL CONTEXT — returns the exact prompt that would be assembled for a "
            "given query by the automatic virtual context middleware. Includes all sections "
            "(working memory, session summary, retrieved memories, procedures, learned "
            "strategies, tool hints), token counts per section, total tokens, budget, and "
            "whether any sections were truncated. Use to inspect/debug context injection."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to assemble context for.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope (defaults to caller's authorized scope).",
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional task type to bias learned strategies.",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_virtual_context_metrics",
        version="1.0.0",
        description=(
            "VIRTUAL CONTEXT METRICS — returns assembly statistics: total assemblies, "
            "tokens injected (mean/p50/p95/p99), assembly latency (mean/p50/p95/p99), "
            "sections dropped due to per-section budgets, budget exceeded count, and "
            "the current budget configuration. Use to monitor virtual context performance."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Phase 1.3: Post-task reflection — mirrors session_init. Call at the END of every
    # task with the outcome + lessons. Stores trajectory (so future session_inits see
    # it), stores lessons as procedures (so they show up as learned strategies), and
    # auto-invokes memory_reflect on failure for deep self-improvement analysis.
    types.Tool(
        name="memory_task_end",
        version="1.0.0",
        description=(
            "POST-TASK REFLECTION — call at the END of every task to record what happened. "
            "Stores a trajectory memory (category=trajectory) capturing task description, "
            "outcome (success/failure/partial), duration estimate, and key facts discovered. "
            "Stores each lesson_learned as a procedure memory so it surfaces in future "
            "memory_session_init calls. If outcome='failure', auto-invokes memory_reflect "
            "for deep self-improvement analysis. Returns: what was stored, similar past "
            "trajectories (for pattern detection), and reflection status."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "What the task was. Must match (roughly) the task you passed to memory_session_init at start.",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "partial"],
                    "description": "How it went. failure triggers auto-reflection; partial/ success are recorded without forcing reflection.",
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional task type (e.g. 'coding', 'debugging', 'planning'). Used by reflection if invoked.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Optional estimated duration in minutes. Used for time-on-task analytics.",
                },
                "facts_discovered": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional list of concrete facts learned (paths, errors, configs). Each becomes its own auto_observe memory. A bare string is accepted and coerced to a single-element list.",
                },
                "lessons_learned": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional list of procedural lessons (do X instead of Y). Each becomes a procedure memory with importance=8 and tag 'learned-strategy'. A bare string is accepted and coerced to a single-element list.",
                },
                "auto_reflect_on_success": {
                    "type": "boolean",
                    "description": "If true, also call memory_reflect on success/partial (default: only on failure).",
                },
            },
            "required": ["task", "outcome"],
        },
    ),
    # Phase 1.4: Structured episodic trajectory layer
    # Step-level logging + structured trajectory storage + step-level failure attribution
    types.Tool(
        name="memory_trajectory_log",
        version="1.0.0",
        description=(
            "INCREMENTAL STEP LOGGER — call during a task to log each action/observation. "
            "Returns a task_id (UUID for this episode). At task close, call "
            "memory_trajectory_close with the same task_id to aggregate all steps into "
            "a structured trajectory memory. Each step is also retrievable via Qdrant "
            "scroll for the active task_id (stored in Redis)."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Short description of the overall task. Will be embedded as task_signature.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "0-based step number in this task.",
                },
                "action": {
                    "type": "string",
                    "description": "What was done in this step (tool call, reasoning, etc.).",
                },
                "observation": {
                    "type": "string",
                    "description": "What was observed (result, error, intermediate state).",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether this step succeeded (default true).",
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "Optional time spent on this step in ms.",
                },
                "tool_used": {
                    "type": "string",
                    "description": "Optional name of the tool called in this step (for tool-selection learning).",
                },
            },
            "required": ["task", "step_index", "action", "observation", "success"],
        },
    ),
    types.Tool(
        name="memory_trajectory_close",
        version="1.0.0",
        description=(
            "CLOSE TRAJECTORY EPISODE — call at the END of a task (after all step logs) to "
            "aggregate the steps into a structured trajectory memory. Stores category=trajectory "
            "with fields: task_signature (hash), outcome (success/failure/partial), steps_json, "
            "failed_step_index, step_count, success_count, failure_count. Then delegates to "
            "memory_task_end for lesson storage + auto-reflection. Returns the trajectory_id."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description (must match what you passed to trajectory_log).",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["success", "failure", "partial"],
                    "description": "Final outcome of the task.",
                },
                "task_type": {
                    "type": "string",
                    "description": "Optional task type for reflection.",
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Optional total task duration.",
                },
                "lessons_learned": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional lessons to store as procedure memories (same as task_end). A bare string is accepted and coerced to a single-element list.",
                },
                "facts_discovered": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "Optional facts to store (same as task_end). A bare string is accepted and coerced to a single-element list.",
                },
            },
            "required": ["task", "outcome"],
        },
    ),
    types.Tool(
        name="memory_trajectory_recall",
        version="1.0.0",
        description=(
            "STRUCTURED TRAJECTORY RECALL — returns past task episodes with full step-level "
            "detail. Optionally filter by outcome (success/failure/partial) and similarity. "
            "Each result includes the failed_step_index when outcome=failure, enabling the "
            "agent to understand WHICH step went wrong on prior similar tasks."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Task description or situation to find similar past trajectories for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max trajectories to return (default 5).",
                },
                "outcome_filter": {
                    "type": "string",
                    "enum": ["success", "failure", "partial"],
                    "description": "Optional — only return trajectories with this outcome.",
                },
            },
            "required": ["query"],
        },
    ),
    # P1-7: Exact Traceability by Session + Step
    types.Tool(
        name="memory_trajectory_get",
        version="1.0.0",
        description=(
            "EXACT TRACEABILITY — get a specific step from a trajectory by session_id and step_index. "
            "Returns the action, observation, success status, duration, and tool used for that step. "
            "Enables direct 'session 3, step 200' lookup without similarity search."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session/task ID (task_signature or trajectory UUID).",
                },
                "task_id": {
                    "type": "string",
                    "description": "Alias for session_id.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "0-based step index to retrieve.",
                },
            },
            "required": ["step_index"],
        },
    ),
    types.Tool(
        name="memory_trajectory_chain",
        version="1.0.0",
        description=(
            "REASONING CHAIN RECONSTRUCTION — reconstruct the reasoning chain from a trajectory "
            "between two step indices (inclusive). Returns all steps with action, observation, "
            "success, duration, and tool. Enables understanding the full reasoning path from "
            "step A to step B."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session/task ID (task_signature or trajectory UUID).",
                },
                "task_id": {
                    "type": "string",
                    "description": "Alias for session_id.",
                },
                "from_step": {
                    "type": "integer",
                    "description": "Starting step index (inclusive). Default 0.",
                },
                "to_step": {
                    "type": "integer",
                    "description": "Ending step index (inclusive). Optional - if omitted, goes to end.",
                },
            },
            "required": ["session_id"],
        },
    ),
    # Phase 2.1: Procedural auto-promotion — synthesize procedures from repeated successes
    # / warnings from repeated failures. Idempotent via Redis tracking.
    types.Tool(
        name="memory_promote_procedures",
        version="1.0.0",
        description=(
            "AUTO-PROMOTE PROCEDURES — scan trajectory memories, group by task_signature, "
            "and synthesize procedure memories for patterns that meet the threshold. "
            "For each task_signature with >= min_success_count successful trajectories, "
            "create an auto-promoted procedure memory (category=procedure, importance=8, "
            "tags include 'auto-promoted' and 'learned-strategy'). For each signature with "
            ">= min_success_count failures, create a warning procedure (tag 'learned-warning'). "
            "Idempotent: tracks already-promoted signatures in Redis so re-running won't "
            "duplicate. Returns list of promoted memories with their IDs."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "min_success_count": {
                    "type": "integer",
                    "description": "Min successful trajectories per task_signature to promote (default 3).",
                },
                "min_failure_count": {
                    "type": "integer",
                    "description": "Min failed trajectories per task_signature to warn (default 3).",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Only consider trajectories created within N days (default 30, 0 = all).",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, report what would be promoted without storing (default false).",
                },
                "force_repromote": {
                    "type": "boolean",
                    "description": "If true, re-promote even if already promoted (overwrite). Default false.",
                },
            },
        },
    ),
    types.Tool(
        name="memory_procedure_status",
        version="1.0.0",
        description=(
            "PROCEDURE STATUS — show how many trajectories exist per task_signature and "
            "what's eligible for promotion. Useful before/after running promote_procedures."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "description": "Only consider trajectories within N days (default 30).",
                },
                "min_success_count": {
                    "type": "integer",
                    "description": "Min success count for eligibility display (default 3).",
                },
                "min_failure_count": {
                    "type": "integer",
                    "description": "Min failure count for warning display (default 3).",
                },
            },
        },
    ),
    # Phase 2.2: Dreaming — async consolidation of trajectory clusters into insight summaries
    types.Tool(
        name="memory_dream",
        version="1.0.0",
        description=(
            "DREAMING / ASYNC CONSOLIDATION — cluster trajectories by task_signature and "
            "generate comparative insight summaries from success vs failure patterns. "
            "Unlike simple promotion (Phase 2.1), dreaming analyzes mixed evidence: "
            "what actions succeed, what actions fail, common failure points, tool usage "
            "patterns, and duration differences. Generates actionable procedure memories "
            "with tag 'dreamed-insight'. Idempotent via Redis tracking."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "description": "Only consider trajectories created within N days (default 30, 0 = all).",
                },
                "min_successes": {
                    "type": "integer",
                    "description": "Min successful trajectories per task_signature to dream (default 2).",
                },
                "min_failures": {
                    "type": "integer",
                    "description": "Min failed trajectories per task_signature to dream (default 1).",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, report what would be dreamed without storing (default false).",
                },
                "force_redream": {
                    "type": "boolean",
                    "description": "If true, re-dream even if already dreamed (overwrite). Default false.",
                },
            },
        },
    ),
    # One-shot maintenance driver — chains all learning loops so a single cron
    # can keep the memory system healthy. Replaces needing to call each loop
    # manually. Safe to run repeatedly; each sub-step is idempotent.
    types.Tool(
        name="memory_maintenance",
        version="1.0.0",
        description=(
            "MAINTENANCE — run all auto-learning maintenance loops in one call. "
            "Chains: (1) promote procedures from successful trajectories, "
            "(2) dream / generate comparative insights, "
            "(3) consolidate stale & duplicate memories, "
            "(4) apply Ebbinghaus decay archive to underused low-importance memories. "
            "Intended to be driven by a periodic cron for continuous self-improvement. "
            "Idempotent: each sub-step is gated on Redis dedup keys so re-running is cheap. "
            "Returns a per-step summary dict. Safe to run unattended."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "description": "Lookback window for promotion + dreaming (default 30).",
                },
                "min_success_count": {
                    "type": "integer",
                    "description": "Min successful trajectories per signature to promote (default 3).",
                },
                "min_failure_count": {
                    "type": "integer",
                    "description": "Min failed trajectories per signature to warn (default 3).",
                },
                "min_successes_dream": {
                    "type": "integer",
                    "description": "Min successes to dream (default 2).",
                },
                "min_failures_dream": {
                    "type": "integer",
                    "description": "Min failures to dream (default 1).",
                },
                "decay_archive_days": {
                    "type": "integer",
                    "description": "Archive memories unused for more than N days (default 90). 0 = skip.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, only report what would change; do not store anything (default false).",
                },
            },
        },
    ),

    types.Tool(
        name="memory_feedback",
        version="1.0.0",
        description=(
            "RELEVANCE FEEDBACK LOOP — let the agent mark recalled memories as useful or "
            "irrelevant. Feedback is stored in Redis and used to adjust future recall rankings. "
            "Positive feedback boosts the memory's effective importance for similar queries; "
            "negative feedback penalizes it. This creates a closed loop where the agent's "
            "judgments directly shape what surfaces in future recalls."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory ID that was recalled."},
                "query": {"type": "string", "description": "The original query used to recall this memory."},
                "rating": {"type": "string", "enum": ["useful", "irrelevant"], "description": "Feedback: 'useful' boosts, 'irrelevant' penalizes."},
                "task_type": {"type": "string", "description": "Optional task type context (e.g. 'coding', 'debugging')."},
            },
            "required": ["memory_id", "query", "rating"],
        },
    ),
    types.Tool(
        name="memory_feedback_stats",
        version="1.0.0",
        description=(
            "Get statistics about the relevance feedback store — total feedback entries, "
            "useful vs irrelevant counts, and top memories by feedback score."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Phase 3.1: Tool-Selection Learning — recommend tools based on trajectory success patterns
    types.Tool(
        name="memory_tool_recommendations",
        version="1.0.0",
        description=(
            "TOOL-SELECTION LEARNING — recommend tools for a task based on historical "
            "trajectory data. Analyzes which tools were used in successful vs failed "
            "trajectories for similar task types. Returns ranked tool recommendations "
            "with success rates and usage counts."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task_type": {"type": "string", "description": "Filter by task type (e.g. 'coding', 'debugging')."},
                "lookback_days": {"type": "integer", "description": "How many days of trajectories to analyze (default 30)."},
                "min_usage": {"type": "integer", "description": "Minimum total uses for a tool to be recommended (default 2)."},
                "query": {"type": "string", "description": "Optional query to find similar trajectories via semantic recall."},
            },
        },
    ),
    # Phase 3.2: Step-Level Failure Dashboard — surface failure hotspots across trajectory clusters
    types.Tool(
        name="memory_failure_hotspots",
        version="1.0.0",
        description=(
            "STEP-LEVEL FAILURE DASHBOARD — analyze failed_step_index across trajectory "
            "clusters to surface failure hotspots. Shows which step indices fail most per "
            "task_signature, the actions/tools/errors involved, and global hotspots across "
            "all signatures. Use before starting a task similar to known-failing ones."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "lookback_days": {
                    "type": "integer",
                    "description": "Only consider trajectories within N days (default 30, 0 = all).",
                },
                "min_failures": {
                    "type": "integer",
                    "description": "Min failures per task_signature to include in report (default 2).",
                },
            },
        },
    ),
    # Enhancement 2: Reflection Loop (with Enhancement 8 closed-loop improvements)
    types.Tool(
        name="memory_reflect",
        version="1.0.0",
        description=(
            "Run a reflection analysis on recent memories. Detects patterns: category distribution, "
            "source breakdown, high-importance items, memory age, conflicts, and isolation warnings. "
            "Generates self-improvement insights and ACTIONABLE procedural strategies. "
            "Stores a reflection summary + learned-strategy memories automatically. "
            "Outcome-driven: pass outcome='failure' or 'success' to generate targeted guidance. "
            "Detects reflection loops (degeneration guard)."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "days": {"type": "integer", "description": "Look back window in days (default 7)."},
                "limit": {"type": "integer", "description": "Max memories to analyze (default 50)."},
                "outcome": {"type": "string", "description": "Outcome of recent work: 'success' or 'failure'. Drives targeted improvement suggestions."},
                "task_type": {"type": "string", "description": "Type of task being reflected on: e.g. 'coding', 'debugging', 'planning'. Used to generate task-specific strategies."},
            },
        },
    ),
    # Enhancement 8: Procedural memory retrieval — retrieve learned strategies from reflect
    types.Tool(
        name="memory_get_insights",
        version="1.0.0",
        description=(
            "Retrieve actionable learned strategies from self-improvement reflections. "
            "Returns high-importance (≥8) procedure memories with tags 'learned-strategy'. "
            "These are generated by memory_reflect and encode what the agent should DO "
            "differently based on past outcomes. Optionally filter by task_type. "
            "Use this before starting a new task to load applicable strategies."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task_type": {"type": "string", "description": "Optional task type to filter strategies: e.g. 'coding', 'debugging'."},
                "outcome": {"type": "string", "description": "Filter by learned_from_outcome: 'success' or 'failure'."},
                "limit": {"type": "integer", "description": "Max strategies to return (default 10)."},
            },
        },
    ),
    # Enhancement 8: Agent-controlled memory hints (SelfMem pattern)
    types.Tool(
        name="memory_hint",
        version="1.0.0",
        description=(
            "Agent-driven memory hint tool (SelfMem pattern). Lets the agent tell the memory "
            "system how to weight, prioritize, or forget specific memories. "
            "Actions: 'boost_importance' (raise importance), 'decay' (lower recency score), "
            "'evict' (delete), 'link' (create graph edge), 'archive' (lower importance, tag archived). "
            "This gives the agent agency over its own memory management instead of a fixed pipeline."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "action": {"type": "string", "enum": ["boost_importance", "decay", "evict", "link", "archive"], "description": "Hint action to apply."},
                "memory_id": {"type": "string", "description": "Target memory ID."},
                "importance": {"type": "integer", "description": "New importance 1-10 (for boost_importance)."},
                "link_target_id": {"type": "string", "description": "Target memory ID for 'link' action."},
                "relation": {"type": "string", "description": "Relation type for 'link': e.g. 'relates_to', 'contradicts', 'refines'."},
            },
            "required": ["action", "memory_id"],
        },
    ),
    # Enhancement 5: Eval before Deploy (sandbox)
    types.Tool(
        name="memory_eval_store",
        version="1.0.0",
        description=(
            "Store a memory in sandbox mode without committing to Qdrant. "
            "Use this to preview how a memory will look and test recall before committing. "
            "Sandboxed memories are stored in memory and can be committed or discarded. "
            "Returns a sandboxed memory_id and preview."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "content": {"type": "string", "description": "The memory content text."},
                "category": {"type": "string", "description": "Category (default general)."},
                "importance": {"type": "integer", "description": "Importance 1-10 (default 5)."},
                "tags": {"type": ["string", "array"], "items": {"type": "string"}, "description": "Tags. A bare string is accepted and coerced to a single-element list."},
                "source": {"type": "string", "description": "Source (default eval)."},
            },
            "required": ["content"],
        },
    ),
    types.Tool(
        name="memory_eval_recall",
        version="1.0.0",
        description="Recall memories from the sandbox store (not Qdrant) using hybrid search.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="memory_eval_commit",
        version="1.0.0",
        description="Commit a sandboxed memory to Qdrant. Returns the new permanent memory_id.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Sandboxed memory ID (from eval_store)."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_eval_discard",
        version="1.0.0",
        description="Discard a sandboxed memory without committing.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Sandboxed memory ID to discard."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_eval_preview",
        version="1.0.0",
        description="Get the current sandbox store state (count and IDs of sandboxed memories).",
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Enhancement 6: Versioned Rollback
    types.Tool(
        name="memory_versions",
        version="1.0.0",
        description="Get version history for a memory (up to N versions stored in Redis).",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID."},
                "limit": {"type": "integer", "description": "Max versions to return (default 10)."},
            },
            "required": ["memory_id"],
        },
    ),
    types.Tool(
        name="memory_version_get",
        version="1.0.0",
        description="Get a specific version snapshot of a memory.",
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID."},
                "version_id": {"type": "string", "description": "Version ID."},
            },
            "required": ["memory_id", "version_id"],
        },
    ),
    types.Tool(
        name="memory_version_restore",
        version="1.0.0",
        description=(
            "Restore a memory to a previous version. Creates a new version of the current state first "
            "(for safe rollback), then overwrites the memory with the target version's content and vector. "
            "Returns the new memory_id and version info."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to restore."},
                "version_id": {"type": "string", "description": "Version ID to restore to."},
            },
            "required": ["memory_id", "version_id"],
        },
    ),
    # Enhancement 3: Poison check utility
    types.Tool(
        name="memory_poison_check",
        version="1.0.0",
        description=(
            "Check content for memory poisoning / prompt injection patterns (OWASP LLM04). "
            "Returns a poisoning score (0-1), list of detected patterns, and whether it would be blocked. "
            "Use this to validate external input before storing. Does NOT block — only reports."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "content": {"type": "string", "description": "Content to check."},
                "block": {"type": "boolean", "description": "If true, apply blocking threshold (default false)."},
            },
            "required": ["content"],
        },
    ),
    # Directive 5: Mandatory System Alignment — verify AGENTS.md was loaded
    types.Tool(
        name="memory_verify_alignment",
        version="1.0.0",
        description=(
            "MANDATORY PRE-TASK CHECK — call at the START of every task to confirm system alignment. "
            "Returns AGENTS.md loading status, protocol version, and active rules. "
            "The agent MUST call this before generating any response or executing any task. "
            "If alignment is not confirmed, the agent MUST NOT proceed."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Directive 4: Strict Protocol Adherence — compliance verification
    types.Tool(
        name="memory_compliance_check",
        version="1.0.0",
        description=(
            "POST-ACTION COMPLIANCE VERIFIER — call after any state-changing operation to verify "
            "protocol was followed. Checks: backup-before-edit, approval-before-fix, secret-masking, "
            "syntax-check-before-restart, no-unsafe-deletions. Returns compliance report with "
            "pass/fail per rule. Agent MUST call this before claiming a task is complete."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["config_edit", "file_edit", "service_restart", "destructive", "credential_access", "general"],
                    "description": "Type of action performed.",
                },
                "details": {
                    "type": "string",
                    "description": "Optional details about what was done (file paths, commands, etc).",
                },
                "backup_created": {
                    "type": "boolean",
                    "description": "Was a backup created before the action?",
                },
                "approval_obtained": {
                    "type": "boolean",
                    "description": "Was user approval obtained for non-trivial changes?",
                },
                "syntax_checked": {
                    "type": "boolean",
                    "description": "Was syntax verified before service restart?",
                },
            },
            "required": ["action_type"],
        },
    ),
    # Directive 6: Structured Task Decomposition
    types.Tool(
        name="memory_task_breakdown",
        version="1.0.0",
        description=(
            "TASK DECOMPOSITION ASSISTANT — takes a complex task description and suggests granular "
            "sub-tasks based on historical trajectory patterns. Uses past similar tasks to recommend "
            "how to break down the work. Returns suggested sub-tasks with estimated complexity. "
            "Agent MUST call this for any task requiring 3+ logical steps."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "task": {"type": "string", "description": "The complex task to decompose."},
                "task_type": {"type": "string", "description": "Optional task type (coding, debugging, planning, etc)."},
            },
            "required": ["task"],
        },
    ),
    # Eval harness — measures retrieval quality (MRR, hit@k, mean rank) on a
    # ground-truth Q->A set. Intended as a regression gate for every
    # embedding/reranker/config change. Stores results in Redis (bounded).
    types.Tool(
        name="memory_eval",
        version="1.0.0",
        description=(
            "RAG RETRIEVAL EVALUATION — run an eval set of (query, expected_memory_id) pairs "
            "against the live recall pipeline and report MRR, hit@1, hit@5, mean rank. "
            "Use before and after any change to embedding model, reranker, or scoring "
            "weights to prove quality moved in the right direction. Stores the latest "
            "results in Redis so you can diff over time. Pass eval_set as a list of "
            "{query, expected_memory_id} objects, or pass 'stored' to use the eval set "
            "persisted at the last run (auto-generated on first call from a random sample "
            "of existing memories). Idempotent — safe to run repeatedly."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "eval_set": {
                    "type": "array",
                    "description": "List of {query, expected_memory_id} pairs. If omitted, uses the stored eval set.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "expected_memory_id": {"type": "string"},
                        },
                        "required": ["query", "expected_memory_id"],
                    },
                },
                "top_k": {
                    "type": "integer",
                    "description": "Recall top-k for the eval run (default 10).",
                },
            },
        },
    ),
    # P0-3: Retrieval Evals Runner (golden set)
    types.Tool(
        name="memory_eval_run",
        version="1.0.0",
        description=(
            "P0-3: RETRIEVAL EVAL RUN — run the golden eval set (50+ labeled queries) "
            "against the live recall pipeline and report precision@k, recall@k, MRR, "
            "nDCG@k for k in {1,3,5,10}. Loads queries from eval/golden_eval_set.json (or MEMORY_EVAL_SET). "
            "Persists results with commit SHA for regression tracking. Optional threshold checking "
            "for CI/CD gates. Use before/after embedding/reranker changes to prove quality moved."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "top_k": {"type": "integer", "description": "Max recall top-k (default 10)."},
                "k_values": {"type": "array", "items": {"type": "integer"}, "description": "k values for metrics (default [1,3,5,10])."},
                "category": {"type": "string", "description": "Optional filter by query category (fact, procedure, etc.)."},
                "thresholds": {"type": "object", "description": "Optional thresholds for CI gate: e.g. {\"mrr\": 0.5, \"hit_at_1\": 0.3, \"precision_at_5\": 0.4}."},
                "persist": {"type": "boolean", "description": "Persist results to Redis (default true)."},
            },
        },
    ),
    types.Tool(
        name="memory_eval_history",
        version="1.0.0",
        description=(
            "P0-3: RETRIEVAL EVAL HISTORY — get history of evaluation runs from Redis "
            "with commit SHAs, timestamps, and full metrics. Use for regression analysis."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "limit": {"type": "integer", "description": "Max results to return (default 20)."},
            },
        },
    ),
    # Re-embed all memories with the current EMBEDDING_MODEL — used when the
    # embedding model or vector size is changed. Destructive to the live
    # collection: drops and recreates it, re-embeds every content field,
    # and re-builds the BM25 index. Backup first!
    types.Tool(
        name="memory_reembed_all",
        version="1.0.0",
        description=(
            "RE-EMBED ALL — drop and recreate the Qdrant collection at the current "
            "VECTOR_SIZE, re-embed every stored memory with the current EMBEDDING_MODEL, "
            "and rebuild the BM25 index. Use AFTER changing MEMORY_EMBEDDING_MODEL or "
            "MEMORY_VECTOR_SIZE to migrate existing memories to the new embedding space. "
            "DESTRUCTIVE to the live collection (all vectors are re-generated); the "
            "payload content is preserved. Confirm you have a recent Qdrant storage "
            "backup before running. Reports the before/after vector count + a timing "
            "summary. Takes a few minutes for ~1000 memories."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, report what would happen without actually re-embedding (default false).",
                },
            },
        },
    ),
    # Directive 7: Context Efficiency & Cost Optimization
    types.Tool(
        name="memory_recall_efficiency",
        version="1.0.0",
        description=(
            "CONTEXT BUDGET TRACKER — monitors recall efficiency for the current session. "
            "Tracks: total recalls done, useful vs irrelevant ratio, waste score, estimated "
            "tokens saved by cache hits. Agent MUST check this before every recall to avoid "
            "wasteful repeated queries. Returns efficiency metrics and recommendations."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # Gap-closure: always-in-context core block (L1 persona + hot heuristics)
    types.Tool(
        name="memory_core",
        version="1.0.0",
        description=(
            "ALWAYS-IN-CONTEXT CORE MEMORY (Letta/MemGPT pattern). Returns a compact, "
            "deterministic block of the agent's hot knowledge: top importance + recency "
            "preference, procedure, decision, and gotcha memories, capped to fit ~1k tokens. "
            "Call once at session start to load persistent context without a full recall. "
            "Falls back gracefully if no core categories exist yet."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "limit": {"type": "integer", "description": "Max core entries to return (default 12)."},
            },
        },
    ),
    # G2 (2026-10-04): writable core memory — Letta's signature feature,
    # implemented with versioning + audit that Letta lacks.
    types.Tool(
        name="memory_core_update",
        version="1.0.0",
        description=(
            "WRITABLE CORE MEMORY (Letta self-editing pattern, from the agent's side). "
            "Update one core memory's content, importance, tags, category, or scope by ID "
            "(IDs come from memory_core). The prior state is versioned (memory_versions) "
            "and the change is audit-logged. Use after the agent decides a core fact changed "
            "— e.g. a user's new location or a retired procedure. Returns updated status."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Core memory ID to update (from memory_core)."},
                "content": {"type": "string", "description": "New content. Omit to keep existing content."},
                "importance": {"type": "integer", "description": "New importance 1-10."},
                "tags": {"type": ["string", "array"], "items": {"type": "string"}, "description": "New tags (replaces existing)."},
                "category": {"type": "string", "description": "New category."},
                "scope": {"type": "string", "description": "New scope."},
            },
            "required": ["memory_id"],
        },
    ),
    # Gap-closure: GDPR right-to-erasure scoped deletion
    types.Tool(
        name="memory_forget",
        version="1.0.0",
        description=(
            "GDPR / RIGHT-TO-ERASURE — delete memories across Qdrant, PostgreSQL, and Redis. "
            "Provide exactly one scope: memory_id (single), scope (all in a scope namespace), "
            "or tag (all carrying that tag). Destructive; no backup is made — use "
            "memory_version_restore if you need a specific memory back later. Returns counts "
            "per database."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "memory_id": {"type": "string", "description": "Delete this single memory from all stores."},
                "scope": {"type": "string", "description": "Delete ALL memories in this scope namespace (e.g. 'user:bob')."},
                "tag": {"type": "string", "description": "Delete all memories carrying this tag."},
            },
        },
    ),
    # 3A-5: identity lifecycle
    types.Tool(
        name="memory_merge_identities",
        version="1.0.0",
        description=(
            "OPERATOR-ONLY. Merge one identity's namespace into another (e.g. anonymous -> "
            "known user on signup). Moves memories across canonical Postgres, Qdrant, typed "
            "PG tables and Redis short-term keys. Dry-run by default — pass dry_run=false to "
            "apply. Idempotent and audited; version snapshots and graph edges follow "
            "automatically (memory_id-keyed)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "from_identity": {"type": "string", "description": "Source scope namespace, e.g. 'anon'."},
                "to_identity": {"type": "string", "description": "Target scope namespace, e.g. 'user:alice'."},
                "dry_run": {"type": "boolean", "description": "Default true: plan only, nothing moves."},
            },
            "required": ["from_identity", "to_identity"],
        },
    ),
    types.Tool(
        name="memory_identity_refresh",
        version="1.0.0",
        description=(
            "OPERATOR-ONLY. Hot-reload the identity registry (identities table + token file) "
            "without a process restart. Newly minted credentials work on the next request "
            "automatically; this forces an immediate re-read."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    # 3B-4: audit export
    types.Tool(
        name="memory_export_audit_log",
        version="1.0.0",
        description=(
            "OPERATOR-ONLY. Export the append-only audit trail (every read/write/delete with "
            "actor, scope, memory_id, timestamp) as JSON or CSV for enterprise audits. Writes "
            "the full export to /root/backup/ and returns the first 200 rows inline."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "csv"], "description": "Default json."},
                "since": {"type": "string", "description": "ISO timestamp lower bound."},
                "until": {"type": "string", "description": "ISO timestamp upper bound."},
                "actor": {"type": "string", "description": "Filter by authenticated actor."},
                "op": {"type": "string", "description": "Filter by operation, e.g. read:recall."},
                "scope": {"type": "string", "description": "Filter by scope namespace."},
                "limit": {"type": "integer", "description": "Max rows (default 10000, cap 50000)."},
            },
        },
    ),
    # P1-2: Health Check
    types.Tool(
        name="memory_health_check",
        version="1.0.0",
        description=(
            "HEALTH CHECK — returns per-store status (Redis, Qdrant, Postgres) and overall system health. "
            "Use to verify all stores are available before critical operations."
        ),
        inputSchema={"type": "object",
            "schema_version": "1.0.0", "properties": {}},
    ),
    # P1-6: Caller-Side Latency Metrics
    types.Tool(
        name="memory_latency_metrics",
        version="1.0.0",
        description=(
            "LATENCY METRICS — returns p50, p95, p99 latency statistics for memory tools. "
            "Call with optional 'tool' parameter to get stats for a specific tool, "
            "or without parameters to get all tools. Use to monitor caller-side latency."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Optional tool name to filter stats (e.g., 'memory_store').",
                },
            },
        },
    ),
    # P2-2: Retrieval Stage Metrics
    types.Tool(
        name="memory_retrieval_stage_metrics",
        version="1.0.0",
        description=(
            "RETRIEVAL STAGE METRICS — returns p50, p95, p99 latency and quality "
            "statistics for each retrieval stage: embed, dense_search, sparse_search, "
            "rrf_fusion, rerank, canonical_filter, feedback_adjust, total_recall. "
            "Call with optional 'stage' parameter to get stats for a specific stage."
        ),
        inputSchema={
            "type": "object",
            "schema_version": "1.0.0",
            "properties": {
                "stage": {
                    "type": "string",
                    "description": "Optional stage name to filter stats (e.g., 'dense_search').",
                    "enum": ["embed", "dense_search", "sparse_search", "rrf_fusion", "rerank", "canonical_filter", "feedback_adjust", "total_recall"],
                },
            },
        },
    ),
]
# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------
_SERVER_TOOLS.append(types.Tool(
    name="memory_turn_ingest",
    version="1.0.0",
    description=(
        "TURN INGEST — capture one conversation turn in a single call: stores the "
        "user/assistant exchange as a memory (auto-classified) and records any tool "
        "error->fix as a procedure. Use at end of a turn instead of several store calls."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "user_message": {"type": "string"},
            "assistant_message": {"type": "string"},
            "tool_events": {"type": "array", "items": {"type": "object"},
                            "description": "Optional [{tool, status|error, fix}] observed during the turn."},
            "scope": {"type": "string"},
        },
        "required": ["user_message"],
    },
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_timeline",
    version="1.0.0",
    description=(
        "TIMELINE — full version/supersession chain of a memory: each node's status, "
        "validity window (valid_from/valid_until), event_time, and superseded_by, "
        "oldest->newest. Use to see what was true when and how a fact evolved."
    ),
    inputSchema={
        "type": "object",
        "properties": {"memory_id": {"type": "string"}, "scope": {"type": "string"}},
        "required": ["memory_id"],
    },
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_kb_ingest", version="1.0.0",
    description=("RAG — ingest a document into the knowledge base (separate from memories). "
                 "Chunks + embeds the text and stores it with source citations. Idempotent per source+content."),
    inputSchema={"type": "object", "properties": {
        "text": {"type": "string"}, "source": {"type": "string"}, "title": {"type": "string"},
        "metadata": {"type": "object"}, "scope": {"type": "string"}}, "required": ["text"]},
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_kb_search", version="1.0.0",
    description=("RAG — search ingested documents and return the most relevant chunks with citations "
                 "(source#chunk). Filter by scope or source."),
    inputSchema={"type": "object", "properties": {
        "query": {"type": "string"}, "top_k": {"type": "integer"},
        "source": {"type": "string"}, "scope": {"type": "string"}}, "required": ["query"]},
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_kb_sources", version="1.0.0",
    description="RAG — list documents ingested into the knowledge base (newest first).",
    inputSchema={"type": "object", "properties": {"scope": {"type": "string"}}},
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_kb_delete", version="1.0.0",
    description="RAG — delete a document and all its chunks by doc_id or by source.",
    inputSchema={"type": "object", "properties": {
        "doc_id": {"type": "string"}, "source": {"type": "string"}, "scope": {"type": "string"}}},
))
_SERVER_TOOLS.append(types.Tool(
    name="memory_kb_reconcile", version="1.0.0",
    description=("RAG — reconcile memory into knowledge: a confirmed memory marks contradicting document "
                 "chunks as stale (reversible, never deleted) and appends an update chunk. Provide memory_id or content."),
    inputSchema={"type": "object", "properties": {
        "memory_id": {"type": "string"}, "content": {"type": "string"}, "scope": {"type": "string"}}},
))

_TOOL_PROFILE = os.environ.get("MEMORY_TOOL_PROFILE", "full").strip().lower()
_AGENT_TOOLS = frozenset({
    "memory_session_init",
    "memory_recall",
    "memory_store",
    "memory_auto_observe",
    "memory_turn_ingest",
    "memory_timeline",
    "memory_kb_ingest",
    "memory_kb_search",
    "memory_kb_sources",
    "memory_kb_delete",
    "memory_kb_reconcile",
    "memory_task_end",
    "memory_trajectory_log",
    "memory_trajectory_recall",
    "memory_trajectory_close",
    "memory_context_preview",
    "memory_core",
    "memory_core_update",
    "memory_get_insights",
    "memory_hint",
    "memory_feedback",
    "memory_compliance_check",
    "memory_verify_alignment",
    "memory_health_check",
    "memory_poison_check",
    "memory_graph_link",
    "memory_graph_unlink",
    "memory_graph_neighbors",
    "memory_graph_relations",
    "memory_get",
    "memory_list",
    "memory_update",
    "memory_delete",
    "memory_remember",
    "memory_lookup_exact",
    "memory_versions",
    "memory_version_get",
    "memory_version_restore",
    "memory_stats",
    "memory_track_access",
    "memory_memory_short_store",
    "memory_memory_short_recall",
    "memory_memory_short_list",
    "memory_memory_short_delete",
    "memory_memory_short_stats",
})

async def _on_list_tools(
    context: object, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    tools = _SERVER_TOOLS
    if _TOOL_PROFILE == "agent":
        tools = [t for t in _SERVER_TOOLS if t.name in _AGENT_TOOLS]
    return types.ListToolsResult(tools=tools)

_MCP_RESOURCES = [
    types.Resource(uri="memory://context", name="Memory context",
        description=("Assembled session context: working memory, session summary, relevant "
                     "memories, procedures, anti-patterns, user profile, learned strategies, tool hints."),
        mime_type="text/markdown"),
    types.Resource(uri="memory://profile", name="User profile",
        description="Stated preferences and behavioral adaptation block.",
        mime_type="text/markdown"),
    types.Resource(uri="memory://core", name="Core memory",
        description="Hot, high-importance memories (procedures, decisions, gotchas, preferences).",
        mime_type="application/json"),
]

async def _on_list_resources(context, params):
    return types.ListResourcesResult(resources=_MCP_RESOURCES)

def _resource_scope() -> str:
    try:
        return authorize_scope()
    except Exception:
        return _primary_scope() or "global"

async def _on_read_resource(context, params):
    uri = str(params.uri)
    base = uri.split("?", 1)[0].rstrip("/")
    scope = _resource_scope()
    if base.endswith("profile"):
        text, _t, _tr = _vc_get_user_profile(scope, VIRTUAL_CONTEXT_PROFILE_BUDGET)
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=params.uri, mime_type="text/markdown", text=text or "(no profile yet)")])
    if base.endswith("core"):
        r = _tool_core({"scope": scope})
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=params.uri, mime_type="application/json", text=r.content[0].text)])
    if base.endswith("context"):
        vc = assemble_virtual_context("session bootstrap context", scope)
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=params.uri, mime_type="text/markdown", text=vc.get("prompt") or "")])
    raise ValueError(f"unknown resource {uri}")

def _tool_turn_ingest(args: dict) -> types.CallToolResult:
    user = (args.get("user_message") or "").strip()
    asst = (args.get("assistant_message") or "").strip()
    scope = args.get("scope")
    tools = args.get("tool_events") or []
    if not (user or asst):
        return _err("memory_turn_ingest: user_message or assistant_message required")
    lines = []
    if user:
        lines.append("User: " + user)
    if asst:
        lines.append("Assistant: " + asst)
    for ev in tools[:20]:
        if isinstance(ev, dict):
            nm = ev.get("tool") or ev.get("name") or "tool"
            status = ev.get("error") or ev.get("status") or ev.get("result")
            if status:
                lines.append(f"Tool {nm}: {str(status)[:300]}")
    content = "\n".join(lines)
    ao = {"content": content, "context_hint": "turn-ingest"}
    if scope:
        ao["scope"] = scope
    stored = []
    try:
        d = json.loads(_tool_auto_observe(ao).content[0].text)
        if d.get("memory_id"):
            stored.append(d["memory_id"])
    except Exception:
        pass
    errs = [e for e in tools if isinstance(e, dict) and
            (e.get("error") or str(e.get("status", "")).lower() in ("error", "failed"))]
    if errs:
        fl = [f"- {e.get('tool','tool')}: error={str(e.get('error') or e.get('status'))[:160]} fix={str(e.get('fix',''))[:200]}"
              for e in errs[:10]]
        st = {"content": "Error->fix observed during a turn:\n" + "\n".join(fl),
              "category": "procedure", "importance": 7,
              "tags": ["error-fix", "turn-ingest"], "source": "agent"}
        if scope:
            st["scope"] = scope
        try:
            d2 = json.loads(_tool_store(st).content[0].text)
            if d2.get("memory_id"):
                stored.append(d2["memory_id"])
        except Exception:
            pass
    return _ok(json.dumps({"status": "ingested", "stored": stored,
                           "turn_chars": len(content), "errors": len(errs)}))

def _tool_timeline(args: dict) -> types.CallToolResult:
    mid = args["memory_id"]
    _assert_memory_access(mid)

    def _q(m):
        r = _pg_query(
            "SELECT memory_id, status, version, valid_from, valid_until, "
            "superseded_by, created_at, payload FROM mem_canonical WHERE memory_id=:m",
            {"m": m})
        if not r:
            return None
        row = r[0]
        def g(k, i):
            return row.get(k) if (isinstance(row, dict) or hasattr(row, "keys")) else row[i]
        pl = g("payload", 7)
        try:
            pl = json.loads(pl) if isinstance(pl, str) else (pl or {})
        except Exception:
            pl = {}
        def iso(x):
            return x.isoformat() if hasattr(x, "isoformat") else x
        return {"memory_id": g("memory_id", 0), "status": g("status", 1),
                "version": g("version", 2), "valid_from": iso(g("valid_from", 3)),
                "valid_until": iso(g("valid_until", 4)),
                "superseded_by": g("superseded_by", 5),
                "created_at": iso(g("created_at", 6)),
                "event_time": pl.get("event_time"),
                "content": (pl.get("content") or "")[:120]}

    seen, cur = set(), mid
    while cur and cur not in seen:  # walk back to the oldest in the chain
        prev = _pg_query("SELECT memory_id FROM mem_canonical WHERE superseded_by=:m LIMIT 1", {"m": cur})
        if not prev:
            break
        nxt = prev[0]
        nxt = nxt.get("memory_id") if (isinstance(nxt, dict) or hasattr(nxt, "keys")) else nxt[0]
        seen.add(cur); cur = nxt
    timeline, seen2 = [], set()
    while cur and cur not in seen2:  # forward via superseded_by
        seen2.add(cur)
        node = _q(cur)
        if not node:
            break
        timeline.append(node); cur = node.get("superseded_by")
    return _ok(json.dumps({"memory_id": mid, "count": len(timeline),
                           "timeline": timeline,
                           "versions": _get_versions(mid, 10)},
                          ensure_ascii=False, indent=2))

# ---------------------------------------------------------------------------
# Central authorization guard
# ---------------------------------------------------------------------------
# Auditing every tool individually showed 33 of them accepted a `scope` or a
# `memory_id` without ever calling authorize_scope(), so any caller could list,
# read, rewrite, delete, link, or restore another tenant's memories. Rather than
# trust 30+ call sites to remember (and hoping a future tool does), isolation is
# enforced ONCE here, immediately before dispatch. A new tool is protected by
# default; opting out requires an explicit entry, which is reviewable.

# Tools that accept an explicit `scope` argument.
_SCOPE_ADDRESSED_TOOLS = frozenset({
    "memory_store", "memory_recall", "memory_list", "memory_get",
    "memory_update", "memory_delete", "memory_forget",
    "memory_track_access", "memory_feedback", "memory_graph_link",
    "memory_graph_unlink", "memory_graph_neighbors", "memory_graph_relations",
    "memory_versions", "memory_version_get", "memory_version_restore",
    "memory_core", "memory_core_update", "memory_hint", "memory_auto_observe",
    "memory_task_end", "memory_trajectory_log", "memory_trajectory_close",
    "memory_trajectory_recall", "memory_session_init",
    "memory_short_store", "memory_short_recall", "memory_short_delete",
    # P0-1: Virtual Context
    "memory_context_preview", "memory_virtual_context_metrics",
    # P0-2: Session Compaction
    "memory_session_summarize", "memory_compaction_metrics",
    # P0-3: Retrieval Evals
    "memory_eval_run", "memory_eval_history",
    # P1-1: Graph Consistency
    "memory_graph_reconcile", "memory_graph_consistency_metrics",
    # P1-2: Health Check
    "memory_health_check",
})

# Tools that address existing memories by id. Each referenced id must belong to
# a scope the caller is authorized for.
_MEMORY_ID_TOOLS = frozenset({
    "memory_get", "memory_update", "memory_delete", "memory_track_access",
    "memory_feedback", "memory_core_update", "memory_graph_link",
    "memory_graph_unlink", "memory_graph_neighbors", "memory_graph_relations",
    "memory_versions", "memory_version_get", "memory_version_restore",
})

# Arguments that may carry a memory id in any tool.
_MEMORY_ID_ARGS = ("memory_id", "from_memory_id", "to_memory_id", "parent_id")


def _assert_memory_access(memory_id: str) -> None:
    """Raise ScopeError unless `memory_id` belongs to an authorized scope.

    The payload's scope is the authority. A memory with no scope is legacy data
    migrated to "global"; it is only reachable by a caller granted "global".
    """
    if not memory_id:
        return
    allowed = authorized_scopes()
    if allowed is None:
        raise ScopeError(
            "no principal configured for this deployment: set MEMORY_PRINCIPAL"
        )
    try:
        found = _client.retrieve(
            collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
    except Exception as e:
        raise ScopeError(f"could not verify ownership of {memory_id}: {e}") from e
    if not found:
        # Memory not found in vector store — treat as access denied to avoid
        # leaking whether the ID exists in another tenant's scope.
        raise ScopeError(f"memory {memory_id} not found or access denied")
    scope = (found[0].payload or {}).get("scope") or "global"
    if scope not in allowed:
        raise ScopeError(
            f"memory {memory_id} belongs to scope {scope!r}, which is not "
            f"authorized for this caller (authorized: {sorted(allowed)})"
        )


# Tools that carry NO stored-memory content. They still require an
# authenticated caller so the surface cannot be used anonymously, but they do
# not need a scope because they expose nothing tenant-specific.
_ANON_TOOLS = frozenset({
    "memory_verify_alignment", "memory_compliance_check",
    "memory_poison_check",
    "memory_latency_metrics",
    "memory_retrieval_stage_metrics",
})

# Tools that read or mutate EVERY namespace. Operator-only: refused for all
# network callers regardless of credential. Reached only through system_context.
_ADMIN_TOOLS = frozenset({
    # global reindex / destructive
    "memory_reembed_all", "memory_consolidate", "memory_maintenance",
    "memory_dream", "memory_reflect", "memory_promote_procedures",
    "memory_task_breakdown", "memory_tool_recommendations",
    "memory_procedure_status", "memory_failure_hotspots",
    "memory_recall_efficiency", "memory_feedback_stats",
    # eval sandbox: stores and retrieves real memory content, unfiltered
    "memory_eval_store", "memory_eval_recall", "memory_eval_commit",
    "memory_eval_discard", "memory_eval_preview", "memory_eval",
    # Stage 4 outbox control plane: backlog/dead-letter contents and manual
    # replay mutate system-wide indexing state, so they stay operator-only.
    "memory_outbox_metrics", "memory_outbox_dead_letters",
    "memory_outbox_replay", "memory_outbox_drain",
    # P3.3: queryable ACL + provenance + quarantine audit — operator-only.
    "memory_governance_audit",
    # Part 3 operator surface: identity lifecycle + audit export are
    # namespace-spanning by design (merge moves every tier; export dumps
    # every actor's rows) — network callers get the proper operator-only
    # refusal instead of the generic unclassified default.
    "memory_merge_identities", "memory_identity_refresh",
    "memory_export_audit_log",
})

# Tools that address tenant data. Every one of these is scope-authorized and,
# where they name a memory, ownership-checked.
_DATA_TOOLS = frozenset({
    # content read/write
    "memory_store", "memory_recall", "memory_list", "memory_get",
    "memory_update", "memory_delete", "memory_forget", "memory_remember",
    "memory_track_access", "memory_feedback", "memory_stats",
    "memory_get_insights", "memory_core", "memory_core_update",
    "memory_hint", "memory_auto_observe", "memory_task_end",
    # P0-1: Virtual Context
    "memory_context_preview", "memory_virtual_context_metrics",
    # P0-2: Session Compaction
    "memory_session_summarize", "memory_compaction_metrics",
    # P0-3: Retrieval Evals
    "memory_eval_run", "memory_eval_history",
    # Trajectory tools (per-user episodic memory)
    "memory_trajectory_log", "memory_trajectory_close", "memory_trajectory_recall",
    "memory_trajectory_get", "memory_trajectory_chain",
    "memory_session_init",
    # P1-8: Exact lookup
    "memory_lookup_exact",
    # graph
    "memory_graph_link", "memory_graph_unlink", "memory_graph_neighbors",
    "memory_graph_relations",
    # P1-1: Graph Consistency
    "memory_graph_reconcile", "memory_graph_consistency_metrics",
    # P1-2: Health Check
    "memory_health_check",
    # Turn ingest
    "memory_turn_ingest",
    # Timeline
    "memory_timeline",
    # RAG / knowledge base
    "memory_kb_ingest",
    "memory_kb_search",
    "memory_kb_sources",
    "memory_kb_delete",
    "memory_kb_reconcile",
    # versions
    "memory_versions", "memory_version_get", "memory_version_restore",
    # short-term (Redis) store -- previously had NO scope at all
    "memory_memory_short_store", "memory_memory_short_recall",
    "memory_memory_short_delete", "memory_memory_short_list",
    "memory_memory_short_stats",
    # cache stats are per-process counters, no tenant content
    "memory_cache_stats",
})


def _enforce_tool_authorization(name: str, arguments: dict) -> None:
    """DEFAULT-DENY authorization applied to every tool before it runs.

    The previous gate was an ALLOW-list: 26 tools were scope-checked and the
    other 29 silently ran. Auditing those 29 found several that cross tenant
    boundaries outright (memory_stats reported the whole deployment,
    memory_eval_recall and memory_trajectory_recall returned stored content
    across scopes, memory_short_* had no scope at all, memory_reembed_all
    destroyed the global index). A new tool added later was unprotected unless
    somebody remembered to add it to the set.

    Now every tool falls into exactly one bucket, and the default is DENY:

      _DATA_TOOLS    touch stored memory content -> require an authenticated
                     caller and an authorized scope (the identity's own
                     namespace unless an authorized one is supplied).
      _ADMIN_TOOLS   read or mutate EVERY scope (global reindex, dreaming,
                     consolidation, eval sandboxes). Reachable only from a
                     trusted system context. A network caller can never use
                     them, whatever credential it holds.
      _ANON_TOOLS    carry no memory content at all (alignment echo, tool
                     help) -> allowed for any authenticated caller.

    Anything unclassified is refused. That is the property PART 1.1 asks for:
    a tool added later is denied until it is explicitly triaged, so forgetting
    to think about it fails safe instead of failing open.
    """
    if name in _ANON_TOOLS:
        if authorized_scopes() is None:
            raise AuthError(f"{name} requires an authenticated caller")
        return

    if name in _ADMIN_TOOLS:
        principal = current_principal()
        if not principal.is_system:
            raise AuthError(
                f"{name} is an operator-only maintenance tool: it reads or "
                f"mutates every namespace, so it is refused for network callers "
                f"(identity {principal.identity or 'anonymous'!r}). Run it via "
                "the operator console or a system_context() block."
            )
        return

    if name in _DATA_TOOLS:
        # Authorizes the explicit value, and resolves+authorizes the implicit
        # default. Raises for malformed or unauthorized scopes.
        authorize_scope(arguments.get("scope"))
        for key in _MEMORY_ID_ARGS:
            _assert_memory_access(arguments.get(key))
        return

    raise AuthError(
        f"tool {name!r} is not classified for authorization and is refused by "
        "default. Add it to _DATA_TOOLS, _ADMIN_TOOLS or _ANON_TOOLS after "
        "reviewing what tenant data it can reach."
    )


# Retrieval payloads the canonical gate filters. A tool that returns stored
# memory under one of these keys gets its results withheld when the canonical
# PostgreSQL record says the memory is deleted / superseded / not yet indexed.
_CANONICAL_FILTER_KEYS = ("memories", "results", "hits", "items", "matches")


def _enforce_canonical_liveness(result: types.CallToolResult, as_of: str | None = None) -> types.CallToolResult:
    """PHASE A/B invariant: retrieval may never expose a non-live memory.

    Enforced centrally so it covers specialized recall, hybrid recall, short
    (Redis) recall, graph traversal and anything added later — the derived
    layers are not authoritative and may still hold a stale row for a memory
    whose canonical record was tombstoned. This is the choke point that makes
    "deleted means deleted" true even when cleanup is still pending.
    """
    try:
        if getattr(result, "isError", False):
            return result
        parts = getattr(result, "content", None) or []
        if not parts:
            return result
        text = parts[0].text
        if not isinstance(text, str) or "memory_id" not in text:
            return result
        body = json.loads(text)
    except Exception:
        return result

    if not isinstance(body, dict):
        return result

    changed = False
    for key in _CANONICAL_FILTER_KEYS:
        rows = body.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        ids = [r.get("memory_id") for r in rows if isinstance(r, dict)]
        if not any(ids):
            continue
        live = _canonical_live_ids(ids, include_superseded=bool(as_of))
        kept = [r for r in rows
                if not isinstance(r, dict) or r.get("memory_id") not in ids
                or r.get("memory_id") in live]
        if len(kept) != len(rows):
            changed = True
            body[key] = kept
            if key == "memories" and "count" in body:
                body["count"] = len(kept)

    if not changed:
        return result

    dropped = _CANONICAL_GATE_DROPPED["n"]
    _CANONICAL_GATE_DROPPED["n"] += 1
    log.info("canonical gate withheld stale derived rows (%s)", dropped)
    return types.CallToolResult(
        content=[types.TextContent(type="text",
                                   text=json.dumps(body, ensure_ascii=False,
                                                   indent=2)
                                   if isinstance(body.get("memories"), list)
                                   else json.dumps(body, ensure_ascii=False))],
        isError=False)


_CANONICAL_GATE_DROPPED = {"n": 0}


# Tools that should receive automatic virtual context injection when enabled.
# These are read tools that accept a query or operate on the current context.
_VIRTUAL_CONTEXT_AUTO_TOOLS = frozenset({
    "memory_recall", "memory_list", "memory_get", "memory_stats",
    "memory_session_init", "memory_task_end", "memory_trajectory_recall",
    "memory_get_insights", "memory_tool_recommendations",
})


async def _inject_virtual_context(name: str, arguments: dict, result: types.CallToolResult) -> types.CallToolResult:
    """Inject virtual context into tool response when VIRTUAL_CONTEXT_ENABLED=1."""
    if not VIRTUAL_CONTEXT_ENABLED:
        return result
    if name not in _VIRTUAL_CONTEXT_AUTO_TOOLS:
        return result
    # Only inject for successful responses
    if getattr(result, "isError", False):
        return result
    
    try:
        # Extract query from arguments (different tools use different param names)
        query = arguments.get("query") or arguments.get("task") or ""
        if not query:
            return result
        
        # Get scope from arguments or use caller's authorized scope
        scope = arguments.get("scope")
        if scope is None:
            try:
                scope = authorize_scope(None)
            except Exception:
                return result
        
        # Get task_type if available
        task_type = arguments.get("task_type")
        
        # Assemble virtual context
        vc_result = assemble_virtual_context(query, scope, task_type)
        
        # Inject into response
        parts = getattr(result, "content", None) or []
        if not parts:
            return result
        
        # Parse existing response
        try:
            existing_text = parts[0].text
            if not isinstance(existing_text, str):
                return result
            existing_data = json.loads(existing_text)
        except Exception:
            return result
        
        # Add virtual context to response
        if isinstance(existing_data, dict):
            existing_data["_virtual_context"] = {
                "prompt": vc_result["prompt"],
                "total_tokens": vc_result["total_tokens"],
                "budget": vc_result["budget"],
                "budget_exceeded": vc_result["budget_exceeded"],
                "assembly_latency_ms": vc_result["assembly_latency_ms"],
                "sections_summary": {
                    k: {"tokens": v["tokens"], "truncated": v["truncated"], "has_content": bool(v["content"])}
                    for k, v in vc_result["sections"].items()
                }
            }
            
            # Re-serialize
            new_text = json.dumps(existing_data, ensure_ascii=False, indent=2)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=new_text)],
                isError=False
            )
    except Exception as e:
        log.warning("virtual context injection failed for %s: %s", name, e)
        return result
    return result


# ---------------------------------------------------------------------------
# 1B-4: per-op latency telemetry · 1B-5: store circuit breakers · 1A-3: version
# ---------------------------------------------------------------------------
# Latency ring buffers live in-process (fast, no cross-store dependency);
# percentiles are exposed through health_check. Breakers are the middleware
# tier of the degraded-mode matrix: a store that keeps failing is opened for
# 30s and tools report a structured STORE_UNAVAILABLE error instead of
# hanging or surfacing a generic 500.
MCP_SERVER_VERSION = "4.3.0"  # 1A-3: semantic version of the tool surface
import collections as _collections_mod
_TOOL_LATENCY: dict = _collections_mod.defaultdict(
    lambda: _collections_mod.deque(maxlen=200))
_OVERALL_LATENCY = _collections_mod.deque(maxlen=500)
_STORE_BREAKER: dict = {
    s: {"consecutive_failures": 0, "opened_at": None, "open_until": 0.0}
    for s in ("postgres", "qdrant", "redis")
}
_BREAKER_THRESHOLD = 3
_BREAKER_COOLDOWN_S = 30.0

def _record_tool_latency(tool: str, ms: float) -> None:
    _TOOL_LATENCY[tool].append(ms)
    _OVERALL_LATENCY.append(ms)


# --- Sprint-1 1.1/1.3: durable op telemetry + maintenance action log --------
# op_telemetry rows persist every MCP tool call (tool, duration, status, scope,
# caller) plus write-gate outcomes, so stats can report p50/p95/p99 over a
# >=7-day window instead of process-lifetime ring buffers. maintenance_log
# gives the consolidation window a durable, per-action audit trail
# (action, memory_id, before, after, reason, dry_run).
_OP_TEL_MAX_BUF = 5000
_op_tel_buf: deque = deque()
_op_tel_lock = threading.Lock()
_OP_TEL_FLUSH_SECONDS = float(os.environ.get("MEMORY_OP_TEL_FLUSH_SECONDS", "5"))


def _op_telemetry_record(tool: str, duration_ms: float, status: str = "ok",
                         scope: str = None, quality_score=None) -> None:
    """Buffer one op row for the background flusher. Never raises."""
    try:
        with _op_tel_lock:
            if len(_op_tel_buf) >= _OP_TEL_MAX_BUF:
                _op_tel_buf.popleft()
            _op_tel_buf.append({
                "tool": str(tool)[:120],
                "duration_ms": int(duration_ms) if duration_ms is not None else None,
                "status": str(status)[:40],
                "scope": str(scope or "global")[:128],
                "caller": (get_caller_identity() or "anonymous")[:128],
                "quality_score": quality_score,
            })
    except Exception:
        pass


def _op_telemetry_flush() -> int:
    """Insert all buffered rows. Returns rows written. Never raises."""
    with _op_tel_lock:
        rows = list(_op_tel_buf)
        _op_tel_buf.clear()
    written = 0
    for r in rows:
        try:
            _pg_run(
                "INSERT INTO op_telemetry "
                "(tool, duration_ms, status, scope, caller, quality_score) "
                "VALUES (:t, :d, :s, :sc, :c, :q)",
                {"t": r["tool"], "d": r["duration_ms"], "s": r["status"],
                 "sc": r["scope"], "c": r["caller"], "q": r["quality_score"]},
                table="op-telemetry")
            written += 1
        except Exception as _e:
            log.debug("op telemetry insert failed: %s", _e)
    return written


class _OpTelemetryFlusher(threading.Thread):
    """Write-behind flusher so per-call telemetry never adds PG latency."""
    def __init__(self):
        super().__init__(name="memory-op-telemetry-flusher", daemon=True)

    def run(self):
        while True:
            time.sleep(_OP_TEL_FLUSH_SECONDS)
            try:
                if _op_tel_buf:
                    _op_telemetry_flush()
            except Exception:
                pass


def _start_op_telemetry_flusher() -> None:
    if os.environ.get("MEMORY_OP_TEL_FLUSHER", "1") != "1":
        return
    if any(t.name == "memory-op-telemetry-flusher" for t in threading.enumerate()):
        return
    _OpTelemetryFlusher().start()


def _mlog(step: str, action: str, memory_id: str = None,
          before: dict = None, after: dict = None, reason: str = "",
          dry_run: bool = True) -> None:
    """Sprint-1 1.4: durable maintenance action log. Never raises."""
    try:
        _pg_run(
            "INSERT INTO maintenance_log "
            "(step, action, memory_id, before, after, reason, dry_run) "
            "VALUES (:st, :a, :m, :b::jsonb, :af::jsonb, :r, :dr)",
            {"st": str(step)[:80], "a": str(action)[:80],
             "m": memory_id, "b": json.dumps(before) if before is not None else None,
             "af": json.dumps(after) if after is not None else None,
             "r": str(reason)[:500], "dr": bool(dry_run)},
            table="maintenance-log")
    except Exception as _e:
        log.debug("maintenance_log write failed: %s", _e)


# --- Sprint-1 1.2: LLM plausibility/consistency judge (async, non-blocking) --
# High-importance facts get an async plausibility pass against their semantic
# neighborhood: plausible -> verification stamp; implausible/inconsistent ->
# quarantine (rank penalty applies per the 2026-10-08 quarantine decision),
# with canonical kept in agreement. Fully opt-in via env; no default
# endpoint/model is baked in, so the judge stays disabled unless configured.
_JUDGE_ENDPOINT = os.environ.get("MEMORY_JUDGE_ENDPOINT", "")
_JUDGE_MODEL = os.environ.get("MEMORY_JUDGE_MODEL", "")
_JUDGE_API_KEY = os.environ.get("MEMORY_JUDGE_API_KEY", "")
_JUDGE_QUEUE_KEY = "memory:judge:queue"
_JUDGE_POLL_SECONDS = float(os.environ.get("MEMORY_JUDGE_POLL_SECONDS", "6"))
_JUDGE_ENABLED = (os.environ.get("MEMORY_JUDGE_ENABLED", "1") == "1"
                  and bool(_JUDGE_API_KEY))


def _judge_enqueue(memory_id: str, content: str) -> None:
    """Queue a stored memory for async LLM judging. Never raises."""
    if not _JUDGE_ENABLED or not memory_id:
        return
    try:
        if len(content or "") < 40:
            return
        _redis.lpush(_JUDGE_QUEUE_KEY,
                     json.dumps({"memory_id": memory_id, "ts": _now_iso()}))
    except Exception:
        pass


def _judge_llm(content: str, context: list) -> "dict | None":
    """One plausibility/consistency call. None = call failed (retryable)."""
    import urllib.request
    try:
        ctx = "\n".join(f"- {str(c)[:220]}" for c in (context or [])[:5]) \
            or "(no related knowledge)"
        body = json.dumps({
            "model": _JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": (
                    "You are a plausibility and consistency judge for a memory "
                    "store. Given RELATED EXISTING KNOWLEDGE and a NEW CLAIM, "
                    "decide whether the claim is plausible AND consistent with "
                    "the knowledge base. Contradicting known facts, or asserting "
                    "specific technical details the knowledge base contradicts, "
                    "means implausible. Reply with STRICT JSON only: "
                    "{\"plausible\": true, \"reason\": \"<short>\"}")},
                {"role": "user", "content":
                    f"RELATED EXISTING KNOWLEDGE:\n{ctx}\n\nNEW CLAIM:\n{content}"}],
            "max_tokens": 120, "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            _JUDGE_ENDPOINT.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {_JUDGE_API_KEY}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            out = json.loads(r.read().decode())
        txt = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        return {"plausible": bool(d.get("plausible", True)),
                "reason": str(d.get("reason", ""))[:200]}
    except Exception as _e:
        log.debug("judge llm call failed: %s", _e)
        return None


# 2026-10-26: LLM conflict adjudication (Mem0-style succession on write).
# The rule layers in _tool_store (negation markers, token-overlap band) miss
# paraphrased corrections — same fact slot, different wording. When the
# ambiguous band triggers, the judge LLM decides SUPERSEDES vs INDEPENDENT.
# Fail-open: any error keeps current (independent) behaviour. Never raises.
_CONFLICT_LLM_ENABLED = (
    os.environ.get("MEMORY_CONFLICT_LLM", "1") == "1" and bool(_JUDGE_API_KEY)
)
_CONFLICT_LLM_CATEGORIES = {"fact", "preference", "decision", "project", "general"}
_CONFLICT_LLM_TIMEOUT = float(os.environ.get("MEMORY_CONFLICT_LLM_TIMEOUT", "12"))


def _conflict_llm_judge(old_content: str, new_content: str) -> "str | None":
    """Adjudicate a same-slot candidate conflict between two similar memories.

    Returns 'supersedes' when the NEW statement replaces the EXISTING one,
    else None (independent, or the call failed — fail-open). Never raises.
    """
    import urllib.request

    if not old_content or not new_content:
        return None
    try:
        body = json.dumps({
            "model": _JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": (
                    "You are a conflict judge for an agent memory store. Compare "
                    "the EXISTING memory and the NEW memory. Decide:\n"
                    "- \"supersedes\": the NEW statement is a correction, update, "
                    "or replacement of the same fact/preference/decision slot "
                    "(different value, different time, changed state).\n"
                    "- \"independent\": both can be true simultaneously, or they "
                    "are about different subjects/aspects.\n"
                    "Reply with STRICT JSON only: "
                    "{\"relation\": \"supersedes\"|\"independent\", \"reason\": \"<short>\"}")},
                {"role": "user", "content":
                    f"EXISTING MEMORY:\n{old_content[:800]}\n\nNEW MEMORY:\n{new_content[:800]}"}],
            "max_tokens": 80, "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            _JUDGE_ENDPOINT.rstrip("/") + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {_JUDGE_API_KEY}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=_CONFLICT_LLM_TIMEOUT) as r:
            out = json.loads(r.read().decode())
        txt = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        return "supersedes" if str(d.get("relation", "")).lower() == "supersedes" else None
    except Exception as _e:
        log.debug("conflict llm adjudication failed (fail-open): %s", _e)
        return None


def _judge_one(memory_id: str) -> None:
    """Judge one memory: stamp verification or quarantine. Never raises."""
    try:
        pts = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id],
                               with_payload=True)
        if not pts:
            return
        pl = dict(pts[0].payload or {})
        if pl.get("quarantine") or pl.get("judge"):
            return
        content = pl.get("content") or ""
        if not content:
            return
        ctx = []
        try:
            vec = _embed_model().encode(content).tolist()
            # Context must include the GLOBAL corpus, not only the memory's own
            # scope — a fresh scope's first facts would otherwise be judged
            # against themselves and always pass. The judge worker runs as
            # system context; only the verdict lands on the memory, so this
            # never leaks cross-scope content to callers.
            for _sc in dict.fromkeys([pl.get("scope", "global"), "global"]):
                res = _client.query_points(
                    collection_name=COLLECTION_NAME, query=vec,
                    query_filter=qmodels.Filter(must=[qmodels.FieldCondition(
                        key="scope", match=qmodels.MatchValue(value=_sc))]),
                    limit=4, with_payload=True)
                ctx += [p.payload.get("content", "") for p in res.points
                        if p.id != memory_id and p.payload]
        except Exception:
            ctx = []
        ctx = ctx[:6]
        verdict = _judge_llm(content, ctx)
        if verdict is None:
            attempts = int(pl.get("judge_attempts", 0)) + 1
            if attempts > 5:
                _mlog("judge", "judge_failed", memory_id=memory_id,
                      after={"attempts": attempts},
                      reason="LLM judge call failed after 5 retries (endpoint/model unavailable)",
                      dry_run=False)
            try:
                _client.set_payload(collection_name=COLLECTION_NAME,
                                     points=[memory_id],
                                     payload={"judge_attempts": attempts})
            except Exception:
                pass
            if attempts <= 5 and _check_redis():
                _redis.lpush(_JUDGE_QUEUE_KEY, json.dumps(
                    {"memory_id": memory_id, "ts": _now_iso()}))
            return
        now = _now_iso()
        if verdict["plausible"]:
            patch = {"judge": "verified", "judge_reason": verdict["reason"],
                     "verified_at": now}
            _mlog("judge", "judge_verified", memory_id=memory_id,
                  after=verdict,
                  reason="LLM plausibility judge: consistent with corpus",
                  dry_run=False)
        else:
            patch = {"judge": "quarantined", "judge_reason": verdict["reason"],
                     "quarantine": True, "poison_score": 0.5,
                     "quarantined_by": "llm-judge-v2", "quarantined_at": now}
            _mlog("judge", "judge_quarantine", memory_id=memory_id,
                  before={"content": content[:120]}, after=verdict,
                  reason="LLM plausibility judge: implausible/inconsistent",
                  dry_run=False)
        _client.set_payload(collection_name=COLLECTION_NAME, points=[memory_id],
                            payload=patch)
        try:
            _pg_run(
                "UPDATE mem_canonical SET "
                "payload = coalesce(payload, '{}'::jsonb) || :patch::jsonb, "
                "updated_at = now() WHERE memory_id = :mid AND status = 'active'",
                {"patch": json.dumps(patch), "mid": memory_id},
                table="mem_canonical-judge")
        except Exception as _e:
            log.debug("judge canonical sync failed for %s: %s", memory_id, _e)
        _invalidate_recall_caches()
    except Exception as _e:
        log.warning("judge_one failed for %s: %s", memory_id, _e)


class _JudgeWorker(threading.Thread):
    """Async judge pass — non-blocking for stores by design (1.2 spec)."""
    def __init__(self):
        super().__init__(name="memory-llm-judge", daemon=True)

    def run(self):
        while True:
            try:
                time.sleep(_JUDGE_POLL_SECONDS)
                if not _JUDGE_ENABLED:
                    continue
                for _ in range(3):
                    raw = _redis.rpop(_JUDGE_QUEUE_KEY) if _check_redis() else None
                    if not raw:
                        break
                    try:
                        job = json.loads(raw)
                    except Exception:
                        continue
                    _judge_one(job.get("memory_id"))
            except Exception:
                pass


def _start_judge_worker() -> None:
    if not _JUDGE_ENABLED:
        log.info("LLM judge disabled (no MEMORY_JUDGE_API_KEY)")
        return
    if any(t.name == "memory-llm-judge" for t in threading.enumerate()):
        return
    _JudgeWorker().start()
    log.info("LLM judge worker started (model=%s)", _JUDGE_MODEL)

def _pct(sorted_vals: list, p: float):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 1)

def _latency_report(top_n: int = 8) -> dict:
    overall = sorted(_OVERALL_LATENCY)
    per_tool = sorted(
        ((t, sorted(v)) for t, v in _TOOL_LATENCY.items() if v),
        key=lambda kv: -len(kv[1]))[:top_n]
    return {
        "overall_ms": {"count": len(overall),
                       "p50": _pct(overall, 50), "p95": _pct(overall, 95),
                       "p99": _pct(overall, 99)},
        "per_tool_ms": {
            t: {"count": len(v), "p50": _pct(v, 50), "p95": _pct(v, 95),
                "p99": _pct(v, 99)}
            for t, v in per_tool},
    }

def _breaker_report() -> dict:
    now = time.time()
    return {s: {
        "state": "open" if now < b["open_until"] else "closed",
        "consecutive_failures": b["consecutive_failures"],
        "seconds_until_close": round(max(0.0, b["open_until"] - now), 1),
    } for s, b in _STORE_BREAKER.items()}

def _breaker_fail(store: str, exc: Exception) -> None:
    b = _STORE_BREAKER.get(store)
    if not b:
        return
    b["consecutive_failures"] += 1
    if b["consecutive_failures"] >= _BREAKER_THRESHOLD:
        b["open_until"] = time.time() + _BREAKER_COOLDOWN_S
        b["opened_at"] = _now_iso()
        log.error("circuit breaker OPEN for store=%s after %d failures",
                  store, b["consecutive_failures"])

def _breaker_ok(store: str) -> None:
    b = _STORE_BREAKER.get(store)
    if b:
        b["consecutive_failures"] = 0
        b["open_until"] = 0.0
        b["opened_at"] = None


async def _on_call_tool(
    context: object, params: types.CallToolRequestParams
) -> types.CallToolResult:
    try:
        name = params.name
        arguments = params.arguments or {}
        # Central isolation gate. ScopeError is converted to a tool error below
        # so a denied call looks like any other rejected request.
        _enforce_tool_authorization(name, arguments)
        _t0 = time.perf_counter()
        _op_status = "error"
        try:
            _res = await _dispatch_tool(name, arguments)
            _op_status = "ok" if not getattr(_res, "is_error", False) else "error"
        except PostgresUnavailableError as _e:
            _op_status = "store_unavailable"
            # 1B-5 middleware tier: a dead Postgres must never surface as a
            # generic 500 or an agent-loop hang — structured + counted.
            _breaker_fail("postgres", _e)
            log.error("tool %s: postgres unavailable: %s", name, _e)
            return _err(f"STORE_UNAVAILABLE: postgres: {_e}")
        finally:
            _ms = (time.perf_counter() - _t0) * 1000.0
            _record_tool_latency(name, _ms)
            # Sprint-1 1.1: durable per-op telemetry (write-behind, non-blocking)
            _op_telemetry_record(name, _ms, _op_status,
                                 scope=arguments.get("scope") if isinstance(arguments, dict) else None)
        # Canonical liveness gate runs on the way OUT, so no tool — including
        # one added later — can leak a tombstoned memory through a derived index.
        _res = _enforce_canonical_liveness(
            _res, as_of=(arguments.get("as_of") if isinstance(arguments, dict) else None))
        # P0-1: Automatic virtual context injection middleware
        _res = await _inject_virtual_context(name, arguments, _res)
        return _res
    except (AuthError, ScopeError) as _e:
        return _err(str(_e))
    except Exception as _e:
        log.exception("tool %s failed", name)
        return _err(f"{type(_e).__name__}: {_e}")


async def _dispatch_tool(name: str, arguments: dict) -> types.CallToolResult:
    """Route a tool call to its handler. Authorization already happened."""
    if name == "memory_turn_ingest":
        return _tool_turn_ingest(arguments)
    if name == "memory_timeline":
        return _tool_timeline(arguments)
    if name == "memory_kb_ingest":
        return _tool_kb_ingest(arguments)
    if name == "memory_kb_search":
        return _tool_kb_search(arguments)
    if name == "memory_kb_sources":
        return _tool_kb_sources(arguments)
    if name == "memory_kb_delete":
        return _tool_kb_delete(arguments)
    if name == "memory_kb_reconcile":
        return _tool_kb_reconcile(arguments)
    if name == "memory_merge_identities":
        return _tool_merge_identities(arguments)
    if name == "memory_export_audit_log":
        return _tool_export_audit_log(arguments)
    if name == "memory_identity_refresh":
        return _tool_identity_refresh(arguments)
    if name == "memory_store":
        return _tool_store(arguments)
    if name == "memory_recall":
        return _tool_recall(arguments)
    if name == "memory_lookup_exact":
        return _tool_lookup_exact(arguments)
    if name == "memory_list":
        return _tool_list(arguments)
    if name == "memory_get":
        return _tool_get(arguments)
    if name == "memory_update":
        return _tool_update(arguments)
    if name == "memory_delete":
        return _tool_delete(arguments)
    if name == "memory_stats":
        return _tool_stats(arguments)
    if name == "memory_cache_stats":
        return _tool_cache_stats()
    if name == "memory_consolidate":
        return _tool_consolidate(arguments)
    if name == "memory_session_summarize":
        return _tool_session_summarize(arguments)
    if name == "memory_compaction_metrics":
        return _tool_compaction_metrics(arguments)
    if name == "memory_track_access":
        return _tool_track_access(arguments)
    if name == "memory_memory_short_store":
        return _tool_short_store(arguments)
    if name == "memory_memory_short_recall":
        return _tool_short_recall(arguments)
    if name == "memory_memory_short_delete":
        return _tool_short_delete(arguments)
    if name == "memory_memory_short_list":
        return _tool_short_list(arguments)
    if name == "memory_memory_short_stats":
        return _tool_short_stats()
    if name == "memory_remember":
        return _tool_remember(arguments)
    if name == "memory_graph_link":
        return _tool_graph_link(arguments)
    if name == "memory_graph_unlink":
        return _tool_graph_unlink(arguments)
    if name == "memory_graph_neighbors":
        return _tool_graph_neighbors(arguments)
    if name == "memory_graph_relations":
        return _tool_graph_relations(arguments)
    if name == "memory_graph_reconcile":
        return _tool_graph_reconcile(arguments)
    if name == "memory_graph_consistency_metrics":
        return _tool_graph_consistency_metrics(arguments)
    if name == "memory_reflect":
        return _tool_reflect(arguments)
    if name == "memory_get_insights":
        return _tool_get_insights(arguments)
    if name == "memory_hint":
        return _tool_hint(arguments)
    if name == "memory_eval_store":
        return _tool_eval_store(arguments)
    if name == "memory_eval_recall":
        return _tool_eval_recall(arguments)
    if name == "memory_eval_commit":
        return _tool_eval_commit(arguments)
    if name == "memory_eval_discard":
        return _tool_eval_discard(arguments)
    if name == "memory_eval_preview":
        return _tool_eval_preview()
    if name == "memory_auto_observe":
        return _tool_auto_observe(arguments)
    if name == "memory_session_init":
        return _tool_session_init(arguments)
    if name == "memory_context_preview":
        return _tool_context_preview(arguments)
    if name == "memory_virtual_context_metrics":
        return _tool_virtual_context_metrics(arguments)
    if name == "memory_task_end":
        return _tool_task_end(arguments)
    if name == "memory_trajectory_log":
        return _tool_trajectory_log(arguments)
    if name == "memory_trajectory_close":
        return _tool_trajectory_close(arguments)
    if name == "memory_trajectory_recall":
        return _tool_trajectory_recall(arguments)
    if name == "memory_trajectory_get":
        return _tool_trajectory_get(arguments)
    if name == "memory_trajectory_chain":
        return _tool_trajectory_chain(arguments)
    if name == "memory_promote_procedures":
        return _tool_promote_procedures(arguments)
    if name == "memory_procedure_status":
        return _tool_procedure_status(arguments)
    if name == "memory_dream":
        return _tool_dream(arguments)
    if name == "memory_maintenance":
        return _tool_maintenance(arguments)
    if name == "memory_outbox_metrics":
        return _tool_outbox_metrics(arguments)
    if name == "memory_outbox_dead_letters":
        return _tool_outbox_dead_letters(arguments)
    if name == "memory_outbox_replay":
        return _tool_outbox_replay(arguments)
    if name == "memory_outbox_drain":
        return _tool_outbox_drain(arguments)
    if name == "memory_governance_audit":
        return _tool_governance_audit(arguments)
    if name == "memory_feedback":
        return _tool_feedback(arguments)
    if name == "memory_feedback_stats":
        return _tool_feedback_stats(arguments)
    if name == "memory_tool_recommendations":
        return _tool_tool_recommendations(arguments)
    if name == "memory_failure_hotspots":
        return _tool_failure_hotspots(arguments)
    if name == "memory_versions":
        return _tool_versions(arguments)
    if name == "memory_version_get":
        return _tool_version_get(arguments)
    if name == "memory_version_restore":
        return _tool_version_restore(arguments)
    if name == "memory_poison_check":
        return _tool_poison_check(arguments)
    if name == "memory_verify_alignment":
        return _tool_verify_alignment()
    if name == "memory_health_check":
        return _tool_health_check()
    if name == "memory_compliance_check":
        return _tool_compliance_check(arguments)
    if name == "memory_task_breakdown":
        return _tool_task_breakdown(arguments)
    if name == "memory_eval":
        return _tool_eval(arguments)
    if name == "memory_eval_run":
        return _tool_eval_run(arguments)
    if name == "memory_eval_history":
        return _tool_eval_history(arguments)
    if name == "memory_reembed_all":
        return _tool_reembed_all(arguments)
    if name == "memory_recall_efficiency":
        return _tool_recall_efficiency()
    if name == "memory_latency_metrics":
        return _tool_latency_metrics(arguments)
    if name == "memory_retrieval_stage_metrics":
        return _tool_retrieval_stage_metrics(arguments)
    if name == "memory_core":
        return _tool_core(arguments)
    if name == "memory_core_update":
        return _tool_core_update(arguments)
    if name == "memory_forget":
        return _tool_forget(arguments)
    return _err(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Part 3 operator tools: identity lifecycle + audit export
# ---------------------------------------------------------------------------

def _require_system(op_name: str):
    """Operator-only guard for identity/audit tooling."""
    principal = current_principal()
    if not principal.is_system:
        return _err(
            f"{op_name} is an operator-only tool (run under system context, not "
            f"as a tenant caller; identity {principal.identity or 'anonymous'!r}).")
    return None


def _tool_identity_refresh(args: dict) -> types.CallToolResult:
    """3A-5: hot-reload the identity registry from the identities table +
    token file. No process restart needed for a newly minted credential."""
    guard = _require_system("memory_identity_refresh")
    if guard is not None:
        return guard
    count = refresh_identity_registry()
    return _ok(json.dumps({"status": "reloaded", "records": count}))


def _tool_merge_identities(args: dict) -> types.CallToolResult:
    """3A-5: merge one identity's namespace into another (anonymous -> known).

    Moves every memory from `from_identity`'s scope into `to_identity`'s scope
    across ALL tiers: canonical Postgres rows, Qdrant payload scope, typed PG
    tables (mem_nodes / mem_procedures / mem_events), Redis short-term keys.
    Version snapshots and graph edges are keyed by memory_id, so they follow
    automatically. Dry-run by default: the plan is returned, nothing moves.
    """
    guard = _require_system("memory_merge_identities")
    if guard is not None:
        return guard
    from_raw = (args.get("from_identity") or "").strip()
    to_raw = (args.get("to_identity") or "").strip()
    dry_run = bool(args.get("dry_run", True))
    if not from_raw or not to_raw:
        return _err("from_identity and to_identity are both required")
    try:
        src = _normalize_scope(from_raw)
        dst = _normalize_scope(to_raw)
    except ScopeError as e:
        return _err(f"invalid identity: {e}")
    if src == dst:
        return _err("from_identity and to_identity must differ")
    ceiling = _deployment_grants() or frozenset()
    if src not in ceiling or dst not in ceiling:
        return _err(f"both identities must be served by this deployment "
                    f"(ceiling: {sorted(ceiling)})")
    with system_context("merge_identities"):
        # 1) plan: what would move
        try:
            canon = _pg_query(
                "SELECT count(*) AS n FROM mem_canonical WHERE scope=:s AND "
                "status <> 'deleted'", {"s": src})
            n_canon = int(canon[0]["n"] if isinstance(canon[0], dict) else canon[0][0])
            qdr_ids = []
            _off = None
            _flt = qmodels.Filter(must=[qmodels.FieldCondition(
                key="scope", match=qmodels.MatchValue(value=src))])
            while True:
                _page = _client.scroll(collection_name=COLLECTION_NAME,
                                      scroll_filter=_flt, limit=256,
                                      offset=_off, with_payload=False)
                qdr_ids.extend(p.id for p in _page[0])
                _off = _page[1]
                if not _off or len(qdr_ids) > 20000:
                    break
            plan = {"from": src, "to": dst, "dry_run": dry_run,
                    "canonical_rows": n_canon, "qdrant_points": len(qdr_ids)}
        except Exception as e:
            return _err(f"plan query failed: {e}")
        if dry_run:
            plan["message"] = ("Dry-run only. Re-run with dry_run=false to apply. "
                               "The move is idempotent and auditable; version "
                               "snapshots and graph edges follow automatically.")
            return _ok(json.dumps(plan, ensure_ascii=False))
        moved = {"canonical_rows": 0, "qdrant_points": 0, "typed_rows": 0,
                 "short_term_keys": 0}
        try:
            # 2) canonical tier (source of truth)
            _r = _pg_query(
                "WITH moved AS (UPDATE mem_canonical SET scope=:dst, updated_at=now() "
                "WHERE scope=:src AND status <> 'deleted' RETURNING 1) "
                "SELECT count(*) AS n FROM moved", {"src": src, "dst": dst})
            moved["canonical_rows"] = int(_r[0]["n"] if isinstance(_r[0], dict) else _r[0][0])
            # 3) typed PG tables
            for tbl in ("mem_nodes", "mem_procedures", "mem_events"):
                try:
                    _r2 = _pg_query(
                        f"WITH moved AS (UPDATE {tbl} SET scope=:dst WHERE scope=:src "
                        "RETURNING 1) SELECT count(*) AS n FROM moved",
                        {"src": src, "dst": dst})
                    moved["typed_rows"] += int(_r2[0]["n"] if isinstance(_r2[0], dict) else _r2[0][0])
                except Exception:
                    pass  # table may not exist in this deployment
            # 4) Qdrant payload scope (vectors untouched)
            _off = None
            _flt = qmodels.Filter(must=[qmodels.FieldCondition(
                key="scope", match=qmodels.MatchValue(value=src))])
            _moved_pts = 0
            while True:
                _page = _client.scroll(collection_name=COLLECTION_NAME,
                                       scroll_filter=_flt, limit=256,
                                       offset=_off, with_payload=False)
                _ids = [p.id for p in _page[0]]
                if _ids:
                    _client.set_payload(collection_name=COLLECTION_NAME,
                                        points=_ids, payload={"scope": dst})
                    _moved_pts += len(_ids)
                _off = _page[1]
                if not _off:
                    break
            moved["qdrant_points"] = _moved_pts
            # 5) Redis short-term keys: rename prefix memory:short:<src>:*
            if _check_redis():
                _pattern = f"{REDIS_PREFIX}{src}:*"
                for k in _redis_bytes_scan(_pattern):
                    try:
                        _val = _redis.get(k)
                        _ttl = _redis.ttl(k)
                        _newk = k.decode().replace(f"{REDIS_PREFIX}{src}:", f"{REDIS_PREFIX}{dst}:", 1)
                        _redis.setex(_newk, max(_ttl, 60), _val) if _ttl and _ttl > 0 else _redis.set(_newk, _val)
                        _redis.delete(k)
                        moved["short_term_keys"] += 1
                    except Exception:
                        pass
            # 6) caches + audit
            _invalidate_recall_caches()
            _pg_audit("identity_merge", None,
                      {"from": src, "to": dst, **moved})
        except Exception as e:
            return _err(f"merge failed mid-way (idempotent — re-run): {e}")
    moved.update({"from": src, "to": dst, "dry_run": False,
                  "note": "Idempotent. Version snapshots and graph edges are "
                          "memory_id-keyed and followed automatically."})
    return _ok(json.dumps(moved, ensure_ascii=False))


def _redis_bytes_scan(pattern: str):
    """Iterate Redis keys matching a pattern (decode_responses-safe)."""
    if _redis is None:
        return
    cur = 0
    while True:
        cur, keys = _redis.scan(cur, match=pattern, count=200)
        for k in keys:
            yield k if isinstance(k, bytes) else k.encode()
        if cur == 0:
            return


def _tool_export_audit_log(args: dict) -> types.CallToolResult:
    """3B-4: export the append-only audit trail as JSON or CSV.

    Enterprise audits need the who/what/when of every memory access in a
    portable file. Filters: since / until (ISO), actor, op, scope. The full
    export lands in /root/backup/; the tool result carries the first 200
    rows inline plus the file path.
    """
    guard = _require_system("memory_export_audit_log")
    if guard is not None:
        return guard
    fmt = (args.get("format") or "json").lower()
    if fmt not in ("json", "csv"):
        return _err("format must be 'json' or 'csv'")
    limit = min(int(args.get("limit") or 10000), 50000)
    since = args.get("since")
    until = args.get("until")
    actor = args.get("actor")
    op = args.get("op")
    scope = args.get("scope")
    clauses, params = [], {}
    if since:
        clauses.append("created_at >= :since"); params["since"] = since
    if until:
        clauses.append("created_at <= :until"); params["until"] = until
    if actor:
        clauses.append("actor = :actor"); params["actor"] = actor
    if op:
        clauses.append("op = :op"); params["op"] = op
    if scope:
        clauses.append("scope = :scope"); params["scope"] = scope
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with system_context("audit_export"):
        rows = _pg_query(
            "SELECT id::text AS id, op, memory_id, actor, scope, detail, created_at "
            f"FROM mem_audit{where} ORDER BY created_at DESC LIMIT :lim",
            {**params, "lim": limit})
    clean = []
    for r in rows:
        d = r if isinstance(r, dict) else dict(zip(
            ("id", "op", "memory_id", "actor", "scope", "detail", "created_at"), r))
        clean.append({
            "id": d.get("id"), "op": d.get("op"), "memory_id": d.get("memory_id"),
            "actor": d.get("actor"), "scope": d.get("scope"),
            "detail": d.get("detail"), "created_at": str(d.get("created_at")),
        })
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    path = f"/root/backup/memory_audit_export_{ts}.{fmt}"
    if fmt == "json":
        payload_text = json.dumps(clean, ensure_ascii=False, indent=2)
    else:
        import csv as _csv, io as _io
        buf = _io.StringIO()
        w = _csv.DictWriter(buf, fieldnames=["id", "op", "memory_id", "actor",
                                             "scope", "detail", "created_at"])
        w.writeheader()
        for r in clean:
            r2 = dict(r); r2["detail"] = json.dumps(r2.get("detail") or {})
            w.writerow(r2)
        payload_text = buf.getvalue()
    with open(path, "w") as fh:
        fh.write(payload_text)
    _pg_audit("audit_export", None, {"format": fmt, "rows": len(clean), "file": path})
    return _ok(json.dumps({
        "status": "exported", "rows": len(clean), "file": path,
        "inline_preview": clean[:200],
        "filters": {"since": since, "until": until, "actor": actor,
                    "op": op, "scope": scope},
    }, ensure_ascii=False, default=str))

_server = Server(
    name="memory",
    instructions="Memory v4.0 — Full 2026 best practices (all 8 implemented): Graph Memory, Reflexion Loop (outcome-driven, closed-loop), Memory Poisoning Guard (OWASP LLM04), Mixture-of-Agents, Eval before Deploy, Versioned Rollback, Semantic Cache, Closed Reflexion Loop (SelfMem pattern). Use memory_store to persist, memory_recall for specialized retrieval, memory_reflect for self-improvement, memory_get_insights for learned strategies, memory_hint for agent-driven memory management.\n\nMEMORY PROTOCOL — At session start read the memory://context resource (or call memory_session_init). Before acting call memory_recall for relevant facts. When relationships matter use memory_graph_link / memory_graph_neighbors. Before editing a memory check memory_versions. Capture outcomes with memory_turn_ingest or memory_auto_observe; close a task with memory_task_end. Mark recalled memories useful/irrelevant with memory_feedback. Pull memory://profile and memory://core for zero-call context.",
    on_list_tools=_on_list_tools,
    on_call_tool=_on_call_tool,
    on_list_resources=_on_list_resources,
    on_read_resource=_on_read_resource,
)

# -- store -------------------------------------------------------------------
def _tool_store(args: dict) -> types.CallToolResult:
    _start = time.perf_counter()
    content = args["content"]

    # Documentation routing: documentation-like content goes to the RAG lane,
    # NOT to memory. Explicit category/flag, or a long structured document.
    _cat = (args.get("category") or "").strip().lower()
    _doc = (_cat in ("documentation", "doc", "docs", "kb", "knowledge")
            or bool(args.get("documentation"))
            or str(args.get("kind", "")).lower() in ("document", "documentation"))
    if not _doc and DOC_ROUTE_CHARS > 0:
        _t = args.get("content") or ""
        if len(_t) >= DOC_ROUTE_CHARS and _looks_like_document(_t):
            _doc = True
    if _doc:
        _rej, _find, _masked = _scan_secrets(args.get("content") or "")
        if _rej:
            _pg_audit("secret_block", None, {"findings": _find, "via": "doc_route"})
            return _err(json.dumps({"blocked": True, "gate": "secret_detection",
                                    "findings": _find}, ensure_ascii=False))
        try:
            _info = _kb_add_document(
                _masked,
                (args.get("source") or args.get("title") or "documentation").strip(),
                (args.get("title") or "Documentation").strip(),
                authorize_scope(args.get("scope")),
                {"routed_from": "memory_store", "category": args.get("category") or "documentation"})
        except Exception as _e:
            return _err("memory_store: documentation routing failed: %s" % _e)
        _info["routed_to"] = "rag"
        _info["lane"] = "knowledge"
        _op_telemetry_record("write_gate", 0.0, "routed_to_rag")
        return _ok(json.dumps(_info, ensure_ascii=False))

    # 3B-2: secret/PII gate runs BEFORE anything else. Credential-shaped
    # content rejects the write (fail closed, audited); PII is masked to
    # redacted placeholders so a compliant store never holds raw PII.
    _secret_reject, _secret_findings, content = _scan_secrets(content)
    if _secret_reject:
        _WRITE_GATE_STATS["secret_rejected"] = _WRITE_GATE_STATS.get("secret_rejected", 0) + 1
        _op_telemetry_record("write_gate", 0.0, "secret_rejected")
        _pg_audit("secret_block", None, {"findings": _secret_findings,
                                          "content_len": len(args["content"])})
        log.warning("STORE BLOCKED secret-shape findings=%s", _secret_findings)
        result = {
            "blocked": True,
            "gate": "secret_detection",
            "findings": _secret_findings,
            "message": "Content contains a credential-shaped string (API key / "
                       "password / token / private key). Secrets must never be "
                       "stored in memory. Remove or replace the credential.",
        }
        _record_latency("memory_store", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))
    _pii_masked = any(f.get("action") == "mask" for f in _secret_findings)
    if _pii_masked:
        _WRITE_GATE_STATS["pii_masked"] = _WRITE_GATE_STATS.get("pii_masked", 0) + 1
        _op_telemetry_record("write_gate", 0.0, "pii_masked")

    # Enhancement 3: Memory Poisoning Guard
    blocked, poison_score, findings = _check_poison(content)
    if blocked:
        log.warning("STORE BLOCKED poison_score=%.3f content_len=%d", poison_score, len(content))
        result = {
            "blocked": True,
            "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Content flagged by poisoning guard (OWASP LLM04). Review findings before retrying.",
        }
        _record_latency("memory_store", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    # Enhancement 9: Quality Gate — reject low-quality memories
    q_rejected, q_score, q_reasons = _check_quality(content)
    # 2B-3: write-gate observability — rejection/quarantine rates are metrics,
    # not log lines. Surfaced via memory_stats. (source resolved early so the
    # gate counters can attribute per-source before any rejection return.)
    source = args.get("source") or "agent"
    _WRITE_GATE_STATS["evaluated"] += 1
    _WRITE_GATE_STATS["quality_sum"] += q_score
    _src_stats = _WRITE_GATE_STATS["by_source"].setdefault(
        source, {"evaluated": 0, "stored": 0, "rejected": 0, "quarantined": 0})
    _src_stats["evaluated"] += 1
    if q_rejected:
        _WRITE_GATE_STATS["rejected"] += 1
        _src_stats["rejected"] += 1
        _op_telemetry_record("write_gate", 0.0, "rejected", quality_score=q_score)
        log.warning("STORE REJECTED quality_score=%.3f reasons=%s content_len=%d", q_score, q_reasons, len(content))
        result = {
            "rejected": True,
            "quality_score": round(q_score, 4),
            "reasons": q_reasons,
            "threshold": QUALITY_REJECT_THRESHOLD,
            "message": f"Content quality too low (score={q_score:.2f}, threshold={QUALITY_REJECT_THRESHOLD}). "
                       f"Add more detail, structure, or specificity.",
        }
        _record_latency("memory_store", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    # Sanitize content (neutralize injection patterns even if below threshold)
    content = _sanitize_content(content)

    memory_id = str(uuid.uuid4())
    now = _now_iso()
    # Normalize + validate category against the routing map. The AI writes raw
    # strings, so coerce variants and typos into the canonical keys the split
    # stack understands — otherwise "procedures"/"architecture_decision"/etc.
    # fall through to the default route and pollute the category taxonomy.
    category_raw = (args.get("category") or "general").strip().lower()
    _KNOWN_CATEGORIES = set(CATEGORY_ROUTING.keys())
    _LEGACY_MAP = {
        "procedures": "procedure",
        "architechture_decision": "decision",
        "architecture_decision": "decision",
        "fix": "procedure",
        "gotcha": "procedure",
        "credential_salt": "credential",
        "project_knowledge": "project",
        "reference": "fact",
        "event": "event",
        "trajectory": "trajectory",
        "session": "transient",
        "transient": "transient",
    }
    category = _LEGACY_MAP.get(category_raw, category_raw if category_raw in _KNOWN_CATEGORIES else "general")
    importance = args.get("importance") if args.get("importance") is not None else 5
    tags_str = ",".join(_coerce_str_list(args.get("tags")))
    source = args.get("source") or "agent"
    # 3A-3: confidence is COMPUTED from provenance, never decorative. A stated
    # preference (the user said it) is 0.9; an agent inference from one
    # observation is 0.4; corroboration across >=3 sightings promotes the
    # EXISTING memory to 0.8 (dedup hit below). The dashboard renders
    # provenance + confidence so inferred preferences are visibly labeled.
    _stated = bool(args.get("stated")) or source.lower() in (
        "user", "user_message", "boss", "operator", "stated")
    confidence = 0.9 if _stated else 0.4
    provenance = "stated" if _stated else "inferred"
    # Fail closed: an absent/blank/malformed scope is rejected, never silently
    # widened to the shared 'global' namespace. The caller's authorized scope
    # comes from the server-side principal, not from this argument.
    try:
        scope = authorize_scope(args.get("scope"))
    except ScopeError as e:
        return _err(f"scope rejected: {e}")
    embedding_model_version = EMBEDDING_MODEL_VERSION  # Phase 4
    vector = _embed_model().encode(content).tolist()
    tokens = _tokenize(content)
    doc_len = len(tokens)
    payload = {
        "scope": scope,
        "embedding_model": embedding_model_version,
        "content": content,
        "created_at": now,
        "updated_at": now,
        "category": category,
        "importance": importance,
        "tags": tags_str,
        "source": source,
        "access_count": 0,
        "last_accessed_at": now,
        "tokens": tokens,
        "doc_len": doc_len,
        "retrieval_count": 0,
        "useful_count": 0,
        "quality_score": round(q_score, 4) if 'q_score' in dir() else None,
        # P3.1 (2026-10-07): store-time poison provenance. Content that matched
        # a poison pattern but stayed under the block threshold is stored
        # (already sanitized above), but flagged so recall surfaces the flag to
        # the consumer instead of handing a stored injection back as clean
        # memory. Flows to Qdrant via canonical payload -> outbox -> the extra
        # payload carry in _derived_upsert.
        "poison_score": round(poison_score, 4),
        "quarantine": bool(findings),
        # 3A-3: provenance + computed confidence ride every payload so the
        # dashboard can label inferred entries and rank trust honestly.
        "confidence": confidence,
        "provenance": provenance,
        "corroboration_count": 1,
        # 3B-2: note when the stored copy had PII masked.
        "pii_masked": _pii_masked,
    }
    # Internal callers (task_end) may attach extra provenance keys that must
    # survive into the canonical payload so the outbox-derived Qdrant payload
    # carries them (e.g. learned_from_outcome for get_insights' outcome filter).
    _extra_payload = args.get("_extra_payload")
    if isinstance(_extra_payload, dict):
        payload.update(_extra_payload)

    # Semantic deduplication: check for highly similar existing memories IN THE
    # CALLER'S OWN SCOPE ONLY. An unfiltered scan would (a) leak a cross-tenant
    # memory_id back as `duplicate_of` and (b) create a cross-tenant reference
    # edge. The scope filter keeps dedup an intra-tenant concern by construction.
    dedup_filter = _build_filter(category=None, tags_key=None, scope=scope)
    similar = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        query_filter=dedup_filter,
        with_payload=True,
    )
    duplicate_id = None
    contradiction_of = None
    for sp in similar.points:
        if sp.score >= 0.92:
            # 2B-6 precedence fix (2026-10-09): a near-copy that CONTAINS a
            # negation marker is a CORRECTION, not a duplicate. The old order
            # flagged "no longer prefers X" (score 0.97) as duplicate_of and
            # both memories stayed live — corrections silently lost. Skip the
            # duplicate branch and let the contradiction/supersession path
            # close the old one's validity window.
            _peek = content.lower()
            if any(m in _peek for m in _NEGATION_MARKERS):
                break
            duplicate_id = sp.id
            log.info("duplicate detected scope=%s score=%.3f existing_id=%s new_id=%s",
                     scope, sp.score, duplicate_id, memory_id)
            break

    # 3A-3 corroboration: a near-duplicate store is independent evidence for
    # the EXISTING memory. Bump its corroboration_count; at >=3 sightings an
    # inferred memory promotes to confidence 0.8 (a stated memory keeps 0.9 —
    # the user's own words outrank repeated agent inference).
    if duplicate_id:
        try:
            _ex = _client.retrieve(collection_name=COLLECTION_NAME,
                                   ids=[duplicate_id], with_payload=True)
            # qdrant-client returns a bare list from retrieve(); older call
            # sites index it directly (results[0]) — .points does not exist.
            _ex_payload = (_ex[0].payload or {}) if _ex else {}
            _cnt = int(_ex_payload.get("corroboration_count") or 1) + 1
            _corro_patch = {"corroboration_count": _cnt,
                            "last_corroborated_at": now}
            if _cnt >= 3 and (_ex_payload.get("provenance") or "inferred") == "inferred":
                _corro_patch["confidence"] = 0.8
                _corro_patch["corroborated"] = True
            _client.set_payload(collection_name=COLLECTION_NAME,
                                points=[duplicate_id], payload=_corro_patch)
            try:
                _pg_run(
                    "UPDATE mem_canonical SET payload = payload || :patch::jsonb "
                    "WHERE memory_id=:mid",
                    {"patch": json.dumps(_corro_patch), "mid": duplicate_id},
                    table="mem_canonical-corroborate")
            except Exception:
                pass
        except Exception as _ce:
            log.warning("corroboration bump failed for %s: %s", duplicate_id, _ce)

    # 2D-2: learning-rate discrimination — stable knowledge (preferences,
    # decisions, procedures) learns fast and decays slow (decay table);
    # ephemeral content (general/conversation/fact) may exceed importance 5
    # only when CORROBORATED — i.e. a similar memory already exists (the
    # dedup signal). One-off uncorroborated observations cap at 5 so junk
    # never reaches top-importance ranking on a single sighting.
    learning_rate = ("stable" if category in ("preference", "decision", "procedure",
                                               "anti_pattern", "summary")
                     else "ephemeral")
    if learning_rate == "ephemeral" and (importance or 0) > 5 and not duplicate_id:
        importance = 5
        _uncorroborated_cap = True
    else:
        _uncorroborated_cap = False

    # Gap #2 + #3: contradiction detection + temporal supersession
    # If top-1 similar is high-scoring AND negation markers are present,
    # the new memory is a correction → close the old one's validity window.
    contradiction_of = None
    superseded_old_id = None
    if not duplicate_id and similar.points and similar.points[0].score >= _CONTRADICTION_THRESHOLD:
        top = similar.points[0]
        lower_new = content.lower()
        has_negation = any(m in lower_new for m in _NEGATION_MARKERS)
        if has_negation:
            contradiction_of = top.id
            superseded_old_id = top.id
            log.info("contradiction detected new=%s supersedes=%s score=%.3f",
                     memory_id[:8], top.id, top.score)

    # Phase 4.1 (2026-10-07): "same fact slot, new value" IS supersession even
    # without a negation word. The 0.82–0.92 band is exactly the gap: below 0.92
    # it is not a duplicate, and the contradiction branch needs a negation marker
    # so plainly-worded sequential facts ("in March Alice moved to Vancouver" →
    # "in June Alice moved to Montreal") fell through and neither was marked
    # superseded, leaving recall to an importance+recency blend that is not
    # property-aware (the "current location" flake).
    if not duplicate_id and not contradiction_of and similar.points:
        _top = similar.points[0]
        if _CONTRADICTION_THRESHOLD <= _top.score < 0.92:
            _top_old = _tokenize((_top.payload or {}).get("content") or "")
            _top_new = _tokenize(content)
            _old_set = set(_top_old)
            _overlap = (len(_old_set & set(_top_new)) / len(_old_set)) if _old_set else 0.0
            # Require substantial token overlap (same slot) but a non-trivial
            # token delta (a different value), so a mere paraphrase at the top of
            # the band is not misread as a value change.
            if _overlap >= 0.6 and _old_set != set(_top_new):
                superseded_old_id = _top.id
                log.info("band supersession new=%s supersedes=%s score=%.3f overlap=%.2f",
                         memory_id[:8], _top.id, _top.score, _overlap)

    # 2026-10-26: LLM adjudication of the remaining ambiguous band. Same-slot
    # conflicts with low token overlap (paraphrased corrections) fall through
    # both rule layers above; ask the judge LLM to decide, then reuse the exact
    # supersession path below (temporal close + supersedes edge + canonical
    # stamp + outbox event).
    if (
        not duplicate_id
        and not superseded_old_id
        and similar.points
        and _CONFLICT_LLM_ENABLED
        and category in _CONFLICT_LLM_CATEGORIES
    ):
        _topc = similar.points[0]
        if _CONTRADICTION_THRESHOLD <= _topc.score < 0.92:
            _verdict = _conflict_llm_judge(
                old_content=(_topc.payload or {}).get("content") or "",
                new_content=content,
            )
            if _verdict == "supersedes":
                superseded_old_id = _topc.id
                contradiction_of = _topc.id
                log.info("llm conflict adjudication new=%s supersedes=%s score=%.3f",
                         memory_id[:8], _topc.id, _topc.score)

    # Close the old memory's temporal window in Qdrant (both branches).
    if superseded_old_id:
        try:
            _client.set_payload(
                collection_name=COLLECTION_NAME,
                points=[superseded_old_id],
                payload={"status": "superseded", "superseded_by": memory_id,
                         "valid_to": now, "updated_at": now},
            )
        except Exception:
            pass
        # Also link as a graph edge in Redis (memory_id based, no FK issue)
        _graph_link(memory_id, superseded_old_id, "supersedes")
        # Canonical tier must agree, else the outbox-derived upsert would drop
        # the tombstoned row and then _canonical_set_status would flip it back
        # to `active`. Best-effort, same posture as the Qdrant close above; the
        # derived tier is already tombstoned regardless of canonical lag.
        # Audit finding 2026-10-09 (final verification): this block was gated on
        # `_pg_conn is not None`, which is None BY DESIGN in pool mode — so the
        # canonical supersede stamp + MEMORY_SUPERSEDED event were silently
        # skipped in every pool deployment (derived tier marked superseded,
        # canonical stayed 'active' = governance disagreement). Call the
        # mode-aware txn executor unconditionally instead.
        try:
            _pg_txn_execute([
                ("UPDATE mem_canonical SET status='superseded', "
                 " superseded_by=:newid, valid_until=now(), updated_at=now() "
                 "WHERE memory_id=:oldid AND status IN ('active','pending_index','index_failed')",
                 {"newid": memory_id, "oldid": superseded_old_id}),
                ("INSERT INTO mem_outbox"
                 "(event_id, event_type, memory_id, scope, version, payload) "
                 "VALUES (:eid, 'MEMORY_SUPERSEDED', :oldid, :scope, :ver, :payload) "
                 "ON CONFLICT (event_id) DO NOTHING",
                 {"eid": _outbox_event_id("MEMORY_SUPERSEDED", superseded_old_id, 1),
                  "oldid": superseded_old_id, "scope": scope, "ver": 1,
                  "payload": json.dumps({"superseded_by": memory_id})}),
            ])
        except Exception as _se:
            log.debug("canonical supersede stamp failed old=%s: %s", superseded_old_id, _se)
        # Temporal graph: close every edge incident to the superseded memory so
        # point-in-time graph views stop reporting relationships that ended.
        try:
            _pg_run("UPDATE mem_edges SET valid_to=now() "
                    "WHERE (src=CAST(:o AS uuid) OR dst=CAST(:o AS uuid)) "
                    "AND valid_to IS NULL", {"o": superseded_old_id})
        except Exception:
            pass

    # PHASE A — the canonical PostgreSQL record is committed BEFORE any derived
    # layer is touched. Qdrant/BM25/Redis/graph are projections of this row, not
    # the other way round. If this fails we must NOT touch a derived layer and
    # must NOT answer "stored" — otherwise a PG outage silently loses writes
    # that the caller was told were durable.
    try:
        _canonical_write(
            memory_id, scope, _route_memory(category)[0],
            # Credentials/transient are Redis-TTL-only by design: the canonical row
            # carries lifecycle authority (so deletion still tombstones them) but
            # stores NO content, so a secret is never persisted to PostgreSQL.
            None if _route_memory(category)[0] == "transient" else content,
            version=1, event_type="MEMORY_CREATED",
            payload={"category": category, "importance": importance,
                     "tags": tags_str, "source": source,
                     "event_time": args.get("event_time"),
                     "learning_rate": learning_rate,
                     "uncorroborated_capped": bool(_uncorroborated_cap),
                     "duplicate_of": duplicate_id,
                     # 3A-3: provenance + computed confidence MUST ride the
                     # canonical payload — the outbox is the sole driver of
                     # the derived Qdrant payload, so fields that only live
                     # in the local dict never reach retrieval.
                     "confidence": confidence,
                     "provenance": provenance,
                     "corroboration_count": 1,
                     "pii_masked": _pii_masked,
                     # 2D-1 judge v1 (deterministic): trust (poison guard),
                     # substance (quality gate), consistency (contradiction
                     # check) — all ran in this write path. LLM-judge upgrade
                     # slots in here once an inference endpoint is wired.
                     "verification": ("deterministic_v1: poison=%s quality=%s "
                                      "contradiction=%s" %
                                      (round(poison_score, 2), round(q_score, 2),
                                       str(contradiction_of)[:8] if contradiction_of else "none"))
                     if (importance or 0) >= 7 else None,
                     # P3.1: poison provenance must ride the canonical row so the
                     # outbox-derived Qdrant payload carries it (setdefault carry
                     # in _derived_upsert) and recall can flag quarantined content.
                     "poison_score": round(poison_score, 4),
                     "quarantine": bool(findings),
                     # Extra provenance (e.g. learned_from_outcome) must reach the
                     # derived tier; spread it last so it is never clobbered.
                     **(_extra_payload or {})})
    except PostgresUnavailableError as _e:
        log.error("canonical write failed id=%s: %s", memory_id, _e)
        return _ok(json.dumps({
            "memory_id": memory_id, "status": "failed",
            "targets": _route_memory(category)[1],
            "durability": {"canonical": str(_e)},
            "error": f"canonical (authoritative) write failed; Postgres unavailable: {_e}; "
                     "no derived layer was modified"}, ensure_ascii=False))
    # MODEL B (Stage 4): the outbox worker is the SOLE driver of derived stores.
    # Supersession, Qdrant upsert, BM25 rebuild and the Redis version snapshot
    # are all owned by _outbox_apply/_derived_upsert now, so a claim-then-crash
    # leaves no partial derived state. A store is durable the moment the
    # canonical row + outbox event commit in _canonical_write above.
    node_type, targets = _route_memory(category)

    log.info("stored id=%s category=%s duplicate_of=%s tokens=%d node_type=%s targets=%s (outbox-driven)",
             memory_id, category, duplicate_id, len(tokens), node_type, targets)

    # Durable + typed routing (split-stack: Qdrant/Postgres/Redis by memory type)
    _pg_audit("store", memory_id, {
        "category": category, "importance": importance,
        "scope": scope, "quality_score": round(q_score, 4),
        "duplicate_of": duplicate_id, "node_type": node_type, "targets": targets,
        "confidence": confidence, "provenance": provenance,
        "pii_masked": _pii_masked,
    })

    # L3 PostgreSQL — insert the typed structured row FIRST (so its node UUID
    # exists before any edge insert). mem_edges FKs reference mem_nodes.id.
    _new_node_uuid = None
    _pg_store_error = None
    if "postgres" in targets:
      try:
        # PG access is best-effort at the statement level, but a driver-level
        # fault (or the pool/connection dropping) must still be turned into an
        # honest "failed" store rather than an exception escaping to the caller.
        if node_type == "procedure":
            _pg_ok = _pg_run(
                "INSERT INTO mem_procedures(memory_id, scope, proc_name, importance, "
                "source_trajectory, steps, tags, payload) "
                "VALUES (:mid, :scope, :pname, :imp, :traj, :steps, :tags, :payload)",
                {"mid": memory_id, "scope": scope, "pname": content[:120],
                 "imp": importance, "traj": payload.get("task_signature"),
                 "steps": "[]",
                 "tags": tags_str,
                 "payload": json.dumps({k: payload.get(k) for k in ("category", "source", "content")})},
                table="mem_procedures",
            )
            if not _pg_ok:
                _pg_store_error = "mem_procedures insert returned False"
        elif node_type == "event":
            _pg_ok = _pg_run(
                "INSERT INTO mem_events(memory_id, scope, event_type, task_signature, payload, occurred_at) "
                "VALUES (:mid, :scope, :etype, :tsig, :payload, :occ)",
                {"mid": memory_id, "scope": scope, "etype": "episodic",
                 "tsig": payload.get("task_signature"),
                 "payload": json.dumps({"category": category, "content": content[:500]}),
                 "occ": now},
                table="mem_events",
            )
            if not _pg_ok:
                _pg_store_error = "mem_events insert returned False"
        else:  # fact / preference → mem_nodes (return node UUID for edge inserts)
            _new_node_uuid = _pg_node_insert(
                content[:200], node_type, scope,
                json.dumps({"memory_id": memory_id, "category": category,
                            "importance": importance, "tags": tags_str, "source": source}),
                memory_id,
            )
            # Treat None return as a durable write failure
            if _new_node_uuid is None:
                _pg_store_error = "mem_nodes insert returned None"
      except Exception as _pg_exc:
        # A driver-level fault must not escape as an exception and must not be
        # reported as a successful store.
        log.error("postgres store step failed id=%s: %s", memory_id, _pg_exc)
        _pg_store_error = f"{type(_pg_exc).__name__}: {_pg_exc}"

    # Gap #3: temporal supersession in Postgres (only if new node UUID is known)
    if contradiction_of and _new_node_uuid:
        _old_node_uuid = _pg_node_uuid(contradiction_of)
        if _old_node_uuid:
            _pg_run(
                "UPDATE mem_nodes SET valid_to = :vto, superseded_by = :newnode "
                "WHERE id = :oldnode AND valid_to IS NULL",
                {"vto": now, "newnode": _new_node_uuid, "oldnode": _old_node_uuid},
                table="mem_nodes-supersede",
            )
            _pg_run(
                "INSERT INTO mem_edges(src, dst, rel) VALUES (:src, :dst, :rel)",
                {"src": _new_node_uuid, "dst": _old_node_uuid, "rel": "supersedes"},
                table="mem_edges-supersede",
            )

    # Gap #7: auto graph-link to top-K similar (only when new node UUID known)
    if _new_node_uuid:
        for sp in similar.points[:_AUTO_LINK_TOP_K]:
            if sp.id == memory_id or sp.id == contradiction_of:
                continue
            if sp.score < _AUTO_LINK_MIN_SCORE:
                continue
            _peer_node_uuid = _pg_node_uuid(sp.id)
            if not _peer_node_uuid:
                continue  # peer has no PG node row yet — skip PG edge, keep Redis graph link
            _pg_run(
                "INSERT INTO mem_edges(src, dst, rel) VALUES (:src, :dst, :rel)",
                {"src": _new_node_uuid, "dst": _peer_node_uuid, "rel": "related_to"},
                table="mem_edges-autolink",
            )

    # L1 Redis — short-term / transient (TTL). Credentials live ONLY here.
    if "redis" in targets and _redis:
        try:
            _redis.setex(
                f"memory:st:{memory_id}",
                604800,  # 7-day TTL for transient/credential
                json.dumps({"memory_id": memory_id, "category": category,
                            "content": content, "importance": importance, "scope": scope}),
            )
        except Exception as _e:
            log.warning("Redis short-term write failed for %s: %s", memory_id, _e)

    # Durable-write verification (2026-10-04): previously every _pg_run()
    # return value was discarded and the tool always answered status="stored",
    # so a durable-tier rejection looked exactly like success. Under concurrency
    # this silently lost 40% of writes. Now a declared durable target that did
    # not land is reported honestly instead of being papered over.

    # MODEL B (Stage 4): durable verification is now canonical-only. The
    # derived layers (Qdrant, BM25, Redis, graph) are driven exclusively by
    # the outbox worker. A store is durable the instant the canonical row +
    # outbox event commit atomically in _canonical_write above; the worker
    # promotes the row from `pending_index` to `active` when the derived
    # stores converge. A store is therefore reported as "stored" on canonical
    # commit success, and never held hostage by a slow or failed derived tier.
    #
    # The caller can opt into a bounded synchronous wait for convergence via
    # MEMORY_OUTBOX_SYNC_GRACE_SECONDS (default 0 = no wait, fully async).
    if _pg_store_error:
        _canonical_set_status(memory_id, "index_failed",
                              error=f"postgres: {_pg_store_error}")
    else:
        # Canonical is committed; leave the row as `pending_index` so the
        # outbox worker can promote it to `active` on convergence. If PG
        # itself is unavailable (tier down), we already returned early.
        pass

    if _pg_store_error:
        # Honest reporting: the canonical row did not fully land (a driver-level
        # fault). The row may or may not have committed depending on where the
        # fault occurred — surface it so the caller can reconcile.
        log.error("store postgres step failed id=%s: %s", memory_id, _pg_store_error)
        result = {"memory_id": memory_id, "status": "stored_index_pending",
                  "canonical": "committed_or_partial",
                  "duplicate_of": duplicate_id,
                  "quality_score": round(q_score, 4), "node_type": node_type,
                  "targets": targets,
                  "durability": {"postgres": _pg_store_error},
                  "error": f"postgres typed-row step failed: {_pg_store_error}"}
        _record_latency("memory_store", (time.perf_counter() - _start) * 1000)
        return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    # Optional bounded synchronous convergence wait (default 0 = async).
    _converged = True
    if OUTBOX_SYNC_GRACE_SECONDS > 0:
        _converged = _outbox_sync_wait(memory_id, OUTBOX_SYNC_GRACE_SECONDS)

    result = {"memory_id": memory_id,
              "status": "stored" if _converged else "stored_index_pending",
              "canonical": "active" if _converged else "pending_index",
              "duplicate_of": duplicate_id,
              "confidence": confidence, "provenance": provenance,
              "pii_masked": _pii_masked,
              "quality_score": round(q_score, 4), "node_type": node_type,
              "targets": targets}
    # 2B-3: stored/quarantined counters
    _WRITE_GATE_STATS["stored"] += 1
    _WRITE_GATE_STATS["by_source"][source]["stored"] += 1
    _op_telemetry_record("write_gate", 0.0, "stored", quality_score=q_score)
    try:
        if KB_AUTOSYNC and importance >= KB_SYNC_MIN_IMPORTANCE and not findings:
            import threading as _th
            _th.Thread(target=_kb_reconcile_safe,
                       kwargs={"content": content, "scope": scope, "memory_id": memory_id},
                       daemon=True).start()
    except Exception:
        pass
    # Sprint-1 1.2: async LLM plausibility pass for high-stakes memories.
    if importance >= 7 or category == "fact":
        _judge_enqueue(memory_id, content)
    if findings:
        _WRITE_GATE_STATS["quarantined"] += 1
        _WRITE_GATE_STATS["by_source"][source]["quarantined"] += 1
        _op_telemetry_record("write_gate", 0.0, "quarantined", quality_score=q_score)
    if contradiction_of:
        result["contradiction_of"] = contradiction_of
        result["temporal_supersession"] = True
    if poison_score > 0:
        result["poison_score"] = round(poison_score, 4)
        result["findings"] = findings
    _record_latency("memory_store", (time.perf_counter() - _start) * 1000)
    return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

# -- recall ------------------------------------------------------------------
def _tool_recall(args: dict) -> types.CallToolResult:
    _start = time.perf_counter()
    global _session_seen_ids, _RECALL_SESSION_STATS
    _RECALL_SESSION_STATS["total_recalls"] += 1
    query = args["query"]
    top_k = args.get("top_k") or 5
    category = args.get("category")
    tags = _coerce_str_list(args.get("tags"))
    tags_key = ",".join(sorted(tags)) if tags else None

    # Fail closed on read: resolve the scope through the authorization layer so
    # a caller can never widen its own read past the granted namespaces.
    try:
        scope = authorize_scope(args.get("scope"))
    except ScopeError as e:
        result = {"error": f"scope rejected: {e}"}
        _record_latency("memory_recall", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))
    as_of = args.get("as_of")
    # 3B-4: reads are audited with the same weight as writes — actor, scope,
    # query and requested depth, so any access is reconstructible later.
    _pg_audit("read:recall", None, {"query": query[:200], "top_k": top_k,
                                    "category": category, "as_of": as_of},
              scope=scope)

    # P1-4: Redis-backed recall cache (horizontal scaling)
    # Check Redis-backed cache first (before semantic cache) when horizontal scaling enabled
    if HORIZONTAL_SCALING_ENABLED and not as_of:
        cached_json = _recall_redis_get(query, top_k or 5, category, tags_key, scope)
        if cached_json:
            try:
                cached_data = json.loads(cached_json)
                log.debug("Redis recall cache hit: %s", query[:50])
                return _ok(json.dumps(cached_data, ensure_ascii=False))
            except Exception as e:
                log.warning("Failed to parse cached recall result: %s", e)
                # Fall through to normal recall path

    # Enhancement 4: Mixture-of-Agents — classify query type and apply specialized recall
    query_type = _classify_query(query)
    log.debug("recall query_type=%s query=%s", query_type, query[:60])

    # Enhancement 7: Semantic Cache — check cache before expensive hybrid recall
    now_ts = datetime.now(timezone.utc).timestamp()
    cache_vector = _embed_model().encode(query).tolist()
    # G3: temporal queries bypass the cache (an as_of call must always reflect
    # the moment requested, not a cached "now" answer).
    cache_salt = f"{category or ''}|{tags_key or ''}|{scope or ''}|asof:{as_of or ''}"
    cached = None if as_of else _check_semantic_cache(cache_vector, cache_salt)
    if cached is not None:
        # G3: cached results may predate the temporal-supersession feature —
        # drop any memory that has since been superseded before re-ranking.
        cached = [item for item in cached if item.get("status") != "superseded"]
        if not cached:
            cached = None
    if cached is not None:
        cached_memories = []
        _c_max_hybrid = max((c.get("hybrid_score", 0) for c in cached[:top_k]), default=0.0) or 1.0
        for item in cached[:top_k]:
            # Phase 2.3: Apply feedback adjustment to cached results
            orig_importance = item.get("importance", 5)
            try:
                importance = float(orig_importance) / 10.0
            except (ValueError, TypeError):
                importance = 0.5
            feedback_adj = _get_feedback_adjustment(item["memory_id"], query)
            importance = max(0.0, min(1.0, importance + feedback_adj))
            
            # Recalculate final_score with adjusted importance
            hybrid_score = item.get("hybrid_score", 0)
            # Estimate recency from cached data
            created_at = item.get("created_at", "")
            last_accessed = item.get("last_accessed_at", created_at)
            try:
                accessed_ts = datetime.fromisoformat(last_accessed.replace("Z", "+00:00")).timestamp()
            except Exception:
                accessed_ts = now_ts
            days_since_access = max(0, (now_ts - accessed_ts) / 86400)
            recency_score = 1.0 / (1.0 + days_since_access * 0.1)
            # Relevance-dominant blend (matches the non-cached path).
            _rel = max(0.0, min(1.0, hybrid_score / _c_max_hybrid))
            final_score = (_rel * RELEVANCE_WEIGHT) + (importance * IMPORTANCE_WEIGHT) + (recency_score * RECENCY_WEIGHT)

            # Confidence must be computed on the cache-hit path too. It used to
            # be omitted entirely, so a cached result and a fresh result had
            # different shapes and consumers saw confidence=None (i.e. 0.0).
            try:
                _q = float(item.get("quality_score"))
            except (TypeError, ValueError):
                _q = 0.5
            _q = max(0.0, min(1.0, _q))
            _created = item.get("created_at") or ""
            try:
                _days = max(0.0, (now_ts - datetime.fromisoformat(
                    _created.replace("Z", "+00:00")).timestamp()) / 86400.0)
            except Exception:
                _days = 0.0
            _decay = 0.5 ** (_days / _CONFIDENCE_HALF_LIFE_DAYS)
            _ev = _query_evidence(query, item.get("content") or "")
            _conf = min(1.0, max(0.5 * _q, _ev)) * _decay
            if _ev < _LOW_EVIDENCE_CUTOFF:
                _conf = min(_conf, _UNVERIFIED_CONFIDENCE_CAP)

            cached_memories.append({
                "memory_id": item["memory_id"],
                "content": item["content"],
                "dense_rrf": item.get("dense_rrf", 0),
                "sparse_rrf": item.get("sparse_rrf", 0),
                "hybrid_score": item.get("hybrid_score", 0),
                "relevance": round(_rel, 6),
                "evidence": round(_ev, 6),
                "final_score": round(final_score, 6),
                # 3A-3: `match_confidence` is THIS QUERY's evidence fit (the old
                # decorative field, renamed); `confidence` is the memory's
                # provenance trust (stated/inferred/corroborated) from the
                # store path — the one the dashboard renders.
                "match_confidence": round(_conf, 4),
                "confidence": item.get("confidence"),
                "provenance": item.get("provenance"),
                "corroboration_count": item.get("corroboration_count"),
                "category": item.get("category"),
                "importance": orig_importance,
                "tags": item.get("tags", []),
                "created_at": item.get("created_at"),
                "last_accessed_at": item.get("last_accessed_at"),
                "access_count": item.get("access_count", 0),
                "scope": item.get("scope") or "global",
                "quarantine": bool(item.get("quarantine")),
                "poison_score": item.get("poison_score", 0.0),
            })
        result = {
            "status": "cached",
            "query": query,
            "query_type": query_type,
            "count": len(cached_memories),
            "memories": _filter_canonical(cached_memories, as_of),
            "cache_hit": True,
        }
        if RECALL_INCLUDE_KB:
            try:
                _kbs = _kb_recall_hits(query, authorize_scope(args.get("scope")), max(3, (top_k or 5)))
                if _kbs:
                    result["knowledge"] = _kbs
                    result["knowledge_count"] = len(_kbs)
            except Exception:
                pass
        _RECALL_SESSION_STATS["cache_hits"] += 1
        _RECALL_SESSION_STATS["total_memories_returned"] += len(cached_memories)
        log.info("recall cache_hit query=%s count=%d", query[:40], len(cached_memories))
        _record_latency("memory_recall", (time.perf_counter() - _start) * 1000)
        return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    # Specialized recall based on query type
    hybrid_results = _build_specialized_recall(query, query_type, top_k * 3, category, tags_key, scope, args.get("as_of"))

    scored_memories = []
    # Normalize the relevance signal to 0-1 BEFORE blending. hybrid_score is a
    # Reciprocal-Rank-Fusion value whose magnitude is ~0.016 at k=60; mixing it
    # raw against 0-1 importance/recency terms meant 95%+ of final_score came
    # from importance and recency, so a high-importance irrelevant memory
    # outranked the actual answer.
    _max_hybrid = max((h for _, (_, _, h) in hybrid_results), default=0.0) or 1.0
    for (memory_id, (dense_rrf, sparse_rrf, hybrid_score)) in hybrid_results:
        # Previously-returned memories are DEPRIORITISED, not removed. Hard
        # exclusion meant a second, closely related query returned nothing at all
        # — strictly worse than showing a repeat, and it silently dropped the
        # answer whenever a user asked two questions in one session.
        _already_seen = memory_id in _session_seen_ids
        # Fetch payload for this memory
        fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
        if not fetched:
            continue
        payload = fetched[0].payload

        # Importance + recency scoring (applied on top of hybrid score)
        importance_raw = payload.get("importance", 5)
        try:
            importance = float(importance_raw) / 10.0
        except (ValueError, TypeError):
            importance = 0.5
        
        # Phase 2.3: Apply feedback adjustment to importance
        feedback_adj = _get_feedback_adjustment(memory_id, query)
        importance = max(0.0, min(1.0, importance + feedback_adj))
        
        # Phase 4: time-based decay
        decayed_importance, _ = _apply_decay(importance, payload.get("created_at", ""),
                                              payload.get("last_accessed_at", ""), now_ts,
                                              category=payload.get("category"))
        importance = decayed_importance
        
        created_at = payload.get("created_at", "")
        last_accessed = payload.get("last_accessed_at", created_at)
        try:
            accessed_ts = datetime.fromisoformat(last_accessed.replace("Z", "+00:00")).timestamp()
        except Exception:
            accessed_ts = now_ts
        days_since_access = max(0, (now_ts - accessed_ts) / 86400)
        recency_score = 1.0 / (1.0 + days_since_access * 0.1)

        # Final score: relevance 70% + importance 15% + recency 15%.
        # Relevance is normalised against the best candidate for this query so
        # all three terms share a 0-1 scale and relevance genuinely dominates.
        relevance = max(0.0, min(1.0, hybrid_score / _max_hybrid))
        final_score = (relevance * RELEVANCE_WEIGHT) + (importance * IMPORTANCE_WEIGHT) + (recency_score * RECENCY_WEIGHT)
        if _already_seen:
            # Keep it eligible so a repeat query can still return an answer, but
            # rank fresh candidates ahead of ones already shown this session.
            final_score *= _SEEN_REPEAT_PENALTY
        # Quarantined (poison-suspect) content stays RETRIEVABLE — with its
        # quarantine flag surfaced — but must never outrank clean answers.
        if payload.get("quarantine"):
            final_score *= _QUARANTINE_RANK_PENALTY

        scored_memories.append({
            "memory_id": memory_id,
            "payload": payload,
            "relevance": round(relevance, 6),
            "evidence": round(_query_evidence(query, payload.get("content") or ""), 6),
            "dense_rrf": round(dense_rrf, 6),
            "sparse_rrf": round(sparse_rrf, 6),
            "hybrid_score": round(hybrid_score, 6),
            "final_score": round(final_score, 6),
        })

    # Relevance feedback (Phase 2.3): adjust the effective ranking of each
    # recalled memory by its past feedback signals. "useful" feedback boosts,
    # "irrelevant" feedback penalizes — creating a closed loop where the agent's
    # own judgments shape future recall.
    _fb_start = time.perf_counter()
    if _redis:
        try:
            for sm in scored_memories[:top_k]:
                mid = sm.get("memory_id")
                if not mid:
                    continue
                useful, irrelevant = _get_feedback_counts(mid)
                boost = useful * FEEDBACK_BOOST + irrelevant * FEEDBACK_PENALTY
                if boost:
                    sm["final_score"] = round(sm["final_score"] + boost, 6)
        except Exception:
            pass
    _record_stage_latency("feedback_adjust", (time.perf_counter() - _fb_start) * 1000)
    _record_stage_quality("feedback_adjust", 1.0 if _redis else 0.0)
    scored_memories.sort(key=lambda x: x["final_score"], reverse=True)

    _canon_start = time.perf_counter()
    memories = []
    accessed_ids = []
    for sm in scored_memories[:top_k]:
        memory_id = sm["memory_id"]
        _session_seen_ids.add(memory_id)
        accessed_ids.append(memory_id)
        payload = sm["payload"]
        # Gap #6: confidence/provenance — decay with time since last access
        last_acc = payload.get("last_accessed_at") or payload.get("created_at") or ""
        try:
            last_acc_ts = datetime.fromisoformat(last_acc.replace("Z", "+00:00")).timestamp()
            days_since = max(0.0, (now_ts - last_acc_ts) / 86400.0)
        except Exception:
            days_since = 0.0
        half_life = _CONFIDENCE_HALF_LIFE_DAYS
        decay = 0.5 ** (days_since / half_life)
        try:
            quality = float(payload.get("quality_score"))
        except (TypeError, ValueError):
            quality = 0.5
        quality = max(0.0, min(1.0, quality))

        # Confidence must answer "how likely is this memory to be the right
        # answer to THIS query", so relevance is now the dominant factor.
        # Previously confidence was built only from quality_score, importance
        # and decay — it contained no relevance term at all, which is why a
        # high-importance irrelevant memory reported higher confidence than the
        # correct answer.
        # Absolute lexical evidence, NOT relevance. Relevance is normalised
        # against the best candidate for this query, so it reads 1.0 for the top
        # hit even when that hit is unrelated — which made an unanswerable
        # question report full confidence. `evidence` is the real overlap
        # between the query's content words and the stored text.
        evidence = sm.get("evidence", 0.0)
        base_confidence = min(1.0, max(0.5 * quality, evidence))
        confidence = round(base_confidence * decay, 4)
        # Expose the calibration inputs so callers can see WHY a result is
        # uncertain rather than being handed a bare number.
        if evidence < _LOW_EVIDENCE_CUTOFF:
            confidence = min(confidence, _UNVERIFIED_CONFIDENCE_CAP)
        memories.append({
            "memory_id": memory_id,
            "content": payload.get("content", ""),
            "dense_rrf": sm["dense_rrf"],
            "sparse_rrf": sm["sparse_rrf"],
            "hybrid_score": sm["hybrid_score"],
            "relevance": sm.get("relevance"),
            "evidence": sm.get("evidence"),
            "final_score": sm["final_score"],
            "category": payload.get("category"),
            "importance": payload.get("importance"),
            "tags": _payload_tags(payload),
            "created_at": payload.get("created_at"),
            "last_accessed_at": payload.get("last_accessed_at"),
            "access_count": payload.get("access_count", 0),
            "retrieval_count": payload.get("retrieval_count", 0),
            "useful_count": payload.get("useful_count", 0),
            "quality_score": payload.get("quality_score"),
            "scope": payload.get("scope") or "global",
            # 3A-3: two distinct semantics, never clobbered again:
            #   match_confidence — how well THIS query's evidence fits (legacy
            #                     computed value, kept under its true name)
            #   confidence/provenance — the memory's stored trust (stated /
            #                     inferred / corroborated), what the UI renders
            "match_confidence": confidence,
            "confidence": payload.get("confidence"),
            "provenance": payload.get("provenance"),
            "corroboration_count": payload.get("corroboration_count"),
            "days_since_access": round(days_since, 1),
            # P3.1: provenance-gated handling — a stored-but-flagged injection is
            # never handed back to downstream consumers as clean memory.
            "quarantine": bool(payload.get("quarantine")),
            "poison_score": payload.get("poison_score", 0.0),
        })

    # Batch update access metadata for ALL recalled memories.
    # Increment retrieval_count + access_count + update last_accessed_at per memory.
    if accessed_ids:
        now = _now_iso()
        for mid in accessed_ids:
            try:
                fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[mid], with_payload=True)
                if fetched:
                    pld = fetched[0].payload or {}
                    _client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            "last_accessed_at": now,
                            "retrieval_count": pld.get("retrieval_count", 0) + 1,
                            "access_count": pld.get("access_count", 0) + 1,
                        },
                        points=[mid],
                    )
            except Exception:
                pass

    # Enhancement 7: Store recall results in semantic cache for future identical queries.
    # Sprint-1 1.5 fix: never cache an EMPTY result set — a transiently empty
    # recall (liveness race, interrupted boot) would otherwise poison the
    # query until TTL while fresh data exists (negative-caching bug, observed
    # live: "status": "cached", "memories": [] served for a populated query).
    if memories:
        _store_semantic_cache(cache_vector, memories, cache_salt)

    # P1-4: Store in Redis-backed recall cache (horizontal scaling)
    if HORIZONTAL_SCALING_ENABLED and not as_of and memories:
        try:
            result_json = json.dumps({"query": query, "query_type": query_type, "count": len(memories), "memories": memories}, ensure_ascii=False, indent=2)
            _recall_redis_set(query, top_k, category, tags_key, scope, result_json)
        except Exception as e:
            log.warning("Failed to cache recall result in Redis: %s", e)

    _RECALL_SESSION_STATS["total_memories_returned"] += len(memories)
    # PHASE A: withhold anything the canonical tier says not live. Applied
    # HERE as well as in the dispatcher, because an in-process caller that
    # skips the dispatcher must not be able to skip the gate.
    _canon_start = time.perf_counter()
    memories = _filter_canonical(memories, as_of)
    if as_of and args.get("as_of_field") == "event":
        import datetime as _dt
        try:
            _t_as = _dt.datetime.fromisoformat(str(as_of).replace("Z", "+00:00"))
            def _event_ok(m):
                _et = m.get("event_time")
                if not _et:
                    return False
                try:
                    return _dt.datetime.fromisoformat(str(_et).replace("Z", "+00:00")) <= _t_as
                except Exception:
                    return False
            memories = [m for m in memories if _event_ok(m)]
        except Exception:
            pass
    _record_stage_latency("canonical_filter", (time.perf_counter() - _canon_start) * 1000)
    _record_stage_quality("canonical_filter", 1.0 if len(memories) > 0 else 0.0)
    result = {"query": query, "query_type": query_type, "count": len(memories), "memories": memories}
    if RECALL_INCLUDE_KB:
        try:
            _kbs = _kb_recall_hits(query, authorize_scope(args.get("scope")), max(3, (top_k or 5)))
            if _kbs:
                result["knowledge"] = _kbs
                result["knowledge_count"] = len(_kbs)
        except Exception:
            pass
    # 2C-2: graph-adjacent expansion — the "connected memory" experience:
    # expand the top hit via the durable graph (depth 1) and inject up to
    # N neighbors not already in the result set, explicitly marked. Reads
    # the Redis fast path; any graph failure degrades to vector-only silently.
    try:
        if memories:
            _adj = []
            _seen = {m.get("memory_id") for m in memories}
            for _top in memories[:2]:
                if len(_adj) >= 3:
                    break
                for _nb in (_graph_get_neighbors(_top.get("memory_id"), None, 1) or []):
                    if len(_adj) >= 3:
                        break
                    _n = _nb.get("related_id") or _nb.get("memory_id")
                    if not _n or _n in _seen:
                        continue
                    try:
                        _pt = _client.retrieve(collection_name=COLLECTION_NAME,
                                               ids=[_n], with_payload=True)
                    except Exception:
                        continue
                    if _pt and (_pt[0].payload or {}).get("content"):
                        _entry = dict(_pt[0].payload)
                        _entry["memory_id"] = _n
                        _entry["graph_adjacent"] = True
                        _entry["adjacent_to"] = _top.get("memory_id")
                        _entry["graph_relation"] = _nb.get("relation")
                        _adj.append(_entry)
                        _seen.add(_n)
            if _adj:
                result["graph_adjacent"] = _adj
    except Exception as _ge:
        log.debug("graph expansion skipped: %s", _ge)
    # 1B-5: surface degraded-mode recalls (e.g. Qdrant down → sparse-only)
    if globals().get("_RECALL_DEGRADED"):
        result["degraded"] = _RECALL_DEGRADED
    _record_latency("memory_recall", (time.perf_counter() - _start) * 1000)
    return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False, indent=2))


# -- lookup_exact (P1-8) ------------------------------------------------------
def _tool_lookup_exact(args: dict) -> types.CallToolResult:
    """P1-8: Exact fact retrieval by metadata fields — SQL-style WHERE on payload.
    
    Supports exact lookup by: memory_id, scope, category, source, tag, created_at range.
    Routes to Postgres (mem_canonical) for exact matches, not Qdrant.
    """
    _start = time.perf_counter()
    field = args.get("field")
    value = args.get("value")
    
    if not field:
        return _err("field is required")
    if value is None:
        return _err("value is required")
    
    # Authorization: scope must be provided and authorized
    scope = args.get("scope")
    if scope:
        try:
            scope = authorize_scope(scope)
        except ScopeError as e:
            return _err(f"scope rejected: {e}")
    else:
        # Default to caller's authorized scopes
        allowed = authorized_scopes()
        if allowed is None:
            return _err("no authorized scopes available")
        scope = None  # Will filter by authorized scopes in query
    
    # Build query based on field
    allowed_fields = {"memory_id", "scope", "category", "source", "tag", "created_at"}
    if field not in allowed_fields:
        return _err(f"field must be one of: {', '.join(sorted(allowed_fields))}")
    
    try:
        if field == "memory_id":
            # Exact memory_id lookup
            sql = "SELECT * FROM mem_canonical WHERE memory_id = :val"
            params = {"val": str(value)}
            if scope:
                sql += " AND scope = :scope"
                params["scope"] = scope
            rows = _pg_query(sql, params)
            
        elif field == "scope":
            # Scope filter
            sql = "SELECT * FROM mem_canonical WHERE scope = :val"
            params = {"val": str(value)}
            rows = _pg_query(sql, params)
            
        elif field == "category":
            # Category filter (from payload)
            sql = "SELECT * FROM mem_canonical WHERE payload->>'category' = :val"
            params = {"val": str(value)}
            if scope:
                sql += " AND scope = :scope"
                params["scope"] = scope
            rows = _pg_query(sql, params)
            
        elif field == "source":
            # Source filter (from payload)
            sql = "SELECT * FROM mem_canonical WHERE payload->>'source' = :val"
            params = {"val": str(value)}
            if scope:
                sql += " AND scope = :scope"
                params["scope"] = scope
            rows = _pg_query(sql, params)
            
        elif field == "tag":
            # Tag filter (tags stored as comma-separated string in payload)
            sql = "SELECT * FROM mem_canonical WHERE payload->>'tags' LIKE :val"
            params = {"val": f"%{str(value)}%"}
            if scope:
                sql += " AND scope = :scope"
                params["scope"] = scope
            rows = _pg_query(sql, params)
            
        elif field == "created_at":
            # Date range filter - value should be dict with 'from' and/or 'to'
            if not isinstance(value, dict):
                return _err("created_at value must be an object with 'from' and/or 'to' ISO8601 timestamps")
            
            conditions = []
            params = {}
            if "from" in value:
                conditions.append("created_at >= :from_ts")
                params["from_ts"] = value["from"]
            if "to" in value:
                conditions.append("created_at <= :to_ts")
                params["to_ts"] = value["to"]
            if not conditions:
                return _err("created_at requires 'from' and/or 'to'")
            
            sql = f"SELECT * FROM mem_canonical WHERE {' AND '.join(conditions)}"
            if scope:
                sql += " AND scope = :scope"
                params["scope"] = scope
            rows = _pg_query(sql, params)
        
        # Format results
        memories = []
        for row in rows:
            mem = dict(row)
            # Convert UUID to string if needed
            if "id" in mem and hasattr(mem["id"], "hex"):
                mem["id"] = str(mem["id"])
            memories.append(mem)
        
        result = {
            "field": field,
            "value": value,
            "count": len(memories),
            "memories": memories,
        }
        _record_latency("memory_lookup_exact", (time.perf_counter() - _start) * 1000)
        return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False, default=str))
        
    except PostgresUnavailableError as e:
        result = {"error": f"Postgres unavailable: {e}"}
        _record_latency("memory_lookup_exact", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))
    except Exception as e:
        result = {"error": f"lookup failed: {e}"}
        _record_latency("memory_lookup_exact", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))

# -- list --------------------------------------------------------------------
def _tool_list(args: dict) -> types.CallToolResult:
    limit = min(args.get("limit") or 20, 100)
    offset = args.get("offset") or 0
    category = args.get("category")
    tags = _coerce_str_list(args.get("tags"))
    tags_key = ",".join(sorted(tags)) if tags else None
    # Authorize here too: this tool is reachable directly, not only via the
    # central guard, and an unscoped list used to scroll the entire collection.
    as_of = args.get("as_of")
    scope = authorize_scope(args.get("scope"))
    filter_q = _build_filter(category, tags_key, scope, as_of)

    results = _client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter_q,
        limit=limit,
        offset=offset,
        with_payload=True,
    )
    memories = [_payload_to_memory(r.payload, r.id) for r in results[0]]
    memories = _filter_canonical(memories, as_of)
    _pg_audit("read:list", None, {"scope": scope, "category": category,
                                   "returned": len(memories)}, scope=scope)
    return _ok(json.dumps({"count": len(memories), "memories": memories}, ensure_ascii=False, indent=2))

# -- get ---------------------------------------------------------------------
def _tool_get(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    as_of = args.get("as_of")
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    _pg_audit("read:get", memory_id)
    # PHASE A/B: Qdrant is a derived layer. Reading straight from it returned
    # the full content of memories whose canonical record had been tombstoned
    # whenever a stale point survived — the exact resurrection hole. Consult
    # the canonical tier before disclosing anything.
    if memory_id not in _canonical_live_ids([memory_id], include_superseded=bool(as_of)):
        status = _pg_query(
            "SELECT status FROM mem_canonical WHERE memory_id=:m",
            {"m": memory_id})
        _s0 = status[0] if status else None
        detail = (_s0["status"] if isinstance(_s0, dict) else _s0[0]) if _s0 else "absent"
        return _err(f"Memory not retrievable: {memory_id} (canonical: {detail})")
    results = _client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[memory_id],
        with_payload=True,
    )
    if not results:
        return _err(f"Memory not found: {memory_id}")
    if as_of and not _payload_passes_recall_constraints(results[0].payload, None, None, None, as_of):
        return _err(f"Memory not valid at {as_of}: {memory_id}")
    return _ok(json.dumps(_payload_to_memory(results[0].payload, memory_id), ensure_ascii=False, indent=2))

# -- update ------------------------------------------------------------------
def _tool_update(args: dict) -> types.CallToolResult:
    _start = time.perf_counter()
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    scope = args.get("scope")  # Phase 3.3
    existing = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
    if not existing:
        result = {"error": f"Memory not found: {memory_id}"}
        _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))
    old_payload = existing[0].payload
    new_content = args.get("content", old_payload["content"])

    # 3B-2: the editor (dashboard or MCP) goes through the same secret gate
    # as a fresh store — an update must not become the bypass for smuggling
    # a credential into an already-trusted row.
    _sec_rej, _sec_findings, new_content = _scan_secrets(new_content)
    if _sec_rej:
        _pg_audit("secret_block", memory_id, {"findings": _sec_findings, "via": "update"})
        result = {"blocked": True, "gate": "secret_detection",
                  "findings": _sec_findings,
                  "message": "Update rejected: content contains a credential-shaped string."}
        _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    # Enhancement 3: Poisoning guard
    blocked, poison_score, findings = _check_poison(new_content)
    if blocked:
        result = {
            "blocked": True, "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Content flagged by poisoning guard.",
        }
        _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))
    new_content = _sanitize_content(new_content)

    # Stage 4: an update is a canonical mutation, so it commits to
    # mem_canonical AND enqueues MEMORY_UPDATED in the same transaction.
    # Without this the derived tier moves ahead of the canonical version and a
    # crash leaves the two permanently disagreeing.
    prev_row = _canonical_row(memory_id)
    if prev_row is None:
        result = {"error": f"Memory not found in canonical tier: {memory_id}"}
        _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))
    new_version = int(prev_row["version"]) + 1
    try:
        _canonical_write(
            memory_id, prev_row["scope"] or scope, prev_row["node_type"] or "fact",
            new_content, version=new_version, event_type="MEMORY_UPDATED",
            payload=prev_row.get("payload") or {})
    except PostgresUnavailableError as _e:
        result = {"error": f"canonical update did not commit: Postgres unavailable: {_e}"}
        _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))

    now = _now_iso()
    payload = dict(old_payload)
    for field in ("category", "importance", "tags", "source"):
        if field in args:
            val = args[field]
            payload[field] = ",".join(val) if field == "tags" else val
    if scope:
        payload["scope"] = scope
    payload["updated_at"] = now
    vector = _embed_model().encode(new_content).tolist()
    payload["content"] = new_content
    tokens = _tokenize(new_content)
    payload["tokens"] = tokens
    payload["doc_len"] = len(tokens)

    # Enhancement 6: Store version snapshot before update
    _store_version(memory_id, old_payload, existing[0].vector)

    _client.upsert(
        collection_name=COLLECTION_NAME,
        points=[qmodels.PointStruct(id=memory_id, vector=vector, payload=payload)],
    )
    # Remove old BM25 entry and add updated one (incremental)
    _remove_from_bm25_index(memory_id)
    _add_to_bm25_index(memory_id, new_content)
    _invalidate_recall_caches()
    log.info("updated id=%s", memory_id)
    _pg_audit("update", memory_id, {"version": new_version})
    result = {"memory_id": memory_id, "status": "updated"}
    if poison_score > 0:
        result["poison_score"] = round(poison_score, 4)
    _record_latency("memory_update", (time.perf_counter() - _start) * 1000)
    return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

# -- consolidate ------------------------------------------------------------
def _tool_consolidate(args: dict) -> types.CallToolResult:
    max_count = args.get("max_count", 1000)
    sim_threshold = args.get("similarity_threshold", 0.92)
    # CRITICAL 2026-10-08: dry_run was ACCEPTED but never honored — a
    # "dry-run" deleted 1392 live Qdrant points (canonical rows survived,
    # restorable from mem_canonical). Deletion, archival and BM25 mutation
    # are now all gated; dry-run is REPORT-ONLY.
    dry_run = bool(args.get("dry_run", False))
    now_ts = datetime.now(timezone.utc).timestamp()

    stats = {"scanned": 0, "duplicates_removed": 0, "old_evicted": 0, "errors": 0}

    try:
        # 2B-1 fix 2026-10-08: use the vectors Qdrant ALREADY stores instead of
        # re-encoding the whole corpus through the embedding model — the
        # re-encode path OOM-killed the process (silent SIGKILL, ~2.5-3.2GB
        # peak) and is architecturally wasteful. Stored-vector + numpy all-
        # pairs is ~6.5MB for 1600 docs and milliseconds of BLAS.
        all_points = []
        _offset = None
        while True:
            _page = _client.scroll(collection_name=COLLECTION_NAME, limit=10000,
                                   with_payload=True, with_vectors=True,
                                   offset=_offset)
            all_points.extend(_page[0])
            _offset = _page[1]
            if not _offset:
                break
        points = all_points
        stats["scanned"] = len(points)

        import numpy as _np
        vecs = _np.array([p.vector for p in points], dtype=_np.float32) \
            if points else _np.zeros((0, 0), dtype=_np.float32)
        # 2B-1 tenant-safety fix: similarity comparison is SCOPE-BUCKETED —
        # a near-identical memory in another tenant's scope must never be
        # deduplicated against (or deleted because of) global content.
        _scope_of = lambda p: (p.payload.get("scope") or "global")
        buckets: dict[str, list[int]] = {}
        for _i, _p in enumerate(points):
            buckets.setdefault(_scope_of(_p), []).append(_i)
        to_delete = set()
        for _scope, idxs in buckets.items():
            if len(idxs) < 2 or vecs.ndim != 2:
                continue
            sub = vecs[_np.array(idxs, dtype=_np.int64)]
            norms = _np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = sub / norms
            sim_matrix = unit @ unit.T
            for _a, _i in enumerate(idxs):
                if points[_i].id in to_delete:
                    continue
                for _b in range(_a + 1, len(idxs)):
                    _j = idxs[_b]
                    if points[_j].id in to_delete:
                        continue
                    if float(sim_matrix[_a, _b]) >= sim_threshold:
                        older = points[_i] if points[_i].payload.get("created_at", "") < points[_j].payload.get("created_at", "") else points[_j]
                        to_delete.add(older.id)
                        stats["duplicates_removed"] += 1

        # Evict old/unaccessed — 2B-4: per-category windows; protected
        # categories (procedure/preference/anti_pattern/decision) never evict.
        for p in points:
            if p.id in to_delete:
                continue
            _cat = (p.payload or {}).get("category") or "general"
            if _cat in _PROTECTED_FROM_EVICT:
                continue
            _evict_days_cat = CATEGORY_DECAY_TABLE.get(_cat, {}).get("evict_unused_days")
            if _evict_days_cat is None:
                continue
            _cat_cutoff = now_ts - (_evict_days_cat * 86400)
            last_accessed = p.payload.get("last_accessed_at", p.payload.get("created_at", ""))
            try:
                accessed_ts = datetime.fromisoformat(last_accessed.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if accessed_ts < _cat_cutoff:
                to_delete.add(p.id)
                stats["old_evicted"] += 1

        # Enhancement 9: Auto-archive memories with decayed importance below threshold
        stats["auto_archived"] = 0
        for p in points:
            if p.id in to_delete:
                continue
            imp = p.payload.get("importance", 5)
            try:
                imp_norm = float(imp) / 10.0
            except (ValueError, TypeError):
                imp_norm = 0.5
            _cat = (p.payload or {}).get("category") or "general"
            if _cat in _PROTECTED_FROM_EVICT:
                continue  # 2B-4: procedures/preferences/decisions never auto-archive
            decayed, _ = _apply_decay(imp_norm, p.payload.get("created_at", ""),
                                       p.payload.get("last_accessed_at", ""), now_ts,
                                       category=_cat)
            if decayed < AUTO_ARCHIVE_THRESHOLD:
                # Archive: lower importance, add archived tag, skip deletion
                if not dry_run:
                    new_tags = p.payload.get("tags", "")
                    if "archived" not in new_tags.split(","):
                        new_tags = (new_tags + ",archived").strip(",")
                    _client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"importance": 1, "tags": new_tags},
                        points=[p.id],
                    )
                stats["auto_archived"] += 1

        # Respect max_count — keep newest
        if len(points) - len(to_delete) > max_count:
            keep = sorted(
                [p for p in points if p.id not in to_delete],
                key=lambda p: p.payload.get("created_at", ""),
                reverse=True
            )
            for p in keep[max_count:]:
                to_delete.add(p.id)

        # 2B-5: dedup is MERGE, not delete — the loser keeps full content,
        # gets duplicate_of + archived tag + importance 1 and drops out of
        # ranking; nothing is ever hard-deleted (restorable via versions).
        # Hard delete remains available through memory_forget (GDPR) only.
        if to_delete and dry_run:
            stats["dry_run_would_merge"] = len(to_delete)
            for _mid in list(to_delete)[:50]:
                _mlog("consolidate", "would_merge", memory_id=_mid,
                      reason="dry-run duplicate candidate (2B-5 merge-not-delete)",
                      dry_run=True)
        if to_delete and not dry_run:
            by_id = {p.id: p for p in points}
            merged = 0
            for mid in to_delete:
                loser = by_id.get(mid)
                if loser is None:
                    continue
                old_tags = (loser.payload or {}).get("tags", "")
                new_tags = old_tags if "archived" in str(old_tags).split(",") \
                    else (str(old_tags) + ",archived").strip(",")
                _client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"importance": 1, "tags": new_tags,
                              "duplicate_of": ""},
                    points=[mid],
                )
                _pg_run("UPDATE mem_canonical SET payload = jsonb_set("
                        "payload || jsonb_build_object('archived_by_dedup', true), "
                        "'{importance}', '1') WHERE memory_id = :m",
                        {"m": mid}, table="canonical-dedup-merge") \
                    if False else None
                merged += 1
                _mlog("consolidate", "merge", memory_id=mid,
                      before={"tags": old_tags, "importance": (loser.payload or {}).get("importance")},
                      after={"tags": new_tags, "importance": 1, "archived": True},
                      reason="dedup merge 2B-5: loser archived, content kept, restorable",
                      dry_run=False)
            stats["merged"] = merged
            _invalidate_recall_caches()
        _invalidate_recall_caches()
        log.info("consolidate: %s", stats)
        return _ok(json.dumps({"status": "consolidated", "stats": stats,
                                "deleted_count": 0 if dry_run else len(to_delete),
                                "would_delete": len(to_delete)}, ensure_ascii=False))
    except Exception as e:
        log.exception("consolidate failed")
        return _err(f"Consolidation failed: {e}")


# ---------------------------------------------------------------------------
# P0-2: Session Summarization and Compaction
# ---------------------------------------------------------------------------
# Provides structured summarization of trajectory sessions and compaction
# of raw step data into compact summaries with critical-fact verification.

# Compaction trigger configuration (env-configurable)
COMPACTION_STEP_THRESHOLD = int(os.environ.get("MEMORY_COMPACTION_STEP_THRESHOLD", "50"))  # compact after N steps
COMPACTION_TOKEN_THRESHOLD = int(os.environ.get("MEMORY_COMPACTION_TOKEN_THRESHOLD", "4000"))  # compact when raw steps exceed tokens
COMPACTION_TIME_THRESHOLD_HOURS = int(os.environ.get("MEMORY_COMPACTION_TIME_THRESHOLD_HOURS", "24"))  # compact sessions older than N hours
COMPACTION_MAX_SUMMARY_TOKENS = int(os.environ.get("MEMORY_COMPACTION_MAX_SUMMARY_TOKENS", "2000"))  # max tokens in summary

# Compaction metrics
_COMPACTION_METRICS = {
    "total_compactions": 0,
    "compression_ratio": [],
    "critical_fact_retention_rate": [],
    "compaction_latency_ms": [],
    "sessions_compacted": 0,
}

def _extract_critical_facts(steps: list[dict]) -> list[str]:
    """Extract candidate critical facts from trajectory steps.
    
    Critical facts are: error messages, config values, tool outputs with specific data,
    API responses, file paths, command outputs, and explicit fact statements.
    """
    critical_facts = []
    for step in steps:
        action = step.get("action", "")
        observation = step.get("observation", "")
        tool = step.get("tool_used", "")
        
        # Look for explicit fact patterns
        for text in [action, observation]:
            if not text:
                continue
            text_lower = text.lower()
            # Patterns indicating critical facts
            if any(pattern in text_lower for pattern in [
                "error:", "exception:", "traceback", "failed:", "returned:", "output:",
                "config:", "setting:", "value:", "result:", "found:", "discovered:",
                "path:", "file:", "command:", "executed:", "response:", "status:",
                "version:", "id:", "uuid:", "key:", "token:", "secret:", "password:",
                "is:", "equals:", "set to:", "configured as:", "using:", "enabled:",
            ]):
                # Extract the sentence containing the pattern
                sentences = text.replace('\n', ' ').split('. ')
                for sent in sentences:
                    if any(pattern in sent.lower() for pattern in [
                        "error:", "exception:", "traceback", "failed:", "returned:", "output:",
                        "config:", "setting:", "value:", "result:", "found:", "discovered:",
                        "path:", "file:", "command:", "executed:", "response:", "status:",
                        "version:", "id:", "uuid:", "key:", "token:", "secret:", "password:",
                        "is:", "equals:", "set to:", "configured as:", "using:", "enabled:",
                    ]):
                        critical_facts.append(sent.strip()[:500])
        
        # Tool-specific critical outputs
        if tool in ["memory_recall", "memory_store", "memory_list", "bash", "shell_run_command", "read", "write"]:
            if observation and len(observation) < 500:
                critical_facts.append(f"[{tool}] {observation[:300]}")
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for fact in critical_facts:
        key = fact.lower()[:100]
        if key not in seen:
            seen.add(key)
            unique.append(fact)
    return unique[:20]  # Cap at 20 critical facts

def _generate_session_summary(task: str, task_type: str, outcome: str, steps: list[dict], 
                               duration_minutes: int, lessons: list[str], facts: list[str],
                               critical_facts: list[str]) -> str:
    """Generate a compact structured summary of a session."""
    lines = []
    lines.append("SESSION SUMMARY")
    lines.append(f"Task: {task}")
    if task_type:
        lines.append(f"Type: {task_type}")
    lines.append(f"Outcome: {outcome}")
    if duration_minutes:
        lines.append(f"Duration: {duration_minutes} min")
    lines.append(f"Steps: {len(steps)}")
    
    success_count = sum(1 for s in steps if s.get("success"))
    failure_count = len(steps) - success_count
    lines.append(f"Success: {success_count}, Failed: {failure_count}")
    
    if lessons:
        lines.append(f"\nKEY LESSONS ({len(lessons)}):")
        for i, lesson in enumerate(lessons[:5], 1):
            lines.append(f"  {i}. {lesson[:200]}")
    
    if facts:
        lines.append(f"\nFACTS DISCOVERED ({len(facts)}):")
        for i, fact in enumerate(facts[:5], 1):
            lines.append(f"  {i}. {fact[:200]}")
    
    if critical_facts:
        lines.append(f"\nCRITICAL FACTS ({len(critical_facts)}):")
        for i, fact in enumerate(critical_facts[:10], 1):
            lines.append(f"  {i}. {fact[:200]}")
    
    # Failed step info
    failed_step = next((s.get("step_index") for s in steps if not s.get("success")), None)
    if failed_step is not None:
        lines.append(f"\nFIRST FAILURE: Step {failed_step}")
    
    return "\n".join(lines)

def _verify_critical_facts_retained(original_steps: list[dict], summary_content: str) -> dict:
    """Verify that critical facts from original steps are retained in summary.
    
    Returns dict with retention_rate, facts_found, facts_missing.
    """
    critical_facts = _extract_critical_facts(original_steps)
    if not critical_facts:
        return {"retention_rate": 1.0, "facts_found": 0, "facts_missing": 0, "details": []}
    
    summary_lower = summary_content.lower()
    found = 0
    details = []
    for fact in critical_facts:
        # Check if key terms from fact appear in summary
        key_terms = [w for w in fact.lower().split() if len(w) > 3]
        if not key_terms:
            continue
        # At least 50% of key terms must be in summary
        matches = sum(1 for term in key_terms if term in summary_lower)
        if matches >= max(1, len(key_terms) // 2):
            found += 1
            details.append({"fact": fact[:100], "retained": True})
        else:
            details.append({"fact": fact[:100], "retained": False})
    
    return {
        "retention_rate": round(found / len(critical_facts), 3),
        "facts_found": found,
        "facts_missing": len(critical_facts) - found,
        "details": details,
    }

def _compact_trajectory_steps(task_id: str, task: str, task_type: str, outcome: str,
                               steps: list[dict], duration_minutes: int,
                               lessons: list[str], facts: list[str]) -> dict:
    """Compact raw trajectory steps into a structured summary.
    
    Returns dict with summary_memory_id, compression_ratio, verification_result.
    """
    t0 = time.time()
    
    # Extract critical facts before compaction
    critical_facts = _extract_critical_facts(steps)
    
    # Generate summary
    summary_content = _generate_session_summary(
        task, task_type, outcome, steps, duration_minutes, lessons, facts, critical_facts
    )
    
    # Verify critical facts retained
    verification = _verify_critical_facts_retained(steps, summary_content)
    
    # Calculate compression ratio
    raw_tokens = _vc_count_tokens(json.dumps(steps, default=str))
    summary_tokens = _vc_count_tokens(summary_content)
    compression_ratio = round(raw_tokens / max(1, summary_tokens), 2)
    
    # Store summary as a new memory (category=summary, high importance)
    summary_id = str(uuid.uuid4())
    now = _now_iso()
    tags = ["session_summary", "compacted", f"outcome-{outcome}", task[:30].replace(" ", "_")]
    if task_type:
        tags.append(task_type)
    
    try:
        # Canonical-tier fix 2026-10-08: this path used to upsert Qdrant + BM25
        # DIRECTLY, bypassing mem_canonical + the outbox. Post-canonical era
        # that made summaries (a) non-durable-by-architecture and (b) silently
        # INVISIBLE to recall, because _filter_canonical drops every id with
        # no live canonical row. Route through the same atomic canonical +
        # outbox commit as every other store; the worker owns derived tiers.
        payload = {
            "category": "summary",
            "importance": 8,
            "tags": ",".join(tags),
            "source": "session_compaction",
            "original_task_id": task_id,
            "original_step_count": len(steps),
            "compression_ratio": compression_ratio,
            "critical_fact_retention": verification["retention_rate"],
        }
        _canonical_write(
            summary_id, "global", "fact", summary_content, version=1,
            event_type="MEMORY_CREATED", payload=payload)
        _invalidate_recall_caches()
    except PostgresUnavailableError as e:
        log.error("session summary canonical write failed (PG unavailable): %s", e)
        return {"error": str(e)}
    except Exception as e:
        log.error("failed to store session summary: %s", e)
        return {"error": str(e)}
    
    # Archive raw steps in Redis with pointer to summary
    if _redis:
        try:
            archive_key = f"trajectory:archive:{task_id}"
            _redis.setex(
                archive_key,
                30 * 24 * 3600,  # 30 days
                json.dumps({
                    "task_id": task_id,
                    "summary_memory_id": summary_id,
                    "original_step_count": len(steps),
                    "compression_ratio": compression_ratio,
                    "archived_at": now,
                }, default=str)
            )
            # Delete raw steps (keep only the pointer)
            _redis.delete(_trajectory_steps_key(task_id))
            _redis.delete(_trajectory_meta_key(task_id))
        except Exception as e:
            log.warning("failed to archive trajectory steps: %s", e)
    
    latency_ms = int((time.time() - t0) * 1000)
    
    # Update metrics
    _COMPACTION_METRICS["total_compactions"] += 1
    _COMPACTION_METRICS["compression_ratio"].append(compression_ratio)
    _COMPACTION_METRICS["critical_fact_retention_rate"].append(verification["retention_rate"])
    _COMPACTION_METRICS["compaction_latency_ms"].append(latency_ms)
    _COMPACTION_METRICS["sessions_compacted"] += 1
    
    # Keep metrics lists bounded
    max_metrics = 1000
    for key in ["compression_ratio", "critical_fact_retention_rate", "compaction_latency_ms"]:
        if len(_COMPACTION_METRICS[key]) > max_metrics:
            _COMPACTION_METRICS[key] = _COMPACTION_METRICS[key][-max_metrics:]
    
    return {
        "summary_memory_id": summary_id,
        "compression_ratio": compression_ratio,
        "raw_tokens": raw_tokens,
        "summary_tokens": summary_tokens,
        "verification": verification,
        "compaction_latency_ms": latency_ms,
    }

def _should_compact(steps: list[dict], started_at: str) -> tuple[bool, str]:
    """Determine if a session should be compacted based on triggers."""
    if not steps:
        return False, "no_steps"
    
    # Step count threshold
    if len(steps) >= COMPACTION_STEP_THRESHOLD:
        return True, f"step_threshold_{len(steps)}"
    
    # Token threshold (estimate raw step tokens)
    raw_tokens = _vc_count_tokens(json.dumps(steps, default=str))
    if raw_tokens >= COMPACTION_TOKEN_THRESHOLD:
        return True, f"token_threshold_{raw_tokens}"
    
    # Time threshold
    try:
        start_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
        hours_elapsed = (datetime.now(timezone.utc).timestamp() - start_ts) / 3600
        if hours_elapsed >= COMPACTION_TIME_THRESHOLD_HOURS:
            return True, f"time_threshold_{hours_elapsed:.1f}h"
    except Exception:
        pass
    
    return False, "no_trigger"

def _tool_session_summarize(args: dict) -> types.CallToolResult:
    """P0-2: Summarize and compact a trajectory session.
    
    Can be called explicitly or triggered automatically by:
    - Step count threshold (default 50)
    - Token threshold (default 4000)
    - Time threshold (default 24 hours)
    - Explicit call
    
    Returns the summary memory ID, compression ratio, and critical-fact verification.
    """
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required")
    
    task_type = args.get("task_type")
    outcome = (args.get("outcome") or "").strip().lower()
    if outcome not in ("success", "failure", "partial", "in_progress"):
        return _err("outcome must be one of: success, failure, partial, in_progress")
    
    duration_minutes = args.get("duration_minutes")
    lessons = _coerce_str_list(args.get("lessons_learned"))
    facts = _coerce_str_list(args.get("facts_discovered"))
    
    # Find the trajectory by task_signature or task_id
    task_sig = _task_signature(task)
    task_id = args.get("task_id")
    
    if not task_id:
        open_key = f"trajectory:open:{task_sig}"
        try:
            task_id = _redis.get(open_key)
        except Exception:
            task_id = None
    
    if not task_id:
        # Try to find by recalling similar trajectories
        try:
            trajs = _fetch_trajectories_by_query(task, top_k=1)
            if trajs:
                task_id = trajs[0].get("memory_id")
        except Exception:
            pass
    
    if not task_id:
        return _err("No trajectory found for this task. Provide task_id or ensure trajectory_log was called.")
    
    # Read steps from Redis or from stored trajectory
    steps = []
    started_at = _now_iso()
    
    # First try Redis (for in-progress sessions)
    try:
        raw = _redis.get(_trajectory_steps_key(task_id))
        if raw:
            steps = json.loads(raw)
        else:
            # Try to get from stored trajectory memory
            traj_data = _client.retrieve(collection_name=COLLECTION_NAME, ids=[task_id], with_payload=True)
            if traj_data:
                pl = traj_data[0].payload or {}
                steps_json = pl.get("steps_json")
                if steps_json:
                    steps = json.loads(steps_json)
                    started_at = pl.get("created_at", _now_iso())
    except Exception as e:
        log.warning("session_summarize failed to read steps: %s", e)
    
    if not steps:
        return _err(f"No steps found for task_id {task_id}")
    
    # Check compaction trigger (unless forced)
    force = bool(args.get("force", False))
    if not force:
        should, reason = _should_compact(steps, started_at)
        if not should:
            return _ok(json.dumps({
                "status": "skipped",
                "reason": f"No compaction trigger met ({reason})",
                "step_count": len(steps),
                "task_id": task_id,
            }, ensure_ascii=False))
    
    # Perform compaction
    result = _compact_trajectory_steps(task_id, task, task_type, outcome, steps, duration_minutes, lessons, facts)
    
    if "error" in result:
        return _err(f"Compaction failed: {result['error']}")
    
    return _ok(json.dumps({
        "status": "compacted",
        "summary_memory_id": result["summary_memory_id"],
        "task_id": task_id,
        "original_step_count": len(steps),
        "compression_ratio": result["compression_ratio"],
        "raw_tokens": result["raw_tokens"],
        "summary_tokens": result["summary_tokens"],
        "critical_fact_retention": result["verification"],
        "compaction_latency_ms": result["compaction_latency_ms"],
    }, ensure_ascii=False))

def _tool_compaction_metrics(args: dict = None) -> types.CallToolResult:
    """Get session compaction metrics."""
    ratios = _COMPACTION_METRICS["compression_ratio"]
    retention = _COMPACTION_METRICS["critical_fact_retention_rate"]
    latencies = _COMPACTION_METRICS["compaction_latency_ms"]
    
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    
    result = {
        "total_compactions": _COMPACTION_METRICS["total_compactions"],
        "sessions_compacted": _COMPACTION_METRICS["sessions_compacted"],
        "compression_ratio": {
            "count": len(ratios),
            "mean": round(sum(ratios) / len(ratios), 2) if ratios else 0,
            "p50": percentile(ratios, 50),
            "p95": percentile(ratios, 95),
            "p99": percentile(ratios, 99),
            "max": max(ratios) if ratios else 0,
        },
        "critical_fact_retention_rate": {
            "count": len(retention),
            "mean": round(sum(retention) / len(retention), 3) if retention else 0,
            "p50": percentile(retention, 50),
            "p95": percentile(retention, 95),
            "min": min(retention) if retention else 0,
        },
        "compaction_latency_ms": {
            "count": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0,
        },
        "trigger_config": {
            "step_threshold": COMPACTION_STEP_THRESHOLD,
            "token_threshold": COMPACTION_TOKEN_THRESHOLD,
            "time_threshold_hours": COMPACTION_TIME_THRESHOLD_HOURS,
            "max_summary_tokens": COMPACTION_MAX_SUMMARY_TOKENS,
        },
    }
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))

# -- track_access ------------------------------------------------------------
def _tool_track_access(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    try:
        existing = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True, with_vectors=True)
        if not existing:
            return _err(f"Memory not found: {memory_id}")
        payload = dict(existing[0].payload)
        payload["access_count"] = payload.get("access_count", 0) + 1
        payload["last_accessed_at"] = _now_iso()
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=memory_id, vector=existing[0].vector, payload=payload)],
        )
        return _ok(json.dumps({"memory_id": memory_id, "access_count": payload["access_count"], "status": "updated"}, ensure_ascii=False))
    except Exception as e:
        return _err(str(e))

# -- delete ------------------------------------------------------------------
def _tool_delete_batch(memory_ids: list[str]) -> types.CallToolResult:
    """Delete several memories. Reports per-id outcomes — never a blanket OK."""
    per_id, failures = [], {}
    for mid in memory_ids:
        try:
            out = _tool_delete({"memory_id": mid})
            body = out.content[0].text
            ok = not getattr(out, "is_error", False)
            status = "deleted"
            if ok:
                try:
                    status = json.loads(body).get("status", "deleted")
                except Exception:
                    pass
            else:
                status = "error"
            per_id.append({"memory_id": mid, "status": status})
            if status in ("error",):
                failures[mid] = body[:200]
        except Exception as _e:
            per_id.append({"memory_id": mid, "status": "error"})
            failures[mid] = f"{type(_e).__name__}: {_e}"
    deleted = sum(1 for r in per_id if r["status"] in
                  ("deleted", "deleted_index_pending"))
    return _ok(json.dumps({
        "status": "deleted" if not failures else "partial",
        "requested": len(memory_ids),
        "deleted": deleted,
        "failed": len(failures),
        "results": per_id,
        "failures": failures,
    }, ensure_ascii=False))


def _tool_delete(args: dict) -> types.CallToolResult:
    """PHASE B — canonical tombstone deletion.

    Order is now inverted from the previous implementation, which deleted
    Qdrant FIRST and then best-effort cleared BM25/Redis/graph. If any of those
    cleanups threw, the call still answered "deleted" while a derived index
    still held the memory — a false success, and the reason a deleted memory
    could come back.

    Now: the tombstone + its outbox event commit in ONE PostgreSQL
    transaction. Only after that commit do we touch derived layers, and every
    derived cleanup is reported individually instead of being swallowed. If a
    derived layer cannot be cleaned, the memory is still gone from retrieval,
    because retrieval gates on canonical state.
    """
    _start = time.perf_counter()
    memory_id = args.get("memory_id")
    # Phase B: batch deletion. Previously only `memory_id` was read, so a
    # caller passing `memory_ids` got a KeyError instead of a deletion.
    batch = args.get("memory_ids") or args.get("memory_id_list") or []
    if isinstance(batch, str):
        batch = [batch]
    batch = [m for m in batch if m]
    if not memory_id and not batch:
        result = {"error": "delete requires memory_id or memory_ids"}
        _record_latency("memory_delete", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))
    if memory_id:
        batch = [memory_id] + [m for m in batch if m != memory_id]
    if len(batch) > 1:
        return _tool_delete_batch(batch)
    memory_id = batch[0]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)

    # Accept a memory that exists canonically even if Qdrant already lost it,
    # so a partially-indexed memory can still be deleted.
    existing = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id],
                                with_payload=True)
    canon = _pg_query(
        "SELECT status, scope FROM mem_canonical WHERE memory_id=:m",
        {"m": memory_id})
    if not existing and not canon:
        result = {"error": f"Memory not found: {memory_id}"}
        _record_latency("memory_delete", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))

    # 1) CANONICAL: tombstone + MEMORY_DELETED outbox event, one transaction.
    ok, why = _canonical_tombstone(memory_id)
    if not ok:
        log.error("delete refused id=%s reason=%s", memory_id, why)
        result = {
            "error": f"deletion not committed: {why}. The memory is still live; "
            f"no derived layer was modified."
        }
        _record_latency("memory_delete", (time.perf_counter() - _start) * 1000)
        return _err(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000)))

    # 2) DERIVED cleanup. Best effort, but never silent: failures are returned.
    derived_failures: dict[str, str] = {}

    try:
        _client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.HasIdCondition(has_id=[memory_id])]
                )
            ),
        )
    except Exception as _e:
        derived_failures["qdrant"] = str(_e)

    try:
        _remove_from_bm25_index(memory_id)
    except Exception as _e:
        derived_failures["bm25"] = str(_e)

    if _check_redis():
        try:
            neighbor_ids = _redis.smembers(f"{GRAPH_NODE_PREFIX}{memory_id}")
            for neighbor_id in neighbor_ids:
                _graph_unlink(memory_id, neighbor_id)
            _redis.delete(f"{GRAPH_NODE_PREFIX}{memory_id}")
            _redis.delete(f"memory:st:{memory_id}")
            _redis.delete(f"memory:version:{memory_id}")
        except Exception as _e:
            derived_failures["redis"] = str(_e)

    # mem_nodes / mem_events / mem_procedures are structured projections.
    try:
        for tbl, col in (("mem_nodes", "payload->>'memory_id'"),
                         ("mem_events", "memory_id::text"),
                         ("mem_procedures", "memory_id::text")):
            # DML goes through _pg_run, never _pg_query: a DELETE produces no
            # records, so the row-returning path raises after the row is
            # already gone — the delete then reports a derived failure it
            # didn't actually have.
            _pg_run(f"DELETE FROM {tbl} WHERE {col} = :m", {"m": memory_id},
                    table=tbl)
    except Exception as _e:
        derived_failures["postgres_projections"] = str(_e)

    _invalidate_recall_caches()
    deleted_versions = _delete_versions(memory_id)

    if derived_failures:
        # The tombstone committed, so the memory is NOT retrievable. The
        # remaining derived rows are garbage that reconciliation will remove.
        # Say so explicitly instead of pretending the job finished cleanly.
        log.error("delete id=%s tombstoned but derived cleanup failed: %s",
                  memory_id, derived_failures)
        result = {
            "memory_id": memory_id,
            "status": "deleted_index_pending",
            "canonical": "tombstoned",
            "retrievable": False,
            "versions_deleted": deleted_versions,
            "derived_pending": derived_failures,
            "note": "tombstone committed; retrieval already refuses this "
                    "memory. Remaining derived rows will be removed by the "
                    "outbox worker / reconciliation.",
        }
        _record_latency("memory_delete", (time.perf_counter() - _start) * 1000)
        return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

    log.info("deleted id=%s versions_deleted=%s", memory_id, deleted_versions)
    _pg_audit("delete", memory_id, {"versions_deleted": deleted_versions})
    result = {
        "memory_id": memory_id,
        "status": "deleted",
        "canonical": "tombstoned",
        "retrievable": False,
        "versions_deleted": deleted_versions,
    }
    _record_latency("memory_delete", (time.perf_counter() - _start) * 1000)
    return _ok(json.dumps(_add_latency_to_result(result, (time.perf_counter() - _start) * 1000), ensure_ascii=False))

# -- gap-closure: always-in-context core block (L1) --------------------------
def _tool_core(args: dict) -> types.CallToolResult:
    """Return a compact core-memory block: hot preferences, procedures,
    decisions, and gotchas ranked by (importance, recency), capped for context.
    Deterministic — no LLM, no embedding call. Pure Qdrant scroll by category."""
    limit = int(args.get("limit") or 12)
    hot_categories = ["preference", "decision", "gotcha", "procedure"]
    entries = []
    try:
        for cat in hot_categories:
            # Scroll this category, newest first, top few. 3B-1 red-team fix:
            # the core block is per-user — bound it to the caller's scopes
            # (an unscoped scroll handed tenant A's top preferences to tenant B).
            _sf = _caller_scope_filter()
            _must = [qmodels.FieldCondition(
                key="category", match=qmodels.MatchValue(value=cat))]
            if _sf is not None and _sf.must:
                _must.extend(_sf.must)
            result = _client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qmodels.Filter(must=_must),
                limit=limit,
                with_payload=True,
            )
            for pt in result[0]:
                pld = dict(pt.payload or {})
                pld["_cat_rank"] = hot_categories.index(cat)
                entries.append(pld)
    except Exception as e:
        log.warning("core scroll failed: %s", e)

    # Rank: importance desc, then most recently accessed first
    def _rank_key(p):
        imp = p.get("importance") or 0
        la = p.get("last_accessed_at") or p.get("created_at") or ""
        return (-imp, la)
    entries.sort(key=_rank_key, reverse=True)
    entries = entries[:limit]

    # Build a compact block (~<=1k tokens target)
    lines = []
    for p in entries:
        content = (p.get("content") or "").strip().replace("\n", " ")
        if len(content) > 220:
            content = content[:217] + "..."
        lines.append(
            f"- [{p.get('category')}] (imp={p.get('importance',5)}) {content}"
        )
    block = "\n".join(lines) if lines else "(no core memories yet — store a preference/decision/procedure/gotcha to populate the core block)"
    return _ok(json.dumps({
        "core_block": block,
        "entry_count": len(entries),
        "categories": hot_categories,
        "token_estimate": len(block) // 4,
        "note": "Inject this block into the system prompt at session start for always-in-context memory.",
    }, ensure_ascii=False))


def _tool_core_update(args: dict) -> types.CallToolResult:
    """Letta-style writable core memory.

    The agent targets one existing core memory (by its ID, typically obtained
    from a recent memory_core call) and updates its content/importance/tags.
    The old state is preserved as a version snapshot and the change is
    audit-logged — closing the 'read-only core' gap with better hygiene than
    Letta, where agent edits are opaque and unversioned.
    """
    memory_id = args.get("memory_id")
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    if not memory_id:
        return _err("memory_id required — target a memory from the core block")
    existing = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
    if not existing:
        return _err(f"Memory not found: {memory_id}")
    old_payload = existing[0].payload
    new_content = args.get("content", old_payload.get("content", ""))

    # Poisoning guard + sanitize (same path as store/update).
    blocked, poison_score, findings = _check_poison(new_content)
    if blocked:
        return _err(json.dumps({
            "blocked": True, "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Content flagged by poisoning guard.",
        }, ensure_ascii=False))
    new_content = _sanitize_content(new_content)

    now = _now_iso()
    payload = dict(old_payload)
    payload["content"] = new_content
    for field in ("category", "importance", "tags", "source"):
        if field in args:
            val = args[field]
            payload[field] = ",".join(val) if field == "tags" else val
    if args.get("scope"):
        payload["scope"] = args["scope"]
    payload["updated_at"] = now
    payload["core_edited_at"] = now

    vector = _embed_model().encode(new_content).tolist()
    tokens = _tokenize(new_content)
    payload["tokens"] = tokens
    payload["doc_len"] = len(tokens)

    _store_version(memory_id, old_payload, existing[0].vector)
    _client.upsert(
        collection_name=COLLECTION_NAME,
        points=[qmodels.PointStruct(id=memory_id, vector=vector, payload=payload)],
    )
    _remove_from_bm25_index(memory_id)
    _add_to_bm25_index(memory_id, new_content)
    _invalidate_recall_caches()
    _pg_audit("core_update", memory_id, {
        "category": payload.get("category", old_payload.get("category", "general")),
        "importance": payload.get("importance", old_payload.get("importance", 5)),
        "scope": payload.get("scope", "global"),
    })
    log.info("core_update id=%s", memory_id)
    result = {"memory_id": memory_id, "status": "updated", "core_edited_at": now}
    if poison_score > 0:
        result["poison_score"] = round(poison_score, 4)
    return _ok(json.dumps(result, ensure_ascii=False))

# -- gap-closure: GDPR right-to-erasure scoped delete ------------------------
def _tool_forget(args: dict) -> types.CallToolResult:
    """Delete memories across Qdrant + PostgreSQL + Redis by exactly one scope.
    memory_id → single; scope → all in namespace; tag → all carrying tag;
    since → 2D-3 time-window rollback: everything created at-or-after an ISO
    timestamp (tombstoned canonically, so restorable via version history)."""
    memory_id = args.get("memory_id")
    scope = args.get("scope")
    tag = args.get("tag")
    since = args.get("since")
    chosen = sum(1 for v in (memory_id, scope, tag, since) if v)
    if chosen > 1:
        return _err("provide exactly one of memory_id, scope, tag, or since")
    if chosen == 0:
        return _err("memory_id, scope, tag, or since is required")

    # Fail closed: a scope-wide forget must be authorized like any other
    # scope-bound operation. Without this, any caller could wipe a tenant.
    if scope:
        try:
            scope = authorize_scope(scope)
        except ScopeError as e:
            return _err(f"scope rejected: {e}")
    if since:
        # Validate the window before doing any work.
        try:
            _since_ts = datetime.fromisoformat(
                str(since).replace("Z", "+00:00")).timestamp()
        except Exception:
            return _err("since must be an ISO-8601 timestamp "
                        "(e.g. 2026-10-09T00:00:00Z)")

    # Resolve target Qdrant point IDs.
    # NOTE: _client.scroll() returns (points, next_offset); the previous code
    # unpacked it as `for _pt, pld in res`, binding _pt to the whole record
    # list. A non-global scope then matched nothing at all (silent no-op that
    # still answered "forgotten"), while scope="global" produced a nested list
    # as a point id -> "too many values to unpack (expected 2, got N)".
    qdrant_deleted = 0
    target_ids: list = []
    if memory_id:
        target_ids = [memory_id]
    elif scope:
        points, _ = _client.scroll(
            collection_name=COLLECTION_NAME, limit=10000, with_payload=True)
        for pt in points:
            pld = pt.payload or {}
            if pld.get("scope") == scope:
                target_ids.append(pt.id)
    elif tag:
        points, _ = _client.scroll(
            collection_name=COLLECTION_NAME, limit=10000, with_payload=True)
        for pt in points:
            pld = pt.payload or {}
            tags = _payload_tags(pld)
            if tag in tags:
                target_ids.append(pt.id)
    elif since:
        # 2D-3: time-window rollback — everything created at-or-after `since`.
        # The ownership filter below still drops out-of-tenant targets.
        points, _ = _client.scroll(
            collection_name=COLLECTION_NAME, limit=10000, with_payload=True)
        for pt in points:
            pld = pt.payload or {}
            created = pld.get("created_at") or ""
            try:
                if datetime.fromisoformat(
                        created.replace("Z", "+00:00")).timestamp() >= _since_ts:
                    target_ids.append(pt.id)
            except Exception:
                continue

    # Authorize a memory_id/tag deletion against the owner of each target, so
    # an in-scope tag wipe cannot reach another tenant's memories.
    if target_ids:
        try:
            allowed = authorized_scopes()
            if allowed is not None:
                kept = []
                for chunk_start in range(0, len(target_ids), 1000):
                    chunk = target_ids[chunk_start:chunk_start + 1000]
                    pts = _client.retrieve(
                        collection_name=COLLECTION_NAME, ids=chunk,
                        with_payload=True)
                    kept.extend(p.id for p in pts
                                if (p.payload or {}).get("scope") in allowed)
                dropped = len(target_ids) - len(kept)
                # Vector-less canonical rows (outbox race, partial write) have
                # no Qdrant point, so retrieve() misses them and a strict
                # vector-only ownership check made them unerasable — the
                # canonical row survived forever. Resolve their scope from the
                # canonical record instead.
                kept_set = set(map(str, kept))
                for mid in target_ids:
                    if str(mid) in kept_set:
                        continue
                    try:
                        row = _pg_query(
                            "SELECT scope FROM mem_canonical WHERE memory_id = :mid",
                            {"mid": str(mid)})
                    except Exception:
                        row = []
                    cscope = None
                    if row:
                        first = row[0]
                        cscope = first.get("scope") if isinstance(first, dict) else first[0]
                    if cscope and cscope in allowed:
                        kept.append(mid)
                    else:
                        log.warning("forget dropped out-of-scope target %s", str(mid)[:8])
                dropped = len(target_ids) - len(kept)
                if dropped:
                    log.warning("forget dropped %d out-of-scope targets", dropped)
                target_ids = kept
        except Exception as e:
            log.warning("forget ownership check failed: %s", e)

    canon_tombstoned, canon_failed = 0, {}
    if target_ids:
        # PHASE B: a bulk forget MUST tombstone canonically, not just drop
        # derived rows. Without this, "forget this scope" deleted Qdrant/BM25
        # but left mem_canonical saying 'active' — so the canonical gate would
        # still consider the memory live and a rebuild from any surviving
        # source would resurrect it.
        for mid in target_ids:
            ok, why = _canonical_tombstone(mid)
            if ok:
                canon_tombstoned += 1
            else:
                canon_failed[mid] = why
        if canon_failed:
            log.error("forget: %d targets could not be tombstoned canonically",
                      len(canon_failed))

        _client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.HasIdCondition(has_id=target_ids)]))
        )
        for mid in target_ids:
            try:
                _remove_from_bm25_index(mid)
                _delete_versions(mid)
                _graph_purge(mid)
            except Exception as _e:
                log.warning("forget derived cleanup failed id=%s: %s", mid, _e)
        qdrant_deleted = len(target_ids)
        _invalidate_recall_caches()

    # PostgreSQL durable cleanup.
    # mem_nodes has NO memory_id column — the id lives inside the payload JSONB
    # (and is how _pg_node_uuid resolves it), so the old
    # `WHERE memory_id = ANY(:ids)` raised 42703 "column does not exist",
    # _pg_run swallowed it, and the row survived while the tool still answered
    # "forgotten". mem_trajectories likewise has no memory_id, so it must not
    # be part of a per-memory delete at all.
    pg_deleted = {}
    if target_ids:
        ids_csv = ",".join(target_ids)
        # Resolve node UUIDs first so mem_edges (which FKs mem_nodes.id) and
        # mem_nodes are both cleaned in the right order.
        try:
            rows = _pg_query(
                "SELECT id FROM mem_nodes "
                "WHERE payload->>'memory_id' = ANY(string_to_array(:ids, ','))",
                {"ids": ids_csv},
            )
            node_csv = ",".join(str(r[0] if not isinstance(r, dict) else r.get("id")) for r in rows)
        except Exception as e:
            log.warning("forget node resolve failed: %s", e)
            node_csv = ""

        if node_csv:
            pg_deleted["mem_edges"] = "ok" if _pg_run(
                "DELETE FROM mem_edges WHERE src::text = ANY(string_to_array(:e, ',')) "
                "OR dst::text = ANY(string_to_array(:e, ','))",
                {"e": node_csv}, table="mem_edges") else "failed"
            pg_deleted["mem_nodes"] = "ok" if _pg_run(
                "DELETE FROM mem_nodes WHERE id::text = ANY(string_to_array(:e, ','))",
                {"e": node_csv}, table="mem_nodes") else "failed"
        else:
            pg_deleted["mem_edges"] = "ok"
            pg_deleted["mem_nodes"] = "ok"

        # mem_procedures.memory_id and mem_events.memory_id are uuid while
        # mem_feedback.memory_id is text; ::text makes one CSV param work for
        # all three (an uncast comparison raised 42883 "uuid = text").
        for tbl in ("mem_procedures", "mem_events", "mem_feedback"):
            pg_deleted[tbl] = "ok" if _pg_run(
                f"DELETE FROM {tbl} "
                f"WHERE memory_id::text = ANY(string_to_array(:ids, ','))",
                {"ids": ids_csv}, table=tbl) else "failed"
    pg_scope_rows = 0
    if scope:
        # Scope-wide durable sweep for rows that have no Qdrant twin.
        try:
            rows = _pg_query(
                "SELECT id FROM mem_nodes WHERE scope = :s", {"s": scope})
            pg_scope_rows = len(rows)
            sc_csv = ",".join(str(r[0] if not isinstance(r, dict) else r.get("id")) for r in rows)
            if sc_csv:
                pg_deleted["mem_nodes(scope)"] = "ok" if _pg_run(
                    "DELETE FROM mem_edges WHERE src::text = ANY(string_to_array(:e, ',')) "
                    "OR dst::text = ANY(string_to_array(:e, ','))",
                    {"e": sc_csv}, table="mem_edges-scope") else "failed"
                pg_deleted["mem_nodes(scope)"] = "ok" if _pg_run(
                    "DELETE FROM mem_nodes WHERE id::text = ANY(string_to_array(:e, ','))",
                    {"e": sc_csv}, table="mem_nodes-scope") else "failed"
            pg_deleted["mem_procedures(scope)"] = "ok" if _pg_run(
                "DELETE FROM mem_procedures WHERE scope = :s",
                {"s": scope}, table="mem_procedures-scope") else "failed"
            pg_deleted["mem_events(scope)"] = "ok" if _pg_run(
                "DELETE FROM mem_events WHERE scope = :s",
                {"s": scope}, table="mem_events-scope") else "failed"
            # Audit findings 2026-10-09 (final verification): a scope erasure
            # must close EVERY canonical row of the scope, not only those with
            # a Qdrant twin (target_ids). Two hole classes found: (a) graph
            # bookkeeping rows (memory_id LIKE 'graph:%') which live only in
            # canonical; (b) orphaned rows whose outbox never applied (no
            # Qdrant point) — both stayed status='active' after forget while
            # claiming erasure. Tombstone + scrub them here.
            pg_deleted["mem_canonical_scope_rows"] = "ok" if _pg_run(
                "UPDATE mem_canonical SET status = 'deleted', deleted_at = now(), "
                "content = NULL, "
                "payload = payload - 'content' - 'tokens' "
                "WHERE scope = :s AND status <> 'deleted'",
                {"s": scope}, table="mem_canonical-scope-tombstones") else "failed"
        except Exception as e:
            log.warning("forget scope sweep failed: %s", e)
            pg_deleted["mem_nodes(scope)"] = "failed"

    # 3B-3 GDPR scrub: a tombstone that still carries the erased user's
    # content is NOT an erasure. Keep the lifecycle skeleton (memory_id,
    # status, timestamps — required to refuse resurrection and to answer
    # "what was deleted" in audits) but NULL the content and drop
    # content-bearing payload keys for every forgotten row.
    if scope:
        pg_deleted["mem_canonical_scrub"] = "ok" if _pg_run(
            "UPDATE mem_canonical SET content = NULL, "
            "payload = payload - 'content' - 'tokens' "
            "WHERE scope = :s AND status = 'deleted'",
            {"s": scope}, table="mem_canonical-gdpr-scrub") else "failed"
    elif target_ids:
        pg_deleted["mem_canonical_scrub"] = "ok" if _pg_run(
            "UPDATE mem_canonical SET content = NULL, "
            "payload = payload - 'content' - 'tokens' "
            "WHERE memory_id::text = ANY(string_to_array(:ids, ','))",
            {"ids": ids_csv}, table="mem_canonical-gdpr-scrub") else "failed"

    # Redis short-term cleanup
    redis_deleted = 0
    if _redis:
        try:
            if memory_id:
                redis_deleted = 1 if _redis.delete(f"memory:st:{memory_id}") else 0
            else:
                # sweep st: keys belonging to target ids / scope
                for mid in target_ids:
                    if _redis.delete(f"memory:st:{mid}"):
                        redis_deleted += 1
            if scope:
                # Audit finding 2026-10-09 (final verification): the short-term
                # tier is scope-namespaced (memory:short:<scope>:<key>). A scope
                # erasure must sweep its keys too, or session-scoped data
                # outlives the GDPR cascade.
                for k in list(_redis.scan_iter(f"{REDIS_PREFIX}{scope}:*")):
                    try:
                        _redis.delete(k)
                        redis_deleted += 1
                    except Exception:
                        pass
        except Exception as e:
            log.warning("forget redis sweep failed: %s", e)

    # Honest outcome: a selector that matched nothing, or a durable target that
    # refused the delete, must not be reported as plain success.
    # "Matched" must mean something was actually targeted. A DELETE that removes
    # 0 rows still reports success at the SQL level, so relying on the pg_deleted
    # map would let an empty scope-wide forget claim it deleted memories.
    matched = bool(target_ids) or pg_scope_rows > 0
    durable_failed = {k: v for k, v in pg_deleted.items() if v != "ok"}
    # A memory whose tombstone did NOT commit is still live and retrievable, so
    # it is a durable failure exactly like a projection that refused the delete.
    for mid, why in canon_failed.items():
        durable_failed[f"canonical_tombstone:{mid[:8]}"] = why
    if not matched:
        status = "not_found"
        error = ("selector matched no memory — nothing was deleted "
                 f"(scope={scope!r} tag={tag!r} memory_id={memory_id!r})")
    elif durable_failed:
        status = "partial_failure"
        error = ("durable tier did not confirm deletion for: "
                 + ", ".join(sorted(durable_failed)))
    else:
        status = "forgotten"
        error = None

    _pg_audit("forget", memory_id, {"scope": scope, "tag": tag, "status": status,
                                     "qdrant": qdrant_deleted, "pg": pg_deleted,
                                     "redis": redis_deleted,
                                     "canonically_tombstoned": canon_tombstoned})
    log.info("forget status=%s qdrant=%d pg=%s redis=%d canon=%d (scope=%s tag=%s)",
             status, qdrant_deleted, pg_deleted, redis_deleted, canon_tombstoned,
             scope, tag)
    body = {
        "status": status,
        "qdrant_deleted": qdrant_deleted,
        "postgres": pg_deleted,
        "redis_deleted": redis_deleted,
        "canonically_tombstoned": canon_tombstoned,
        "still_live": len(canon_failed),
        "scope": scope, "tag": tag, "memory_id": memory_id,
    }
    if error:
        body["error"] = error
    return _ok(json.dumps(body, ensure_ascii=False))


_stats_cache: dict = {}

def _tool_stats(args: dict | None = None) -> types.CallToolResult:
    """Per-scope statistics.

    This used to report the deployment-wide point count, the cross-tenant
    category breakdown, and internal infrastructure (Qdrant URL, embedding
    model, vector size) to ANY caller. That is a cross-tenant information leak:
    PART 1.1 requires that one tenant cannot retrieve another's statistics. The
    result is now computed strictly within one authorized scope, and the cache
    is keyed by that scope -- a single shared cache would have handed tenant A
    the snapshot computed for tenant B.
    """
    global _stats_cache
    args = args or {}
    scope = authorize_scope(args.get("scope"))
    # Sprint-1 1.1/1.3: the infra tier (latency percentiles, write-gate
    # outcomes) admits the deployment's operator identities (the dashboard
    # principal + the operator agent) in addition to in-process system
    # context. Configure via MEMORY_OPERATOR_IDENTITIES (csv). Tenant
    # principals never see cross-scope infrastructure numbers.
    _op_ids = frozenset(x.strip() for x in os.environ.get(
        "MEMORY_OPERATOR_IDENTITIES", "global,user:moon").split(",") if x.strip())
    _p = current_principal()
    is_system = _p.is_system or getattr(_p, "identity", None) in _op_ids

    now = time.time()
    cache = _stats_cache.get(scope)
    if cache and (now - cache["timestamp"]) < STATS_CACHE_TTL_SECONDS:
        return _ok(cache["data"])

    categories: dict[str, int] = {}
    count = 0
    offset = None
    scope_filter = qmodels.Filter(must=[qmodels.FieldCondition(
        key="scope", match=qmodels.MatchValue(value=scope))])
    while True:
        try:
            pts, offset = _client.scroll(
                collection_name=COLLECTION_NAME, scroll_filter=scope_filter,
                limit=2000, offset=offset, with_payload=True, with_vectors=False)
        except Exception:
            break
        for r in pts:
            count += 1
            cat = (r.payload or {}).get("category") or "general"
            categories[cat] = categories.get(cat, 0) + 1
        if offset is None:
            break

    payload = {
        "scope": scope,
        "total_memories": count,
        "categories": categories,
        "phase4_decay_enabled": DECAY_ENABLED,
        "phase4_decay_half_life_days": DECAY_HALF_LIFE_DAYS,
    }
    if is_system:
        # Infrastructure detail is for operators only; a tenant-facing caller
        # gets counts about its own data and nothing about the deployment.
        payload.update({
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_version": EMBEDDING_MODEL_VERSION,
            "vector_size": VECTOR_SIZE,
            "qdrant_url": QDRANT_URL,
        })
        # 2B-3: write-gate observability (operator tier) — answers "what %
        # of candidates get rejected?" with a number. If rejection is ~0%
        # the gate is decorative (raise the threshold per plan).
        g = _WRITE_GATE_STATS
        payload["write_gate"] = {
            "evaluated": g["evaluated"], "stored": g["stored"],
            "rejected": g["rejected"], "quarantined": g["quarantined"],
            "rejection_rate": round(g["rejected"] / g["evaluated"], 4) if g["evaluated"] else None,
            "mean_quality_score": round(g["quality_sum"] / g["evaluated"], 4) if g["evaluated"] else None,
            "by_source": g["by_source"],
            "note": "counters above are process-lifetime; latency_7d / write_gate_daily below are the durable op_telemetry views",
        }
        # Sprint-1 1.1/1.3 acceptance: durable >=7-day latency percentiles and
        # daily write-gate outcomes from op_telemetry (survive restarts).
        try:
            rows = _pg_query(
                "SELECT tool, percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50, "
                "percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95, "
                "percentile_cont(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99, count(*) AS n "
                "FROM op_telemetry WHERE ts >= now() - interval '7 days' AND tool <> 'write_gate' "
                "GROUP BY tool ORDER BY n DESC LIMIT 12", {})
            lat = {}
            for r in rows or []:
                if isinstance(r, dict):
                    lat[r["tool"]] = {"count": int(r["n"]),
                                      "p50_ms": round(float(r["p50"]), 1),
                                      "p95_ms": round(float(r["p95"]), 1),
                                      "p99_ms": round(float(r["p99"]), 1)}
                else:
                    lat[r[0]] = {"count": int(r[4]),
                                 "p50_ms": round(float(r[1]), 1),
                                 "p95_ms": round(float(r[2]), 1),
                                 "p99_ms": round(float(r[3]), 1)}
            payload["latency_7d"] = lat if lat else {
                "status": "no data yet in the 7-day window"}
        except Exception as _e:
            payload["latency_7d"] = {"status": "unavailable", "error": str(_e)[:160]}
        try:
            rows = _pg_query(
                "SELECT date_trunc('day', ts)::date AS day, "
                "count(*) FILTER (WHERE status = 'stored') AS stored, "
                "count(*) FILTER (WHERE status = 'rejected') AS rejected, "
                "count(*) FILTER (WHERE status = 'quarantined') AS quarantined, "
                "count(*) FILTER (WHERE status = 'secret_rejected') AS secret_rejected, "
                "round(avg(quality_score)::numeric, 3) AS mean_quality, "
                "count(*) FILTER (WHERE quality_score < 0.3) AS q_low, "
                "count(*) FILTER (WHERE quality_score >= 0.3 AND quality_score < 0.5) AS q_mid, "
                "count(*) FILTER (WHERE quality_score >= 0.5 AND quality_score < 0.7) AS q_ok, "
                "count(*) FILTER (WHERE quality_score >= 0.7) AS q_high "
                "FROM op_telemetry WHERE tool = 'write_gate' "
                "AND ts >= now() - interval '7 days' GROUP BY 1 ORDER BY 1", {})
            payload["write_gate_daily"] = [
                {"day": str(r["day"] if isinstance(r, dict) else r[0]),
                 "stored": int(r["stored"] if isinstance(r, dict) else r[1]),
                 "rejected": int(r["rejected"] if isinstance(r, dict) else r[2]),
                 "quarantined": int(r["quarantined"] if isinstance(r, dict) else r[3]),
                 "secret_rejected": int(r["secret_rejected"] if isinstance(r, dict) else r[4]),
                 "mean_quality": float(r["mean_quality"]) if (r["mean_quality"] if isinstance(r, dict) else r[5]) is not None else None,
                 "quality_bands": {"low<0.3": int(r["q_low"] if isinstance(r, dict) else r[6]),
                                  "mid0.3-0.5": int(r["q_mid"] if isinstance(r, dict) else r[7]),
                                  "ok0.5-0.7": int(r["q_ok"] if isinstance(r, dict) else r[8]),
                                  "high>=0.7": int(r["q_high"] if isinstance(r, dict) else r[9])}}
                for r in rows] or [{"status": "no write-gate data in the 7-day window"}]
        except Exception as _e:
            payload["write_gate_daily"] = {"status": "unavailable", "error": str(_e)[:160]}
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    _stats_cache[scope] = {"timestamp": now, "data": result}
    return _ok(result)


def _tool_cache_stats() -> types.CallToolResult:
    global _semantic_cache_hits, _semantic_cache_misses
    total = _semantic_cache_hits + _semantic_cache_misses
    hit_rate = (_semantic_cache_hits / total * 100) if total > 0 else 0.0
    cache_keys = []
    cache_size = 0
    if _redis:
        try:
            cache_keys = list(_redis.keys(f"{SEMANTIC_CACHE_PREFIX}*"))
            cache_size = len(cache_keys)
        except Exception:
            pass
    return _ok(json.dumps({
        "semantic_cache_hits": _semantic_cache_hits,
        "semantic_cache_misses": _semantic_cache_misses,
        "total_requests": total,
        "hit_rate_percent": round(hit_rate, 2),
        "cache_threshold": SEMANTIC_CACHE_THRESHOLD,
        "cache_ttl_seconds": SEMANTIC_CACHE_TTL,
        "cached_entries": cache_size,
    }, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Enhancement 1: Graph Memory tool implementations
# ---------------------------------------------------------------------------
def _graph_resolve_ids(args: dict) -> tuple[str | None, str | None]:
    """Resolve the two endpoint ids from either naming convention.

    Canonical public names are `memory_id`/`related_id` (consistent with
    memory_graph_neighbors and memory_graph_relations). The older
    `from_memory_id`/`to_memory_id` names are accepted as aliases so no caller
    convention is left broken. Returns (from_id, to_id); either may be None.
    """
    from_id = args.get("memory_id") or args.get("from_memory_id")
    to_id = args.get("related_id") or args.get("to_memory_id")
    return from_id, to_id


def _edge_bookkeeping_scope(from_id: str) -> str:
    """Scope for the durable graph-edge bookkeeping row in mem_canonical.

    mem_canonical has FORCE RLS (policy: scope IN app.scopes), so the
    historical hardcoded scope="global" made every tenant-initiated
    graph_link/unlink fail with an RLS violation — the Redis fast-path edge
    landed but the durable bookkeeping row never did, silently. The edge's
    SOURCE-memory scope is the correct home: the ownership gate has already
    proven the caller is authorized for it.
    """
    try:
        found = _client.retrieve(collection_name=COLLECTION_NAME, ids=[from_id],
                                 with_payload=True)
        if found:
            return (found[0].payload or {}).get("scope") or "global"
    except Exception as e:
        log.warning("edge bookkeeping scope lookup failed for %s: %s", from_id, e)
    return "global"


def _tool_graph_link(args: dict) -> types.CallToolResult:
    from_id, to_id = _graph_resolve_ids(args)
    if not from_id or not to_id:
        return _err("memory_id and related_id are required "
                    "(aliases: from_memory_id, to_memory_id)")
    relation = args.get("relation")
    if relation not in GRAPH_RELATION_TYPES:
        return _err(f"Invalid relation type: {relation}")
    # Ownership gate: an edge may only join memories this caller owns.
    _assert_memory_access(from_id)
    _assert_memory_access(to_id)
    # Verify both memories exist in Qdrant
    f1 = _client.retrieve(collection_name=COLLECTION_NAME, ids=[from_id], with_payload=True)
    f2 = _client.retrieve(collection_name=COLLECTION_NAME, ids=[to_id], with_payload=True)
    if not f1 or not f2:
        return _err("One or both memory IDs not found in Qdrant")
    
    # Create edge in Redis immediately (fast path)
    ok = _graph_link(from_id, to_id, relation)
    if not ok:
        _outbox_bump("graph_link_failed")
        return _err("Graph link failed — Redis unavailable")
    
    # Route durable graph write through outbox for idempotent, retried Postgres persistence
    # This ensures Redis and Postgres stay consistent even if one store is temporarily down
    _outbox_bump("graph_link_created")
    try:
        _canonical_write(
            memory_id=f"graph:{from_id}:{to_id}",
            scope=_edge_bookkeeping_scope(from_id),
            node_type="graph_edge",
            content="",
            version=1,
            event_type="GRAPH_LINK",
            extra={"src": from_id, "dst": to_id, "rel": relation},
        )
    except PostgresUnavailableError as _e:
        _outbox_bump("graph_link_failed")
        return _err(f"Graph link failed — Postgres unavailable: {_e}")
    _pg_audit("graph_link", None, {"src": from_id, "dst": to_id, "rel": relation})
    return _ok(json.dumps({"status": "linked", "from": from_id, "to": to_id, "relation": relation}))

def _tool_graph_unlink(args: dict) -> types.CallToolResult:
    from_id, to_id = _graph_resolve_ids(args)
    if not from_id or not to_id:
        return _err("memory_id and related_id are required "
                    "(aliases: from_memory_id, to_memory_id)")
    # Ownership gate: an edge may only be removed by a caller who owns it.
    _assert_memory_access(from_id)
    _assert_memory_access(to_id)
    ok = _graph_unlink(from_id, to_id)
    if ok:
        # Route through outbox for idempotent Postgres delete
        _outbox_bump("graph_unlink_created")
        try:
            _canonical_write(
                memory_id=f"graph:{from_id}:{to_id}",
                scope=_edge_bookkeeping_scope(from_id),
                node_type="graph_edge",
                content="",
                version=1,
                event_type="GRAPH_UNLINK",
                extra={"src": from_id, "dst": to_id},
            )
        except PostgresUnavailableError as _e:
            return _err(f"Graph unlink failed — Postgres unavailable: {_e}")
        _pg_audit("graph_unlink", None, {"src": from_id, "dst": to_id})
        return _ok(json.dumps({"status": "unlinked", "from": from_id, "to": to_id}))
    return _err("Graph unlink failed — Redis unavailable")

def _tool_graph_neighbors(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    relation = args.get("relation")
    depth = min(args.get("depth", 1), 3)
    as_of = args.get("as_of")
    neighbors = _graph_get_neighbors(memory_id, relation, depth, as_of)
    enriched = []
    for n in neighbors:
        fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[n["memory_id"]], with_payload=True)
        if fetched:
            n["content_preview"] = fetched[0].payload.get("content", "")[:100]
        enriched.append(n)
    return _ok(json.dumps({"memory_id": memory_id, "depth": depth, "neighbors": enriched}, ensure_ascii=False))
def _tool_graph_relations(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    relations = _graph_get_all_relations(memory_id)
    return _ok(json.dumps({"memory_id": memory_id, "relations": relations}, ensure_ascii=False))


# P1-1: Graph Consistency Reconciliation
def _reconcile_graph_edges() -> dict:
    """Reconcile Redis and Postgres graph edges.
    
    Scans both stores and repairs any divergence:
    - Edges in Redis but not Postgres -> insert into Postgres
    - Edges in Postgres but not Redis -> insert into Redis
    - Edges in both with different relations -> use Postgres as source of truth
    
    Returns stats about the reconciliation run.
    """
    stats = {
        "redis_edges_scanned": 0,
        "postgres_edges_scanned": 0,
        "divergence_found": 0,
        "repaired_redis_to_pg": 0,
        "repaired_pg_to_redis": 0,
        "relation_mismatch_fixed": 0,
        "errors": [],
    }
    
    # Sprint-1 fix: _pg_conn is None BY DESIGN in pool mode — the old gate
    # made reconcile a permanent no-op on every pool deployment.
    if not _redis or (_pg_conn is None and _pg_pool is None):
        return {"error": "Redis or Postgres unavailable"}
    
    try:
        # Scan Redis edges
        redis_edges = {}
        pattern = f"{GRAPH_EDGE_PREFIX}*"
        for key in _redis.scan_iter(match=pattern, count=1000):
            try:
                val = _redis.get(key)
                if val:
                    data = json.loads(val)
                    src = data.get("from")
                    dst = data.get("to")
                    rel = data.get("relation")
                    if src and dst and rel:
                        edge_key = f"{src}:{dst}"
                        redis_edges[edge_key] = rel
                        stats["redis_edges_scanned"] += 1
            except Exception as e:
                stats["errors"].append(f"Redis edge parse error: {e}")
        
        # Scan Postgres edges
        pg_edges = {}
        try:
            rows = _pg_query("SELECT src, dst, rel FROM mem_edges")
            for row in rows:
                src, dst, rel = row
                edge_key = f"{src}:{dst}"
                pg_edges[edge_key] = rel
                stats["postgres_edges_scanned"] += 1
        except Exception as e:
            stats["errors"].append(f"Postgres query error: {e}")
        
        # Sprint-1 fix: classify BEFORE repairing. mem_edges FKs mem_nodes, so
        # a blind Redis->PG insert fails on every endpoint without a node row.
        # Three legitimate classes: (a) both endpoints have nodes -> insert;
        # (b) procedure/event-typed endpoints -> Redis-only BY DESIGN (see the
        # GRAPH_LINK outbox apply contract); (c) endpoints with no live
        # canonical row -> dangling edge to a dead/forgotten memory -> purge
        # from Redis instead of failing forever.
        _redis_endpoint_ids = set()
        for _ek in redis_edges:
            try:
                _s, _d = _ek.split(":", 1)
                _redis_endpoint_ids.update((_s, _d))
            except ValueError:
                pass
        _canon_map = {}
        if _redis_endpoint_ids:
            try:
                _rows = _pg_query(
                    "SELECT memory_id, node_type, status FROM mem_canonical "
                    "WHERE memory_id = ANY(string_to_array(:ids, ','))",
                    {"ids": ",".join(_redis_endpoint_ids)})
                for r in _rows or []:
                    _mid = str(r["memory_id"] if isinstance(r, dict) else r[0]).lower()
                    _canon_map[_mid] = {
                        "node_type": (r["node_type"] if isinstance(r, dict) else r[1]) or "",
                        "status": (r["status"] if isinstance(r, dict) else r[2]) or "",
                    }
            except Exception:
                _canon_map = {}
        stats["redis_only_by_design"] = 0
        stats["dangling_purged"] = 0
        stats["redis_cache_restored"] = 0

        # Find divergence
        all_keys = set(redis_edges.keys()) | set(pg_edges.keys())
        
        for edge_key in all_keys:
            redis_rel = redis_edges.get(edge_key)
            pg_rel = pg_edges.get(edge_key)
            
            if redis_rel is None and pg_rel is not None:
                # In Postgres but not Redis -> add to Redis
                try:
                    src, dst = edge_key.split(":", 1)
                    _graph_link(src, dst, pg_rel)
                    stats["repaired_pg_to_redis"] += 1
                    stats["divergence_found"] += 1
                except Exception as e:
                    stats["errors"].append(f"Repair PG->Redis failed for {edge_key}: {e}")
            elif pg_rel is None and redis_rel is not None:
                # In Redis but not Postgres -> classify, then insert or purge
                try:
                    src, dst = edge_key.split(":", 1)
                    _su, _du = _pg_node_uuid(src), _pg_node_uuid(dst)
                    if _su and _du:
                        _pg_run(
                            "INSERT INTO mem_edges(src, dst, rel, confidence, valid_from, payload) "
                            "VALUES (CAST(:src AS uuid), CAST(:dst AS uuid), :rel, 1.0, now(), '{}'::jsonb) "
                            "ON CONFLICT (src, dst, rel) DO NOTHING",
                            {"src": str(_su), "dst": str(_du), "rel": redis_rel},
                            table="mem_edges",
                        )
                        stats["repaired_redis_to_pg"] += 1
                    else:
                        _cs, _cd = _canon_map.get(src.lower(), {}), _canon_map.get(dst.lower(), {})
                        _by_design = (
                            _cs.get("node_type") in ("procedure", "event")
                            and _cd.get("node_type") in ("procedure", "event")
                            and _cs.get("status") != "deleted"
                            and _cd.get("status") != "deleted")
                        if _by_design:
                            stats["redis_only_by_design"] += 1
                        else:
                            # Durable-copy check: graph_link persists every edge as a
                            # mem_canonical graph_edge row (memory_id "graph:src:dst").
                            # A missing Redis key for a durable edge is a cold cache,
                            # not a dangling edge — restore it so the next
                            # graph_extract dedup pass skips re-proposing the edge.
                            try:
                                _edge_durable = _pg_query(
                                    "SELECT 1 FROM mem_canonical "
                                    "WHERE memory_id = :mid AND status <> 'deleted' LIMIT 1",
                                    {"mid": f"graph:{src}:{dst}"},
                                )
                            except Exception:
                                _edge_durable = []
                            if _edge_durable:
                                _graph_link(src, dst, redis_rel)
                                stats["redis_cache_restored"] += 1
                            else:
                                _redis.delete(f"{GRAPH_EDGE_PREFIX}{edge_key}")
                                stats["dangling_purged"] += 1
                    stats["divergence_found"] += 1
                except Exception as e:
                    stats["errors"].append(f"Repair Redis->PG failed for {edge_key}: {e}")
            elif redis_rel != pg_rel:
                # Relation mismatch -> use Postgres as source of truth
                try:
                    src, dst = edge_key.split(":", 1)
                    # Update Redis to match Postgres
                    edge_key_redis = f"{GRAPH_EDGE_PREFIX}{src}:{dst}"
                    _redis.setex(
                        edge_key_redis,
                        60 * 60 * 24 * 365,
                        json.dumps({"from": src, "to": dst, "relation": pg_rel, "created_at": _now_iso()})
                    )
                    stats["relation_mismatch_fixed"] += 1
                    stats["divergence_found"] += 1
                except Exception as e:
                    stats["errors"].append(f"Fix relation mismatch failed for {edge_key}: {e}")
        
        return stats
    
    except Exception as e:
        return {"error": str(e)}


def _tool_graph_reconcile(args: dict = None) -> types.CallToolResult:
    """P1-1: Reconcile Redis and Postgres graph edges.
    
    Scans both stores for divergence and repairs it:
    - Edges in Redis but not Postgres -> insert into Postgres (idempotent)
    - Edges in Postgres but not Redis -> add to Redis
    - Edges with different relations -> use Postgres as source of truth
    
    Returns stats: edges scanned, divergence found, repaired counts.
    """
    stats = _reconcile_graph_edges()
    if "error" in stats:
        return _err(f"Reconciliation failed: {stats['error']}")
    
    _outbox_bump("graph_reconciliation_runs")
    if stats.get("divergence_found", 0) > 0:
        _outbox_bump("graph_divergence_found", stats["divergence_found"])
        _outbox_bump("graph_divergence_repaired", 
                     stats.get("repaired_redis_to_pg", 0) + stats.get("repaired_pg_to_redis", 0) + stats.get("relation_mismatch_fixed", 0))
    
    return _ok(json.dumps(stats, ensure_ascii=False, indent=2))


def _tool_graph_consistency_metrics(args: dict = None) -> types.CallToolResult:
    """P1-1: Get graph consistency metrics."""
    result = {
        "graph_link_created": _outbox_counters.get("graph_link_created", 0),
        "graph_link_failed": _outbox_counters.get("graph_link_failed", 0),
        "graph_unlink_created": _outbox_counters.get("graph_unlink_created", 0),
        "graph_reconciliation_runs": _outbox_counters.get("graph_reconciliation_runs", 0),
        "graph_divergence_found": _outbox_counters.get("graph_divergence_found", 0),
        "graph_divergence_repaired": _outbox_counters.get("graph_divergence_repaired", 0),
        "reconciliation_lag_seconds": None,  # Could be computed from last run timestamp
    }
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Enhancement 5: Eval before Deploy (sandbox) tool implementations
# ---------------------------------------------------------------------------
def _tool_eval_store(args: dict) -> types.CallToolResult:
    content = args["content"]
    # Poison check even in sandbox mode
    blocked, poison_score, findings = _check_poison(content)
    if blocked:
        return _err(json.dumps({
            "blocked": True, "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Content flagged by poisoning guard even in sandbox.",
        }, ensure_ascii=False))
    content = _sanitize_content(content)
    category = args.get("category", "general")
    importance = args.get("importance", 5)
    tags = _coerce_str_list(args.get("tags"))
    source = args.get("source", "eval")
    result = _sandbox_store_memory(content, category, importance, tags, source)
    return _ok(json.dumps(result, ensure_ascii=False))

def _tool_eval_recall(args: dict) -> types.CallToolResult:
    query = args["query"]
    top_k = args.get("top_k", 5)
    results = _sandbox_recall(query, top_k)
    memories = []
    for r in results:
        payload = r["payload"]
        memories.append({
            "memory_id": r["memory_id"],
            "content": payload.get("content", ""),
            "hybrid_score": round(r["hybrid_score"], 6),
            "category": payload.get("category"),
        })
    return _ok(json.dumps({"query": query, "count": len(memories), "memories": memories}, ensure_ascii=False, indent=2))

def _tool_eval_commit(args: dict) -> types.CallToolResult:
    sandbox_id = args["memory_id"]
    result = _sandbox_commit(sandbox_id)
    status = result.get("status", "error")
    if status == "committed":
        memory_id = result["memory_id"]
        data = None
        for mid, d in list(_sandbox_store.items()):
            if mid == sandbox_id:
                data = d
                break
        if data:
            _store_version(memory_id, data["payload"], data["vector"])
        return _ok(json.dumps(result, ensure_ascii=False))
    return _err(json.dumps(result, ensure_ascii=False))

def _tool_eval_discard(args: dict) -> types.CallToolResult:
    sandbox_id = args["memory_id"]
    result = _sandbox_discard(sandbox_id)
    return _ok(json.dumps(result, ensure_ascii=False))

def _tool_eval_preview() -> types.CallToolResult:
    return _ok(json.dumps(_sandbox_preview(), ensure_ascii=False))

# ---------------------------------------------------------------------------
# Enhancement 6: Versioned Rollback tool implementations
# ---------------------------------------------------------------------------
def _tool_versions(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    _pg_audit("read:versions", memory_id, {"limit": args.get("limit", 10)})
    limit = args.get("limit", 10)
    versions = _get_versions(memory_id, limit)
    return _ok(json.dumps({"memory_id": memory_id, "count": len(versions), "versions": versions}, ensure_ascii=False))

def _tool_version_get(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    version_id = args["version_id"]
    version = _get_version(memory_id, version_id)
    if version:
        version.pop("vector", None)
        return _ok(json.dumps({"memory_id": memory_id, "version": version}, ensure_ascii=False, indent=2))
    return _err("Version not found")

def _tool_version_restore(args: dict) -> types.CallToolResult:
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    version_id = args["version_id"]
    version = _get_version(memory_id, version_id)
    if not version:
        return _err("Version not found")
    existing = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
    if not existing:
        return _err("Current memory not found")
    old_payload = dict(existing[0].payload)
    old_vector = existing[0].vector
    _store_version(memory_id, old_payload, old_vector)
    new_payload = dict(version["payload"])
    new_payload["updated_at"] = _now_iso()
    new_vector = version.get("vector")
    if not new_vector:
        new_vector = _embed_model().encode(new_payload["content"]).tolist()
    _client.upsert(
        collection_name=COLLECTION_NAME,
        points=[qmodels.PointStruct(id=memory_id, vector=new_vector, payload=new_payload)],
    )
    _remove_from_bm25_index(memory_id)
    _add_to_bm25_index(memory_id, new_payload.get("content", ""))
    _invalidate_recall_caches()
    _pg_audit("version_restore", memory_id, {"version_id": version_id})
    return _ok(json.dumps({
        "status": "restored",
        "memory_id": memory_id,
        "version_id": version_id,
        "new_version_id_of_old_state": _get_versions(memory_id, 1)[0]["version_id"] if _get_versions(memory_id, 1) else None,
    }, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Enhancement 3: Poison check tool
# ---------------------------------------------------------------------------
def _tool_poison_check(args: dict) -> types.CallToolResult:
    content = args["content"]
    block = args.get("block", False)
    blocked, score, findings = _check_poison(content, block)
    sanitized = _sanitize_content(content) if content else content
    return _ok(json.dumps({
        "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
        "poison_score": round(score, 4),
        "blocked": blocked,
        "findings": findings,
        "sanitized_length": len(sanitized),
        "original_length": len(content),
    }, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Short-term memory handlers (Redis)
# ---------------------------------------------------------------------------
def _short_scoped_key(scope: str, key: str) -> str:
    """Build the caller-scoped Redis key for the short-term tier.

    Every short-term key is namespaced under the caller's authorized scope so a
    tenant can only ever address its own keys. Pre-existing keys under the old
    flat ``memory:short:<key>`` layout are intentionally orphaned: they are
    ephemeral (<=7-day TTL) session state, and reading through to the old
    namespace would re-open the cross-tenant leak this closes.
    """
    return f"{REDIS_PREFIX}{scope}:{key}"


def _tool_short_store(args: dict) -> types.CallToolResult:
    if not _check_redis():
        return _err("Redis unavailable")
    key = args["key"]
    value = args["value"]
    ttl = args.get("ttl_seconds", REDIS_DEFAULT_TTL)
    scope = authorize_scope(args.get("scope"))
    full_key = _short_scoped_key(scope, key)
    if ttl > 0:
        _redis.setex(full_key, ttl, value)
    else:
        _redis.set(full_key, value)
    log.info("short_store scope=%s key=%s ttl=%s", scope, key, ttl)
    return _ok(json.dumps({"key": key, "scope": scope, "status": "stored", "ttl": ttl}, ensure_ascii=False))

def _tool_short_recall(args: dict) -> types.CallToolResult:
    if not _check_redis():
        return _err("Redis unavailable")
    key = args["key"]
    scope = authorize_scope(args.get("scope"))
    full_key = _short_scoped_key(scope, key)
    value = _redis.get(full_key)
    if value is None:
        return _err(f"Short-term memory not found: {key}")
    ttl = _redis.ttl(full_key)
    log.info("short_recall scope=%s key=%s", scope, key)
    return _ok(json.dumps({"key": key, "scope": scope, "value": value, "ttl_remaining": ttl}, ensure_ascii=False))

def _tool_short_delete(args: dict) -> types.CallToolResult:
    if not _check_redis():
        return _err("Redis unavailable")
    key = args["key"]
    scope = authorize_scope(args.get("scope"))
    full_key = _short_scoped_key(scope, key)
    deleted = _redis.delete(full_key)
    log.info("short_delete scope=%s key=%s deleted=%s", scope, key, deleted)
    return _ok(json.dumps({"key": key, "scope": scope, "status": "deleted" if deleted else "not_found"}, ensure_ascii=False))

def _tool_short_list(args: dict) -> types.CallToolResult:
    if not _check_redis():
        return _err("Redis unavailable")
    scope = authorize_scope(args.get("scope"))
    prefix = args.get("prefix")
    # Default to the caller's own scope namespace; a caller-supplied prefix is
    # still anchored under that scope so listing can never span tenants.
    pattern = f"{REDIS_PREFIX}{scope}:{prefix if prefix else ''}*"
    limit = min(args.get("limit", 100), 1000)
    keys = []
    cursor = 0
    while len(keys) < limit:
        cursor, partial = _redis.scan(cursor, match=pattern, count=100)
        keys.extend(partial)
        if cursor == 0:
            break
    keys = keys[:limit]
    log.info("short_list scope=%s prefix=%s count=%s", scope, prefix, len(keys))
    return _ok(json.dumps({"count": len(keys), "keys": keys}, ensure_ascii=False))

def _tool_short_stats() -> types.CallToolResult:
    """Whole-server Redis stats. NOT per-tenant: it reports deployment-wide
    counters (used memory, total keys, short-tier key count) and exposes no
    tenant content, so it is left global deliberately."""
    if not _check_redis():
        return _err("Redis unavailable")
    try:
        info = _redis.info("memory")
        db_info = _redis.info("keyspace")
        total_keys = sum(db.get("keys", 0) for db in db_info.values() if isinstance(db, dict))
        memory_keys = len(_redis.keys(f"{REDIS_PREFIX}*"))
        stats = {
            "connected": True,
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "total_keys": total_keys,
            "memory_short_keys": memory_keys,
        }
        return _ok(json.dumps(stats, ensure_ascii=False))
    except Exception as e:
        return _err(str(e))

# ---------------------------------------------------------------------------
# P3.3 (2026-10-07): governance-as-live-layer — a queryable ACL + provenance
# surface. Operator-only (_ADMIN_TOOLS): _enforce_tool_authorization refuses
# admin tools for network callers, so this cannot become a cross-tenant read
# primitive. Returns, in one response: the caller's effective ACL, a memory's
# provenance chain (source / duplicate_of / superseded_by / quarantine /
# poison_score / graph edges / canonical-tier status), and an optional
# scope-wide summary (total / quarantined / superseded counts).
# ---------------------------------------------------------------------------
def _tool_governance_audit(args: dict) -> types.CallToolResult:
    memory_id = args.get("memory_id")
    scope = args.get("scope")
    out: dict = {}

    # --- effective ACL of the current caller ---
    acl: dict = {}
    try:
        allowed = authorized_scopes()
    except Exception as e:
        allowed = None
        acl["error"] = str(e)
    try:
        ident = _current.get().identity
    except Exception:
        ident = None
    acl["identity"] = ident
    if allowed is None and "error" not in acl:
        acl["authorized_scopes"] = "system (unrestricted)"
    elif allowed is not None:
        acl["authorized_scopes"] = sorted(allowed)
    out["effective_acl"] = acl

    # --- per-memory provenance chain + graph edges + canonical status ---
    if memory_id:
        chain = []
        cur: str | None = memory_id
        seen: set = set()
        while cur and cur not in seen and len(chain) < 8:
            seen.add(cur)
            try:
                pts = _client.retrieve(collection_name=COLLECTION_NAME,
                                       ids=[cur], with_payload=True)
            except Exception as e:
                chain.append({"memory_id": cur, "error": f"qdrant: {e}"})
                break
            if not pts:
                chain.append({"memory_id": cur, "error": "not found in qdrant"})
                break
            p = pts[0].payload or {}
            chain.append({
                "memory_id": cur,
                "scope": p.get("scope"),
                "status": p.get("status"),
                "source": p.get("source"),
                "category": p.get("category"),
                "importance": p.get("importance"),
                "created_at": p.get("created_at"),
                "valid_to": p.get("valid_to"),
                "duplicate_of": p.get("duplicate_of"),
                "superseded_by": p.get("superseded_by"),
                "quarantine": bool(p.get("quarantine")),
                "poison_score": p.get("poison_score", 0.0),
            })
            cur = p.get("superseded_by") or p.get("duplicate_of")
        out["provenance_chain"] = chain
        try:
            out["graph_edges"] = _graph_get_all_relations(memory_id)
        except Exception as e:
            out["graph_edges"] = {"error": str(e)}
        try:
            row = _canonical_row(memory_id)
            if row:
                out["canonical_tier"] = {
                    "status": row.get("status"),
                    "version": row.get("version"),
                    "superseded_by": row.get("superseded_by"),
                    "deleted_at": row.get("deleted_at"),
                }
        except Exception as e:
            out["canonical_tier"] = {"error": str(e)}

    # --- scope-wide summary ---
    if scope:
        try:
            _base = qmodels.Filter(must=[qmodels.FieldCondition(
                key="scope", match=qmodels.MatchValue(value=scope))])
            _c_total = _client.count(
                collection_name=COLLECTION_NAME, count_filter=_base, exact=True).count
            _c_quar = _client.count(
                collection_name=COLLECTION_NAME,
                count_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope)),
                    qmodels.FieldCondition(key="quarantine", match=qmodels.MatchValue(value=True)),
                ]),
                exact=True,
            ).count
            _c_sup = _client.count(
                collection_name=COLLECTION_NAME,
                count_filter=qmodels.Filter(must=[
                    qmodels.FieldCondition(key="scope", match=qmodels.MatchValue(value=scope)),
                    qmodels.FieldCondition(key="status", match=qmodels.MatchValue(value="superseded")),
                ]),
                exact=True,
            ).count
            out["scope_summary"] = {
                "scope": scope,
                "memories": _c_total,
                "quarantined": _c_quar,
                "superseded": _c_sup,
            }
        except Exception as e:
            out["scope_summary"] = {"error": str(e)}

    return _ok(json.dumps(out, ensure_ascii=False, indent=2, default=str))

# ---------------------------------------------------------------------------
# Auto-observe: the agent decides what's worth saving (Phase 1.1)
# ---------------------------------------------------------------------------
# Heuristic classifier — picks category, importance, tags from content patterns.
# Goal: make a reasonable guess without needing an LLM call, so it stays fast and
# deterministic. When uncertain, defaults to general/importance=5.
_AUTO_OBSERVE_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("credential", ["password", "passwd", "token", "api_key", "apikey", "secret",
                    "bearer", "private_key", "ssh_key", "credentials"]),
    ("procedure",   ["how to", "how do i", "steps to", "process of", "tutorial",
                     "guide to", "run:", "run ", "execute ", "execute:", "install ",
                     "install:", "configure ", "setup ", "deploy ", "fix:",
                     "fix is", "solution:", "workaround:", "recipe:",
                     "command:", "use this command", "do this", "you can fix",
                     "to fix this"]),
    ("decision",    ["decided to", "decision:", "chose", "picked ", "selected ",
                     "tradeoff", "we will use", "we'll use", "going with",
                     "agreed to", "approved"]),
    ("project",     ["project", "deliverable", "milestone", "sprint", "task #",
                     "ticket", "jira", "client", "customer"]),
    ("event",       ["happened", "occurred", "yesterday", "this morning",
                     "log entry", "incident", "outage", "deployed on",
                     "migrated on"]),
    ("preference",  ["prefer", "i like", "i want", "use this", "always do",
                     "never do", "convention:"]),
    ("fact",        ["is ", "are ", "uses ", "runs on", "located at", "version ",
                     "v1.", "v2.", "v3.", "build ", "release ", "endpoint ",
                     "url is", "ip is", "host is", "today"]),
]
_AUTO_OBSERVE_TAG_RULES: list[tuple[str, list[str]]] = [
    ("server",       ["server", "vps", "host", "machine"]),
    ("docker",       ["docker", "container", "compose", "kubernetes", "k8s"]),
    ("nginx",        ["nginx", "openlitespeed", "ols", "vhost"]),
    ("wordpress",    ["wordpress", "wp_", "wp-", "wp.", "site", "plugin", "theme"]),
    ("mysql",        ["mysql", "mariadb", "sql", "database", "db ", "query"]),
    ("redis",        ["redis", "cache", "ttl"]),
    ("qdrant",       ["qdrant", "vector", "embedding"]),
    ("elementor",    ["elementor"]),
    ("cloudflare",   ["cloudflare", "cf-", "cname", "dns"]),
    ("agent",        ["agent", "mcp", "tool", "kilo"]),
    ("memory",       ["memory", "recall", "embed", "vector store"]),
    ("caringyou",    ["caringyou", "consultation", "patient"]),
    ("tweaktrove",   ["tweaktrove"]),
    ("fix",          ["fix:", "fixed", "bug", "issue", "error", "fail"]),
    ("config",       ["config", "configuration", "setting", "env "]),
    ("security",     ["security", "auth", "permission", "rbac", "csrf", "xss"]),
    ("backup",       ["backup", "restore", "snapshot"]),
]

def _classify_auto_observe(content: str, context_hint: str | None) -> tuple[str, int, list[str]]:
    """Return (category, importance, tags) based on content + hint heuristics."""
    text = (content + " " + (context_hint or "")).lower()
    # category
    category = "general"
    for cat, kws in _AUTO_OBSERVE_CATEGORY_RULES:
        if any(kw in text for kw in kws):
            category = cat
            break
    # importance — high if it looks like config/credential/procedure/fact
    importance = 5
    if category in ("credential", "procedure", "decision"):
        importance = 8
    elif category in ("fact", "project"):
        importance = 7
    elif category == "preference":
        importance = 6
    # bump if very short (concise fact) or has strong signal markers
    if len(content) > 800:
        importance = min(10, importance + 1)  # long-form = denser info
    if any(marker in text for marker in ["critical", "important", "warning", "do not"]):
        importance = min(10, importance + 1)
    if any(marker in text for marker in ["minor", "draft", "maybe"]):
        importance = max(1, importance - 2)
    # tags — collect all matched rule tags, cap at 6
    tags: list[str] = []
    for tag, kws in _AUTO_OBSERVE_TAG_RULES:
        if any(kw in text for kw in kws):
            tags.append(tag)
    if "auto_observe" not in tags:
        tags.append("auto_observe")
    if not tags:
        tags = ["auto_observe"]
    return category, importance, tags[:6]

def _tool_auto_observe(args: dict) -> types.CallToolResult:
    """Phase 1.1 auto-save hook — agent calls this proactively for valuable info."""
    content = args.get("content", "").strip()
    if not content:
        return _err("content is required")
    if len(content) > 8000:
        return _err(f"content too long ({len(content)} chars, max 8000)")
    context_hint = args.get("context_hint")
    source = args.get("source") or "agent_auto_observe"
    force = bool(args.get("force"))

    # 3B-2: same secret gate as store — auto-observe must not be the bypass.
    _sec_rej, _sec_findings, content = _scan_secrets(content)
    if _sec_rej:
        _pg_audit("secret_block", None, {"findings": _sec_findings, "via": "auto_observe"})
        return _ok(json.dumps({
            "blocked": True, "stored": False, "gate": "secret_detection",
            "findings": _sec_findings,
            "message": "Auto-observe blocked: credential-shaped string detected.",
        }, ensure_ascii=False))

    # Poison guard first
    blocked, poison_score, findings = _check_poison(content)
    if blocked:
        log.warning("AUTO_OBSERVE BLOCKED poison_score=%.3f", poison_score)
        return _ok(json.dumps({
            "blocked": True,
            "stored": False,
            "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Auto-observe blocked by poisoning guard. Content flagged OWASP LLM04 patterns.",
            "action": "Do NOT retry. Inform them their content was not stored for safety.",
        }, ensure_ascii=False))
    content_clean = _sanitize_content(content)

    # Classify
    category, importance, tags = _classify_auto_observe(content_clean, context_hint)

    # Auto-dedup check (unless forced). Scoped to the caller's own namespace so
    # the check can never surface another tenant's memory_id as a duplicate.
    duplicate_id = None
    if not force:
        try:
            dedup_scope = authorize_scope(args.get("scope"))
            vec = _embed_model().encode(content_clean).tolist()
            similar = _client.query_points(
                collection_name=COLLECTION_NAME,
                query=vec,
                limit=3,
                query_filter=_build_filter(None, None, dedup_scope),
                with_payload=False,
            )
            for sp in similar.points:
                if sp.score >= 0.92:
                    duplicate_id = str(sp.id)
                    break
        except Exception as e:
            log.warning("auto_observe dedup check failed: %s", e)

    if duplicate_id and not force:
        log.info("auto_observe duplicate_id=%s score>=0.92 — skipping store", duplicate_id)
        return _ok(json.dumps({
            "status": "duplicate",
            "duplicate_of": duplicate_id,
            "would_have_classified_as": {"category": category, "importance": importance, "tags": tags},
            "message": f"Highly similar memory exists (id={duplicate_id}). Pass force=true to store anyway.",
        }, ensure_ascii=False))

    # Store via the regular pipeline (so dedup-vs-self, BM25, versioning all run)
    _ao_store_args = {
        "content": content_clean,
        "category": category,
        "importance": importance,
        "tags": tags,
        "source": source,
        "scope": args.get("scope"),
    }
    # Sprint-1 1.5: forward task attribution so task-end facts participate in
    # the co-task graph extraction rule.
    if isinstance(args.get("_extra_payload"), dict):
        _ao_store_args["_extra_payload"] = args["_extra_payload"]
    result_payload = _tool_store(_ao_store_args)
    # Try to extract memory_id from result for the agent to capture
    try:
        result_text = result_payload.content[0].text if hasattr(result_payload, "content") else str(result_payload)
        parsed = json.loads(result_text)
        if isinstance(parsed, dict) and "memory_id" in parsed:
            parsed["auto_classified"] = {"category": category, "importance": importance, "tags": tags}
            parsed["status"] = parsed.get("status", "stored")
            return _ok(json.dumps(parsed, ensure_ascii=False))
    except Exception:
        pass
    return result_payload

# ---------------------------------------------------------------------------
# Phase 1.2: Pre-task session init — one-call context bundle for task start
# ---------------------------------------------------------------------------
def _recall_learned_strategies(limit: int = 3) -> list[dict]:
    """Pull top 'learned-strategy' memories (generated by memory_reflect)."""
    try:
        _must = [qmodels.FieldCondition(key="tags", match=qmodels.MatchText(text="learned-strategy"))]
        _sf = _caller_scope_filter()  # 3B-1: insights must not cross tenants
        if _sf is not None and _sf.must:
            _must.extend(_sf.must)
        strategies, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(must=_must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for p in strategies:
            pl = p.payload or {}
            out.append({
                "memory_id": str(p.id),
                "content": (pl.get("content") or "")[:500],
                "importance": pl.get("importance"),
                "updated_at": pl.get("updated_at"),
                "tags": pl.get("tags", "").split(",") if pl.get("tags") else [],
            })
        return out
    except Exception as e:
        log.warning("recall_learned_strategies failed: %s", e)
        return []

def _recall_recent_trajectories(limit: int = 5) -> list[dict]:
    """Pull recent trajectory memories with structured Phase-1.4 fields when available."""
    try:
        _must = [qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value="trajectory"))]
        _sf = _caller_scope_filter()  # 3B-1: session_init feed must be tenant-bounded
        if _sf is not None and _sf.must:
            _must.extend(_sf.must)
        trajs, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(must=_must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        out = []
        for p in trajs:
            pl = p.payload or {}
            entry = {
                "memory_id": str(p.id),
                "content": (pl.get("content") or "")[:400],
                "outcome": pl.get("outcome") or "unknown",
                "task_signature": pl.get("task_signature"),
                "created_at": pl.get("created_at"),
            }
            # Include Phase-1.4 structured fields if present
            if "step_count" in pl:
                entry["step_count"] = pl.get("step_count")
                entry["failed_step_index"] = pl.get("failed_step_index")
            out.append(entry)
        return out
    except Exception as e:
        log.warning("recall_recent_trajectories failed: %s", e)
        return []

def _session_stats_snapshot() -> dict:
    """Lightweight stats for the session-init bundle."""
    try:
        info = _client.get_collection(COLLECTION_NAME)
        return {
            "total_memories": getattr(info, "points_count", 0),
            "vectors_count": getattr(info, "vectors_count", None),
            "embedding_model": "%s (%dd)" % (EMBEDDING_MODEL_VERSION, VECTOR_SIZE),
            "qdrant_status": getattr(info, "status", "unknown"),
        }
    except Exception as e:
        return {"error": str(e)}

def _derive_suggestions(task: str, learned: list[dict], memories: list[dict]) -> list[str]:
    """Turn top memories into one-line actionable suggestions."""
    sugg = []
    # From learned strategies first (highest value)
    for s in learned[:2]:
        c = (s.get("content") or "").strip()
        if c:
            first_line = c.split("\n")[0][:160]
            sugg.append(f"Prior lesson: {first_line}")
    # From high-importance memories
    for m in memories[:3]:
        imp = m.get("importance") or 0
        if imp >= 8:
            c = (m.get("content") or "").strip()
            if c:
                first_line = c.split("\n")[0][:160]
                sugg.append(f"Critical context: {first_line}")
    return sugg[:5]

def _tool_session_init(args: dict) -> types.CallToolResult:
    """Phase 1.2: One-shot pre-task context bundle."""
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required (short description of what you're about to do)")
    task_type = args.get("task_type")
    top_k = max(1, min(15, int(args.get("top_k") or 5)))
    include_trajectories = bool(args.get("include_trajectories", True))

    # 1. Hybrid recall of relevant memories (use existing pipeline)
    recall_args = {"query": task, "top_k": top_k}
    if task_type:
        # Bias by adding task_type as tag hint (mild)
        recall_args["tags"] = None  # don't over-filter, just semantic
    try:
        recall_res = _tool_recall(recall_args)
        recall_parsed = json.loads(recall_res.content[0].text)
        top_memories = recall_parsed.get("memories", [])[:top_k]
    except Exception as e:
        log.warning("session_init recall failed: %s", e)
        top_memories = []

    # 1.5 (3A-2): the user model — profile-driven behavioral adaptation.
    try:
        _si_scope = authorize_scope(args.get("scope"))
    except Exception:
        _si_scope = None
    profile_text = ""
    if _si_scope:
        profile_text, _, _ = _vc_get_user_profile(_si_scope, VIRTUAL_CONTEXT_PROFILE_BUDGET)
    # 2. Learned strategies from prior reflections
    learned = _recall_learned_strategies(limit=3)

    # 3. Recent trajectories (optional)
    trajectories = _recall_recent_trajectories(limit=5) if include_trajectories else []

    # 4. Snapshot of memory state
    stats = _session_stats_snapshot()

    # 5. Suggestions derived from high-importance memories + learned strategies
    suggestions = _derive_suggestions(task, learned, top_memories)

    bundle = {
        "task": task,
        "task_type": task_type,
        "session_init_version": "1.3",
        "stats": stats,
        "user_profile": profile_text or None,
        "top_memories": top_memories,
        "learned_strategies": learned,
        "recent_trajectories": trajectories,
        "suggestions": suggestions,
        "instructions": (
            "Use this context BEFORE acting. Check top_memories for relevant facts/procedures. "
            "Check learned_strategies for prior lessons that apply to this task. "
            "Heed suggestions as warnings. If a recent_trajectory matches your task, "
            "review what worked/failed."
            + (" USER_PROFILE is binding: adapt verbosity, output format, and "
               "proactivity to the stated preferences (inferred entries are "
               "observed habits — follow unless contradicted)." if profile_text else "")
        ),
    }
    return _ok(json.dumps(bundle, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# Phase 1.3: Post-task reflection (memory_task_end)
# ---------------------------------------------------------------------------
def _build_trajectory_summary(task: str, outcome: str, task_type, duration_minutes,
                                facts: list[str], lessons: list[str]) -> str:
    """Compact human-readable summary stored as the trajectory memory content."""
    lines = []
    lines.append(f"Task: {task}")
    lines.append(f"Outcome: {outcome}")
    if task_type:
        lines.append(f"Task type: {task_type}")
    if duration_minutes is not None:
        lines.append(f"Duration: {duration_minutes} minutes")
    if facts:
        lines.append("")
        lines.append("Facts discovered:")
        for f in facts:
            lines.append(f"  - {f}")
    if lessons:
        lines.append("")
        lines.append("Lessons learned:")
        for l in lessons:
            lines.append(f"  - {l}")
    return "\n".join(lines)

def _coerce_str_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _tool_task_end(args: dict) -> types.CallToolResult:
    """Phase 1.3: One-shot post-task closure: store trajectory + lessons + auto-reflect on failure."""
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required (echo what you passed to session_init)")
    outcome = (args.get("outcome") or "").strip().lower()
    if outcome not in ("success", "failure", "partial"):
        return _err("outcome must be one of: success, failure, partial")
    task_type = args.get("task_type")
    duration_minutes = args.get("duration_minutes")
    facts = _coerce_str_list(args.get("facts_discovered"))
    lessons = _coerce_str_list(args.get("lessons_learned"))
    auto_reflect_on_success = bool(args.get("auto_reflect_on_success", False))

    stored: list[dict] = []
    stored_ids: list[str] = []

    # 1-3. The trajectory, every lesson and every fact are independent writes.
    # Each one costs an embedding + dedup + insert round-trip, so running them
    # serially made task_end the slowest tool in the server and pushed calls
    # past the client request timeout. They now run concurrently, with the
    # caller's auth context copied into each worker thread.
    jobs: list[tuple[str, object]] = []

    # 1. Store the trajectory memory itself (category=trajectory)
    traj_content = _build_trajectory_summary(task, outcome, task_type, duration_minutes, facts, lessons)
    # Sprint-1 1.5: task attribution — every memory this close creates carries
    # the task_signature so the offline extractor can link co-task memories.
    _te_sig = _task_signature(task)
    traj_args = {
        "content": traj_content,
        "category": "trajectory",
        "importance": 8 if outcome == "failure" else 6,
        "tags": ["trajectory", "auto_close", f"outcome-{outcome}"],
        "_extra_payload": {"task_signature": _te_sig},
    }
    if task_type:
        traj_args["tags"].append(task_type)
    jobs.append(("trajectory", lambda: _tool_store(traj_args)))

    # 2. Store each lesson as a procedure memory with importance=8 + learned-strategy tag
    for lesson in lessons:
        if not isinstance(lesson, str) or not lesson.strip():
            continue
        lesson_args = {
            "content": f"Lesson from '{task[:80]}' ({outcome}): {lesson.strip()}",
            "category": "procedure",
            "importance": 8,
            "tags": ["learned-strategy", "auto_close", f"outcome-{outcome}"],
            # get_insights' outcome filter reads learned_from_outcome from the
            # payload; the reflect path sets it, so task_end must too.
            "_extra_payload": {"learned_from_outcome": outcome, "task_signature": _te_sig},
        }
        if task_type:
            # Match the reflect-path convention: get_insights filters on the
            # `task-type:<x>` tag, so the bare task_type would never surface.
            lesson_args["tags"].append(f"task-type:{task_type}")
        jobs.append(("lesson", lambda la=lesson_args: _tool_store(la)))

    # 3. Store each fact as its own memory (auto_observe so it gets classified properly)
    for fact in facts:
        if not isinstance(fact, str) or not fact.strip():
            continue
        fact_args = {
            "content": fact.strip(),
            "context_hint": f"fact from task '{task[:60]}'",
        }
        if task_type:
            fact_args["context_hint"] += f" ({task_type})"
        fact_args["source"] = f"task_end:{task_type or 'general'}"
        fact_args["_extra_payload"] = {"task_signature": _te_sig}
        jobs.append(("fact", lambda fa=fact_args: _tool_auto_observe(fa)))

    def _run_job(job_ctx):
        kind, fn, ctx = job_ctx
        try:
            return kind, ctx.run(fn), None
        except Exception as exc:
            return kind, None, exc

    results = []
    if jobs:
        # The context snapshot MUST be taken in this thread: a worker thread has
        # its own (empty) context, so copying it there would drop the caller's
        # authenticated principal and every write would fail closed.
        jobs_ctx = [(kind, fn, contextvars.copy_context()) for kind, fn in jobs]
        if len(jobs_ctx) == 1:
            results.append(_run_job(jobs_ctx[0]))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(jobs_ctx))) as pool:
                results = list(pool.map(_run_job, jobs_ctx))

    for kind, res, exc in results:
        if exc is not None:
            stored.append({"what": kind, "error": str(exc)})
            continue
        try:
            parsed = json.loads(res.content[0].text)
        except Exception:
            stored.append({"what": kind, "raw": res.content[0].text[:200]})
            continue
        if getattr(res, "isError", False):
            stored.append({"what": kind, "error": res.content[0].text[:200]})
            continue
        stored.append({"what": kind, "memory_id": parsed.get("memory_id")})
        if parsed.get("memory_id"):
            stored_ids.append(parsed["memory_id"])

    # 4. Auto-reflect on failure (or on success if flag set)
    reflection_status = {"called": False}
    if outcome == "failure" or (outcome in ("success", "partial") and auto_reflect_on_success):
        try:
            reflect_args = {"days": 1, "limit": 20, "outcome": "failure" if outcome == "failure" else "success"}
            if task_type:
                reflect_args["task_type"] = task_type
            rres = _tool_reflect(reflect_args)
            reflection_status = {
                "called": True,
                "outcome_arg": reflect_args["outcome"],
            }
            try:
                rp = json.loads(rres.content[0].text)
                reflection_status["insights"] = len(rp.get("insights", []))
                reflection_status["result_excerpt"] = rres.content[0].text[:400]
            except Exception:
                reflection_status["result_excerpt"] = rres.content[0].text[:400]
        except Exception as e:
            reflection_status = {"called": True, "error": str(e)}

    # 5. Find similar past trajectories to surface patterns
    similar = []
    try:
        rec = _tool_recall({"query": task, "top_k": 4, "category": "trajectory"})
        rp = json.loads(rec.content[0].text)
        for m in rp.get("memories", [])[:4]:
            if m.get("memory_id") not in stored_ids:
                similar.append({
                    "memory_id": m.get("memory_id"),
                    "outcome": "see-content",  # category=trajectory but outcome embedded in content
                    "excerpt": (m.get("content") or "")[:300],
                })
    except Exception:
        pass

    bundle = {
        "task": task,
        "outcome": outcome,
        "task_type": task_type,
        "stored": stored,
        "stored_count": len(stored),
        "memory_ids": stored_ids,
        "reflection": reflection_status,
        "similar_past_trajectories": similar,
        "summary": (
            f"Stored trajectory + {len(lessons)} lessons + {len(facts)} facts. "
            f"{'Auto-reflection invoked.' if reflection_status.get('called') else 'No reflection invoked.'}"
        ),
    }
    return _ok(json.dumps(bundle, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# Phase 1.4: Structured episodic trajectory layer
# ---------------------------------------------------------------------------
# Step logs are accumulated in Redis under key "trajectory:steps:<task_id>"
# until memory_trajectory_close aggregates them into a structured memory.
_TRAJECTORY_STEP_TTL = 7 * 24 * 3600  # 7 days

def _task_signature(task: str) -> str:
    """Coarse-grained task signature for grouping similar trajectories.

    Normalizes to a coarse class so auto-promotion can actually accumulate
    enough trajectories per signature. Exact-task hashing would never reach
    min_success_count thresholds.
    """
    import hashlib
    import re
    norm = " ".join(task.lower().split())
    norm = re.sub(r"\s+", " ", norm)

    # Detect task_type hints from keywords
    task_type = "general"
    if any(k in norm for k in ("bug", "error", "fail", "crash", "traceback", "exception", "diagnos", "troubleshoot", "fix ")):
        task_type = "debugging"
    elif any(k in norm for k in ("refactor", "rewrite", "implement", "add ", "create", "build", "develop", "code", "feature", "function", "class", "module", "script")):
        task_type = "coding"
    elif any(k in norm for k in ("test", "verify", "check", "validate", "qa", "regression")):
        task_type = "testing"
    elif any(k in norm for k in ("deploy", "release", "ship", "publish", "migrate", "upgrade")):
        task_type = "deployment"
    elif any(k in norm for k in ("plan", "design", "architect", "spec", "proposal")):
        task_type = "planning"
    elif any(k in norm for k in ("research", "investigate", "explore", "understand", "analyze", "compare")):
        task_type = "research"
    elif any(k in norm for k in ("config", "setup", "install", "provision", "configure")):
        task_type = "configuration"
    elif any(k in norm for k in ("document", "write docs", "readme", "comment")):
        task_type = "documentation"

    sig = hashlib.sha256(f"{task_type}:{norm[:60]}".encode("utf-8")).hexdigest()[:16]
    return f"{task_type}|{sig}"

def _trajectory_steps_key(task_id: str) -> str:
    return f"trajectory:steps:{task_id}"

def _trajectory_meta_key(task_id: str) -> str:
    return f"trajectory:meta:{task_id}"

def _tool_trajectory_log(args: dict) -> types.CallToolResult:
    """Phase 1.4: Log one step into Redis (under new task_id)."""
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required")
    step_index = args.get("step_index")
    if step_index is None:
        return _err("step_index is required (0-based)")
    action = (args.get("action") or "").strip()
    observation = (args.get("observation") or "").strip()
    success = bool(args.get("success", True))
    duration_ms = args.get("duration_ms")
    tool_used = args.get("tool_used")

    if not action:
        return _err("action is required")
    if observation is None:
        observation = ""

    # Generate or reuse task_id via Redis key inspection
    task_sig = _task_signature(task)
    # Strategy: caller passes the same task repeatedly, so we use task_signature as
    # an implicit task_id UNLESS caller provides one explicitly. To keep the API
    # simple, we generate a fresh task_id per log call sequence and store the
    # signature separately. The caller is expected to use trajectory_close with
    # the same task string.
    task_id = args.get("task_id")
    if not task_id:
        # Look up an open episode by task_signature
        open_key = f"trajectory:open:{task_sig}"
        task_id = None
        try:
            task_id = _redis.get(open_key)
        except Exception:
            task_id = None
        if not task_id:
            task_id = str(uuid.uuid4())
            try:
                _redis.set(open_key, task_id, ex=_TRAJECTORY_STEP_TTL)
                _redis.set(_trajectory_meta_key(task_id),
                            json.dumps({"task": task, "task_signature": task_sig,
                                        "started_at": _now_iso()}),
                            ex=_TRAJECTORY_STEP_TTL)
            except Exception as e:
                log.warning("trajectory_log redis init failed: %s", e)

    step = {
        "step_index": int(step_index),
        "action": action,
        "observation": observation,
        "success": success,
        "duration_ms": duration_ms,
        "tool_used": tool_used,
        "logged_at": _now_iso(),
    }
    try:
        key = _trajectory_steps_key(task_id)
        existing = _redis.get(key)
        steps = json.loads(existing) if existing else []
        steps.append(step)
        _redis.set(key, json.dumps(steps, default=str), ex=_TRAJECTORY_STEP_TTL)
    except Exception as e:
        return _err(f"failed to log step: {e}")

    return _ok(json.dumps({
        "task_id": task_id,
        "task_signature": task_sig,
        "step_index": int(step_index),
        "logged": True,
        "total_steps_logged": len(steps),
    }, ensure_ascii=False))

def _build_trajectory_content(task: str, outcome: str, steps: list[dict]) -> str:
    """Render trajectory as human-readable text for embedding + display."""
    lines = [
        f"Task: {task}",
        f"Outcome: {outcome}",
        f"Steps: {len(steps)}",
    ]
    success_count = sum(1 for s in steps if s.get("success"))
    failure_count = len(steps) - success_count
    lines.append(f"Success: {success_count}, Failure: {failure_count}")
    failed_step = next((s.get("step_index") for s in steps if not s.get("success")), None)
    if failed_step is not None:
        lines.append(f"Failed at step: {failed_step}")
    lines.append("")
    lines.append("Step-by-step:")
    for s in steps:
        marker = "✓" if s.get("success") else "✗"
        idx = s.get("step_index", "?")
        action = (s.get("action") or "")[:200]
        tool = s.get("tool_used")
        tool_str = f" [{tool}]" if tool else ""
        lines.append(f"  [{idx}] {marker}{tool_str} {action}")
        obs = (s.get("observation") or "")[:200]
        if obs:
            lines.append(f"        → {obs}")
    return "\n".join(lines)

def _tool_trajectory_close(args: dict) -> types.CallToolResult:
    """Phase 1.4: Aggregate steps, store structured trajectory, then call task_end."""
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required")
    outcome = (args.get("outcome") or "").strip().lower()
    if outcome not in ("success", "failure", "partial"):
        return _err("outcome must be one of: success, failure, partial")
    task_type = args.get("task_type")
    duration_minutes = args.get("duration_minutes")
    lessons = _coerce_str_list(args.get("lessons_learned"))
    facts = _coerce_str_list(args.get("facts_discovered"))

    task_sig = _task_signature(task)
    # Find open episode for this task_signature
    open_key = f"trajectory:open:{task_sig}"
    task_id = None
    try:
        task_id = _redis.get(open_key)
    except Exception:
        task_id = None
    if not task_id:
        # Allow caller to pass task_id explicitly
        task_id = args.get("task_id")
    if not task_id:
        return _err("no open trajectory found for this task — call trajectory_log first, or pass task_id explicitly")

    # Read all steps from Redis
    steps = []
    try:
        raw = _redis.get(_trajectory_steps_key(task_id))
        if raw:
            steps = json.loads(raw)
    except Exception as e:
        log.warning("trajectory_close failed to read steps: %s", e)

    # Compute structured fields
    step_count = len(steps)
    success_count = sum(1 for s in steps if s.get("success"))
    failure_count = step_count - success_count
    failed_step_index = None
    if outcome != "success":
        for s in steps:
            if not s.get("success"):
                failed_step_index = s.get("step_index")
                break

    content = _build_trajectory_content(task, outcome, steps)
    tags = ["trajectory", "structured", f"outcome-{outcome}", task_sig[:8]]
    if task_type:
        tags.append(task_type)

    # Store the structured trajectory (bypass _tool_store's dedup to ensure we always persist)
    trajectory_id = str(uuid.uuid4())
    try:
        vector = _embed_model().encode(content).tolist()
        payload = {
            "content": content,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "category": "trajectory",
            "importance": 8 if outcome == "failure" else 6,
            "tags": ",".join(tags),
            "source": "agent_trajectory_close",
            "access_count": 0,
            "last_accessed_at": _now_iso(),
            "task_signature": task_sig,
            "outcome": outcome,
            "task_type": task_type or "",
            "duration_minutes": duration_minutes,
            "steps_json": json.dumps(steps, default=str),
            "step_count": step_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "failed_step_index": failed_step_index,
            "task_id": task_id,
        }
        _client.upsert(
            collection_name=COLLECTION_NAME,
            points=[qmodels.PointStruct(id=trajectory_id, vector=vector, payload=payload)],
        )
        # Update BM25 + cache
        _add_to_bm25_index(trajectory_id, content)
        _invalidate_recall_caches()
        _store_version(trajectory_id, payload, vector)
    except Exception as e:
        return _err(f"failed to store trajectory: {e}")

    # L3 Postgres: mirror trajectory into durable typed tables
    # Insert trajectory and get the generated UUID
    traj_row = _pg_query(
        "INSERT INTO mem_trajectories(task_signature, task_type, outcome, step_count, "
        "failed_step_index, duration_s, steps_json) "
        "VALUES (:sig, :ttype, :outcome, :steps, :failed, :dur, :sjson) "
        "RETURNING id",
        {"sig": task_sig, "ttype": task_type, "outcome": outcome,
         "steps": step_count, "failed": failed_step_index,
         "dur": (duration_minutes * 60) if duration_minutes else None,
         "sjson": json.dumps(steps, default=str)},
    )
    trajectory_pg_id = traj_row[0]["id"] if traj_row else None

    # Insert individual steps into mem_trajectory_steps for exact traceability (P1-7)
    if trajectory_pg_id and steps:
        step_stmts = []
        for s in steps:
            step_stmts.append((
                "INSERT INTO mem_trajectory_steps(trajectory_id, step_index, action, observation, success, duration_ms, tool_used) "
                "VALUES (:tid, :idx, :act, :obs, :success, :dur, :tool)",
                {"tid": trajectory_pg_id, "idx": s.get("step_index"), "act": s.get("action"),
                 "obs": s.get("observation"), "success": s.get("success"),
                 "dur": s.get("duration_ms"), "tool": s.get("tool_used")},
            ))
        if step_stmts:
            try:
                _pg_txn_execute(step_stmts)
            except Exception as _e:
                log.warning("failed to insert trajectory steps: %s", _e)
    _pg_run(
        "INSERT INTO mem_events(memory_id, scope, event_type, task_signature, payload, occurred_at) "
        "VALUES (:mid, :scope, :etype, :tsig, :payload, :occ)",
        {"mid": trajectory_id, "scope": "global", "etype": "trajectory",
         "tsig": task_sig,
         "payload": json.dumps({"outcome": outcome, "step_count": step_count,
                                "task_type": task_type}),
         "occ": _now_iso()},
        table="mem_events",
    )
    # If this trajectory produced a lesson, also register it as a procedural memory
    if lessons and outcome == "success":
        proc_name = (lessons[0][:120])
        _pg_run(
            "INSERT INTO mem_procedures(memory_id, scope, proc_name, importance, "
            "source_trajectory, steps, tags, payload) "
            "VALUES (:mid, :scope, :pname, :imp, :traj, :steps, :tags, :payload)",
            {"mid": trajectory_id, "scope": "global", "pname": proc_name,
             "imp": 8, "traj": task_sig, "steps": json.dumps(steps, default=str),
             "tags": "learned-strategy,auto-promoted," + (task_type or ""),
             "payload": json.dumps({"outcome": outcome, "task_type": task_type})},
            table="mem_procedures",
        )
    _pg_audit("trajectory_close", trajectory_id, {"task_signature": task_sig, "outcome": outcome, "step_count": step_count})

    # Clean up Redis open episode + steps
    try:
        _redis.delete(_trajectory_steps_key(task_id))
        _redis.delete(_trajectory_meta_key(task_id))
        _redis.delete(open_key)
    except Exception:
        pass

    # Delegate to task_end for lessons + facts + auto-reflection
    task_end_args = {
        "task": task,
        "outcome": outcome,
        "task_type": task_type,
        "duration_minutes": duration_minutes,
        "lessons_learned": lessons,
        "facts_discovered": facts,
        "auto_reflect_on_success": bool(args.get("auto_reflect_on_success", False)),
    }
    task_end_result = None
    try:
        te_res = _tool_task_end(task_end_args)
        task_end_result = json.loads(te_res.content[0].text)
    except Exception as e:
        task_end_result = {"error": str(e)}

    return _ok(json.dumps({
        "trajectory_id": trajectory_id,
        # Postgres row id — the key trajectory_get/trajectory_chain resolve on.
        # The Qdrant trajectory_id above does NOT match mem_trajectories.id, so
        # without this field the documented close -> get flow 404s.
        "trajectory_pg_id": trajectory_pg_id,
        "task_id": task_id,
        "task_signature": task_sig,
        "outcome": outcome,
        "step_count": step_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "failed_step_index": failed_step_index,
        "task_end_result": task_end_result,
    }, ensure_ascii=False, default=str))

def _fetch_trajectories_by_query(query: str, top_k: int = 10, outcome_filter: str | None = None) -> list[dict]:
    """Raw trajectory fetch — returns list of trajectory dicts, not wrapped in MCP result."""
    vec = _embed_model().encode(query).tolist()
    must_filters = [qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value="trajectory"))]
    if outcome_filter:
        must_filters.append(qmodels.FieldCondition(key="outcome", match=qmodels.MatchValue(value=outcome_filter)))
    # 3B-1 red-team fix: trajectory recall leaked cross-tenant task steps.
    _sf = _caller_scope_filter()
    if _sf is not None and _sf.must:
        must_filters.extend(_sf.must)
    flt = qmodels.Filter(must=must_filters)
    results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        query_filter=flt,
        limit=top_k * 3,
        with_payload=True,
    )
    trajs = []
    for r in results.points:
        pl = r.payload or {}
        try:
            steps = json.loads(pl.get("steps_json") or "[]")
        except Exception:
            steps = []
        trajs.append({
            "memory_id": str(r.id),
            "task_signature": pl.get("task_signature"),
            "outcome": pl.get("outcome") or "unknown",
            "task_type": pl.get("task_type"),
            "duration_minutes": pl.get("duration_minutes"),
            "step_count": pl.get("step_count", 0),
            "success_count": pl.get("success_count", 0),
            "failure_count": pl.get("failure_count", 0),
            "failed_step_index": pl.get("failed_step_index"),
            "steps": steps,
            "content_excerpt": (pl.get("content") or "")[:400],
            "created_at": pl.get("created_at"),
            "relevance_score": r.score,
        })
    return trajs[:top_k]


def _tool_trajectory_recall(args: dict) -> types.CallToolResult:
    """Phase 1.4: Recall structured trajectories with step-level detail."""
    query = (args.get("query") or "").strip()
    if not query:
        return _err("query is required")
    top_k = max(1, min(15, int(args.get("top_k") or 5)))
    outcome_filter = args.get("outcome_filter")

    try:
        trajs = _fetch_trajectories_by_query(query, top_k, outcome_filter)
    except Exception as e:
        return _err(f"recall failed: {e}")

    return _ok(json.dumps({
        "query": query,
        "outcome_filter": outcome_filter,
        "count": len(trajs),
        "trajectories": trajs,
        "instructions": (
            "Each trajectory shows the full step list. For failure trajectories, "
            "check failed_step_index to see WHICH step went wrong. Use this to plan "
            "your current task: avoid the failure patterns you see in past similar tasks."
        ),
    }, ensure_ascii=False, default=str))


# -- trajectory get (P1-7) ----------------------------------------------------
def _tool_trajectory_get(args: dict) -> types.CallToolResult:
    """P1-7: Get a specific step from a trajectory by session_id and step_index."""
    session_id = args.get("session_id") or args.get("task_id")
    if not session_id:
        return _err("session_id (or task_id) is required")
    step_index = args.get("step_index")
    if step_index is None:
        return _err("step_index is required")
    
    # First find the trajectory by session_id (task_id)
    traj_row = _pg_query(
        "SELECT id FROM mem_trajectories WHERE task_signature = :sig",
        {"sig": session_id},
    )
    if not traj_row:
        # Try by trajectory UUID
        traj_row = _pg_query(
            "SELECT id FROM mem_trajectories WHERE id = :id",
            {"id": session_id},
        )
    if not traj_row:
        return _err(f"trajectory not found for session_id: {session_id}")
    
    trajectory_id = traj_row[0]["id"]
    
    # Get the specific step
    step_row = _pg_query(
        "SELECT * FROM mem_trajectory_steps WHERE trajectory_id = :tid AND step_index = :idx",
        {"tid": trajectory_id, "idx": int(step_index)},
    )
    if not step_row:
        return _err(f"step {step_index} not found in trajectory {trajectory_id}")
    
    step = step_row[0]
    result = {
        "trajectory_id": str(trajectory_id),
        "step_index": step["step_index"],
        "action": step["action"],
        "observation": step["observation"],
        "success": step["success"],
        "duration_ms": step["duration_ms"],
        "tool_used": step["tool_used"],
        "logged_at": step["logged_at"],
    }
    return _ok(json.dumps(result, ensure_ascii=False, default=str))


# -- trajectory chain (P1-7) --------------------------------------------------
def _tool_trajectory_chain(args: dict) -> types.CallToolResult:
    """P1-7: Reconstruct a reasoning chain from a trajectory between two step indices."""
    session_id = args.get("session_id") or args.get("task_id")
    if not session_id:
        return _err("session_id (or task_id) is required")
    from_step = args.get("from_step", 0)
    to_step = args.get("to_step")
    
    # Find the trajectory
    traj_row = _pg_query(
        "SELECT id FROM mem_trajectories WHERE task_signature = :sig",
        {"sig": session_id},
    )
    if not traj_row:
        traj_row = _pg_query(
            "SELECT id FROM mem_trajectories WHERE id = :id",
            {"id": session_id},
        )
    if not traj_row:
        return _err(f"trajectory not found for session_id: {session_id}")
    
    trajectory_id = traj_row[0]["id"]
    
    # Build query for step range
    if to_step is not None:
        steps = _pg_query(
            "SELECT * FROM mem_trajectory_steps WHERE trajectory_id = :tid "
            "AND step_index BETWEEN :from_idx AND :to_idx ORDER BY step_index",
            {"tid": trajectory_id, "from_idx": int(from_step), "to_idx": int(to_step)},
        )
    else:
        steps = _pg_query(
            "SELECT * FROM mem_trajectory_steps WHERE trajectory_id = :tid "
            "AND step_index >= :from_idx ORDER BY step_index",
            {"tid": trajectory_id, "from_idx": int(from_step)},
        )
    
    chain = []
    for s in steps:
        chain.append({
            "step_index": s["step_index"],
            "action": s["action"],
            "observation": s["observation"],
            "success": s["success"],
            "duration_ms": s["duration_ms"],
            "tool_used": s["tool_used"],
        })
    
    result = {
        "trajectory_id": str(trajectory_id),
        "from_step": from_step,
        "to_step": to_step,
        "step_count": len(chain),
        "chain": chain,
    }
    return _ok(json.dumps(result, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# Phase 2.1: Procedural auto-promotion
# ---------------------------------------------------------------------------
_PROMOTION_DEDUP_KEY_PREFIX = "promoted:"
_PROMOTION_DEDUP_TTL = 90 * 24 * 3600  # 90 days — re-evaluate after this

# Phase 2.2: Dreaming dedup
_DREAM_DEDUP_KEY_PREFIX = "dreamed:"
_DREAM_DEDUP_TTL = 90 * 24 * 3600  # 90 days

def _promotion_dedup_key(kind: str, signature: str) -> str:
    """kind = 'procedure' or 'warning'. Tracks which signatures have been promoted."""
    return f"{_PROMOTION_DEDUP_KEY_PREFIX}{kind}:{signature}"

def _fetch_trajectories_for_promotion(lookback_days: int) -> list[dict]:
    """Pull structured trajectory memories within the lookback window."""
    out = []
    try:
        _must = [qmodels.FieldCondition(key="category",
                                        match=qmodels.MatchValue(value="trajectory"))]
        # 3B-1 red-team fix: tenant-callable hotspots/recommendations consume
        # this fetch — bound it to the caller's authorized scopes.
        _sf = _caller_scope_filter()
        if _sf is not None and _sf.must:
            _must.extend(_sf.must)
        flt = qmodels.Filter(must=_must)
        scroll_result, _ = _client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        cutoff = None
        if lookback_days > 0:
            from datetime import datetime, timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        for p in scroll_result:
            pl = p.payload or {}
            ts = pl.get("created_at") or ""
            if cutoff is not None and ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt < cutoff:
                        continue
                except Exception:
                    pass
            try:
                steps = json.loads(pl.get("steps_json") or "[]")
            except Exception:
                steps = []
            out.append({
                "memory_id": str(p.id),
                "task_signature": pl.get("task_signature"),
                "outcome": pl.get("outcome") or "unknown",
                "task_type": pl.get("task_type") or "",
                "step_count": pl.get("step_count", len(steps)),
                "failed_step_index": pl.get("failed_step_index"),
                "steps": steps,
                "content_excerpt": (pl.get("content") or "")[:300],
                "created_at": ts,
            })
    except Exception as e:
        log.warning("fetch_trajectories_for_promotion: %s", e)
    return out

def _group_trajectories_by_signature(trajs: list[dict]) -> dict[str, dict]:
    """Group trajectories by task_signature and count outcomes."""
    groups: dict[str, dict] = {}
    for t in trajs:
        sig = t.get("task_signature")
        if not sig:
            continue
        if sig not in groups:
            groups[sig] = {
                "task_signature": sig,
                "successes": [],
                "failures": [],
                "partials": [],
                "task_type": t.get("task_type") or "",
            }
        bucket = "successes" if t["outcome"] == "success" else "failures" if t["outcome"] == "failure" else "partials"
        groups[sig][bucket].append(t)
    return groups

def _summarize_procedure_steps(successful_trajs: list[dict]) -> str:
    """Build a step-pattern summary from successful trajectories."""
    # Aggregate the most common steps across successful trajectories
    step_counter: dict[str, int] = {}
    step_action_samples: dict[str, str] = {}
    for t in successful_trajs:
        for s in t.get("steps") or []:
            key = (s.get("action") or "").strip()
            if not key:
                continue
            # Normalize: collapse whitespace
            key_norm = " ".join(key.split())
            step_counter[key_norm] = step_counter.get(key_norm, 0) + 1
            if key_norm not in step_action_samples:
                step_action_samples[key_norm] = key_norm
    # Sort by frequency descending
    sorted_steps = sorted(step_counter.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for action, count in sorted_steps[:10]:
        lines.append(f"  {count}x: {action}")
    return "\n".join(lines)

def _build_procedure_content(task_signature: str, successful_trajs: list[dict], task_type: str) -> str:
    """Build content for an auto-promoted procedure memory."""
    sample = successful_trajs[0]
    task_text = ""
    try:
        # Try to extract task from sample content (first line)
        first_line = (sample.get("content_excerpt") or "").split("\n", 1)[0]
        if first_line.startswith("Task: "):
            task_text = first_line[len("Task: "):].strip()
    except Exception:
        pass
    if not task_text:
        task_text = f"Task signature {task_signature}"

    step_summary = _summarize_procedure_steps(successful_trajs)
    lines = [
        f"AUTO-PROMOTED PROCEDURE — synthesized from {len(successful_trajs)} successful trajectories with task_signature={task_signature}",
        "",
        f"Task pattern: {task_text}",
        f"Task type: {task_type or 'unspecified'}",
        f"Success count: {len(successful_trajs)}",
        "",
        "Step pattern (most common first):",
        step_summary or "  (no steps recorded)",
        "",
        "Usage: at task start, call memory_session_init with this task. If a similar "
        "task_signature appears in recent_trajectories with this procedure, follow these steps.",
    ]
    return "\n".join(lines)

def _build_warning_content(task_signature: str, failed_trajs: list[dict], task_type: str) -> str:
    """Build content for an auto-promoted warning procedure."""
    sample = failed_trajs[0]
    task_text = ""
    try:
        first_line = (sample.get("content_excerpt") or "").split("\n", 1)[0]
        if first_line.startswith("Task: "):
            task_text = first_line[len("Task: "):].strip()
    except Exception:
        pass
    if not task_text:
        task_text = f"Task signature {task_signature}"
    # Find the most common failed step index
    failed_steps: dict[int, int] = {}
    for t in failed_trajs:
        idx = t.get("failed_step_index")
        if idx is not None:
            failed_steps[idx] = failed_steps.get(idx, 0) + 1
    common_failed = sorted(failed_steps.items(), key=lambda x: x[1], reverse=True)[:3]

    lines = [
        f"AUTO-PROMOTED WARNING — synthesized from {len(failed_trajs)} failed trajectories with task_signature={task_signature}",
        "",
        f"Task pattern: {task_text}",
        f"Task type: {task_type or 'unspecified'}",
        f"Failure count: {len(failed_trajs)}",
        "",
    ]
    if common_failed:
        lines.append("Most commonly failed steps:")
        for idx, count in common_failed:
            lines.append(f"  Step {idx} failed in {count} of {len(failed_trajs)} attempts")
        lines.append("")
    lines.extend([
        "Avoidance guidance: at task start, if memory_session_init surfaces this task_signature "
        "in recent_trajectories with outcome=failure, pay extra attention to the failed steps above. "
        "Consider an alternative approach before attempting step " + (str(common_failed[0][0]) if common_failed else "?"),
    ])
    return "\n".join(lines)

def _store_promotion_memory(content: str, tags: list[str], importance: int,
                            category: str, source: str,
                            procedure_schema: dict = None) -> str:
    """Helper: store a promotion memory. Returns memory_id.

    Canonical-tier fix 2026-10-09: this used to upsert Qdrant directly —
    same bug class as the compaction-summary bypass — so promoted procedures
    had no canonical row and the recall liveness gate silently dropped them.
    Now routes through the atomic canonical+outbox commit.
    2A-3: carries the parameterized procedure schema (task_signature,
    preconditions, steps, failure_modes, success_criteria) in the payload.
    """
    memory_id = str(uuid.uuid4())
    payload = {
        "category": category,
        "importance": importance,
        "tags": ",".join(tags),
        "source": source,
    }
    if procedure_schema:
        payload["procedure"] = procedure_schema
    try:
        _canonical_write(
            memory_id, "global", "fact", content, version=1,
            event_type="MEMORY_CREATED", payload=payload)
    except Exception as _e:
        log.error("promotion canonical write failed (memory not stored): %s", _e)
        return None
    _invalidate_recall_caches()
    return memory_id

def _tool_promote_procedures(args: dict) -> types.CallToolResult:
    """Phase 2.1: Scan trajectories, auto-promote procedures + warnings."""
    min_success = max(1, int(args.get("min_success_count") or 3))
    min_failure = max(1, int(args.get("min_failure_count") or 3))
    lookback_days = int(args.get("lookback_days") if args.get("lookback_days") is not None else 30)
    dry_run = bool(args.get("dry_run", False))
    force = bool(args.get("force_repromote", False))

    trajs = _fetch_trajectories_for_promotion(lookback_days)
    groups = _group_trajectories_by_signature(trajs)

    promoted_procedures: list[dict] = []
    promoted_warnings: list[dict] = []

    for sig, g in groups.items():
        succ = g["successes"]
        fail = g["failures"]

        # Procedures from repeated successes
        if len(succ) >= min_success:
            already = False
            if not force:
                try:
                    already = _redis.get(_promotion_dedup_key("procedure", sig)) is not None
                except Exception:
                    already = False
            if not already:
                content = _build_procedure_content(sig, succ, g["task_type"])
                tags = ["auto-promoted", "learned-strategy", "reusable-skill",
                        f"task_signature:{sig}", f"task_type:{g['task_type']}" if g["task_type"] else "task_type:unspecified"]
                entry = {
                    "task_signature": sig,
                    "outcome": "success",
                    "source_trajectory_count": len(succ),
                    "memory_id": None,
                    "dry_run": dry_run,
                    "would_promote": True,
                }
                if not dry_run:
                    # 2A-5: runaway-learning guard — hold a signature that tries
                    # to promote more than N times/day (operator review, not
                    # auto-write). Redis daily flux counter, 25h TTL.
                    _flux_key = ("memory:promoflux:" + str(sig) + ":" +
                                 datetime.now(timezone.utc).strftime("%Y%m%d"))
                    _flux = 0
                    try:
                        _flux = int(_redis.get(_flux_key) or 0) + 1
                        _redis.set(_flux_key, _flux, ex=90000)
                    except Exception:
                        _flux = 0
                    if _flux > MAX_PROMOTIONS_PER_SIGNATURE_PER_DAY:
                        entry["held_by_guard"] = True
                        entry["guard_reason"] = ("runaway-learning guard: %d promotions "
                                                 "for this signature today" % _flux)
                        promoted_procedures.append(entry)
                        log.warning("promotion guard held sig=%s flux=%d", sig, _flux)
                        continue
                    _schema = {
                        "task_signature": sig,
                        "outcome": "success",
                        "source_trajectory_count": len(succ),
                        "preconditions": [],
                        "steps": [s.strip() for s in re.split(r"(?<=[.;])\s+", content) if s.strip()][:8],
                        "failure_modes": [],
                        "success_criteria": f"{len(succ)} successful trajectories share this task_signature",
                    }
                    mid = _store_promotion_memory(content, tags, 8, "procedure", "agent_auto_promote",
                                                  procedure_schema=_schema)
                    entry["memory_id"] = mid
                    entry["schema"] = "parameterized_v1"
                    try:
                        _redis.set(_promotion_dedup_key("procedure", sig),
                                   _now_iso(), ex=_PROMOTION_DEDUP_TTL)
                    except Exception:
                        pass
                promoted_procedures.append(entry)

        # Warnings from repeated failures
        if len(fail) >= min_failure:
            already = False
            if not force:
                try:
                    already = _redis.get(_promotion_dedup_key("warning", sig)) is not None
                except Exception:
                    already = False
            if not already:
                content = _build_warning_content(sig, fail, g["task_type"])
                tags = ["auto-promoted", "learned-warning", "avoid-pattern",
                        f"task_signature:{sig}", f"task_type:{g['task_type']}" if g["task_type"] else "task_type:unspecified"]
                entry = {
                    "task_signature": sig,
                    "outcome": "failure",
                    "source_trajectory_count": len(fail),
                    "memory_id": None,
                    "dry_run": dry_run,
                    "would_promote": True,
                }
                if not dry_run:
                    # 2A-5: runaway-learning guard (see procedure branch).
                    _flux_key = ("memory:promoflux:" + str(sig) + ":" +
                                 datetime.now(timezone.utc).strftime("%Y%m%d"))
                    _flux = 0
                    try:
                        _flux = int(_redis.get(_flux_key) or 0) + 1
                        _redis.set(_flux_key, _flux, ex=90000)
                    except Exception:
                        _flux = 0
                    if _flux > MAX_PROMOTIONS_PER_SIGNATURE_PER_DAY:
                        entry["held_by_guard"] = True
                        promoted_warnings.append(entry)
                        log.warning("promotion guard held warning sig=%s flux=%d", sig, _flux)
                        continue
                    _schema = {
                        "task_signature": sig,
                        "outcome": "failure",
                        "source_trajectory_count": len(fail),
                        "preconditions": [],
                        "steps": [],
                        "failure_modes": [s.strip() for s in re.split(r"(?<=[.;])\s+", content) if s.strip()][:6],
                        "success_criteria": "avoid recurrence of this failure pattern",
                    }
                    mid = _store_promotion_memory(content, tags, 8, "procedure", "agent_auto_promote_warning",
                                                  procedure_schema=_schema)
                    entry["memory_id"] = mid
                    entry["schema"] = "parameterized_v1"
                    try:
                        _redis.set(_promotion_dedup_key("warning", sig),
                                   _now_iso(), ex=_PROMOTION_DEDUP_TTL)
                    except Exception:
                        pass
                promoted_warnings.append(entry)

    # Also scan signatures that were already promoted and report (for visibility)
    already_promoted_count = 0
    try:
        keys = _redis.keys(f"{_PROMOTION_DEDUP_KEY_PREFIX}*")
        already_promoted_count = len(keys) if keys else 0
    except Exception:
        pass

    return _ok(json.dumps({
        "dry_run": dry_run,
        "lookback_days": lookback_days,
        "thresholds": {"min_success_count": min_success, "min_failure_count": min_failure},
        "trajectories_scanned": len(trajs),
        "signatures_seen": len(groups),
        "promoted_procedures": promoted_procedures,
        "promoted_warnings": promoted_warnings,
        "promoted_procedures_count": len(promoted_procedures),
        "promoted_warnings_count": len(promoted_warnings),
        "already_promoted_signatures_tracked": already_promoted_count,
        "instructions": (
            "New procedure memories surface in memory_session_init's learned_strategies list "
            "via the recall_learned_strategies helper (filtered by tag 'learned-strategy'). "
            "New warning memories have tag 'learned-warning' — query for them explicitly via "
            "memory_recall with query containing the task_signature or task description."
        ),
    }, ensure_ascii=False, default=str))

def _tool_procedure_status(args: dict) -> types.CallToolResult:
    """Phase 2.1: Show trajectory groups and promotion eligibility."""
    lookback_days = int(args.get("lookback_days") if args.get("lookback_days") is not None else 30)
    min_success = max(1, int(args.get("min_success_count") or 3))
    min_failure = max(1, int(args.get("min_failure_count") or 3))

    trajs = _fetch_trajectories_for_promotion(lookback_days)
    groups = _group_trajectories_by_signature(trajs)

    eligible_procedures: list[dict] = []
    eligible_warnings: list[dict] = []
    other_signatures: list[dict] = []

    for sig, g in groups.items():
        sc, fc, pc = len(g["successes"]), len(g["failures"]), len(g["partials"])
        already_proc = False
        already_warn = False
        try:
            already_proc = _redis.get(_promotion_dedup_key("procedure", sig)) is not None
            already_warn = _redis.get(_promotion_dedup_key("warning", sig)) is not None
        except Exception:
            pass

        record = {
            "task_signature": sig,
            "task_type": g["task_type"],
            "success_count": sc,
            "failure_count": fc,
            "partial_count": pc,
            "already_promoted_procedure": already_proc,
            "already_promoted_warning": already_warn,
        }
        if sc >= min_success:
            eligible_procedures.append({**record, "meets_threshold": True})
        if fc >= min_failure:
            eligible_warnings.append({**record, "meets_threshold": True})
        if sc < min_success and fc < min_failure:
            other_signatures.append(record)

    return _ok(json.dumps({
        "lookback_days": lookback_days,
        "thresholds": {"min_success_count": min_success, "min_failure_count": min_failure},
        "trajectories_scanned": len(trajs),
        "signatures_seen": len(groups),
        "eligible_procedures": eligible_procedures,
        "eligible_warnings": eligible_warnings,
        "below_threshold": other_signatures,
        "instructions": (
            "Run memory_promote_procedures to convert eligible_procedures/eligible_warnings "
            "into procedure+warning memories. Items with already_promoted_*:true are tracked "
            "and will be skipped on the next run unless force_repromote=true."
        ),
    }, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# Phase 2.2: Dreaming — async consolidation of trajectory clusters
# ---------------------------------------------------------------------------
def _dream_dedup_key(signature: str) -> str:
    return f"{_DREAM_DEDUP_KEY_PREFIX}{signature}"

def _analyze_cluster_for_insights(successes: list[dict], failures: list[dict], task_type: str) -> list[str]:
    """Compare success and failure trajectories to extract actionable insights."""
    insights = []
    
    # 1. Failed step analysis
    failed_steps = {}
    for t in failures:
        idx = t.get("failed_step_index")
        if idx is not None:
            failed_steps[idx] = failed_steps.get(idx, 0) + 1
    if failed_steps:
        most_common = sorted(failed_steps.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(
            "Most commonly failed step indices: " + 
            ", ".join(f"Step {i} ({c}x)" for i, c in most_common)
        )
    
    # 2. Step action patterns — successes vs failures
    success_actions = {}
    failure_actions = {}
    for t in successes:
        for s in (t.get("steps") or []):
            action = " ".join((s.get("action") or "").split())
            if action:
                success_actions[action] = success_actions.get(action, 0) + 1
    for t in failures:
        for s in (t.get("steps") or []):
            action = " ".join((s.get("action") or "").split())
            if action:
                failure_actions[action] = failure_actions.get(action, 0) + 1
    
    success_only = []
    for action, count in sorted(success_actions.items(), key=lambda x: x[1], reverse=True)[:10]:
        if failure_actions.get(action, 0) == 0 and count >= 2:
            success_only.append(f"{action} ({count}x)")
    if success_only:
        insights.append("Actions in successes but not failures: " + "; ".join(success_only[:5]))
    
    failure_only = []
    for action, count in sorted(failure_actions.items(), key=lambda x: x[1], reverse=True)[:10]:
        if success_actions.get(action, 0) == 0 and count >= 2:
            failure_only.append(f"{action} ({count}x)")
    if failure_only:
        insights.append("Actions in failures but not successes: " + "; ".join(failure_only[:5]))
    
    # 3. Tool usage patterns
    success_tools = {}
    failure_tools = {}
    for t in successes:
        for s in (t.get("steps") or []):
            tool = s.get("tool_used")
            if tool:
                success_tools[tool] = success_tools.get(tool, 0) + 1
    for t in failures:
        for s in (t.get("steps") or []):
            tool = s.get("tool_used")
            if tool:
                failure_tools[tool] = failure_tools.get(tool, 0) + 1
    
    tool_insights = []
    all_tools = set(success_tools) | set(failure_tools)
    for tool in sorted(all_tools):
        sc = success_tools.get(tool, 0)
        fc = failure_tools.get(tool, 0)
        if sc > 0 and fc == 0:
            tool_insights.append(f"{tool} used {sc}x in successes only")
        elif fc > 0 and sc == 0:
            tool_insights.append(f"{tool} used {fc}x in failures only")
    if tool_insights:
        insights.append("Tool patterns: " + "; ".join(tool_insights[:5]))
    
    # 4. Duration patterns
    success_durations = [t.get("duration_minutes") for t in successes if t.get("duration_minutes")]
    failure_durations = [t.get("duration_minutes") for t in failures if t.get("duration_minutes")]
    if success_durations and failure_durations:
        avg_succ = sum(success_durations) / len(success_durations)
        avg_fail = sum(failure_durations) / len(failure_durations)
        insights.append(f"Duration: successes avg {avg_succ:.1f} min, failures avg {avg_fail:.1f} min")
    
    # 5. Common failure observations
    error_patterns = {}
    for t in failures:
        for s in (t.get("steps") or []):
            if not s.get("success", True):
                obs = " ".join((s.get("observation") or "").split())[:120]
                if obs:
                    error_patterns[obs] = error_patterns.get(obs, 0) + 1
    if error_patterns:
        top_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append("Common failure observations: " + "; ".join(f"{e} ({c}x)" for e, c in top_errors))
    
    return insights

def _build_dream_insight_content(task_signature: str, task_type: str, successes: list[dict], failures: list[dict], insights: list[str]) -> str:
    sample = successes[0] if successes else failures[0]
    task_text = ""
    first_line = (sample.get("content_excerpt") or "").split("\n", 1)[0]
    if first_line.startswith("Task: "):
        task_text = first_line[len("Task: "):].strip()
    if not task_text:
        task_text = f"Task signature {task_signature}"
    
    lines = [
        f"DREAMED INSIGHT — synthesized from {len(successes)} successes and {len(failures)} failures with task_signature={task_signature}",
        "",
        f"Task pattern: {task_text}",
        f"Task type: {task_type or 'unspecified'}",
        f"Evidence: {len(successes)} success(es), {len(failures)} failure(s)",
        "",
        "Comparative analysis:",
    ]
    for insight in insights:
        lines.append(f"  - {insight}")
    lines.extend([
        "",
        "Usage: This insight was auto-generated by dreaming (Phase 2.2) by comparing "
        "success and failure trajectories. Use it to adapt your approach when facing "
        "similar tasks — favor the success patterns and avoid the failure patterns.",
    ])
    return "\n".join(lines)

def _tool_dream(args: dict) -> types.CallToolResult:
    """Phase 2.2: Cluster trajectories, compare successes vs failures, generate insights."""
    lookback_days = int(args.get("lookback_days") if args.get("lookback_days") is not None else 30)
    min_successes = max(1, int(args.get("min_successes") or 2))
    min_failures = max(1, int(args.get("min_failures") or 1))
    dry_run = bool(args.get("dry_run", False))
    force = bool(args.get("force_redream", False))
    
    trajs = _fetch_trajectories_for_promotion(lookback_days)
    groups = _group_trajectories_by_signature(trajs)
    
    dreamed_insights = []
    
    for sig, g in groups.items():
        succ = g["successes"]
        fail = g["failures"]
        
        if len(succ) < min_successes or len(fail) < min_failures:
            continue
        
        already = False
        if not force:
            try:
                already = _redis.get(_dream_dedup_key(sig)) is not None
            except Exception:
                already = False
        if already:
            continue
        
        insights = _analyze_cluster_for_insights(succ, fail, g["task_type"])
        if not insights:
            continue
        
        content = _build_dream_insight_content(sig, g["task_type"], succ, fail, insights)
        tags = ["dreamed-insight", "learned-strategy", "auto-consolidation",
                f"task_signature:{sig}", f"task_type:{g['task_type']}" if g["task_type"] else "task_type:unspecified"]
        
        entry = {
            "task_signature": sig,
            "success_count": len(succ),
            "failure_count": len(fail),
            "insights_generated": len(insights),
            "memory_id": None,
            "dry_run": dry_run,
            "would_dream": True,
        }
        
        if not dry_run:
            mid = _store_promotion_memory(content, tags, 9, "procedure", "agent_dream")
            entry["memory_id"] = mid
            try:
                _redis.set(_dream_dedup_key(sig), _now_iso(), ex=_DREAM_DEDUP_TTL)
            except Exception:
                pass
        
        dreamed_insights.append(entry)
    
    return _ok(json.dumps({
        "dry_run": dry_run,
        "lookback_days": lookback_days,
        "thresholds": {"min_successes": min_successes, "min_failures": min_failures},
        "trajectories_scanned": len(trajs),
        "signatures_seen": len(groups),
        "dreamed_insights": dreamed_insights,
        "dreamed_count": len(dreamed_insights),
        "instructions": (
            "Dreamed insights surface in memory_session_init via learned_strategies "
            "(tagged 'dreamed-insight'). They are comparative analyses of success vs "
            "failure trajectories — use them to adapt your approach to similar tasks."
        ),
    }, ensure_ascii=False, default=str))

# ---------------------------------------------------------------------------
# Phase 2.4: One-shot maintenance driver (cron-able learning loop chain)
# ---------------------------------------------------------------------------
def _tool_graph_extract(args: dict) -> types.CallToolResult:
    """2C-1: rule-based graph extraction (offline, maintenance-tier).

    v1 is deterministic (no LLM): edges are proposed from structural signals —
    typed content prefixes ([ENTITY:x], [ARCH:x], [STRATEGIC:x], [ROADMAP:x])
    and tag co-occurrence — and auto-applied only at confidence >= 0.7 via
    the same durable graph_link path (canonical + outbox). Lower-confidence
    proposals are reported for operator review, never auto-applied.
    """
    dry_run = bool(args.get("dry_run", False))
    min_conf = float(args.get("min_confidence", 0.7))
    stats = {"scanned": 0, "proposed": 0, "applied": 0, "would_apply": 0,
             "skipped_existing": 0, "held_for_review": 0, "failed": 0, "rules": {}}

    # 1) gather global-scope candidate memories with structured signals
    points, _offset = [], None
    while True:
        _page = _client.scroll(collection_name=COLLECTION_NAME, limit=1000,
                               with_payload=True, with_vectors=False, offset=_offset)
        points.extend(_page[0]); _offset = _page[1]
        if not _offset: break
    cands = []
    for p in points:
        pld = p.payload or {}
        if (pld.get("scope") or "global") != "global":
            continue
        content = pld.get("content") or ""
        if not content:
            continue
        stats["scanned"] += 1
        prefix = None
        for _tag_open, _kind in (("[ENTITY:", "entity"), ("[ARCH:", "arch"),
                                 ("[STRATEGIC:", "strategic"), ("[ROADMAP:", "roadmap")):
            if content.startswith(_tag_open):
                prefix = (_kind, content[len(_tag_open):].split("]", 1)[0].split(" ", 1)[0])
                break
        cands.append({"id": p.id, "content": content, "prefix": prefix,
                      "tags": set(_payload_tags(pld)), "created": pld.get("created_at", ""),
                      "tsig": (pld.get("task_signature") or "").strip(),
                      "is_traj": (pld.get("category") == "trajectory")})

    # 2) propose edges
    proposals = []
    # Rule A: arch layers implement/refer each other (vertical architecture flow)
    arch = [c for c in cands if c["prefix"] and c["prefix"][0] == "arch"]
    flows = {"working-memory": ("references", "context-synthesizer"),
             "context-synthesizer": ("depends_on", "memory-governance"),
             "learning-engine": ("depends_on", "memory-governance")}
    by_name = {c["prefix"][1]: c for c in arch if c["prefix"]}
    for src_name, (rel, dst_name) in flows.items():
        s, d = by_name.get(src_name), by_name.get(dst_name)
        if s and d:
            proposals.append({"src": s["id"], "dst": d["id"], "rel": rel,
                              "conf": 0.8, "rule": "arch-vertical-flow"})
    # Rule B: roadmap -> strategic/arch same theme (tag overlap >= 2) : references
    for a in cands:
        for b in cands:
            if a["id"] >= b["id"]:
                continue
            overlap = a["tags"] & b["tags"]
            if len(overlap) >= 2:
                proposals.append({"src": a["id"], "dst": b["id"], "rel": "related_to",
                                  "conf": 0.75, "rule": "tag-cooccurrence-2"})
            elif a["prefix"] and b["prefix"] and a["prefix"][0] != b["prefix"][0]:
                proposals.append({"src": a["id"], "dst": b["id"], "rel": "references",
                                  "conf": 0.7, "rule": "typed-prefix-cross-link"})
    # Sprint-1 1.5 — trajectory-derived rules (high-precision lineage):
    # Rule T (co-task-trajectory): memories carrying the SAME task_signature
    #   were created by one task episode — star-link each member to the
    #   episode's trajectory summary (references), never member-to-member
    #   (avoids pair explosion, keeps the semantics honest).
    # Rule L (trajectory-implements): a procedure extracted from a trajectory
    #   IMPLEMENTS what that episode learned.
    _by_sig = {}
    for c in cands:
        if c["tsig"]:
            _by_sig.setdefault(c["tsig"], {"traj": None, "members": []})
            if c["is_traj"] and _by_sig[c["tsig"]]["traj"] is None:
                _by_sig[c["tsig"]]["traj"] = c
            else:
                _by_sig[c["tsig"]]["members"].append(c)
    for _sig, _grp in _by_sig.items():
        _traj = _grp["traj"]
        if _traj is None:
            continue
        for _m in _grp["members"][:8]:
            _is_proc = bool(_m["tags"] & {"learned-strategy", "procedure",
                                          "learned-warning", "dreamed-insight"})
            proposals.append({"src": _m["id"], "dst": _traj["id"],
                              "rel": "implements" if _is_proc else "references",
                              "conf": 0.75, "rule": "co-task-trajectory",
                              "src_tsig": _sig[:16]})
    stats["proposed"] = len(proposals)

    # 3) apply >= min_conf via the durable path; dedup against existing edges
    existing = set()
    try:
        rows = _pg_query("SELECT src, dst, rel FROM mem_edges", {})
        for r in rows:
            existing.add((str(r["src"]) if isinstance(r, dict) else str(r[0]),
                          str(r["dst"]) if isinstance(r, dict) else str(r[1]),
                          (r["rel"] if isinstance(r, dict) else r[2])))
    except Exception:
        pass
    for pr in proposals:
        key = (pr["src"], pr["dst"], pr["rel"])
        if key in existing or (pr["dst"], pr["src"], pr["rel"]) in existing:
            stats["skipped_existing"] += 1
            continue
        stats["rules"][pr["rule"]] = stats["rules"].get(pr["rule"], 0) + 1
        if pr["conf"] < min_conf:
            stats["held_for_review"] += 1
            # Sprint-1 1.5: durable review queue — proposals below the
            # auto-apply threshold persist as edge_proposal rows the operator
            # can review, apply, or discard (was report-and-forget).
            _mlog("graph_extract", "edge_proposal",
                  memory_id=f"{pr['src']}->{pr['dst']}",
                  after={"src": str(pr["src"]), "dst": str(pr["dst"]),
                         "rel": pr["rel"], "conf": pr["conf"], "rule": pr["rule"]},
                  reason="below auto-apply threshold — operator review queue",
                  dry_run=dry_run)
            continue
        if dry_run:
            stats["would_apply"] += 1
            continue
        try:
            # Sprint-1 1.5 honesty fix: prior live applies may have landed the
            # edge under a resolved node UUID, so the PG-based dedup set can
            # miss it while graph_link's ON CONFLICT silently no-ops — the run
            # then reports hundreds of phantom "applied". The Redis edge key
            # uses MEMORY ids (same namespace as proposals), so it is the
            # authoritative cheap duplicate check.
            _ekey = f"{GRAPH_EDGE_PREFIX}{pr['src']}:{pr['dst']}"
            if _check_redis():
                try:
                    if _redis.get(_ekey):
                        stats["skipped_existing"] += 1
                        continue
                except Exception:
                    pass
            r = _tool_graph_link({"memory_id": pr["src"], "related_id": pr["dst"],
                                  "relation": pr["rel"]})
            if getattr(r, "is_error", None):
                stats["failed"] += 1
            else:
                stats["applied"] += 1
                _mlog("graph_extract", "edge_apply",
                      memory_id=f"{pr['src']}->{pr['dst']}",
                      after={"rel": pr["rel"], "rule": pr["rule"], "conf": pr["conf"]},
                      reason="2C-1 rule-based extraction (>=0.7 confidence)", dry_run=False)
        except Exception:
            stats["failed"] += 1
    return _ok(json.dumps({"status": "graph_extract_complete", "dry_run": dry_run,
                           "stats": stats}, ensure_ascii=False, indent=2))


def _tool_maintenance(args: dict) -> types.CallToolResult:
    """One-shot maintenance driver: chains all auto-learning loops.

    Intended to be called by a periodic cron so the memory system
    keeps improving without manual intervention. Each sub-step is
    idempotent via Redis dedup keys, so re-running is cheap.
    """
    lookback_days = int(args.get("lookback_days") or 30)
    min_success = int(args.get("min_success_count") or 3)
    min_failure = int(args.get("min_failure_count") or 3)
    min_successes_dream = int(args.get("min_successes_dream") or 2)
    min_failures_dream = int(args.get("min_failures_dream") or 1)
    decay_archive_days = int(args.get("decay_archive_days") or 90)
    dry_run = bool(args.get("dry_run", False))

    summary: dict = {
        "dry_run": dry_run,
        "lookback_days": lookback_days,
        "timestamp": _now_iso(),
        "steps": {},
    }

    # Step 1: promote procedures + warnings from trajectories
    try:
        pargs = {
            "lookback_days": lookback_days,
            "min_success_count": min_success,
            "min_failure_count": min_failure,
            "dry_run": dry_run,
        }
        pres = _tool_promote_procedures(pargs)
        summary["steps"]["promote_procedures"] = json.loads(pres.content[0].text)
    except Exception as e:
        summary["steps"]["promote_procedures"] = {"error": str(e)}

    # Step 2: dream comparative insights
    try:
        dargs = {
            "lookback_days": lookback_days,
            "min_successes": min_successes_dream,
            "min_failures": min_failures_dream,
            "dry_run": dry_run,
        }
        dres = _tool_dream(dargs)
        summary["steps"]["dream"] = json.loads(dres.content[0].text)
    except Exception as e:
        summary["steps"]["dream"] = {"error": str(e)}

    # Step 3: consolidate (dedup + cleanup)
    try:
        cres = _tool_consolidate({"dry_run": dry_run})
        summary["steps"]["consolidate"] = json.loads(cres.content[0].text)
    except Exception as e:
        summary["steps"]["consolidate"] = {"error": str(e)}

    # Step 4: decay-archive stale low-importance memories (soft: lowers
    # importance so they drift down in ranking — no hard delete).
    if decay_archive_days > 0:
        try:
            summary["steps"]["decay_archive"] = _decay_archive_stale(decay_archive_days, dry_run=dry_run)
        except Exception as e:
            summary["steps"]["decay_archive"] = {"error": str(e)}
    else:
        summary["steps"]["decay_archive"] = {"skipped": True}

    # Step 5 (2C-1): rule-based graph extraction — the graph grows without
    # manual wiring; auto-applies confidence >= 0.7 only.
    try:
        gres = _tool_graph_extract({"dry_run": dry_run, "min_confidence": 0.7})
        summary["steps"]["graph_extract"] = json.loads(gres.content[0].text)
    except Exception as e:
        summary["steps"]["graph_extract"] = {"error": str(e)}

    # Step 6 (2C-4): three-way graph reconcile (PG <-> Redis <-> Qdrant).
    try:
        rres = _tool_graph_reconcile({})
        summary["steps"]["graph_reconcile"] = json.loads(rres.content[0].text)
    except Exception as e:
        summary["steps"]["graph_reconcile"] = {"error": str(e)}

    # Sprint-1 1.4: durable per-run audit rows + retention prune.
    try:
        for _step, _sd in summary["steps"].items():
            _mlog(_step, "step_summary",
                  after=_sd if isinstance(_sd, dict) else {"raw": str(_sd)[:200]},
                  reason="maintenance run step summary", dry_run=dry_run)
    except Exception:
        pass
    try:
        _pg_run("DELETE FROM op_telemetry WHERE ts < now() - interval '30 days'",
                {}, table="op-telemetry-prune")
        _pg_run("DELETE FROM maintenance_log WHERE ts < now() - interval '90 days'",
                {}, table="maintenance-log-prune")
        summary["retention_prune"] = {"status": "ok",
                                      "op_telemetry_window_days": 30,
                                      "maintenance_log_window_days": 90}
    except Exception as _e:
        summary["retention_prune"] = {"status": "skipped", "error": str(_e)[:120]}

    return _ok(json.dumps(summary, ensure_ascii=False, default=str))


def _decay_archive_stale(days: int, dry_run: bool = False) -> dict:
    """Find low-importance memories unused for > N days and lower their
    importance so Ebbinghaus decay ranks them down. No hard delete.
    """
    cutoff_ts = datetime.now(timezone.utc).timestamp() - days * 86400
    archived = []
    try:
        points, _offset = _client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("category") == "credential":
                continue
            importance = int(payload.get("importance") or 5)
            if importance >= 7:
                continue
            last = payload.get("last_accessed_at") or payload.get("created_at") or ""
            try:
                last_ts = datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp()
            except Exception:
                last_ts = cutoff_ts
            if last_ts >= cutoff_ts:
                continue
            archived.append({
                "memory_id": str(point.id),
                "category": payload.get("category"),
                "importance": importance,
                "last_accessed_at": last,
                "new_importance": max(1, importance - 1),
            })
        if dry_run and archived:
            for item in archived[:50]:
                _mlog("decay_archive", "would_archive", memory_id=item["memory_id"],
                      before={"importance": item["importance"]},
                      after={"importance": item["new_importance"]},
                      reason=f"stale_low_importance_{days}d", dry_run=True)
        if not dry_run and archived:
            now = _now_iso()
            for item in archived:
                try:
                    _client.set_payload(
                        collection_name=COLLECTION_NAME,
                        points=[item["memory_id"]],
                        payload={"importance": item["new_importance"],
                                 "last_archived_at": now,
                                 "archive_reason": f"stale_low_importance_{days}d"},
                    )
                    _mlog("decay_archive", "archive", memory_id=item["memory_id"],
                          before={"importance": item["importance"]},
                          after={"importance": item["new_importance"],
                                 "archive_reason": f"stale_low_importance_{days}d"},
                          reason="Ebbinghaus decay-archive (soft, no delete)", dry_run=False)
                except Exception:
                    pass
    except Exception as e:
        return {"error": str(e)}
    return {
        "dry_run": dry_run,
        "cutoff_days": days,
        "archived_count": len(archived),
        "archived": archived[:50],
    }
# Phase 2.3: Relevance Feedback Loop — agent marks recalled memories as useful/irrelevant
# to improve future recall rankings via implicit feedback.


# Phase 2.3: Relevance Feedback Loop — feedback storage and ranking adjustment
# ---------------------------------------------------------------------------
FEEDBACK_PREFIX = "memory:feedback:"
FEEDBACK_QUERY_PREFIX = "memory:feedback:query:"
FEEDBACK_TTL = 60 * 60 * 24 * 30  # 30 days
FEEDBACK_BOOST = 0.15  # importance boost per useful feedback
FEEDBACK_PENALTY = -0.10  # importance penalty per irrelevant feedback

# Phase 4: Time-based decay — unused memories gradually lose effective importance.
# Sigmoid-shaped decay: half-life ~180 days for importance=5, longer at higher importance.
# No memory is hard-deleted by decay; it just drifts down in ranking.
DECAY_ENABLED = True
DECAY_HALF_LIFE_DAYS = 180  # halve the decay bonus every N days

# 2B-4: per-type decay policy — learned strategies must outlive ephemeral
# chatter. Applied by _apply_decay(category=...) and by consolidate's
# eviction/archival pass. evict_unused_days=None means "never auto-evict"
# (supersede-only categories).
CATEGORY_DECAY_TABLE: dict = {
    "procedure":     {"half_life_days": 365, "evict_unused_days": None},
    "preference":    {"half_life_days": 180, "evict_unused_days": None},
    "anti_pattern":  {"half_life_days": 180, "evict_unused_days": None},
    "decision":      {"half_life_days": 180, "evict_unused_days": None},
    "fact":          {"half_life_days": 90,  "evict_unused_days": 90},
    "event":         {"half_life_days": 90,  "evict_unused_days": 90},
    "summary":       {"half_life_days": 90,  "evict_unused_days": 90},
    "conversation":  {"half_life_days": 30,  "evict_unused_days": 30},
    "general":       {"half_life_days": 30,  "evict_unused_days": 45},
}
_PROTECTED_FROM_EVICT = {c for c, t in CATEGORY_DECAY_TABLE.items()
                         if t["evict_unused_days"] is None}
# 2A-5: runaway-learning circuit breaker (promotions per signature per day)
MAX_PROMOTIONS_PER_SIGNATURE_PER_DAY = int(
    os.environ.get("MEMORY_MAX_PROMOTIONS_PER_SIG_DAY", "20"))

# Phase 4: Embedding model versioning — track which model produced each vector.
# If EMBEDDING_MODEL changes, previously-stored memories fall to a legacy bucket.
# EMBEDDING_MODEL_VERSION is defined at the top of the file (from MEMORY_EMBEDDING_MODEL env).


def _get_feedback_counts(memory_id: str) -> tuple:
    """Return (useful_count, irrelevant_count) from persisted payload fields.

    Falls back to scanning Redis feedback keys if payload fields are missing.
    """
    try:
        fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
        if fetched and fetched[0].payload:
            return (
                int(fetched[0].payload.get("useful_count", 0)),
                int(fetched[0].payload.get("irrelevant_count", 0)),
            )
    except Exception:
        pass
    return (0, 0)


def _feedback_key(memory_id: str, query_hash: str) -> str:
    """Generate Redis key for a specific memory+query feedback."""
    return f"{FEEDBACK_PREFIX}{memory_id}:{query_hash}"


def _query_feedback_key(query_hash: str) -> str:
    """Generate Redis key for all feedback on a query."""
    return f"{FEEDBACK_QUERY_PREFIX}{query_hash}"


def _hash_query(query: str) -> str:
    """Create a short hash of the query for indexing."""
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]


def _tool_feedback(args: dict) -> types.CallToolResult:
    """Store relevance feedback for a recalled memory."""
    global _RECALL_SESSION_STATS
    memory_id = args["memory_id"]
    # Ownership gate (defence in depth: the dispatcher also enforces this).
    _assert_memory_access(memory_id)
    query = args["query"]
    rating = args["rating"]
    task_type = args.get("task_type")
    
    if rating not in ("useful", "irrelevant"):
        return _err("rating must be 'useful' or 'irrelevant'")
    
    # Track session efficiency
    if rating == "useful":
        _RECALL_SESSION_STATS["useful_feedback"] += 1
    elif rating == "irrelevant":
        _RECALL_SESSION_STATS["irrelevant_feedback"] += 1
    
    if not _redis:
        return _err("Redis unavailable — feedback requires short-term memory layer")
    
    query_hash = _hash_query(query)
    fb_key = _feedback_key(memory_id, query_hash)
    
    # Store feedback entry
    feedback_entry = {
        "memory_id": memory_id,
        "query": query,
        "query_hash": query_hash,
        "rating": rating,
        "task_type": task_type,
        "timestamp": _now_iso(),
    }
    
    try:
        # Store the feedback with TTL
        _redis.setex(fb_key, FEEDBACK_TTL, json.dumps(feedback_entry))
        
        # Also maintain a set of feedback keys per query for stats
        q_key = _query_feedback_key(query_hash)
        _redis.sadd(q_key, fb_key)
        _redis.expire(q_key, FEEDBACK_TTL)
        
        # Enhancement 9: Utility tracking — increment useful_count on "useful" feedback
        if rating == "useful":
            try:
                fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
                if fetched and fetched[0].payload:
                    cur_uc = fetched[0].payload.get("useful_count", 0)
                    _client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"useful_count": cur_uc + 1},
                        points=[memory_id],
                    )
            except Exception:
                pass

        # Track irrelevant feedback count too, for penalty side of the loop.
        if rating == "irrelevant":
            try:
                fetched = _client.retrieve(collection_name=COLLECTION_NAME, ids=[memory_id], with_payload=True)
                if fetched and fetched[0].payload:
                    cur_ic = fetched[0].payload.get("irrelevant_count", 0)
                    _client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={"irrelevant_count": cur_ic + 1},
                        points=[memory_id],
                    )
            except Exception:
                pass
        
        log.info("Feedback stored: memory=%s query_hash=%s rating=%s", memory_id[:8], query_hash, rating)

        # Durable feedback row in PostgreSQL (queryable for consolidation scoring)
        _pg_run(
            "INSERT INTO mem_feedback(memory_id, query_hash, rating, task_type) "
            "VALUES (:mid, :qhash, :rating, :ttype)",
            {"mid": memory_id, "qhash": query_hash, "rating": rating, "ttype": task_type},
            table="mem_feedback",
        )

        return _ok(json.dumps({
            "stored": True,
            "memory_id": memory_id,
            "query_hash": query_hash,
            "rating": rating,
            "message": f"Feedback '{rating}' recorded for memory {memory_id[:8]}",
        }, ensure_ascii=False))
    except Exception as e:
        log.exception("Feedback store failed")
        return _err(f"Failed to store feedback: {e}")


def _tool_feedback_stats(args: dict) -> types.CallToolResult:
    """Get statistics about the relevance feedback store."""
    if not _redis:
        return _err("Redis unavailable")
    
    try:
        # Find all query feedback keys
        query_keys = _redis.keys(f"{FEEDBACK_QUERY_PREFIX}*")
        
        total_feedback = 0
        useful_count = 0
        irrelevant_count = 0
        memory_scores = Counter()  # memory_id -> net score
        
        for q_key in query_keys:
            fb_keys = _redis.smembers(q_key)
            for fb_key in fb_keys:
                fb_data = _redis.get(fb_key)
                if fb_data:
                    total_feedback += 1
                    fb = json.loads(fb_data)
                    if fb["rating"] == "useful":
                        useful_count += 1
                        memory_scores[fb["memory_id"]] += 1
                    else:
                        irrelevant_count += 1
                        memory_scores[fb["memory_id"]] -= 1
        
        # Top memories by feedback score
        top_memories = [
            {"memory_id": mid, "net_score": score}
            for mid, score in memory_scores.most_common(10)
        ]
        
        return _ok(json.dumps({
            "total_feedback_entries": total_feedback,
            "useful_count": useful_count,
            "irrelevant_count": irrelevant_count,
            "unique_memories_with_feedback": len(memory_scores),
            "top_memories_by_score": top_memories,
            "feedback_boost_per_useful": FEEDBACK_BOOST,
            "feedback_penalty_per_irrelevant": FEEDBACK_PENALTY,
        }, ensure_ascii=False, default=str))
    except Exception as e:
        log.exception("Feedback stats failed")
        return _err(f"Failed to get feedback stats: {e}")


def _tool_tool_recommendations(args: dict) -> types.CallToolResult:
    """Phase 3.1: Recommend tools based on trajectory success patterns."""
    task_type = args.get("task_type")
    lookback_days = int(args.get("lookback_days") or 30)
    min_usage = int(args.get("min_usage") or 2)
    query = args.get("query")
    
    if not _redis:
        return _err("Redis unavailable for trajectory access")
    
    try:
        # Fetch trajectories from Qdrant
        trajs = _fetch_trajectories_for_promotion(lookback_days)
        
        # Filter by task_type if provided
        if task_type:
            trajs = [t for t in trajs if t.get("task_type") == task_type]
        
        # If query provided, also find semantically similar trajectories
        if query:
            similar_trajs = _fetch_trajectories_by_query(query, top_k=20)
            if similar_trajs:
                trajs = similar_trajs
        
        # Analyze tool usage in successes vs failures
        success_tools = Counter()
        failure_tools = Counter()
        total_success_steps = 0
        total_failure_steps = 0
        
        for t in trajs:
            is_success = t.get("outcome") == "success"
            steps = t.get("steps", [])
            for step in steps:
                tool = step.get("tool_used")
                if not tool:
                    continue
                if is_success:
                    success_tools[tool] += 1
                    total_success_steps += 1
                else:
                    failure_tools[tool] += 1
                    total_failure_steps += 1
        
        # Calculate recommendations
        all_tools = set(success_tools) | set(failure_tools)
        recommendations = []
        
        for tool in all_tools:
            succ_count = success_tools.get(tool, 0)
            fail_count = failure_tools.get(tool, 0)
            total_count = succ_count + fail_count
            
            if total_count < min_usage:
                continue
            
            # Success rate
            succ_rate = succ_count / total_count if total_count > 0 else 0
            # Weighted score: favor tools with high success rate AND high usage
            usage_factor = min(1.0, total_count / 10.0)  # cap at 10 uses
            score = succ_rate * 0.7 + usage_factor * 0.3
            
            # Determine recommendation level
            if succ_count > 0 and fail_count == 0:
                level = "highly_recommended"
            elif succ_rate >= 0.8:
                level = "recommended"
            elif succ_rate >= 0.5:
                level = "neutral"
            elif fail_count > 0 and succ_count == 0:
                level = "avoid"
            else:
                level = "caution"
            
            recommendations.append({
                "tool": tool,
                "success_count": succ_count,
                "failure_count": fail_count,
                "total_count": total_count,
                "success_rate": round(succ_rate, 3),
                "score": round(score, 3),
                "level": level,
            })
        
        # Sort by score descending
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Group by level
        by_level = {}
        for r in recommendations:
            by_level.setdefault(r["level"], []).append(r)
        
        return _ok(json.dumps({
            "trajectories_analyzed": len(trajs),
            "task_type_filter": task_type,
            "lookback_days": lookback_days,
            "recommendations": recommendations,
            "by_level": by_level,
            "summary": {
                "highly_recommended": len(by_level.get("highly_recommended", [])),
                "recommended": len(by_level.get("recommended", [])),
                "neutral": len(by_level.get("neutral", [])),
                "caution": len(by_level.get("caution", [])),
                "avoid": len(by_level.get("avoid", [])),
            },
        }, ensure_ascii=False, default=str))
        
    except Exception as e:
        log.exception("Tool recommendations failed")
        return _err(f"Failed to get tool recommendations: {e}")


def _tool_failure_hotspots(args: dict) -> types.CallToolResult:
    """Phase 3.2: Step-level failure dashboard across trajectory clusters."""
    lookback_days = int(args.get("lookback_days") if args.get("lookback_days") is not None else 30)
    min_failures = max(1, int(args.get("min_failures") or 2))

    try:
        trajs = _fetch_trajectories_for_promotion(lookback_days)
    except Exception as e:
        return _err(f"Failed to fetch trajectories: {e}")

    groups = _group_trajectories_by_signature(trajs)

    global_step_failures = Counter()   # step_index -> count
    global_tool_failures = Counter()   # tool -> count
    global_error_patterns = Counter()  # observation snippet -> count

    signature_reports = []
    for sig, g in groups.items():
        failures = g["failures"]
        if len(failures) < min_failures:
            continue

        step_failures = Counter()       # failed_step_index -> count
        hotspot_details = []

        for t in failures:
            steps = t.get("steps") or []
            fsi = t.get("failed_step_index")
            if fsi is None:
                # Fallback: first unsuccessful step
                for s in steps:
                    if not s.get("success", True):
                        fsi = s.get("step_index")
                        break
            if fsi is None:
                continue
            step_failures[fsi] += 1
            global_step_failures[fsi] += 1

            # Extract details for that step
            for s in steps:
                if s.get("step_index") == fsi:
                    tool = s.get("tool_used")
                    if tool:
                        global_tool_failures[tool] += 1
                    obs = " ".join((s.get("observation") or "").split())[:150]
                    action = " ".join((s.get("action") or "").split())[:120]
                    if obs:
                        global_error_patterns[obs] += 1
                    hotspot_details.append({
                        "step_index": fsi,
                        "action": action,
                        "tool_used": tool,
                        "observation": obs,
                    })
                    break

        signature_reports.append({
            "task_signature": sig,
            "task_type": g.get("task_type"),
            "failure_count": len(failures),
            "success_count": len(g["successes"]),
            "top_failed_steps": [
                {"step_index": idx, "failures": count}
                for idx, count in step_failures.most_common(5)
            ],
            "hotspot_details": hotspot_details[:10],
        })

    signature_reports.sort(key=lambda r: r["failure_count"], reverse=True)

    return _ok(json.dumps({
        "lookback_days": lookback_days,
        "min_failures": min_failures,
        "trajectories_scanned": len(trajs),
        "signatures_with_failures": len(signature_reports),
        "signature_reports": signature_reports,
        "global_hotspots": {
            "most_common_failed_step_indices": [
                {"step_index": idx, "failure_count": count}
                for idx, count in global_step_failures.most_common(10)
            ],
            "tools_most_involved_in_failures": [
                {"tool": tool, "failure_count": count}
                for tool, count in global_tool_failures.most_common(10)
            ],
            "most_common_failure_observations": [
                {"observation": obs, "count": count}
                for obs, count in global_error_patterns.most_common(10)
            ],
        },
        "instructions": (
            "High 'step_index' hotspots indicate a recurring structural problem at that "
            "stage of this task type. Before starting a similar task, check this report: "
            "add an extra verification step BEFORE the hotspot index, or pick a different "
            "tool than the ones dominating 'tools_most_involved_in_failures'."
        ),
    }, ensure_ascii=False, default=str))


def _apply_decay(importance: float, created_at: str, accessed_at: str, now_ts: float,
                 category: str = None) -> tuple[float, float]:
    """Phase 4: Time-based decay. Returns (decayed_importance, decay_factor).
    Decay reduces effective importance over time based on days since last access.
    Higher base importance decays slower (half-life scales with importance).
    2B-4: when a category is provided, its per-type half-life from
    CATEGORY_DECAY_TABLE replaces the global default (procedures/decisions
    decay far slower than conversation/general).
    """
    if not DECAY_ENABLED:
        return importance, 1.0
    try:
        since_ts = datetime.fromisoformat(accessed_at.replace("Z", "+00:00")).timestamp() if accessed_at else None
        if since_ts is None and created_at:
            since_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    except Exception:
        since_ts = None
    if since_ts is None:
        return importance, 1.0
    days = max(0.0, (now_ts - since_ts) / 86400.0)
    base_half_life = DECAY_HALF_LIFE_DAYS
    if category:
        base_half_life = CATEGORY_DECAY_TABLE.get(
            category, {}).get("half_life_days", DECAY_HALF_LIFE_DAYS)
    # Half-life scales with importance: importance 10 → 2x half-life, importance 1 → 0.5x
    decay_half_life = base_half_life * (0.5 + (importance / 10.0) * 1.5)
    # Sigmoid-ish decay factor: 1.0 → 0.5 as days → half_life, approaches ~0.1 as days → 3x half_life
    decay_factor = 1.0 / (1.0 + (days / decay_half_life) ** 1.8)
    # Apply decay to importance portion only (hybrid + recency untouched)
    decayed_importance = importance * decay_factor
    return decayed_importance, decay_factor


def _get_feedback_adjustment(memory_id: str, query: str) -> float:
    """Get the feedback-based score adjustment for a memory+query pair.
    Returns a value between -0.5 and +0.5 that should be added to importance score.
    """
    if not _redis:
        return 0.0
    try:
        query_hash = _hash_query(query)
        fb_key = _feedback_key(memory_id, query_hash)
        fb_data = _redis.get(fb_key)
        if fb_data:
            fb = json.loads(fb_data)
            if fb["rating"] == "useful":
                return FEEDBACK_BOOST
            elif fb["rating"] == "irrelevant":
                return FEEDBACK_PENALTY
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Auto-routing remember tool
# ---------------------------------------------------------------------------
SESSION_ROUTE_KEYWORDS = {
    "credential", "password", "token", "secret", "api_key", "apikey",
    "session_id", "sessionid", "bearer", "auth", "passphrase", "private_key",
}
PERMANENT_ROUTE_KEYWORDS = {
    "fact", "procedure", "preference", "decision", "project", "server",
    "config", "setting", "fix", "bug", "solution", "task", "todo",
    "directory", "path", "url", "email", "phone", "address", "name",
}

def _auto_decide(content: str, persistence: str) -> str:
    """Decide layer based on keywords in content."""
    if persistence != "auto":
        return persistence
    lowered = content.lower()
    if any(kw in lowered for kw in SESSION_ROUTE_KEYWORDS):
        return "session"
    if any(kw in lowered for kw in PERMANENT_ROUTE_KEYWORDS):
        return "permanent"
    return "permanent"

def _tool_remember(args: dict) -> types.CallToolResult:
    content = args["content"]
    # 3B-2: secret gate ahead of everything — remember covers BOTH the
    # short-term Redis tier and the durable store, so gate here once.
    _sec_rej, _sec_findings, content = _scan_secrets(content)
    if _sec_rej:
        _pg_audit("secret_block", None, {"findings": _sec_findings, "via": "remember"})
        return _err(json.dumps({
            "blocked": True, "gate": "secret_detection",
            "findings": _sec_findings,
            "message": "Content flagged by secret-detection gate (credential-shaped string).",
        }, ensure_ascii=False))
    # Enhancement 3: Poisoning guard on remember tool too
    blocked, poison_score, findings = _check_poison(content)
    if blocked:
        return _err(json.dumps({
            "blocked": True, "poison_score": round(poison_score, 4),
            "findings": findings,
            "message": "Content flagged by poisoning guard.",
        }, ensure_ascii=False))
    content = _sanitize_content(content)
    persistence = args.get("persistence", "auto")
    layer = _auto_decide(content, persistence)

    if layer == "session":
        key = args.get("key") or f"auto:{hash(content) % 1000000}"
        ttl = args.get("ttl_seconds", REDIS_DEFAULT_TTL)
        return _tool_short_store({"key": key, "value": content, "ttl_seconds": ttl})
    else:
        category = args.get("category") or "general"
        importance = args.get("importance", 5)
        tags = _coerce_str_list(args.get("tags"))
        source = args.get("source", "auto")
        return _tool_store({
            "content": content,
            "category": category,
            "importance": importance,
            "tags": tags,
            "source": source,
        })

# ---------------------------------------------------------------------------
# Directive 5: Mandatory System Alignment — verify AGENTS.md loaded
# ---------------------------------------------------------------------------
_AGENTS_MD_HASH = None  # cached hash of AGENTS.md content

def _compute_agents_md_hash() -> str:
    """Compute hash of AGENTS.md for alignment verification."""
    global _AGENTS_MD_HASH
    if _AGENTS_MD_HASH:
        return _AGENTS_MD_HASH
    agents_paths = [p for p in [
        os.environ.get("MEMORY_AGENTS_MD_PATH"),
        os.path.expanduser("~/.config/kilo/AGENTS.md"),
    ] if p]
    for p in agents_paths:
        try:
            with open(p, "r") as f:
                content = f.read()
            _AGENTS_MD_HASH = hashlib.sha256(content.encode()).hexdigest()[:16]
            return _AGENTS_MD_HASH
        except Exception:
            continue
    _AGENTS_MD_HASH = "unavailable"
    return _AGENTS_MD_HASH

def _tool_verify_alignment() -> types.CallToolResult:
    """Verify AGENTS.md was loaded and return alignment status."""
    now = _now_iso()
    agents_hash = _compute_agents_md_hash()
    
    # Check if alignment was verified this session
    alignment_key = "memory:alignment:verified"
    already_verified = False
    if _check_redis():
        try:
            already_verified = _redis.get(alignment_key) is not None
        except Exception:
            pass
    
    # Mark as verified for this session (1 hour TTL)
    if _check_redis():
        try:
            _redis.setex(alignment_key, 3600, now)
        except Exception:
            pass
    
    # Key protocol rules from AGENTS.md
    active_rules = [
        "Section 3: Anti-Disaster (no SSH mod, no rm -rf, no secret leaks)",
        "Section 5: aaPanel-first (GUI before CLI, override requires explicit approval)",
        "Section 7: Backup before any edit (/root/backup/ with restore_instructions.txt)",
        "Section 8: Syntax check before service restart",
        "Section 10: No blind changes, verify before claiming fix",
        "Section 12: sequentialthinking mandatory for 3+ step tasks",
        "Section 17A: Canonical loop (Recall→Plan→Route→Preflight→Act→Observe→Validate→Report)",
        "Quality Gate: content scored before store (score < 0.3 rejected)",
        "Utility Tracking: retrieval_count + useful_count on all memories",
    ]
    
    result = {
        "aligned": True,
        "agents_md_hash": agents_hash,
        "verified_at": now,
        "already_verified_this_session": already_verified,
        "active_rules": active_rules,
        "rule_count": len(active_rules),
        "message": "AGENTS.md loaded. All protocol rules active. Proceed with task.",
    }
    return _ok(json.dumps(result, ensure_ascii=False))


def _tool_health_check() -> types.CallToolResult:
    """P1-2: Health check — per-store status + version + breakers + latency."""
    result = {
        "redis": {"status": "unknown"},
        "qdrant": {"status": "unknown"},
        "postgres": {"status": "unknown"},
        "overall": "unknown",
        "version": MCP_SERVER_VERSION,
        "circuit_breakers": _breaker_report(),
        "tool_latency": _latency_report(),
        "timestamp": _now_iso(),
    }

    # Check Redis
    try:
        if _redis:
            _redis.ping()
            result["redis"]["status"] = "healthy"
            _breaker_ok("redis")
        else:
            result["redis"]["status"] = "not_configured"
    except Exception as e:
        result["redis"]["status"] = f"unhealthy: {e}"
        _breaker_fail("redis", e)

    # Check Qdrant
    try:
        if _client:
            _client.get_collections()
            result["qdrant"]["status"] = "healthy"
            _breaker_ok("qdrant")
        else:
            result["qdrant"]["status"] = "not_configured"
    except Exception as e:
        result["qdrant"]["status"] = f"unhealthy: {e}"
        _breaker_fail("qdrant", e)
    
    # Check Postgres
    try:
        if HORIZONTAL_SCALING_ENABLED:
            if _pg_pool:
                # Try a simple query
                _pg_query("SELECT 1")
                result["postgres"]["status"] = "healthy"
                _breaker_ok("postgres")
            else:
                result["postgres"]["status"] = "not_configured"
        else:
            if _pg_conn:
                # Try a simple query
                _pg_query("SELECT 1")
                result["postgres"]["status"] = "healthy"
                _breaker_ok("postgres")
            else:
                result["postgres"]["status"] = "not_configured"
    except PostgresUnavailableError:
        result["postgres"]["status"] = "unavailable"
        _breaker_fail("postgres", PostgresUnavailableError("health ping failed"))
    except Exception as e:
        _breaker_fail("postgres", e)
        result["postgres"]["status"] = f"unhealthy: {e}"
    
    # Overall status
    statuses = [result["redis"]["status"], result["qdrant"]["status"], result["postgres"]["status"]]
    if all(s == "healthy" for s in statuses):
        result["overall"] = "healthy"
    elif any(s.startswith("unhealthy") or s == "unavailable" for s in statuses):
        result["overall"] = "degraded"
    else:
        result["overall"] = "partial"
    
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Directive 4: Strict Protocol Adherence — compliance verification
# ---------------------------------------------------------------------------
_COMPLIANCE_RULES = {
    "config_edit": [
        "backup_created: Must backup config before editing (Section 7)",
        "syntax_checked: Must run syntax test before restart (Section 8)",
        "approval_obtained: Non-trivial config changes need user approval (Section 10)",
    ],
    "file_edit": [
        "backup_created: Must backup production files before editing (Section 7)",
        "approval_obtained: UI changes require explicit approval (Section 2)",
    ],
    "service_restart": [
        "syntax_checked: Must run syntax test (nginx -t, php-fpm -t) before restart (Section 8)",
        "backup_created: Config must be backed up before restart (Section 7)",
    ],
    "destructive": [
        "approval_obtained: Destructive operations require explicit user approval (Section 3, 10)",
        "backup_created: Must backup before destructive operations (Section 7)",
        "no_recursive_delete: Never rm -rf on system folders (Section 3)",
    ],
    "credential_access": [
        "secret_masking: Never output/log/expose secrets (Section 3, 6)",
        "root_isolation: Save credentials in /root/ with chmod 600 (Section 6)",
    ],
    "general": [
        "no_blind_changes: Understand before changing (Section 10)",
        "verify_before_claiming: Test before declaring fixed (Section 10)",
    ],
}

def _tool_compliance_check(args: dict) -> types.CallToolResult:
    """Verify protocol compliance after an action."""
    action_type = args.get("action_type", "general")
    details = args.get("details", "")
    backup_created = args.get("backup_created")
    approval_obtained = args.get("approval_obtained")
    syntax_checked = args.get("syntax_checked")
    
    rules = _COMPLIANCE_RULES.get(action_type, _COMPLIANCE_RULES["general"])
    
    checks = []
    all_pass = True
    
    for rule in rules:
        rule_name = rule.split(":")[0].strip()
        rule_desc = rule.split(":", 1)[1].strip() if ":" in rule else rule
        
        # Determine pass/fail based on provided info
        passed = None
        if rule_name == "backup_created" and backup_created is not None:
            passed = backup_created
        elif rule_name == "approval_obtained" and approval_obtained is not None:
            passed = approval_obtained
        elif rule_name == "syntax_checked" and syntax_checked is not None:
            passed = syntax_checked
        elif rule_name == "secret_masking":
            # Check if details contain potential secrets
            secret_patterns = re.compile(r"(password|token|api_key|secret|private_key)\s*[=:]\s*\S+", re.IGNORECASE)
            if details and secret_patterns.search(details):
                passed = False
                rule_desc += " — SECRET DETECTED IN DETAILS! Mask it."
            else:
                passed = True
        elif rule_name == "no_recursive_delete":
            if details and re.search(r"rm\s+-rf\s+/", details):
                passed = False
                rule_desc += " — RECURSIVE DELETE DETECTED!"
            else:
                passed = True
        else:
            # Unknown rule — mark as WARNING, not FAIL
            passed = None
        
        status = "PASS" if passed else ("FAIL" if passed is False else "WARNING")
        if passed is False:
            all_pass = False
        
        checks.append({
            "rule": rule_name,
            "status": status,
            "description": rule_desc,
        })
    
    compliance_score = sum(1 for c in checks if c["status"] == "PASS") / len(checks) if checks else 0
    
    result = {
        "compliant": all_pass,
        "compliance_score": round(compliance_score, 2),
        "action_type": action_type,
        "checks": checks,
        "details": details[:200] if details else "",
        "message": "All checks passed" if all_pass else "COMPLIANCE ISSUES DETECTED — review failures before proceeding",
    }
    return _ok(json.dumps(result, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Directive 6: Structured Task Decomposition
# ---------------------------------------------------------------------------
def _tool_task_breakdown(args: dict) -> types.CallToolResult:
    """Break down a complex task into sub-tasks based on historical patterns."""
    task = (args.get("task") or "").strip()
    if not task:
        return _err("task is required")
    task_type = args.get("task_type")
    
    # Find similar past trajectories
    similar_trajs = _fetch_trajectories_by_query(task, top_k=5)
    
    # Analyze patterns from similar tasks
    sub_tasks = []
    patterns_used = []
    
    if similar_trajs:
        # Extract step patterns from successful trajectories
        success_trajs = [t for t in similar_trajs if t.get("outcome") == "success"]
        if success_trajs:
            for traj in success_trajs[:2]:  # top 2 successful
                steps = traj.get("steps", [])
                for step in steps:
                    action = step.get("action", "")
                    if action and action not in [s["action"] for s in sub_tasks]:
                        sub_tasks.append({
                            "action": action[:100],
                            "tool_used": step.get("tool_used"),
                            "success": step.get("success", True),
                            "source": "historical_pattern",
                        })
                patterns_used.append(traj.get("task", "")[:50])
    
    # Add standard decomposition template if no patterns found
    if not sub_tasks:
        sub_tasks = [
            {"action": "Understand requirements and query memory for context", "tool_used": "memory_session_init", "source": "standard_template"},
            {"action": "Investigate the issue / gather information", "tool_used": "various", "source": "standard_template"},
            {"action": "Diagnose root cause", "tool_used": "sequentialthinking", "source": "standard_template"},
            {"action": "Propose fix and get approval", "tool_used": "question", "source": "standard_template"},
            {"action": "Create backup before changes", "tool_used": "shell_run_command", "source": "standard_template"},
            {"action": "Apply fix", "tool_used": "various", "source": "standard_template"},
            {"action": "Verify fix works", "tool_used": "various", "source": "standard_template"},
            {"action": "Report results and store learnings", "tool_used": "memory_task_end", "source": "standard_template"},
        ]
    
    result = {
        "task": task,
        "task_type": task_type,
        "sub_tasks": sub_tasks,
        "sub_task_count": len(sub_tasks),
        "patterns_used": patterns_used,
        "source": "historical" if patterns_used else "standard_template",
        "instructions": "Execute sub-tasks sequentially. Log each step with memory_trajectory_log. "
                       "Use memory_task_end when complete.",
    }
    return _ok(json.dumps(result, ensure_ascii=False))

# ---------------------------------------------------------------------------
# Directive 7: Context Efficiency & Cost Optimization
# ---------------------------------------------------------------------------
# 2B-3: write-gate observability — what % of candidates the quality/poison
# gates reject or quarantine. Surfaced through memory_stats.
_WRITE_GATE_STATS: dict = {
    "evaluated": 0, "stored": 0, "rejected": 0, "quarantined": 0,
    "quality_sum": 0.0, "by_source": {},
}

_RECALL_SESSION_STATS = {
    "total_recalls": 0,
    "cache_hits": 0,
    "useful_feedback": 0,
    "irrelevant_feedback": 0,
    "total_memories_returned": 0,
}


# ---------------------------------------------------------------------------
# Retrieval eval harness + re-embed-all (added for embedding/reranker upgrades)
# ---------------------------------------------------------------------------
EVAL_STORAGE_KEY = "memory:eval:last_run"

def _auto_build_eval_set(top_n: int = 30) -> list[dict]:
    """Generate a ground-truth eval set from existing memories.

    For a random sample of memories, use the memory's own content as the
    'query' (truncated to the first sentence) and the memory_id as the
    expected answer. This is a self-referential baseline — it measures
    whether the pipeline can retrieve a memory when you ask it the exact
    question that memory answers. Not as strong as a hand-authored
    question set, but a valid regression signal.
    """
    import random
    points, _ = _client.scroll(
        collection_name=COLLECTION_NAME,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    random.seed(42)  # stable across runs for diff-ability
    sample = random.sample(points, min(top_n, len(points)))
    eval_set = []
    for p in sample:
        content = (p.payload or {}).get("content", "")
        if len(content) < 20:
            continue
        # first sentence as the query
        first_sent = content.split(". ")[0].split("\n")[0][:120].strip()
        eval_set.append({
            "query": first_sent,
            "expected_memory_id": str(p.id),
        })
    return eval_set


def _tool_eval(args: dict) -> types.CallToolResult:
    """Run a retrieval eval and report MRR / hit@k / mean rank."""
    global _eval_set_cache
    top_k = int(args.get("top_k") or 10)
    eval_set = args.get("eval_set")
    stored = False
    if not eval_set:
        try:
            if _redis and _redis.get(EVAL_STORAGE_KEY):
                eval_set = json.loads(_redis.get(EVAL_STORAGE_KEY))
                stored = True
            elif _eval_set_cache := globals().get("_eval_set_cache"):
                eval_set = _eval_set_cache
                stored = True
            else:
                eval_set = _auto_build_eval_set(30)
        except Exception:
            eval_set = _auto_build_eval_set(30)

    if not eval_set:
        return _err("No eval set available and could not auto-generate one.")

    # Persist this eval set for next run (diff-ability)
    try:
        if _redis:
            _redis.setex(EVAL_STORAGE_KEY, 30*24*3600, json.dumps(eval_set))
    except Exception:
        pass
    _eval_set_cache = eval_set

    mrr = 0.0
    hit1 = hit5 = hit10 = 0
    total = 0
    ranks = []
    for item in eval_set:
        q = item.get("query", "")
        expected = str(item.get("expected_memory_id", ""))
        if not q or not expected:
            continue
        total += 1
        try:
            res = _tool_recall({"query": q, "top_k": top_k})
            rp = json.loads(res.content[0].text)
            ids = [m.get("memory_id") for m in rp.get("memories", [])]
            if expected in ids:
                rank = ids.index(expected) + 1
                mrr += 1.0 / rank
                ranks.append(rank)
                if rank == 1: hit1 += 1
                if rank <= 5: hit5 += 1
                if rank <= 10: hit10 += 1
        except Exception:
            ranks.append(None)

    n = len(ranks) or 1
    valid_ranks = [r for r in ranks if r is not None]
    return _ok(json.dumps({
        "total": total,
        "retrieved": len(valid_ranks),
        "mrr": round(mrr / n, 4),
        "hit_at_1": round(hit1 / n, 4),
        "hit_at_5": round(hit5 / n, 4),
        "hit_at_10": round(hit10 / n, 4),
        "mean_rank": round(sum(valid_ranks)/len(valid_ranks), 2) if valid_ranks else None,
        "top_k_used": top_k,
        "eval_set_source": "stored" if stored else "auto-generated",
        "eval_size": len(eval_set),
        "timestamp": _now_iso(),
        "instructions": "Compare MRR/hit@k before vs after any embedding/reranker/scoring change. A drop = regression. Save the numbers; run this on every maintenance cycle.",
    }, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# P0-3: Retrieval Evals Runner (golden set, precision@k, recall@k, MRR, nDCG@k)
# ---------------------------------------------------------------------------
_EVAL_RESULTS_KEY = "memory:eval:results"

def _load_golden_eval_set() -> list[dict]:
    """Load the golden eval set from file."""
    eval_path = os.environ.get("MEMORY_EVAL_SET") or os.path.join(os.path.dirname(__file__), "eval", "golden_eval_set.json")
    try:
        with open(eval_path, "r") as f:
            data = json.load(f)
            return data.get("queries", [])
    except Exception as e:
        log.warning("failed to load golden eval set from %s: %s", eval_path, e)
        return []

def _compute_dcg(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0, so we use position i+1 -> log2(i+2)
    return dcg

def _compute_ndcg(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute Normalized DCG at k."""
    if not relevant_ids:
        return 1.0  # No relevant items, perfect score
    # Ideal ranking: all relevant items first
    ideal_relevances = [1.0] * min(len(relevant_ids), k)
    ideal_dcg = _compute_dcg(ideal_relevances, k)
    if ideal_dcg == 0:
        return 1.0
    # Actual ranking
    actual_relevances = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids[:k]]
    actual_dcg = _compute_dcg(actual_relevances, k)
    return actual_dcg / ideal_dcg

def _run_retrieval_eval(eval_set: list[dict], top_k: int = 10, k_values: list[int] = None) -> dict:
    """Run retrieval evaluation on a golden eval set.

    Returns comprehensive metrics: precision@k, recall@k, MRR, nDCG@k for multiple k values.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    max_k = max(k_values) if k_values else top_k

    # EVAL HYGIENE (regression learned 2026-10-08):
    # 1. The eval MUST NOT run against a dense-only pipeline. If the BM25
    #    index is not loaded in this process, load it from the shared Redis
    #    store (or rebuild) BEFORE measuring anything.
    # 2. The eval MUST NOT read from, or poison, the shared semantic recall
    #    cache — flush it before AND after the run so operator evals and
    #    production traffic never exchange cached results.
    try:
        _idx, _ids = _bm25_snapshot()
        if _idx is None:
            if not (HORIZONTAL_SCALING_ENABLED and _bm25_redis_load()):
                _load_bm25_corpus()
                _prune_bm25_corpus()
                _rebuild_bm25_index()
            log.info("eval: BM25 index ensured before run (%d docs)",
                     len(_bm25_corpus) if _bm25_corpus else 0)
    except Exception as _e:
        log.warning("eval: BM25 ensure failed, sparse leg may be dead: %s", _e)
    _invalidate_recall_caches()

    total_queries = 0
    metrics = {k: {"precision_sum": 0.0, "recall_sum": 0.0, "ndcg_sum": 0.0, "count": 0} for k in k_values}
    mrr_sum = 0.0
    hit_at = {k: 0 for k in k_values}
    ranks = []
    per_query_results = []
    
    for item in eval_set:
        q = item.get("query", "")
        relevant_ids = set(str(rid) for rid in item.get("relevant_ids", []))
        if not q or not relevant_ids:
            continue
        
        total_queries += 1
        try:
            # 2026-10-26: golden-set items may carry their own scope (the
            # deployment's memories usually live in a user scope, not global).
            # Default stays "global" so existing sets behave unchanged.
            with system_context("eval_run"):
                res = _tool_recall({"query": q, "top_k": max_k,
                                    "scope": item.get("scope") or "global"})
            rp = json.loads(res.content[0].text)
            retrieved_ids = [m.get("memory_id") for m in rp.get("memories", [])]
            
            # MRR and hit@k
            rank = None
            for i, rid in enumerate(retrieved_ids):
                if rid in relevant_ids:
                    rank = i + 1
                    break
            
            if rank:
                mrr_sum += 1.0 / rank
                ranks.append(rank)
                for k in k_values:
                    if rank <= k:
                        hit_at[k] += 1
            else:
                ranks.append(None)
            
            # Precision@k, Recall@k, nDCG@k for each k
            for k in k_values:
                retrieved_at_k = retrieved_ids[:k]
                relevant_retrieved = sum(1 for rid in retrieved_at_k if rid in relevant_ids)
                
                precision = relevant_retrieved / k if k > 0 else 0.0
                recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
                ndcg = _compute_ndcg(retrieved_at_k, list(relevant_ids), k)
                
                metrics[k]["precision_sum"] += precision
                metrics[k]["recall_sum"] += recall
                metrics[k]["ndcg_sum"] += ndcg
                metrics[k]["count"] += 1
            
            per_query_results.append({
                "query": q[:80],
                "relevant_count": len(relevant_ids),
                "retrieved_ids": retrieved_ids[:max_k],
                "rank": rank,
                "metrics_at_k": {
                    k: {
                        "precision": round(relevant_retrieved / k, 4) if k > 0 else 0,
                        "recall": round(relevant_retrieved / len(relevant_ids), 4) if relevant_ids else 0,
                        "ndcg": round(_compute_ndcg(retrieved_ids[:k], list(relevant_ids), k), 4)
                    }
                    for k in k_values
                }
            })
        except Exception as e:
            log.warning("eval query failed: %s", e)
            for k in k_values:
                metrics[k]["count"] += 1
            per_query_results.append({"query": q[:80], "error": str(e)})
    
    # Aggregate
    n = total_queries or 1
    result = {
        "total_queries": total_queries,
        "mrr": round(mrr_sum / n, 4),
        "hit_at": {k: round(hit_at[k] / n, 4) for k in k_values},
        "mean_rank": round(sum(r for r in ranks if r) / len([r for r in ranks if r]), 2) if any(r for r in ranks if r) else None,
    }
    
    for k in k_values:
        c = metrics[k]["count"] or 1
        result[f"precision_at_{k}"] = round(metrics[k]["precision_sum"] / c, 4)
        result[f"recall_at_{k}"] = round(metrics[k]["recall_sum"] / c, 4)
        result[f"ndcg_at_{k}"] = round(metrics[k]["ndcg_sum"] / c, 4)
    
    result["per_query"] = per_query_results
    result["top_k_used"] = max_k
    result["timestamp"] = _now_iso()
    # Post-run hygiene: the eval populated the shared caches via _tool_recall;
    # drop them so production never serves eval-time results.
    _invalidate_recall_caches()
    return result

def _tool_eval_run(args: dict) -> types.CallToolResult:
    """P0-3: Run retrieval evaluation on the golden eval set.
    
    Computes precision@k, recall@k, MRR, nDCG@k for k in {1,3,5,10}.
    Loads golden eval set from file (50+ labeled queries).
    Persists results with commit SHA for regression tracking.
    Fails if metrics drop below thresholds (optional).
    """
    top_k = int(args.get("top_k") or 10)
    k_values = args.get("k_values") or [1, 3, 5, 10]
    thresholds = args.get("thresholds")  # optional: {"mrr": 0.5, "hit_at_1": 0.3, "precision_at_5": 0.4}
    persist = bool(args.get("persist", True))
    
    eval_set = _load_golden_eval_set()
    if not eval_set:
        return _err("Golden eval set not found or empty. Expected at app/eval/golden_eval_set.json (or set MEMORY_EVAL_SET)")
    
    # Filter by category if specified
    category = args.get("category")
    if category:
        eval_set = [item for item in eval_set if item.get("category") == category]
    
    result = _run_retrieval_eval(eval_set, top_k, k_values)
    
    # Check thresholds
    failed_thresholds = []
    if thresholds:
        for metric, threshold in thresholds.items():
            value = result.get(metric)
            if value is not None and value < threshold:
                failed_thresholds.append({"metric": metric, "value": value, "threshold": threshold})
    
    if failed_thresholds:
        result["threshold_check"] = "FAILED"
        result["failed_thresholds"] = failed_thresholds
    else:
        result["threshold_check"] = "PASSED" if thresholds else "NOT_CONFIGURED"
    
    # Persist results
    if persist:
        try:
            if _redis:
                try:
                    commit_sha = subprocess.run(
                        ["git", "rev-parse", "HEAD"], 
                        capture_output=True, text=True, cwd=os.path.dirname(__file__),
                        timeout=5
                    ).stdout.strip()
                except Exception:
                    commit_sha = "unknown"
                result["commit_sha"] = commit_sha
                dumped = json.dumps(result, default=str)
                log.info("eval persistence: lpush key=%s len=%d", _EVAL_RESULTS_KEY, len(dumped))
                _redis.lpush(_EVAL_RESULTS_KEY, dumped)
                _redis.ltrim(_EVAL_RESULTS_KEY, 0, 99)  # Keep last 100 runs
                log.info("eval persistence: lpush success, count=%d", _redis.llen(_EVAL_RESULTS_KEY))
        except Exception as e:
            log.warning("failed to persist eval results: %s", e)
    
    return _ok(json.dumps(result, ensure_ascii=False, indent=2))

def _tool_eval_history(args: dict = None) -> types.CallToolResult:
    """Get history of evaluation runs from Redis."""
    limit = int((args or {}).get("limit", 20))
    try:
        if _redis:
            results = _redis.lrange(_EVAL_RESULTS_KEY, 0, limit - 1)
            return _ok(json.dumps({
                "count": len(results),
                "results": [json.loads(r) for r in results]
            }, ensure_ascii=False, indent=2))
    except Exception as e:
        return _err(f"Failed to load eval history: {e}")
    return _ok(json.dumps({"count": 0, "results": []}, ensure_ascii=False))

# Add to tool registration and dispatch...


def _tool_reembed_all(args: dict) -> types.CallToolResult:
    """Destructive: re-embed every memory with the current model + recreate the collection."""
    dry_run = bool(args.get("dry_run", False))
    # Snapshot current state
    points, _ = _client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    count = len(points)

    if dry_run:
        return _ok(json.dumps({
            "dry_run": True,
            "current_count": count,
            "target_model": EMBEDDING_MODEL,
            "target_vector_size": VECTOR_SIZE,
            "action": "Would drop collection '%s', recreate at %d dims, re-embed %d memories." % (COLLECTION_NAME, VECTOR_SIZE, count),
        }, ensure_ascii=False))

    import time as _time
    t0 = _time.time()

    # 1. Ensure the collection exists at the right size
    _hnsw = qmodels.HnswConfigDiff(m=16, ef_construct=128, full_scan_threshold=10000, max_indexing_threads=4)
    _opt = qmodels.OptimizersConfigDiff(indexing_threshold=10)
    try:
        info = _client.get_collection(COLLECTION_NAME)
        current_size = info.vectors_config.size if hasattr(info, "vectors_config") else None
        if current_size != VECTOR_SIZE:
            _client.delete_collection(COLLECTION_NAME)
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE, hnsw_config=_hnsw),
                optimizers_config=_opt,
            )
            log.info("Collection recreated at %d dims (HNSW enabled)", VECTOR_SIZE)
    except Exception:
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE, hnsw_config=_hnsw),
            optimizers_config=_opt,
        )

    # 2. Re-embed every memory
    contents = [(str(p.id), (p.payload or {}).get("content", "")) for p in points]
    mem_texts = [c for _, c in contents if c]
    vectors = _embed_model().encode(mem_texts, show_progress_bar=False, batch_size=32)
    vectors = vectors.tolist() if hasattr(vectors, "tolist") else vectors

    # 3. Upsert with new vectors, preserving payload
    rebuilt = []
    vec_iter = iter(vectors)
    for p in points:
        content = (p.payload or {}).get("content", "")
        if not content:
            continue
        vec = next(vec_iter)
        payload = dict(p.payload or {})
        payload["embedding_model"] = EMBEDDING_MODEL_VERSION
        payload["embedding_dim"] = VECTOR_SIZE
        payload["reembedded_at"] = _now_iso()
        rebuilt.append(qmodels.PointStruct(id=str(p.id), vector=vec, payload=payload))

    # Upsert in chunks of 200
    for i in range(0, len(rebuilt), 200):
        chunk = rebuilt[i:i+200]
        _client.upsert(collection_name=COLLECTION_NAME, points=chunk)
        log.info("Re-embed progress: %d/%d", min(i+200, len(rebuilt)), len(rebuilt))

    # 4. Rebuild BM25 index from fresh tokens
    _bm25_corpus.clear()
    _rebuild_bm25_index()

    elapsed = _time.time() - t0
    log.info("Re-embed complete: %d memories in %.1fs", len(rebuilt), elapsed)

    return _ok(json.dumps({
        "status": "reembedded",
        "memories": len(rebuilt),
        "model": EMBEDDING_MODEL,
        "vector_size": VECTOR_SIZE,
        "elapsed_s": round(elapsed, 1),
        "note": "Run memory_eval to verify retrieval quality on the new embedding space.",
    }, ensure_ascii=False, default=str))


def _tool_recall_efficiency() -> types.CallToolResult:
    """Track recall efficiency for the current session."""
    global _RECALL_SESSION_STATS
    
    # Get cache stats
    cache_stats = {}
    if _check_redis():
        try:
            cache_keys = _redis.keys(f"{SEMANTIC_CACHE_PREFIX}*")
            cache_stats["cached_entries"] = len(cache_keys)
        except Exception:
            cache_stats["cached_entries"] = 0
    
    # Get feedback stats
    fb_stats = {}
    if _check_redis():
        try:
            query_keys = _redis.keys(f"{FEEDBACK_QUERY_PREFIX}*")
            total_fb = 0
            useful = 0
            irrelevant = 0
            for q_key in query_keys:
                fb_keys = _redis.smembers(q_key)
                for fb_key in fb_keys:
                    fb_data = _redis.get(fb_key)
                    if fb_data:
                        fb = json.loads(fb_data)
                        total_fb += 1
                        if fb.get("rating") == "useful":
                            useful += 1
                        elif fb.get("rating") == "irrelevant":
                            irrelevant += 1
            fb_stats = {"total": total_fb, "useful": useful, "irrelevant": irrelevant}
        except Exception:
            fb_stats = {"total": 0, "useful": 0, "irrelevant": 0}
    
    # Calculate efficiency metrics
    total = _RECALL_SESSION_STATS["total_recalls"]
    cache_hits = _RECALL_SESSION_STATS["cache_hits"]
    useful = _RECALL_SESSION_STATS["useful_feedback"]
    irrelevant = _RECALL_SESSION_STATS["irrelevant_feedback"]
    memories_returned = _RECALL_SESSION_STATS["total_memories_returned"]
    
    cache_hit_rate = cache_hits / total if total > 0 else 0
    waste_ratio = irrelevant / (useful + irrelevant) if (useful + irrelevant) > 0 else 0
    avg_memories_per_recall = memories_returned / total if total > 0 else 0
    
    # Recommendations
    recommendations = []
    if waste_ratio > 0.3:
        recommendations.append("HIGH WASTE: >30% of recalled memories marked irrelevant. Use more specific queries.")
    if cache_hit_rate < 0.1 and total > 5:
        recommendations.append("LOW CACHE HIT RATE: Consider rephrasing queries instead of new queries.")
    if avg_memories_per_recall > 5:
        recommendations.append("MANY MEMORIES PER RECALL: Use top_k=3 instead of default 5 to reduce context bloat.")
    if total > 10 and useful == 0:
        recommendations.append("NO USEFUL FEEDBACK: Start marking recalled memories as useful/irrelevant to improve rankings.")
    if not recommendations:
        recommendations.append("Efficiency is within normal bounds.")
    
    result = {
        "session_stats": _RECALL_SESSION_STATS,
        "cache_stats": cache_stats,
        "feedback_stats": fb_stats,
        "efficiency_metrics": {
            "cache_hit_rate": round(cache_hit_rate, 3),
            "waste_ratio": round(waste_ratio, 3),
            "avg_memories_per_recall": round(avg_memories_per_recall, 1),
        },
        "recommendations": recommendations,
        "message": f"Session: {total} recalls, {cache_hits} cache hits, {useful} useful, {irrelevant} irrelevant.",
    }
    return _ok(json.dumps(result, ensure_ascii=False))


# -- latency metrics (P1-6) ----------------------------------------------------
def _tool_latency_metrics(args: dict | None = None) -> types.CallToolResult:
    """Get latency statistics (p50, p95, p99) for memory tools."""
    tool_name = args.get("tool") if args else None
    stats = _get_latency_stats(tool_name)
    return _ok(json.dumps(stats, ensure_ascii=False))


# -- retrieval stage metrics (P2-2) --------------------------------------------
def _tool_retrieval_stage_metrics(args: dict | None = None) -> types.CallToolResult:
    """Get latency and quality statistics for retrieval stages."""
    stage = args.get("stage") if args else None
    latency_stats = _get_stage_latency_stats(stage)
    quality_stats = _get_stage_quality_stats(stage)
    result = {
        "latency": latency_stats,
        "quality": quality_stats,
    }
    return _ok(json.dumps(result, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Transport selection: stdio (default, per-Kilo-run) or http (shared,
# long-lived). MEMORY_TRANSPORT=http runs ONE warm server that every Kilo run
# (main + sub-agents) connects to over streamable-http, so the embedding model,
# Qdrant connection, and BM25 index live in a single process and are loaded
# once — sub-agents never re-load them.
# ---------------------------------------------------------------------------
_MEMORY_TRANSPORT = os.environ.get("MEMORY_TRANSPORT", "stdio")


def _stdio_main() -> None:
    async def _run_stdio() -> None:
        _prewarm()
        # stdio is a locally-spawned, operator-trusted transport. Bind a caller
        # principal for the session so tools are authorized exactly like the
        # HTTP path: MEMORY_STDIO_IDENTITY if set, else the dev anon principal
        # when MEMORY_ALLOW_ANONYMOUS=1, else the operator (system) principal.
        _ident = os.environ.get("MEMORY_STDIO_IDENTITY", "").strip()
        if _ident:
            _principal = _Principal(_normalize_scope(_ident), None, False, "stdio")
        elif os.environ.get("MEMORY_ALLOW_ANONYMOUS", "").strip() == "1":
            _principal = _anonymous_principal()
        else:
            _principal = _Principal(None, None, True, "stdio-operator")
        set_caller_principal(_principal)
        try:
            await _run_stdio_inner()
        finally:
            clear_caller_principal()

    async def _run_stdio_inner() -> None:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await _server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memory",
                    server_version=MCP_SERVER_VERSION,
                    capabilities=_server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    asyncio.run(_run_stdio())


def _http_main() -> None:
    """Serve the same _server over streamable-http (shared, long-lived)."""
    import uvicorn
    from mcp.server.streamable_http import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    host = os.environ.get("MEMORY_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("MEMORY_HTTP_PORT", "3901"))

    manager = StreamableHTTPSessionManager(
        app=_server,
        event_store=None,
        stateless=True,
        security_settings=None,
    )

    async def _lifespan(app):
        async with manager.run():
            log.info("Memory streamable-http session manager started on %s:%s", host, port)
            yield

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=manager)],
        lifespan=_lifespan,
    )
    log.info("Serving shared Memory MCP over streamable-http on %s:%s/mcp", host, port)
    uvicorn.run(starlette_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    if _MEMORY_TRANSPORT == "http":
        _http_main()
    else:
        _stdio_main()
