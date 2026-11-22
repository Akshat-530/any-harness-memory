# Benchmarks

`memory_bench.py` is a reproducible accuracy harness (recall@k / MRR / hit@1)
that runs against the live MCP endpoint.

```bash
# quick self-test (built-in sample)
docker run --rm --network memory-mcp_memory --entrypoint python \
  -v "$PWD/benchmarks/memory_bench.py:/b.py:ro" memory:latest /b.py --self-test

# your dataset
python benchmarks/memory_bench.py dataset.json --url http://127.0.0.1:4000/mcp --k 10
```

Dataset: `{"memories":[{"content":..}], "queries":[{"query":..,"expect":["substring"]}]}`.

## Official benchmarks (LoCoMo / LongMemEval / BEAM)
Those need their datasets and an LLM judge. Adapter outline:
1. Ingest each conversation turn with `memory_turn_ingest` (or `memory_store`).
2. For each QA, call `memory_recall(as_of=<session time>)` if temporal.
3. Grade with the benchmark's judge; report the same metric.

Point this repo at LoCoMo and you get a number directly comparable to Mem0/Zep —
which is the one thing competitors publish and this project still needs to.