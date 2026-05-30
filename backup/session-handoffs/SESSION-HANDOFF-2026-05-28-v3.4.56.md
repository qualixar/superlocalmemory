# Session Handoff — 2026-05-28 — SLM v3.4.51 → v3.4.56

**Date:** 2026-05-27/28 (overnight session)
**Releases shipped:** 6 (v3.4.51, v3.4.52, v3.4.53, v3.4.54, v3.4.55, v3.4.56)
**Published:** GitHub (58f5876), PyPI 3.4.56, npm 3.4.56 — all live

---

## Release Chain

### v3.4.51 — Recency Intelligence
- FSRS v5 + Ebbinghaus exponential decay formula in engine.py
- `recency_boost = 0.8 + 0.3 * exp(-(ln(2)/S) * age_days)` where S = 30d base
- access_count stabilizes half-life up to 60d
- Fixed hardcoded `age_days: 0` in recall_pipeline.py v1 + v2
- Added `created_at` to pool recall protocol (unified_daemon + PoolFact)
- `max_age_days=30` age gate in session_init MCP tool
- `--max-age-days`, `--full`, `--json` CLI flags
- Reranker Python 3.14 crash detection

### v3.4.52 — Warm Memory, No Cold Starts
- `keep_alive: -1` on all Ollama embedding calls
- Eager async pre-warm at daemon boot
- `/health` reports `embedding_warm` flag
- Emergency FTS5 BM25 fallback with `degraded_mode` flag
- 4 covering indexes on graph_edges + association_edges (weight DESC)
- max_neighbors_per_node: 100 → 30 (GAM ICLR 2026)
- JSON control character sanitization in recall responses
- DaemonPoolProxy timeout 30s → 60s

### v3.4.53 — Parallel Channels
- Channels execute in parallel via ThreadPoolExecutor (EverMemOS pattern)
- Non-blocking reranker: try_lock, skip if busy → fallback to fusion scores
- `asyncio.to_thread` in daemon recall handler (prevents event loop blocking)
- Recall semaphore=3 for concurrency control
- Reranker subprocess timeout 180s → 15s
- Performance: 5-10s → 474-623ms sequential warm, 0% spikes concurrent

### v3.4.54 — 3-Mode Config System
- Three config files: mode_a.json, mode_b.json, mode_c.json
- `current_mode` pointer file
- `SLMConfig.switch_mode()` — full provider stack switches atomically
- Auto-migration from legacy config.json on first daemon boot
- Reranker stays ON for ALL modes
- Backward compatible — user customizations preserved per-mode

### v3.4.55 — Dashboard + Installer
- 7 new `/api/v3/` routes: mode switching, provider detection, embedding test
- install.sh interactive mode selection (A/B/C with descriptions)
- npm postinstall already uses `slm setup` for mode selection
- Provider auto-detection from environment keys

---

## Performance Benchmarks (v3.4.53, Mode B)

| Metric | Before | After |
|---|---|---|
| Sequential warm recall (all 6 channels) | 5-10s | 474-623ms |
| Concurrent 5-parallel recall | 90% spikes, 10-26s | 0% spikes, 1.8-2.4s |
| Health endpoint during recall | Dead | Always responsive |
| Spike rate | 45-90% | 0% |

---

## Files Modified Across All Releases

**Core:**
- `retrieval/engine.py` — parallel channels + FSRS formula
- `core/recall_pipeline.py` — age_days fix v1 + v2
- `core/config.py` — 3-mode system + migration + logger
- `core/ollama_embedder.py` — keep_alive=-1
- `retrieval/reranker.py` — non-blocking lock + crash detection + timeout 15s
- `retrieval/spreading_activation.py` — max_neighbors_per_node 100→30
- `server/unified_daemon.py` — asyncio.to_thread, semaphore, /api/v3/ routes, sanitize, covering indexes, migration
- `mcp/tools_active.py` — age gate, emergency FTS5, degraded_mode
- `mcp/_pool_adapter.py` — PoolFact.created_at
- `mcp/_daemon_proxy.py` — timeout 30s→60s
- `cli/commands.py` — mode switching, max_age_days
- `cli/main.py` — CLI flags

**Config:**
- `__init__.py`, `pyproject.toml`, `package.json` — version bumps
- `install.sh` — interactive mode selection
- `CHANGELOG.md` — entries for 3.4.51, 3.4.52

**Tests:**
- `test_mcp_pool_adapter.py` — fast flag updates
- `test_mcp_session_init_tool.py` — fast flag updates

---

## Environment Requirements

```bash
# ~/.zshrc additions
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_QUEUE=256
```

---

## Next Session Priorities

1. **Mode C end-to-end testing** — tested basic recall/remember, needs full QA with real API usage
2. **Mode A live testing** — SQLite-only mode not live tested
3. **Dashboard UI mode switching** — API routes exist, verify UI JS calls work end-to-end
4. **Reranker stability** — occasional 10s spikes during high concurrency (intermittent, pre-existing)
5. **Recall quality audit** — verify FSRS formula + age gate produce relevant memories, not random data
6. **RAM audit** — ensure ≤2GB total under concurrent load

---

## Quick Reference

```bash
# Install latest
pip3 install superlocalmemory==3.4.55 --break-system-packages && slm restart

# Switch modes
slm mode a  # Zero-Cloud
slm mode b  # Local AI (Ollama)
slm mode c  # Cloud Power (OpenRouter)

# Dashboard
slm dashboard  # Web UI at http://localhost:8765
```
