# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Subprocess recall worker — runs the full recall pipeline in isolation.

The dashboard/MCP main process NEVER imports torch, numpy, or the engine.
All heavy work (engine init, embedding, retrieval, reranking) happens here.

Protocol (JSON over stdin/stdout):
  Request:  {"cmd": "recall", "query": "...", "limit": 10}
  Response: {"ok": true, "results": [...], "query_type": "...", ...}

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import os
import signal
import sys

# Force CPU BEFORE any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEM_LIMIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE"] = "cpu"
# V3.4.37: Disable CoreML EP — uses 3-5GB on ARM64 Mac.
os.environ["ORT_DISABLE_COREML"] = "1"

# SIGTERM bridge: Docker/systemd send SIGTERM to stop processes.
# Without this, the worker ignores SIGTERM and becomes a zombie.
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _start_parent_watchdog() -> None:
    """Monitor parent process — self-terminate if parent dies.

    V3.4.24: Delegates to platform_utils.start_parent_watchdog().
    """
    from superlocalmemory.core.platform_utils import start_parent_watchdog
    start_parent_watchdog()

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.engine import MemoryEngine
        config = SLMConfig.load()
        _engine = MemoryEngine(config)
        _engine.initialize()
    return _engine


def _handle_recall(
    query: str, limit: int, session_id: str = "", fast: bool = False,
    include_global: bool | None = None, include_shared: bool | None = None,
) -> dict:
    engine = _get_engine()
    # v3.6.15 multi-scope: None flags let engine.recall resolve the configured
    # default (shared-off). The subprocess loads its own SLMConfig, so the
    # resolution is identical to the in-process / daemon paths.
    response = engine.recall(
        query, limit=limit, session_id=session_id or None, fast=bool(fast),
        include_global=include_global, include_shared=include_shared,
    )

    # Batch-fetch original memory text for all results
    memory_ids = list({r.fact.memory_id for r in response.results[:limit] if r.fact.memory_id})
    memory_map = engine._db.get_memory_content_batch(memory_ids) if memory_ids else {}

    # v3.6.6: same shared chokepoint as the daemon HTTP route + CLI fallback,
    # so the MCP WorkerPool subprocess path returns identical budgeted output.
    from superlocalmemory.server.recall_serializer import serialize_recall_response
    _rc = getattr(engine._config, "retrieval", None)
    results, no_confident_match = serialize_recall_response(
        response,
        limit=limit,
        memory_map=memory_map,
        per_fact_max=getattr(_rc, "recall_per_fact_max_chars", 2400),
        total_max=getattr(_rc, "recall_total_max_chars", 12000),
    )
    return {
        "ok": True,
        "query": query,
        "query_type": response.query_type,
        "result_count": len(results),
        "retrieval_time_ms": round(response.retrieval_time_ms, 1),
        "channel_weights": {
            k: round(v, 3) for k, v in (response.channel_weights or {}).items()
        },
        "total_candidates": getattr(response, "total_candidates", 0),
        "results": results,
        "no_confident_match": no_confident_match,
    }


def _handle_store(content: str, metadata: dict) -> dict:
    engine = _get_engine()
    values = dict(metadata or {})
    session_id = str(values.pop("session_id", "") or "")
    scope = str(values.pop("scope", "personal") or "personal")
    shared_with = list(values.pop("shared_with", []) or [])
    idempotency_key = str(values.pop("idempotency_key", "") or "")
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )
    fact_ids = canonical_store(
        engine,
        content,
        source_type="mcp-offline-worker",
        trusted_actor_id=local_trusted_actor_id("recall-worker"),
        metadata=values,
        scope=scope,
        shared_with=shared_with,
        session_id=session_id,
        idempotency_key=idempotency_key,
    )

    # Generate and persist summary immediately after store (Mode A heuristic, B/C LLM)
    if fact_ids:
        try:
            from superlocalmemory.core.summarizer import Summarizer
            summarizer = Summarizer(engine._config)
            summary = summarizer.summarize_cluster([{"content": content}])
            if summary:
                # Get the memory_id from the first stored fact
                rows = engine._db.execute(
                    "SELECT memory_id FROM atomic_facts WHERE fact_id = ? LIMIT 1",
                    (fact_ids[0],),
                )
                if rows:
                    memory_id = dict(rows[0])["memory_id"]
                    engine._db.update_memory_summary(memory_id, summary)
        except Exception:
            pass  # Summary is non-critical

    return {"ok": True, "fact_ids": fact_ids, "count": len(fact_ids)}


def _handle_get_memory_facts(memory_id: str) -> dict:
    engine = _get_engine()
    pid = engine.profile_id
    # Get original memory content
    mem_map = engine._db.get_memory_content_batch([memory_id])
    original = mem_map.get(memory_id, "")
    # Get child facts
    facts = engine._db.get_facts_by_memory_id(memory_id, pid)
    fact_list = []
    for f in facts:
        fact_list.append({
            "fact_id": f.fact_id,
            "content": f.content,
            "fact_type": f.fact_type.value if hasattr(f.fact_type, 'value') else str(f.fact_type),
            "confidence": round(f.confidence, 3),
            "created_at": f.created_at,
        })
    return {
        "ok": True,
        "memory_id": memory_id,
        "original_content": original,
        "facts": fact_list,
        "fact_count": len(fact_list),
    }


def _handle_delete_memory(
    fact_id: str,
    source_agent_id: str = "system",
) -> dict:
    """Delete a fact after capability-derived authorization."""
    engine = _get_engine()
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import delete_fact_authorized

    return delete_fact_authorized(
        engine,
        fact_id,
        trusted_actor_id=local_trusted_actor_id("recall-worker"),
        source_agent_id=source_agent_id,
    )


def _handle_update_memory(
    fact_id: str,
    content: str,
    source_agent_id: str = "system",
) -> dict:
    """Update a fact after capability-derived authorization."""
    engine = _get_engine()
    from superlocalmemory.core.engine_ingestion import local_trusted_actor_id
    from superlocalmemory.core.mutations import update_fact_authorized

    return update_fact_authorized(
        engine,
        fact_id,
        content,
        trusted_actor_id=local_trusted_actor_id("recall-worker"),
        source_agent_id=source_agent_id,
    )


def _handle_summarize(texts: list[str], mode: str) -> dict:
    """Generate summary using heuristic (A) or LLM (B/C)."""
    from superlocalmemory.core.summarizer import Summarizer
    engine = _get_engine()
    summarizer = Summarizer(engine._config)
    summary = summarizer.summarize_cluster(
        [{"content": t} for t in texts],
    )
    return {"ok": True, "summary": summary}


def _handle_synthesize(query: str, facts: list[dict]) -> dict:
    """Generate synthesized answer from query + facts."""
    from superlocalmemory.core.summarizer import Summarizer
    engine = _get_engine()
    summarizer = Summarizer(engine._config)
    synthesis = summarizer.synthesize_answer(query, facts)
    return {"ok": True, "synthesis": synthesis}


def _handle_status() -> dict:
    engine = _get_engine()
    pid = engine.profile_id
    fact_count = engine._db.get_fact_count(pid)
    return {
        "ok": True,
        "mode": engine._config.mode.value,
        "profile": pid,
        "fact_count": fact_count,
    }


def _worker_main() -> None:
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    _start_parent_watchdog()
    from superlocalmemory.core.platform_utils import get_rss_mb

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "error": "Invalid JSON"})
            continue

        cmd = req.get("cmd", "")

        if cmd == "quit":
            break

        if cmd == "ping":
            _respond({"ok": True})
            continue

        if cmd == "warmup":
            # Pre-load engine + database + embeddings only.
            # V3.3.2: Do NOT run a dummy recall — it triggers the ONNX
            # cross-encoder export (~30s) which combined with engine init
            # exceeds the worker timeout. The cross-encoder loads lazily
            # in a background thread on the first real recall instead.
            try:
                engine = _get_engine()
                fact_count = engine._db.get_fact_count(engine._profile_id) if engine._db else 0
                _respond({"ok": True, "message": "Engine warm", "facts": fact_count})
            except Exception as exc:
                _respond({"ok": False, "error": f"Warmup failed: {exc}"})
            continue

        try:
            if cmd == "recall":
                result = _handle_recall(
                    req.get("query", ""), req.get("limit", 10),
                    req.get("session_id", ""), bool(req.get("fast", False)),
                    include_global=req.get("include_global"),
                    include_shared=req.get("include_shared"),
                )
                _respond(result)
            elif cmd == "store":
                result = _handle_store(req.get("content", ""), req.get("metadata", {}))
                _respond(result)
            elif cmd == "delete_memory":
                result = _handle_delete_memory(
                    req.get("fact_id", ""),
                    req.get("source_agent_id", req.get("agent_id", "system")),
                )
                _respond(result)
            elif cmd == "update_memory":
                result = _handle_update_memory(
                    req.get("fact_id", ""),
                    req.get("content", ""),
                    req.get("source_agent_id", req.get("agent_id", "system")),
                )
                _respond(result)
            elif cmd == "get_memory_facts":
                result = _handle_get_memory_facts(req.get("memory_id", ""))
                _respond(result)
            elif cmd == "summarize":
                result = _handle_summarize(req.get("texts", []), req.get("mode", "a"))
                _respond(result)
            elif cmd == "synthesize":
                result = _handle_synthesize(req.get("query", ""), req.get("facts", []))
                _respond(result)
            elif cmd == "status":
                _respond(_handle_status())
            else:
                _respond({"ok": False, "error": f"Unknown command: {cmd}"})
        except Exception as exc:
            _respond({"ok": False, "error": str(exc)})

        # V3.3.16: RSS watchdog — V3.4.24: cross-platform via platform_utils.
        rss_mb = get_rss_mb()
        if rss_mb > 0 and rss_mb > 1500:
            sys.exit(0)


def _respond(data: dict) -> None:
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        _worker_main()
    except KeyboardInterrupt:
        sys.exit(0)
