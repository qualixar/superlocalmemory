# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
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
import sys

# Force CPU BEFORE any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEM_LIMIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE"] = "cpu"

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


def _handle_recall(query: str, limit: int) -> dict:
    engine = _get_engine()
    response = engine.recall(query, limit=limit)

    # Batch-fetch original memory text for all results
    memory_ids = list({r.fact.memory_id for r in response.results[:limit] if r.fact.memory_id})
    memory_map = engine._db.get_memory_content_batch(memory_ids) if memory_ids else {}

    results = []
    for r in response.results[:limit]:
        results.append({
            "fact_id": r.fact.fact_id,
            "memory_id": r.fact.memory_id,
            "content": r.fact.content[:300],
            "source_content": memory_map.get(r.fact.memory_id, ""),
            "score": round(r.score, 4),
            "confidence": round(r.confidence, 4),
            "trust_score": round(r.trust_score, 4),
            "channel_scores": {
                k: round(v, 4) for k, v in (r.channel_scores or {}).items()
            },
        })
    return {
        "ok": True,
        "query": query,
        "query_type": response.query_type,
        "result_count": len(results),
        "retrieval_time_ms": round(response.retrieval_time_ms, 1),
        "results": results,
    }


def _handle_store(content: str, metadata: dict) -> dict:
    engine = _get_engine()
    session_id = metadata.pop("session_id", "")
    fact_ids = engine.store(content, session_id=session_id, metadata=metadata)
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

        try:
            if cmd == "recall":
                result = _handle_recall(req.get("query", ""), req.get("limit", 10))
                _respond(result)
            elif cmd == "store":
                result = _handle_store(req.get("content", ""), req.get("metadata", {}))
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


def _respond(data: dict) -> None:
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _worker_main()
