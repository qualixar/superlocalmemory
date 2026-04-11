# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the Elastic License 2.0 - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Subprocess embedding worker — isolates PyTorch memory from main process.

The main process (dashboard/MCP) stays at ~60 MB. All PyTorch/model memory
lives in this worker subprocess, which auto-kills after idle timeout.

Protocol (JSON over stdin/stdout):
  Request:  {"cmd": "embed",  "texts": ["hello"]}
  Response: {"ok": true, "vectors": [[0.1, ...]], "dim": 768}

  Request:  {"cmd": "ping"}
  Response: {"ok": true}

  Request:  {"cmd": "quit"}
  (worker exits)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading

# Force CPU BEFORE any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEM_LIMIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE"] = "cpu"
# V3.3.17: Disable CoreML EP for ONNX Runtime — uses 3-5GB on ARM64 Mac.
os.environ["ORT_DISABLE_COREML"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# SIGTERM bridge: Docker/systemd send SIGTERM to stop processes.
# Without this, the worker ignores SIGTERM and becomes a zombie.
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _start_parent_watchdog() -> None:
    """Monitor parent process — self-terminate if parent dies.

    Prevents orphaned workers that consume 500-800 MB each when the parent
    process crashes, is killed, or exits without cleanup.

    V3.3.7: Added after incident where orphaned workers consumed 33 GB.
    """
    parent_pid = os.getppid()

    def _watch() -> None:
        import time
        while True:
            time.sleep(5)
            try:
                os.kill(parent_pid, 0)
            except OSError:
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watchdog")
    t.start()


def _load_embedding_model(name: str) -> tuple:
    """Load embedding model. ONNX first (no memory leak), PyTorch fallback.

    V3.3.17: PyTorch SentenceTransformer on ARM64 Mac leaks memory —
    grows from 300MB to 17GB after ~200 encode calls. ONNX Runtime
    has no such issue. Same approach as CrossEncoder ONNX migration.

    Returns (model, backend_name) or (None, "").
    """
    from sentence_transformers import SentenceTransformer

    # Tier 1: ONNX (stable memory, ~200MB footprint)
    try:
        m = SentenceTransformer(name, backend="onnx", trust_remote_code=True)
        return m, "onnx"
    except Exception:
        pass

    # Tier 2: PyTorch CPU (stable at ~1.4GB after 100+ calls, verified)
    try:
        import torch
        with torch.inference_mode():
            m = SentenceTransformer(name, trust_remote_code=True, device="cpu")
        return m, "pytorch"
    except Exception:
        return None, ""


def _worker_main() -> None:
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    if sys.platform != "win32":
        _start_parent_watchdog()  # V3.3.7: self-terminate if parent dies

    import numpy as np

    model = None
    model_name = None
    dim = 0

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

        if cmd == "load":
            name = req.get("model_name", "nomic-ai/nomic-embed-text-v1.5")
            expected_dim = req.get("dimension", 768)
            model, active_backend = _load_embedding_model(name)
            if model is not None:
                dim = model.get_sentence_embedding_dimension()
                if dim != expected_dim:
                    _respond({"ok": False, "error": f"Dimension mismatch: {dim} != {expected_dim}"})
                    model = None
                    continue
                model_name = name
                _respond({"ok": True, "dim": dim, "model": name, "backend": active_backend})
            else:
                _respond({"ok": False, "error": "Model load failed"})
            continue

        if cmd == "embed":
            texts = req.get("texts", [])
            if not texts:
                _respond({"ok": False, "error": "No texts provided"})
                continue
            if model is None:
                name = req.get("model_name", "nomic-ai/nomic-embed-text-v1.5")
                model, active_backend = _load_embedding_model(name)
                if model is not None:
                    dim = model.get_sentence_embedding_dimension()
                    model_name = name
                else:
                    _respond({"ok": False, "error": "Model load failed"})
                    continue
            try:
                vecs = model.encode(texts, normalize_embeddings=True)
                if isinstance(vecs, np.ndarray) and vecs.ndim == 2:
                    result = [vecs[i].tolist() for i in range(vecs.shape[0])]
                else:
                    result = [np.asarray(v, dtype=np.float32).tolist() for v in vecs]
                _respond({"ok": True, "vectors": result, "dim": dim})
            except Exception as exc:
                _respond({"ok": False, "error": str(exc)})

            # V3.3.16: RSS watchdog — self-terminate if memory exceeds limit.
            # PyTorch on ARM64 Mac never returns memory to OS. After ~200 embeds
            # a worker that started at 300MB grows to 17GB+. Parent auto-respawns
            # a fresh worker on next request (existing mechanism in embeddings.py).
            # V3.3.21: Configurable via SLM_EMBED_WORKER_RSS_LIMIT_MB (default 2500MB).
            _rss_limit = int(os.environ.get("SLM_EMBED_WORKER_RSS_LIMIT_MB", 2500))
            rss_mb = 0.0
            if sys.platform != "win32":
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            else:
                try:
                    import psutil
                    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
                except ImportError:
                    pass  # psutil not available, skip RSS check on Windows

            if rss_mb > 0 and rss_mb > _rss_limit:
                sys.exit(0)

            continue

        _respond({"ok": False, "error": f"Unknown command: {cmd}"})


def _respond(data: dict) -> None:
    """Write JSON response to stdout, flush immediately."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        _worker_main()
    except KeyboardInterrupt:
        # V3.3.13: Windows CI sends KeyboardInterrupt on test completion.
        sys.exit(0)
