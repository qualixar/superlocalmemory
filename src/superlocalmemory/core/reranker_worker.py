# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Subprocess reranker worker — isolates PyTorch/ONNX from main process.

Same pattern as embedding_worker.py. The main process stays at ~60 MB.
All cross-encoder model memory lives in this worker subprocess.

Protocol (JSON over stdin/stdout):
  Request:  {"cmd": "rerank", "query": "...", "documents": ["...", ...]}
  Response: {"ok": true, "scores": [0.95, 0.32, ...]}

  Request:  {"cmd": "score", "query": "...", "document": "..."}
  Response: {"ok": true, "score": 0.87}

  Request:  {"cmd": "ping"}
  Response: {"ok": true, "backend": "onnx", "model": "..."}

  Request:  {"cmd": "quit"}
  (worker exits)

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import os
import platform
import signal
import struct
import sys
import threading

# Force CPU BEFORE any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEM_LIMIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE"] = "cpu"

# SIGTERM bridge for Docker/systemd
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _start_parent_watchdog() -> None:
    """Monitor parent process — self-terminate if parent dies.

    Prevents orphaned workers that consume 1+ GB each when the parent
    process crashes, is killed, or exits without cleanup.

    V3.3.7: Added after incident where ~30 orphaned workers consumed 33 GB.
    """
    parent_pid = os.getppid()

    def _watch() -> None:
        import time
        while True:
            time.sleep(5)
            try:
                os.kill(parent_pid, 0)  # Check if parent is alive (signal 0)
            except OSError:
                # Parent is dead — self-terminate
                os._exit(0)

    t = threading.Thread(target=_watch, daemon=True, name="parent-watchdog")
    t.start()


def _detect_onnx_variant() -> str:
    """Auto-detect the best ONNX model variant for the current platform."""
    arch = platform.machine().lower()
    is_64bit = struct.calcsize("P") * 8 == 64

    if sys.platform == "darwin" and arch in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    if arch in ("x86_64", "amd64") and is_64bit:
        return "onnx/model_quint8_avx2.onnx"
    return "onnx/model.onnx"


def _worker_main() -> None:
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    _start_parent_watchdog()  # V3.3.7: self-terminate if parent dies

    model = None
    active_backend = ""
    model_name = ""

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
            _respond({
                "ok": True,
                "loaded": model is not None,
                "backend": active_backend,
                "model": model_name,
            })
            continue

        if cmd == "load":
            name = req.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2")
            backend = req.get("backend", "onnx")
            model, active_backend, model_name = _load_model(name, backend)
            # V3.3.16: Run real inference to trigger ONNX CoreML JIT compilation.
            # Without this, first real rerank call triggers 30-60s compilation
            # that exceeds the caller's timeout, killing the worker.
            warmup_ok = False
            if model is not None:
                try:
                    dummy_pairs = [
                        ("What is the capital of France?", "Paris is the capital of France."),
                        ("Who wrote Hamlet?", "Shakespeare wrote many plays."),
                        ("What color is the sky?", "The sky is blue on a clear day."),
                    ]
                    try:
                        import torch
                        with torch.inference_mode():
                            _scores = model.predict(dummy_pairs)
                    except ImportError:
                        _scores = model.predict(dummy_pairs)
                    warmup_ok = True
                except Exception:
                    pass
            _respond({
                "ok": model is not None,
                "backend": active_backend,
                "model": model_name,
                "warmup_inference": warmup_ok,
            })
            continue

        if cmd == "rerank":
            query = req.get("query", "")
            documents = req.get("documents", [])
            if not query or not documents:
                _respond({"ok": False, "error": "Missing query or documents"})
                continue
            if model is None:
                # Auto-load with defaults
                name = req.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                backend = req.get("backend", "onnx")
                model, active_backend, model_name = _load_model(name, backend)
            if model is None:
                _respond({"ok": False, "error": "Model load failed"})
                continue
            try:
                pairs = [(query, doc) for doc in documents]
                try:
                    import torch
                    with torch.inference_mode():
                        scores = model.predict(pairs)
                except ImportError:
                    scores = model.predict(pairs)
                _respond({
                    "ok": True,
                    "scores": [float(s) for s in scores],
                })
            except Exception as exc:
                _respond({"ok": False, "error": str(exc)})
            continue

        if cmd == "score":
            query = req.get("query", "")
            document = req.get("document", "")
            if not query or not document:
                _respond({"ok": False, "error": "Missing query or document"})
                continue
            if model is None:
                name = req.get("model_name", "cross-encoder/ms-marco-MiniLM-L-12-v2")
                backend = req.get("backend", "onnx")
                model, active_backend, model_name = _load_model(name, backend)
            if model is None:
                _respond({"ok": False, "error": "Model load failed"})
                continue
            try:
                try:
                    import torch
                    with torch.inference_mode():
                        scores = model.predict([(query, document)])
                except ImportError:
                    scores = model.predict([(query, document)])
                _respond({"ok": True, "score": float(scores[0])})
            except Exception as exc:
                _respond({"ok": False, "error": str(exc)})
            continue

        _respond({"ok": False, "error": f"Unknown command: {cmd}"})


def _load_model(
    name: str, backend: str,
) -> tuple:
    """Load cross-encoder model. Returns (model, backend_name, model_name).

    V3.3.13: sentence-transformers 5.x+ supports backend='onnx' for
    CrossEncoder. We use a 3-tier fallback chain:

      1. ONNX + platform-quantized model (fastest, ~200MB, 2.4ms/pair)
      2. ONNX + generic model (fast, auto-exported on first use)
      3. PyTorch (always works, ~500MB, 6ms/pair)

    Cross-platform:
      Mac ARM64 → model_qint8_arm64.onnx
      x86_64    → model_quint8_avx2.onnx
      Fallback  → model.onnx (generic)
    """
    try:
        from sentence_transformers import CrossEncoder

        if backend == "onnx":
            # Tier 1: Platform-specific quantized ONNX (fastest)
            try:
                onnx_file = _detect_onnx_variant()
                m = CrossEncoder(
                    name, backend="onnx",
                    model_kwargs={"file_name": onnx_file},
                )
                return m, f"onnx-quantized({onnx_file})", name
            except Exception:
                pass

            # Tier 2: Generic ONNX (auto-exported by optimum)
            try:
                m = CrossEncoder(name, backend="onnx")
                return m, "onnx", name
            except Exception:
                pass

        # Tier 3: PyTorch (always works, no ONNX dependency needed)
        m = CrossEncoder(name)
        return m, "pytorch", name
    except ImportError:
        return None, "", ""
    except Exception:
        return None, "", ""


def _respond(data: dict) -> None:
    """Write JSON response to stdout, flush immediately."""
    sys.stdout.write(json.dumps(data) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        _worker_main()
    except KeyboardInterrupt:
        # V3.3.13: Windows CI sends KeyboardInterrupt on test completion.
        # Exit cleanly instead of printing a traceback that fails CI.
        sys.exit(0)
