# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
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

import ctypes
import json
import os
import platform
import signal
import struct
import sys

# Force CPU BEFORE any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_MEM_LIMIT"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE"] = "cpu"
# V3.3.17: Disable CoreML EP for ONNX Runtime. CoreML compiles execution
# plans that consume 3-5GB on ARM64 Mac. CPU EP is ~500MB and fast enough.
os.environ["ORT_DISABLE_COREML"] = "1"

# SIGTERM bridge for Docker/systemd
if sys.platform != "win32":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _start_parent_watchdog() -> None:
    """Monitor parent process — self-terminate if parent dies.

    V3.4.24: Delegates to platform_utils.start_parent_watchdog().
    """
    from superlocalmemory.core.platform_utils import start_parent_watchdog
    start_parent_watchdog()


def _detect_onnx_variant(model_name: str = "") -> str:
    """Auto-detect the best ONNX model variant for the current platform.

    V3.4.2: Supports both legacy ms-marco-MiniLM (platform-specific quantized)
    and new gte-modernbert-base (int8/uint8 quantized). Falls back to generic
    model.onnx if platform-specific variant unavailable.
    """
    arch = platform.machine().lower()
    is_64bit = struct.calcsize("P") * 8 == 64

    # Legacy ms-marco-MiniLM models have platform-specific quantized variants
    if "ms-marco" in model_name or "MiniLM" in model_name:
        if sys.platform == "darwin" and arch in ("arm64", "aarch64"):
            return "onnx/model_qint8_arm64.onnx"
        if arch in ("x86_64", "amd64") and is_64bit:
            return "onnx/model_quint8_avx2.onnx"
        return "onnx/model.onnx"

    # gte-modernbert-base and other modern models: int8 for ARM64, uint8 for x86
    if sys.platform == "darwin" and arch in ("arm64", "aarch64"):
        return "onnx/model_int8.onnx"
    if arch in ("x86_64", "amd64") and is_64bit:
        return "onnx/model_uint8.onnx"
    return "onnx/model.onnx"


def _worker_main() -> None:
    """Main loop: read JSON requests from stdin, write responses to stdout."""
    _start_parent_watchdog()
    from superlocalmemory.core.platform_utils import get_rss_mb

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
            model, active_backend, model_name, load_error = _load_model(
                name, backend,
            )
            # V3.3.16: Run real inference to trigger ONNX CoreML JIT compilation.
            # Without this, first real rerank call triggers 30-60s compilation
            # that exceeds the caller's timeout, killing the worker.
            warmup_ok = False
            if model is not None:
                try:
                    # Use 60 pairs (realistic batch size) to trigger CoreML
                    # compilation for the actual workload. 3 pairs compiled a
                    # different execution plan that got recompiled on 60 pairs.
                    dummy_pairs = [
                        (
                            f"What happened to person {i}?",
                            f"Person {i} went to location {i} and did activity {i} last summer with friends.",
                        )
                        for i in range(60)
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
                _release_gpu_cache(model, rebase=True)
            _respond({
                "ok": model is not None,
                "backend": active_backend,
                "model": model_name,
                "warmup_inference": warmup_ok,
                # Carries the real reason to the parent so the warmup log can
                # print it instead of a generic timeout message (issue #103).
                "error": load_error,
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
                model, active_backend, model_name, load_error = _load_model(
                    name, backend,
                )
            if model is None:
                _respond({
                    "ok": False,
                    "error": load_error or "Model load failed",
                })
                continue
            try:
                pairs = [(query, doc) for doc in documents]
                try:
                    import torch
                    with torch.inference_mode():
                        scores = predict_by_length(model, pairs)
                except ImportError:
                    scores = predict_by_length(model, pairs)
                _respond({
                    "ok": True,
                    "scores": [float(s) for s in scores],
                })
            except Exception as exc:
                _respond({"ok": False, "error": str(exc)})
            _release_gpu_cache(model)

            # V3.3.16: RSS watchdog — V3.4.24: cross-platform via platform_utils.
            rss_mb = get_rss_mb()
            if rss_mb > 0 and rss_mb > 2500:
                sys.exit(0)

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
                model, active_backend, model_name, load_error = _load_model(
                    name, backend,
                )
            if model is None:
                _respond({
                    "ok": False,
                    "error": load_error or "Model load failed",
                })
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
            _release_gpu_cache(model)
            continue

        _respond({"ok": False, "error": f"Unknown command: {cmd}"})


_KNOWN_BACKENDS = ("onnx", "", "pytorch", "torch")
# Backends this worker can never serve — they are handled over HTTP by
# superlocalmemory.retrieval.remote_reranker in the parent process (#105).
# Duplicated as a literal on purpose: this module runs as a bare subprocess
# and must not import the retrieval package (or, transitively, httpx).
_REMOTE_BACKENDS = ("openai", "remote")


#: Pairs per forward pass when scoring by length (see predict_by_length).
LENGTH_BATCH = 8


def predict_by_length(model, pairs: list[tuple[str, str]],
                      batch_size: int = LENGTH_BATCH) -> list[float]:
    """Score ``pairs`` shortest first, in small batches; scores in input order.

    A batch is padded to its longest pair. Recall's 30-60 candidates mix short
    facts (median 64 tokens) with whole saved texts (512, the limit), so one
    batch padded every pair to 512: measured on a live-store copy, 287 ms
    instead of 170 ms per recall. Scores differ only by float rounding
    (<= 2e-6 measured; the single batch's scores already depended on which
    long text happened to share it) and never changed an order in testing.
    """
    if not pairs:
        return []
    order = sorted(range(len(pairs)), key=lambda i: (len(pairs[i][1]), i))
    ordered = [pairs[i] for i in order]
    try:
        ranked = model.predict(ordered, batch_size=batch_size)
    except TypeError:  # a scorer without batch_size: one call, same order
        ranked = model.predict(ordered)
    scores = [0.0] * len(pairs)
    for slot, i in enumerate(order):
        scores[i] = float(ranked[slot])
    return scores


def _load_model(
    name: str, backend: str,
) -> tuple:
    """Load cross-encoder model. Returns (model, backend_name, model_name, error).

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
    # v3.8.11 (issue #103): an unrecognised backend used to fall through to
    # the PyTorch tier and fail there with a confusing model-load error. A
    # user who set backend="openai" expecting a remote reranker got five
    # silent failures and no hint that the value meant nothing. Name it.
    #
    # v3.8.12 (issue #105): remote reranking now EXISTS, but it is served in
    # the parent process — this worker holds torch/ONNX and cannot forward an
    # HTTP request. Reaching here with a remote backend means the parent
    # routed wrong (or a caller drove the worker directly), so the message
    # points at the config keys that select the remote path.
    if backend in _REMOTE_BACKENDS:
        return None, "", "", (
            f"unknown backend {backend!r} for the LOCAL reranker worker. "
            f"{backend!r} selects the remote reranker, which runs in the "
            f"parent process — set retrieval.cross_encoder_endpoint (e.g. "
            f"\"http://127.0.0.1:8041/v1/rerank\") so SuperLocalMemory routes "
            f"reranking over HTTP instead of spawning this worker."
        )
    if backend not in _KNOWN_BACKENDS:
        return None, "", "", (
            f"unknown backend {backend!r}; supported values are 'onnx' or ''"
            f" (PyTorch) for local reranking, or 'openai'/'remote' with "
            f"retrieval.cross_encoder_endpoint set for a remote "
            f"OpenAI-compatible /v1/rerank endpoint."
        )

    tier_errors: list[str] = []
    try:
        from sentence_transformers import CrossEncoder

        if backend == "onnx":
            # Tier 1: Platform-specific quantized ONNX (fastest)
            try:
                onnx_file = _detect_onnx_variant(name)
                m = CrossEncoder(
                    name, backend="onnx",
                    model_kwargs={"file_name": onnx_file},
                )
                return m, f"onnx-quantized({onnx_file})", name, ""
            except Exception as exc:
                tier_errors.append(f"onnx-quantized: {exc}")

            # Tier 2: Generic ONNX (auto-exported by optimum)
            try:
                m = CrossEncoder(name, backend="onnx")
                return m, "onnx", name, ""
            except Exception as exc:
                tier_errors.append(f"onnx: {exc}")

        # Tier 3: PyTorch (always works, no ONNX dependency needed)
        m = CrossEncoder(name)
        return m, "pytorch", name, ""
    except ImportError as exc:
        # Previously indistinguishable from a bad model name.
        return None, "", "", (
            f"sentence-transformers is not installed ({exc}); "
            f"install it or set retrieval.use_cross_encoder=false"
        )
    except Exception as exc:
        tier_errors.append(f"pytorch: {exc}")
        # Every tier's real error, propagated instead of discarded. Before
        # 3.8.11 this returned (None, "", "") and the operator saw only a
        # generic "did not confirm ready" line from the parent process.
        return None, "", "", (
            f"could not load cross-encoder model {name!r} "
            f"(backend={backend or 'pytorch'}): " + "; ".join(tier_errors)
        )


#: How far the worker may grow above what it held right after loading before
#: the GPU allocator's cache is handed back. Below this, keeping the cache costs
#: little and reusing it is what keeps requests fast.
GPU_CACHE_HEADROOM_MB = 256      # graphics memory the driver holds
FOOTPRINT_HEADROOM_MB = 512      # the whole process, as Activity Monitor counts it

# Post-load (or post-release) levels, bytes: {"driver": ..., "footprint": ...}.
_baseline: dict[str, int] = {}


class _RusageInfoV0(ctypes.Structure):
    """``struct rusage_info_v0`` from <sys/resource.h> (macOS)."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


def _phys_footprint() -> int | None:
    """This process's physical footprint in bytes (macOS), else None.

    The figure ``/usr/bin/footprint`` and Activity Monitor report, which -- unlike
    RSS -- includes the GPU memory a process holds on Apple Silicon.
    """
    if sys.platform != "darwin":
        return None
    try:
        info = _RusageInfoV0()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
            return None
        return int(info.ri_phys_footprint)
    except Exception:
        return None


def _levels(held) -> dict[str, int]:
    levels = {"driver": int(held())}
    footprint = _phys_footprint()
    if footprint is not None:
        levels["footprint"] = footprint
    return levels


def _release_gpu_cache(model, *, rebase: bool = False) -> None:
    """Hand back the GPU memory the allocator kept, once it has grown.

    The cross-encoder runs on Apple's GPU (the library picks it; the variables
    set above do not stop it). Its allocator keeps freed buffers for reuse and,
    with ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``, never gives any back.
    Measured with the real model and public text (250 requests of mixed
    length): graphics memory 1,126 -> 2,201 MB and the process 1.6 -> 3.6 GB.

    Releasing after EVERY request held it at about 1.6 GB but made short
    requests about 1.7x slower: each then had to reallocate. So the cache is
    released only when the process has grown past a headroom above what it
    held after loading (``rebase=True``) or after the last release -- whichever
    is higher, so memory that a release cannot return is never chased on every
    request. Scores are untouched: this moves memory, not arithmetic.

    ``torch.mps.empty_cache()`` and ``torch.mps.driver_allocated_memory()`` are
    the library's own controls ("releases all unoccupied cached memory currently
    held by the caching allocator"; "total GPU memory allocated by Metal driver
    for the process ... includes cached allocations"). The block holding the
    weights stays: it is in use. Nothing here can fail a request.
    """
    device = getattr(getattr(model, "device", None), "type", "")
    if device != "mps":
        return
    try:
        import torch

        mps = getattr(torch, "mps", None)
        empty_cache = getattr(mps, "empty_cache", None)
        if not callable(empty_cache):
            return
        held = getattr(mps, "driver_allocated_memory", None)
        if not callable(held):
            empty_cache()  # growth cannot be measured: memory first
            return
        if rebase or not _baseline:
            empty_cache()
            _baseline.clear()
            _baseline.update(_levels(held))
            return
        now = _levels(held)
        limits = {"driver": GPU_CACHE_HEADROOM_MB, "footprint": FOOTPRINT_HEADROOM_MB}
        if not any(now[k] > _baseline[k] + limits[k] * 1024 * 1024
                   for k in now if k in _baseline):
            return
        empty_cache()
        for key, value in _levels(held).items():
            _baseline[key] = max(_baseline.get(key, 0), value)
    except Exception:
        pass


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
