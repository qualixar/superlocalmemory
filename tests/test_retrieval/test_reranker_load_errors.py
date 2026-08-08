# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Issue #103 — the reranker must say WHY it failed to load.

The reported symptom was five identical warmup failures per daemon start, each
logged as "did not confirm ready (timeout=90s)" even though the reporter's own
timestamps were 3-12 seconds apart. The real cause — an unsupported backend
value, which sent an unloadable model down the PyTorch tier — was destroyed by
a bare ``except Exception: return None, "", ""`` and never reached any log.

These tests pin the diagnostics, not the model loading.
"""

from __future__ import annotations

from superlocalmemory.core.reranker_worker import _load_model
from superlocalmemory.retrieval.reranker import _is_permanent_load_error


def test_unknown_backend_is_named_not_silently_downgraded() -> None:
    """backend='openai' must be rejected by name.

    It previously fell through to the PyTorch tier, so the operator saw a
    confusing model-load failure rather than "this backend does not exist".
    """
    model, backend, name, error = _load_model("/root/model/reranker.gguf", "openai")

    assert model is None
    assert error, "an unknown backend must produce an error string"
    assert "unknown backend" in error.lower()
    assert "openai" in error


def test_remote_backend_error_points_at_the_endpoint_key() -> None:
    """v3.8.12 (#105) changed the truth this error has to tell.

    Remote reranking now EXISTS, but it is served in the parent process — this
    worker holds torch/ONNX and cannot forward HTTP. Reaching it with
    backend='openai' means ``cross_encoder_endpoint`` was never set, so the
    error must name that key rather than declare remote reranking impossible.
    """
    _, _, _, error = _load_model("/root/model/reranker.gguf", "openai")

    assert "cross_encoder_endpoint" in error
    assert "remote" in error.lower()
    assert "parent process" in error.lower()
    assert "no remote" not in error.lower(), (
        "the pre-3.8.12 claim that remote reranking does not exist is now false"
    )


def test_known_backends_are_not_rejected() -> None:
    """Guard against the validation over-reaching onto supported values."""
    for backend in ("onnx", "", "pytorch", "torch"):
        _, _, _, error = _load_model("definitely-not-a-real-model", backend)
        assert "unknown backend" not in (error or "").lower(), (
            f"backend {backend!r} is supported and must not be rejected"
        )


def test_model_load_failure_propagates_the_underlying_reason() -> None:
    """A bad model name must surface the loader's own error text."""
    model, _, _, error = _load_model("definitely-not-a-real-model", "")

    assert model is None
    assert error
    assert "definitely-not-a-real-model" in error


def test_permanent_errors_are_distinguished_from_transient_ones() -> None:
    """Config errors must not be retried; transient ones must be.

    Retrying an unfixable misconfiguration cost the reporter ~7.5 minutes of
    every daemon startup and produced no new information.
    """
    assert _is_permanent_load_error("unknown backend 'openai'; supported ...")
    assert _is_permanent_load_error("sentence-transformers is not installed")

    assert not _is_permanent_load_error(
        "could not load cross-encoder model 'x': connection reset by peer"
    )
    assert not _is_permanent_load_error("")
