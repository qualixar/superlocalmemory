"""Embedding backend order must avoid the observed Apple Silicon ONNX stall."""

from __future__ import annotations

from unittest.mock import patch


def test_apple_silicon_prefers_stable_pytorch_cpu_backend() -> None:
    from superlocalmemory.core import embedding_worker

    with patch.object(embedding_worker.sys, "platform", "darwin"), patch(
        "platform.machine", return_value="arm64",
    ):
        assert embedding_worker._embedding_backend_order() == ("pytorch", "onnx")


def test_other_platforms_retain_onnx_first_order() -> None:
    from superlocalmemory.core import embedding_worker

    with patch.object(embedding_worker.sys, "platform", "linux"), patch(
        "platform.machine", return_value="x86_64",
    ):
        assert embedding_worker._embedding_backend_order() == ("onnx", "pytorch")

