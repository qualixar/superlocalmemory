"""The real self-check, wired as the environment's canary; and the two ports."""

import sys
from pathlib import Path

from superlocalmemory.runtimes import media_canary, ports
from superlocalmemory.runtimes.media_env import MEDIA_ENV


def test_canary_passes_with_the_fake_worker():
    assert media_canary.media_canary(Path(sys.executable), model_id="fake:768", revision="") is True


def test_canary_fails_on_wrong_dimension():
    assert media_canary.media_canary(Path(sys.executable), model_id="fake:512", revision="") is False


def test_canary_fails_when_python_cannot_run():
    assert media_canary.media_canary(Path("/nonexistent/python"), model_id="fake:768", revision="") is False


def test_canary_is_wired_into_the_media_environment():
    assert MEDIA_ENV.canary is media_canary.media_canary


def test_the_png_is_a_valid_8x8_image(tmp_path):
    import struct
    import zlib
    data = media_canary.tiny_png()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (8, 8)
    idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
    assert len(zlib.decompress(idat)) == 8 * (1 + 8 * 3)


def test_embedding_service_satisfies_the_text_port():
    from superlocalmemory.core.embeddings import EmbeddingService
    from superlocalmemory.core.embeddings import EmbeddingConfig

    adapter = ports.EmbeddingServiceText(EmbeddingService(EmbeddingConfig()))
    assert isinstance(adapter, ports.TextEmbedderPort)
    assert isinstance(adapter.model_id, str) and adapter.dim == adapter._service.dimension
    assert callable(EmbeddingService.embed_batch)


def test_the_media_client_satisfies_the_media_port(stub_env):
    from superlocalmemory.runtimes.worker_client import MediaWorkerClient
    c = MediaWorkerClient(stub_env, model_id="fake:768", revision="")
    assert isinstance(c, ports.MediaEmbedderPort)
