"""Text embedders never take the media stop hook: the picture channel owns it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.runtimes import worker_client


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(worker_client, "_CLIENTS", {})


def _env(tmp_path):
    return SimpleNamespace(root=tmp_path, status=lambda: SimpleNamespace(state="ready"))


@pytest.mark.parametrize("pictures_on", [True, False])
def test_a_text_embedder_never_registers_the_stop_hook(tmp_path, monkeypatch, pictures_on):
    hooks: list = []
    monkeypatch.setattr(worker_client, "register_media_stop_hook", hooks.append)
    monkeypatch.setattr(worker_client, "media_enabled", lambda root=None: pictures_on)
    client = worker_client.text_embedder(env=_env(tmp_path), data_root=tmp_path,
                                         model_id="google/embeddinggemma-2")
    assert client is not None
    assert hooks == []


def test_paired_mode_keeps_the_picture_workers_hook(tmp_path, monkeypatch):
    hooks: list = []
    monkeypatch.setattr(worker_client, "register_media_stop_hook", hooks.append)
    monkeypatch.setattr(worker_client, "media_enabled", lambda root=None: True)
    picture = worker_client._shared_client(_env(tmp_path), "google/embeddinggemma-2", "", "image")
    assert hooks == [picture.stop]
    worker_client.text_embedder(env=_env(tmp_path), data_root=tmp_path, model_id="google/embeddinggemma-2")
    assert hooks == [picture.stop]
