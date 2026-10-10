"""The per-model table: pinned revision, memory cap, evidence floor."""

from __future__ import annotations

import re

import pytest

import importlib

from superlocalmemory.runtimes import media_models, space_plan, worker_client

media_env = importlib.import_module("superlocalmemory.runtimes.media_env")  # the package re-exports a function of that name

EG2 = "google/embeddinggemma-2"
NOMIC_VISION = "nomic-ai/nomic-embed-vision-v1.5"


def test_revision_is_a_full_commit_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", media_env.MEDIA_MODEL_REVISION)
    assert media_env.MEDIA_MODEL_REVISION == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert media_env.MEDIA_ENV.model_source.revision == media_env.MEDIA_MODEL_REVISION


def test_profiles_for_the_shipped_models():
    eg2 = media_models.MODEL_PROFILES[EG2]
    assert (eg2.repo, eg2.revision, eg2.dim) == (EG2, media_env.MEDIA_MODEL_REVISION, 768)
    assert eg2.rss_limit_mb == 4500 and eg2.media_min_score == 0.69 and eg2.image_max_pixels == 0
    vision = media_models.MODEL_PROFILES[NOMIC_VISION]
    assert vision.dim == 768 and vision.media_min_score == 0.084


def test_unknown_and_fake_models_have_no_profile():
    assert media_models.profile_for("fake:768") is None
    assert media_models.profile_for("someone/else") is None
    assert media_models.rss_limit_mb_for("fake:768") == 1600
    assert media_models.min_score_for("someone/else") is None


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", raising=False)


def test_default_cap_comes_from_the_model(stub_env, no_env):
    assert worker_client.MediaWorkerClient(stub_env, model_id=EG2, revision="").rss_limit_mb == 4500
    assert worker_client.MediaWorkerClient(stub_env, model_id="fake:768", revision="").rss_limit_mb == 1600


def test_env_still_overrides_the_model_cap(stub_env, monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "123")
    assert worker_client.MediaWorkerClient(stub_env, model_id=EG2, revision="").rss_limit_mb == 123


def test_explicit_cap_wins(stub_env, no_env):
    assert worker_client.MediaWorkerClient(stub_env, model_id=EG2, revision="", rss_limit_mb=9).rss_limit_mb == 9


def test_separate_plan_takes_its_floor_from_the_model(monkeypatch):
    monkeypatch.delenv(space_plan.MODE_ENV, raising=False)
    nomic = "nomic-ai/nomic-embed-text-v1.5"
    p = space_plan.resolve_space_plan(nomic, 768, separate_model=(EG2, media_env.MEDIA_MODEL_REVISION, 768))
    assert p.mode == "separate" and p.min_score == 0.69
    other = space_plan.resolve_space_plan(nomic, 768, separate_model=("x/y", "", 768))
    assert other.min_score is None  # unknown model: the configured floor stays


def test_plan_floor_reaches_the_channel_floor():
    from superlocalmemory.retrieval.media_channel import MediaChannel  # noqa: F401
    p = space_plan.resolve_space_plan("nomic-ai/nomic-embed-text-v1.5", 768, requested="separate",
                                      separate_model=(EG2, "", 768))
    assert p.min_score == 0.69


def test_watchdog_leaves_the_picture_worker_its_own_larger_cap(monkeypatch):
    from superlocalmemory.server.unified_daemon import _worker_limit_mb

    monkeypatch.delenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", raising=False)
    other = ["/venv/bin/python", "-m", "superlocalmemory.core.embedding_worker"]
    media = ["/media/venv/bin/python", "-I", "/site/runtimes/multimodal_worker.py"]
    assert _worker_limit_mb(other, 2500) == 2500
    assert _worker_limit_mb(media, 2500) == 4500
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "6000")
    assert _worker_limit_mb(media, 2500) == 6000
    monkeypatch.setenv("SLM_MEDIA_WORKER_RSS_LIMIT_MB", "0")
    assert _worker_limit_mb(media, 2500) == 0


def test_text_floor_comes_from_the_table_and_unknown_models_have_none():
    assert media_models.text_min_semantic_for(EG2) == media_models.MODEL_PROFILES[EG2].text_min_semantic
    assert isinstance(media_models.text_min_semantic_for(EG2), float)
    assert media_models.text_min_semantic_for(NOMIC_VISION) is None  # a picture model has no text floor
    assert media_models.text_min_semantic_for("fake:768") is None
    assert media_models.text_min_semantic_for("nomic-ai/nomic-embed-text-v1.5") is None
