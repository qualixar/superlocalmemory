# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``Upgrade memory engine``: the one target, the plan, and the daemon's pick-up of a request."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_upgrade as eu
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_models

EG2 = "google/embeddinggemma-2"
NOMIC = EmbeddingConfig(provider="sentence-transformers", model_name="nomic-ai/nomic-embed-text-v1.5",
                        dimension=768, ollama_model="nomic-embed-text")
EG2_LIVE = EmbeddingConfig(provider="slm-media", model_name=EG2, dimension=768)


def _plan(live=NOMIC, memories=1000, state="ready", **kw):
    return eu.plan(live, memories, state, **kw)


# -- the target ---------------------------------------------------------------

def test_the_target_is_the_managed_model_and_drops_endpoint_and_key():
    live = EmbeddingConfig(provider="openai", model_name="text-embedding-3-small", dimension=1536,
                           api_endpoint="https://example.invalid/v1", api_key="sk-secret",
                           ollama_model="keep-me")
    target = eu.upgrade_target(live)
    assert (target.provider, target.model_name, target.dimension) == ("slm-media", EG2, 768)
    assert target.api_endpoint == "" and target.api_key == ""
    assert target.ollama_model == "keep-me", "the rest of the live config is carried over"
    assert target.is_cloud is False


def test_the_target_does_not_change_the_live_config():
    before = NOMIC
    eu.upgrade_target(before)
    assert before.provider == "sentence-transformers" and before.dimension == 768


# -- the plan numbers ---------------------------------------------------------

def test_a_ready_plan_has_the_numbers():
    p = _plan(memories=1000)
    assert p["available"] is True and p["reason"] == "" and p["already"] is False
    assert p["memories"] == 1000
    assert p["ram_mb"] == media_models.load_mb_for(EG2)
    new_mb = math.ceil(1000 * 768 * 4 / 1024 ** 2)
    kept_mb = math.ceil(1000 * 768 * 4 / 1024 ** 2)
    assert p["disk_new_mb"] == new_mb and p["disk_kept_mb"] == kept_mb
    assert p["disk_mb"] == new_mb + kept_mb
    assert p["minutes"] == round(1000 * media_models.PROVISIONAL_SECONDS_PER_MEMORY / 60)
    assert p["from"]["model"] == NOMIC.model_name and p["to"]["model"] == EG2
    assert p["to"]["provider"] == "slm-media" and p["to"]["dimension"] == 768


def test_the_kept_space_is_sized_by_the_live_width():
    cloud = EmbeddingConfig(provider="openai", model_name="big", dimension=3072)
    p = _plan(cloud, memories=2000)
    assert p["disk_kept_mb"] == math.ceil(2000 * 3072 * 4 / 1024 ** 2)


def test_a_small_store_still_says_at_least_one_minute_and_an_empty_one_says_zero():
    assert _plan(memories=3)["minutes"] == 1
    assert _plan(memories=0)["minutes"] == 0 and _plan(memories=0)["available"] is True


def test_the_time_estimate_is_labelled_provisional_and_about():
    p = _plan(memories=600)
    assert p["minutes_label"].startswith("about ")
    assert "minute" in p["minutes_label"]


def test_the_plan_promises_recall_and_rollback_in_plain_words():
    p = _plan()
    text = p["explain"].lower()
    assert "recall keeps working" in text and "roll back" in text
    assert p["rollback"] is True


# -- reasons ------------------------------------------------------------------

def test_already_upgraded():
    p = _plan(EG2_LIVE)
    assert p["available"] is False and p["already"] is True
    assert "already" in p["reason"].lower()


def test_a_running_change_blocks_and_names_how_to_stop_it():
    p = _plan(job_running=True)
    assert p["available"] is False and "slm embedder cancel" in p["reason"]


def test_images_and_documents_off_says_exactly_that_and_the_command():
    p = _plan(state="not_installed", media_enabled=False)
    assert p["available"] is False and p["needs_media"] is True
    assert "images and documents" in p["reason"].lower() and "slm media enable" in p["reason"]
    assert p["turn_on_command"] == "slm media enable"


@pytest.mark.parametrize("state,needle", [
    ("installing", "still being set up"),
    ("failed", "slm doctor"),
    ("unsupported", "can't be set up on this computer"),
    ("not_installed", "not set up yet"),
])
def test_media_on_but_environment_not_ready_says_the_state(state, needle):
    p = _plan(state=state, media_enabled=True)
    assert p["available"] is False and p["needs_media"] is False
    assert needle in p["reason"]


def test_already_wins_over_everything_and_a_running_job_wins_over_media():
    assert _plan(EG2_LIVE, media_enabled=False, state="failed", job_running=True)["already"] is True
    p = _plan(job_running=True, media_enabled=False, state="failed")
    assert "slm embedder cancel" in p["reason"]


def test_no_text_for_a_non_technical_person_says_reindex():
    cases = [_plan(), _plan(EG2_LIVE), _plan(job_running=True), _plan(media_enabled=False),
             _plan(state="installing"), _plan(state="failed"), _plan(state="unsupported"),
             _plan(state="not_installed")]
    for p in cases:
        for key in ("reason", "explain", "minutes_label"):
            assert "re-index" not in p[key].lower() and "reindex" not in p[key].lower(), (key, p[key])


# -- the daemon picks up a recorded request -----------------------------------

class _Runner:
    def __init__(self, error=None):
        self.switched, self.error = [], error

    def request_switch(self, target, **kw):
        if self.error:
            raise self.error
        self.switched.append((target, kw))
        return {"job_id": 1, "state": "queued"}


def _app(runner):
    return SimpleNamespace(embedding_reindex=runner)


def _config(live=NOMIC):
    return SimpleNamespace(embedding=live, db_path="x")


def _write(root, data):
    (root / "features.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture()
def wired(tmp_path, monkeypatch):
    state = {"media": (True, "ready"), "job": False, "count": 40}
    monkeypatch.setattr(eu, "media_state", lambda data_root=None, env=None: state["media"])
    monkeypatch.setattr(eu, "count_memories", lambda db_path: state["count"])
    monkeypatch.setattr(eu, "job_is_running", lambda db_path: state["job"])
    return tmp_path, state


def test_a_request_with_a_ready_environment_queues_one_upgrade_and_clears_the_flag(wired):
    root, _state = wired
    _write(root, {"schema": 1, "engine_upgrade": {"requested": True, "choice_source": "npm"}})
    runner = _Runner()
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "queued"
    assert len(runner.switched) == 1
    assert runner.switched[0][0].model_name == EG2 and runner.switched[0][1].get("kind", "switch") == "switch"
    assert feat.read_features(root).get("engine_upgrade", {}).get("requested") is not True
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "none"
    assert len(runner.switched) == 1, "never queued twice"


def test_a_request_while_the_environment_is_not_ready_changes_nothing(wired):
    root, state = wired
    state["media"] = (True, "installing")
    _write(root, {"schema": 1, "engine_upgrade": {"requested": True}})
    before = (root / "features.json").read_bytes()
    runner = _Runner()
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "waiting"
    assert runner.switched == [] and (root / "features.json").read_bytes() == before


def test_a_request_while_another_change_runs_waits_for_the_next_start(wired):
    root, state = wired
    state["job"] = True
    _write(root, {"schema": 1, "engine_upgrade": {"requested": True}})
    runner = _Runner()
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "waiting"
    assert runner.switched == []
    assert feat.read_features(root)["engine_upgrade"]["requested"] is True


def test_a_request_when_already_upgraded_is_just_cleared(wired):
    root, _state = wired
    _write(root, {"schema": 1, "engine_upgrade": {"requested": True}})
    runner = _Runner()
    assert eu.apply_on_start(_app(runner), _config(EG2_LIVE), data_root=root) == "cleared"
    assert runner.switched == []
    assert feat.read_features(root).get("engine_upgrade", {}).get("requested") is not True


def test_no_request_touches_nothing_and_creates_no_file(wired):
    root, _state = wired
    runner = _Runner()
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "none"
    assert list(root.iterdir()) == [] and runner.switched == []
    _write(root, {"schema": 1, "media": {"enabled": True}})
    before = (root / "features.json").read_bytes()
    assert eu.apply_on_start(_app(runner), _config(), data_root=root) == "none"
    assert (root / "features.json").read_bytes() == before


def test_no_runner_or_a_failing_runner_never_raises_and_keeps_the_flag(wired):
    root, _state = wired
    _write(root, {"schema": 1, "engine_upgrade": {"requested": True}})
    assert eu.apply_on_start(SimpleNamespace(), _config(), data_root=root) == "waiting"
    assert eu.apply_on_start(_app(_Runner(error=RuntimeError("boom"))), _config(), data_root=root) == "waiting"
    assert feat.read_features(root)["engine_upgrade"]["requested"] is True


def test_the_daemon_start_calls_the_pickup_after_the_media_request():
    import inspect

    from superlocalmemory.server import unified_daemon

    source = inspect.getsource(unified_daemon)
    assert source.index("_features.apply_requested(source=\"npm\")") < source.index("apply_on_start")
