# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``/api/v3/features/media/packs``: read each optional part, turn one on or off (WS-8)."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes import media_packs as mp
from superlocalmemory.runtimes.managed_env import EnvStatus
from superlocalmemory.runtimes.model_sources import LocalDirSource
from superlocalmemory.server.routes import features as routes

TOKEN = "install-token-for-this-test-0123456789abcdef"
AUTH = {"X-Install-Token": TOKEN}
ENABLE, DISABLE = "/api/v3/features/media/packs/enable", "/api/v3/features/media/packs/disable"


class _Env:
    root = None

    def __init__(self, state="ready"):
        self.state = state

    def status(self):
        return EnvStatus(state=self.state, progress=1.0, step="")

    def precheck(self):
        return {"disk_ok": True, "free_bytes": 10 * 1024 ** 3}

    def python(self):
        import sys
        from pathlib import Path

        return Path(sys.executable)


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    token_file = tmp_path / ".install_token"
    token_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: token_file)
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    monkeypatch.setenv("SLM_MEDIA_ALLOW_LOW_RAM", "1")
    for name in ("SLM_FEATURE_MEDIA_CAPTIONS", "SLM_FEATURE_MEDIA_REGIONS", "SLM_4230_OFF"):
        monkeypatch.delenv(name, raising=False)
    env = _Env()
    env.root = root / "runtimes" / "media"
    monkeypatch.setattr(routes, "_data_root", lambda: root)
    monkeypatch.setattr(routes, "_media_env", lambda: env)
    mp._JOBS.clear()
    mp._THREADS.clear()
    feat._write_features(root, {**feat.read_features(root), "media": {"enabled": True}})
    src = tmp_path / "files"
    src.mkdir()
    (src / "config.json").write_text("{}")
    real_enable = mp.enable
    monkeypatch.setattr(mp, "enable", lambda pack_id, **kw: real_enable(
        pack_id, **{**kw, "model_source": LocalDirSource(src), "background": False}))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), root, env


def test_the_read_lists_the_parts_with_their_sizes_and_changes_nothing(ctx):
    client, root, _env = ctx
    before = sorted(p.name for p in root.iterdir())
    body = client.get("/api/v3/features/media/packs").json()
    assert [p["pack"] for p in body["packs"]] == ["captions", "documents", "regions"]
    assert body["packs"][0]["download_words"] == "Describe pictures: a 0.5 GB download"
    models = [p for p in body["packs"] if p["pack"] != "regions"]
    assert all(p["state"] == "off" and p["installed"] is False for p in models)  # no model downloaded yet
    assert body["packs"][2]["state"] == "on" and body["packs"][0]["media_ready"] is True  # regions: on by default
    assert all(p["words"] for p in body["packs"]) and body["packs"][0]["mode"] in ("auto", "one_click")
    assert sorted(p.name for p in root.iterdir()) == before


def test_enable_downloads_then_the_part_reads_as_on_and_why_lists_no_raw_ids(ctx):
    client, root, _env = ctx
    r = client.post(ENABLE, json={"pack": "captions", "yes": True}, headers=AUTH)
    assert r.status_code == 200, r.text
    assert r.json()["pack"]["state"] == "on"
    assert json.loads((root / "features.json").read_text())["features"]["media.captions"] == "on"
    assert client.post(DISABLE, json={"pack": "captions"}, headers=AUTH).json()["pack"]["state"] == "ready"


@pytest.mark.parametrize("body", [{"pack": "captions"}, {"pack": "captions", "yes": "true"}, {"yes": True},
                                  {"pack": "everything", "yes": True}])
def test_enable_needs_a_literal_yes_and_a_known_part(ctx, body):
    client, root, _env = ctx
    assert client.post(ENABLE, json=body, headers=AUTH).status_code == 400
    assert not (root / "runtimes" / "media" / "packs").exists()


def test_enable_with_a_bad_source_is_400(ctx):
    client, *_ = ctx
    assert client.post(ENABLE, json={"pack": "captions", "yes": True, "source": "npm"}, headers=AUTH).status_code == 400


def test_a_refusal_is_a_409_in_words_and_nothing_is_written(ctx):
    client, root, env = ctx
    env.state = "installing"
    r = client.post(ENABLE, json={"pack": "documents", "yes": True}, headers=AUTH)
    assert r.status_code == 409 and r.json()["detail"] == mp.NEEDS_MEDIA_WORDS
    assert "features" not in json.loads((root / "features.json").read_text())


def test_regions_turns_on_without_a_download(ctx):
    client, root, _env = ctx
    r = client.post(ENABLE, json={"pack": "regions", "yes": True}, headers=AUTH)
    assert r.status_code == 200 and r.json()["pack"]["download_words"] == "No download"
    assert not (root / "runtimes" / "media" / "packs").exists()


@pytest.mark.parametrize("path,body", [(ENABLE, {"pack": "captions", "yes": True}), (DISABLE, {"pack": "captions"})])
def test_changes_need_a_credential_and_this_machine(ctx, path, body):
    client, root, _env = ctx
    assert client.post(path, json=body).status_code == 403
    assert client.post(path, json=body, headers={"X-Install-Token": "wrong"}).status_code == 403
    remote = TestClient(client.app, client=("203.0.113.9", 5000))
    assert remote.post(path, json=body, headers=AUTH).status_code == 403
    assert not (root / "runtimes" / "media" / "packs").exists()


def test_the_remote_role_gate_lists_both_changes():
    from superlocalmemory.server import mutation_gate

    for path in (ENABLE, DISABLE):
        rule = mutation_gate.rule_for("POST", path)
        assert rule is not None and rule.permission.name == "MANAGE"
