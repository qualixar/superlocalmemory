# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/whatsnew/state``: which 4.2.30 features are on, nothing else."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core.feature_flags import FLAGS, FlagState, write_flag
from superlocalmemory.outcomes import disclosure
from superlocalmemory.server.routes import whatsnew

URL = "/api/whatsnew/state"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    for name in [k for k in __import__("os").environ if k.startswith("SLM_FEATURE_")] + ["SLM_4230_OFF"]:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture()
def client(home):
    app = FastAPI()
    app.include_router(whatsnew.router)
    return TestClient(app)


def test_every_entry_names_a_real_flag_or_none() -> None:
    for entry, flag in whatsnew.ENTRIES.items():
        assert flag is None or flag in FLAGS, entry


def test_a_fresh_install_has_nothing_on(client) -> None:
    body = client.get(URL).json()
    assert body["success"] and body["kill_switch"] is False
    assert {k: v["on"] for k, v in body["features"].items()} == {k: False for k in whatsnew.ENTRIES}
    assert body["features"]["need_gate"]["flag"] == "delivery.need_gate"
    assert body["features"]["why"]["flag"] is None


def test_turning_a_flag_on_turns_only_its_entries_on(client, home) -> None:
    write_flag(home, "trust.taint", FlagState.ON)
    write_flag(home, "delivery.cues", FlagState.ON)
    on = {k for k, v in client.get(URL).json()["features"].items() if v["on"]}
    assert on == {"origin", "firewall", "cues"}


def test_shadow_is_not_on(client, home) -> None:
    write_flag(home, "trust.taint", FlagState.SHADOW)
    features = client.get(URL).json()["features"]
    assert not features["origin"]["on"] and not features["firewall"]["on"]


def test_measuring_is_on_once_its_notice_was_read(client, home) -> None:
    assert client.get(URL).json()["features"]["measurement"]["on"] is False
    disclosure.acknowledge(home, "dashboard")
    assert client.get(URL).json()["features"]["measurement"]["on"] is True


def test_the_kill_switch_reads_everything_off_and_says_so(client, home, monkeypatch) -> None:
    write_flag(home, "trust.signing", FlagState.ON)
    monkeypatch.setenv("SLM_4230_OFF", "1")
    body = client.get(URL).json()
    assert body["kill_switch"] is True and body["features"]["signing"]["on"] is False


def test_reading_never_writes(client, home) -> None:
    client.get(URL)
    assert not (home / "features.json").exists()


def test_an_unreadable_state_is_an_error_not_a_guess(client, monkeypatch) -> None:
    from superlocalmemory.outcomes import disclosure as d

    monkeypatch.setattr(d, "pending", lambda home: (_ for _ in ()).throw(RuntimeError("boom")))
    r = client.get(URL)
    assert r.status_code == 500 and "boom" not in r.text
