# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The answer check's settings have one home, and a mode switch never touches it.

A withdrawn consent must stay withdrawn. Before this, the answer check's
choice, consent and reordering lived in config.json AND in a per-mode copy
(mode_a/b/c.json): switching modes reloaded the copy, so a consent withdrawn
in one mode came back on switching back, and a dashboard mode switch left the
routes writing one file while every decision read another — so unticking
consent answered 200 and the online check kept running.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from superlocalmemory.core import engine_wiring, judge_selection
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode

# Fixtures and fakes of the answer-check API suite (imported, not duplicated).
from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    FAKE_KEY,
    _FakeKeyStore,
    _reset_fake_key_store,
    call_order,
    client,
)

STATE_FILE = "answer_check.json"


class _FakeJev:
    backend = "jev"

    def __init__(self, rerank_k: int = 0) -> None:
        self.rerank_k = rerank_k
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True


class _FakeLaya:
    backend = "laya"

    def __init__(self) -> None:
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True


@pytest.fixture()
def live(client, monkeypatch):  # noqa: F811
    """A retrieval engine whose judge is whatever the config asks for."""
    engine = SimpleNamespace(_sufficiency_judge=None)
    client.app.state.engine = SimpleNamespace(_retrieval_engine=engine)
    built: list[tuple] = []

    def _attach(retrieval_engine, retrieval_config):
        mode = retrieval_config.sufficiency_judge
        k = judge_selection.jev_rerank_k(retrieval_config)
        built.append((mode, retrieval_config.sufficiency_jev_consent, k))
        old = retrieval_engine._sufficiency_judge
        if old is not None:
            old.shutdown()
        if mode == "jev" and retrieval_config.sufficiency_jev_consent is True:
            retrieval_engine._sufficiency_judge = _FakeJev(k)
        else:
            retrieval_engine._sufficiency_judge = None
        return judge_selection.backend_of(retrieval_engine._sufficiency_judge)

    monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge", _attach, raising=False)
    return SimpleNamespace(engine=engine, built=built)


def _answer_check(cfg: SLMConfig) -> dict:
    r = cfg.retrieval
    return {
        "judge": r.sufficiency_judge,
        "consent": r.sufficiency_jev_consent,
        "rerank": r.sufficiency_jev_rerank,
        "rerank_consent": r.sufficiency_jev_rerank_consent,
    }


def _turn_jev_and_reordering_on(client) -> None:  # noqa: F811
    _FakeKeyStore._by_test["typesafe"] = FAKE_KEY
    assert client.post("/api/v3/answer-check/jev/consent",
                       json={"accepted": True, "provider": "typesafe"}).status_code == 200
    assert client.post("/api/v3/answer-check/mode", json={"mode": "jev"}).status_code == 200
    rerank = client.post("/api/v3/answer-check/jev/rerank",
                         json={"enabled": True, "accepted": True, "provider": "typesafe"})
    assert rerank.status_code == 200, rerank.text


def _withdraw(client):  # noqa: F811
    return client.post("/api/v3/answer-check/jev/consent",
                       json={"accepted": False, "provider": "typesafe"})


@pytest.fixture()
def three_modes(tmp_path):
    """A real 3-mode install in mode B (current_mode + mode_a/b/c.json)."""
    SLMConfig.migrate_to_3mode(tmp_path)
    SLMConfig.switch_mode("b", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# (a) a mode switch never brings a withdrawn consent back
# ---------------------------------------------------------------------------


def test_a_withdrawal_survives_switching_back_to_the_mode_it_was_given_in(
        client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    SLMConfig.switch_mode("a", three_modes)        # `slm mode a` / MCP set_mode
    assert _withdraw(client).status_code == 200
    SLMConfig.switch_mode("b", three_modes)        # back, with no new consent

    after = _answer_check(SLMConfig.load())
    assert after == {"judge": "off", "consent": False,
                     "rerank": False, "rerank_consent": False}
    assert judge_selection.resolve_judge_mode(SLMConfig.load().retrieval) == "off"


def test_switching_modes_does_not_switch_the_answer_check_off_either(
        client, live, three_modes):  # noqa: F811
    """One home means the choice follows the person, not the mode."""
    _turn_jev_and_reordering_on(client)
    SLMConfig.switch_mode("a", three_modes)
    assert _answer_check(SLMConfig.load()) == {
        "judge": "jev", "consent": True, "rerank": True, "rerank_consent": True}
    SLMConfig.switch_mode("c", three_modes)
    assert _answer_check(SLMConfig.load())["consent"] is True


def test_the_dashboard_mode_template_carries_the_answer_check_choice(
        client, live, three_modes):  # noqa: F811
    """PUT /api/v3/mode builds the new engine from ``for_mode(...).retrieval``."""
    _turn_jev_and_reordering_on(client)
    template = SLMConfig.for_mode(Mode.C)
    assert template.retrieval.sufficiency_judge == "jev"
    assert template.retrieval.sufficiency_jev_consent is True
    assert _withdraw(client).status_code == 200
    assert SLMConfig.for_mode(Mode.C).retrieval.sufficiency_jev_consent is False


def test_no_mode_file_keeps_its_own_copy_of_the_consent(client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    SLMConfig.switch_mode("a", three_modes)
    SLMConfig.switch_mode("b", three_modes)
    for name in ("config.json", "mode_a.json", "mode_b.json", "mode_c.json"):
        retrieval = json.loads((three_modes / name).read_text(encoding="utf-8")).get("retrieval", {})
        leaked = sorted(k for k in retrieval if k.startswith("sufficiency_"))
        assert leaked == [], f"{name} still carries {leaked}"


# ---------------------------------------------------------------------------
# (b) the routes read what they write, whichever file load() prefers
# ---------------------------------------------------------------------------


def _drift_config_json_away_from_current_mode(root) -> None:
    """What a dashboard mode switch leaves behind: config.json says one mode,
    ``current_mode`` another, so ``SLMConfig.load()`` reads mode_<current>.json."""
    path = root / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["mode"] = "c"
    path.write_text(json.dumps(data), encoding="utf-8")


def test_withdrawing_after_a_dashboard_mode_switch_really_turns_jev_off(
        client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    _drift_config_json_away_from_current_mode(three_modes)
    assert isinstance(live.engine._sufficiency_judge, _FakeJev)

    response = _withdraw(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jev"]["consent"] is False
    assert body["jev"]["rerank"]["enabled"] is False
    assert live.engine._sufficiency_judge is None, "the online check is still attached"
    assert _answer_check(SLMConfig.load())["consent"] is False


def test_a_withdrawal_never_answers_success_while_jev_is_still_running(
        client, live, monkeypatch, three_modes):  # noqa: F811
    """A switch that fails to take Jev off is not a withdrawal: the route
    takes it off itself, and checks, before it says yes."""
    _turn_jev_and_reordering_on(client)
    stuck = live.engine._sufficiency_judge
    monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge",
                        lambda retrieval_engine, cfg: "jev", raising=False)

    response = _withdraw(client)
    assert response.status_code == 200, response.text
    assert live.engine._sufficiency_judge is None
    assert stuck.stopped is True


def test_a_withdrawal_that_cannot_be_confirmed_is_reported_not_hidden(
        client, live, monkeypatch, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    monkeypatch.setattr(engine_wiring, "attach_sufficiency_judge",
                        lambda retrieval_engine, cfg: "jev", raising=False)
    monkeypatch.setattr(judge_selection, "swap_sufficiency_judge",
                        lambda retrieval_engine, build: "jev")

    response = _withdraw(client)
    assert response.status_code == 500
    assert "restart" in response.json()["error"].lower()
    # The saved choice is already off, so a restart honours the withdrawal.
    assert _answer_check(SLMConfig.load())["consent"] is False


def test_turning_reordering_off_takes_it_off_the_running_check(
        client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    assert live.engine._sufficiency_judge.rerank_k == 20
    response = client.post("/api/v3/answer-check/jev/rerank",
                           json={"enabled": False, "accepted": False, "provider": "typesafe"})
    assert response.status_code == 200, response.text
    assert live.engine._sufficiency_judge.rerank_k == 0


# ---------------------------------------------------------------------------
# saves elsewhere in SLM: stale copies never overwrite, explicit edits land
# ---------------------------------------------------------------------------


def test_saving_a_config_loaded_before_the_withdrawal_does_not_undo_it(
        client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    stale = SLMConfig.load()                  # e.g. the daemon's copy from boot
    assert _withdraw(client).status_code == 200
    stale.save()                              # an unrelated settings save
    stale.save(mode_change=True)
    assert _answer_check(SLMConfig.load())["consent"] is False


def test_an_explicit_choice_made_in_code_and_saved_takes_effect(
        client, live, three_modes):  # noqa: F811
    """The setup wizard sets the field and saves — that must still work."""
    _turn_jev_and_reordering_on(client)
    cfg = SLMConfig.load()
    cfg.retrieval.sufficiency_judge = "laya"
    cfg.retrieval.sufficiency_python = "/opt/laya/venv/bin/python"
    cfg.save()
    after = SLMConfig.load().retrieval
    assert after.sufficiency_judge == "laya"
    assert after.sufficiency_python == "/opt/laya/venv/bin/python"


def test_a_damaged_settings_file_never_reads_as_consent(client, live, three_modes):  # noqa: F811
    """Even with a copy elsewhere still saying yes (a hand edit, or a copy that
    could not be tidied away), a damaged file reads as no — never as "absent,
    so use whatever the config file says"."""
    _turn_jev_and_reordering_on(client)
    path = three_modes / "config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("retrieval", {}).update(
        sufficiency_judge="jev", sufficiency_jev_consent=True,
        sufficiency_jev_rerank=True, sufficiency_jev_rerank_consent=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    (three_modes / STATE_FILE).write_text("{not json", encoding="utf-8")
    after = _answer_check(SLMConfig.load())
    assert after["consent"] is False
    assert after["rerank_consent"] is False
    assert judge_selection.resolve_judge_mode(SLMConfig.load().retrieval) == "off"


def test_the_settings_file_is_private_to_its_owner(client, live, three_modes):  # noqa: F811
    _turn_jev_and_reordering_on(client)
    mode = (three_modes / STATE_FILE).stat().st_mode & 0o777
    assert os.name == "nt" or mode == 0o600, oct(mode)  # no mode bits on Windows
