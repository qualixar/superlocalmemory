# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9: the cases that mutation checks showed the first tests did not pin down."""

from __future__ import annotations

import subprocess
import sys

import pytest

from superlocalmemory.portability import readers, registry, runner
from superlocalmemory.portability.readers import Doc
from tests.test_portability import fixtures as fx


@pytest.fixture(autouse=True)
def _own_data_elsewhere(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path_factory.mktemp("slm-home")))


def test_a_claude_conversation_in_two_files_counts_once() -> None:
    conv = fx.claude_list()[0]
    parsed = registry.parse_any([Doc("a.json", fx.dumps([conv])), Doc("b.json", fx.dumps([conv]))])
    assert parsed.conversations == 1 and len(parsed.items) == 2


def test_a_link_is_refused_by_name_not_by_accident(tmp_path) -> None:
    real = tmp_path / "real.md"
    real.write_text("# A\nbody")
    link = tmp_path / "shortcut.md"
    link.symlink_to(real)
    with pytest.raises(readers.IntakeRefused, match="That path is a link"):
        readers.from_path(link)


def test_a_busy_writer_is_waited_for_and_the_import_finishes(world, monkeypatch) -> None:
    from superlocalmemory.core.remember_runtime import CanonicalRememberBusy

    real = runner.submit_memory_settled
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CanonicalRememberBusy("too many saves are arriving at once")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "submit_memory_settled", flaky)
    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    assert run["state"] == "done" and run["saved_items"] == 2 and calls["n"] == 3


def test_the_real_command_line_has_the_import_command() -> None:
    out = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "import", "--help"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    for word in ("preview", "run", "status", "list", "cancel", "undo"):
        assert word in out.stdout
    bad = subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", "import", "run", "x",
                          "--format", "notion"], capture_output=True, text=True, timeout=120)
    assert bad.returncode != 0 and "invalid choice" in bad.stderr


def test_a_relative_path_is_sent_whole_and_a_failed_run_exits_with_an_error(monkeypatch, tmp_path,
                                                                           capsys) -> None:
    from argparse import Namespace

    from superlocalmemory.cli import daemon, import_cmd

    sent = []
    failed = {"run_id": "imp_x", "state": "failed", "format": "mem0", "saved_items": 1,
              "total_items": 2, "source_label": "m.json", "note": "SuperLocalMemory was too busy.",
              "skipped_words": [], "active": False, "disputes_opened": 0, "disputes_unlisted": 0}
    monkeypatch.setattr(daemon, "daemon_request", lambda m, p, b=None, **k: sent.append((m, p, b)) or
                        {"success": True, "run": failed})
    monkeypatch.setattr(import_cmd, "_daemon", lambda: True)
    monkeypatch.chdir(tmp_path)
    args = Namespace(import_command="run", profile="", json=False, format="", file="exports/m.json",
                     run_id="", wait=0.0)
    assert import_cmd.cmd_import(args) == 1
    assert sent[0][2]["path"] == str(tmp_path / "exports" / "m.json")
    assert "SuperLocalMemory was too busy." in capsys.readouterr().out


def test_undo_and_status_still_work_after_the_pack_is_switched_off(world, monkeypatch) -> None:
    """Switching the pack off must never trap someone with an import they cannot take back."""
    from superlocalmemory.core.feature_flags import env_name
    from superlocalmemory.portability import service

    run = world.run("m.json", fx.dumps(fx.MEM0_ITEMS))
    monkeypatch.setenv(env_name("portability.import"), "off")
    assert service.get_run(world.engine, world.profile, run["run_id"])["state"] == "done"
    assert service.overview(world.engine, world.profile)["enabled"] is False
    assert world.undo(run["run_id"])["state"] == "rolled_back"
    assert world.count("memories") == 0


def test_erasing_a_profile_deletes_every_staged_upload(world, tmp_path) -> None:
    from superlocalmemory.compliance.erasure_sidecars import erase_profile_sidecars
    from superlocalmemory.portability import staging

    first = staging.put([Doc("a.json", b"[1]")])
    second = staging.put([Doc("b.json", b"[2]")])
    counts: dict = {}
    erase_profile_sidecars(tmp_path / "slm-home", None, "work", counts)
    assert counts["import_staging"] == 2
    for gone in (first, second):
        with pytest.raises(readers.IntakeRefused, match="no longer here"):
            staging.get(gone)
    assert staging.drop_all(tmp_path / "slm-home") == 0


def test_staged_uploads_expire_and_only_a_few_are_kept(tmp_path, monkeypatch) -> None:
    import os
    import time

    from superlocalmemory.portability import constants as C
    from superlocalmemory.portability import staging

    old = staging.put([Doc("a.json", b"[1]")])
    folder = staging._root() / old
    stamp = time.time() - C.STAGING_TTL_S - 10
    os.utime(folder, (stamp, stamp))
    fresh = [staging.put([Doc(f"{n}.json", b"[1]")]) for n in range(staging.MAX_STAGED + 2)]
    with pytest.raises(readers.IntakeRefused):
        staging.get(old)  # expired
    kept = [u for u in fresh if (staging._root() / u).is_dir()]
    assert len(kept) <= staging.MAX_STAGED and fresh[-1] in kept
    victim = staging._root().parent / "victim"  # a folder shaped like a staged upload, elsewhere
    victim.mkdir()
    (victim / "names.json").write_text('["secret.json"]')
    (victim / "00000.bin").write_bytes(b"[1]")
    for not_an_id in ("../victim", "../../etc", "", "g" * 32, fresh[-1].upper()):
        with pytest.raises(readers.IntakeRefused, match="no longer here"):
            staging.get(not_an_id)  # an id is 32 hex characters, never a path


def test_even_an_administrator_from_another_machine_cannot_switch_the_pack(world, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER

    world.app.state.rbac = RbacEngine(str(world.db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")
    admin = world.app.state.rbac.create_user("ada", "password-1234")
    for workspace in ("default", "work"):
        world.app.state.rbac.set_membership(workspace, admin["user_id"], "admin")
    headers = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard",
               "X-SLM-User-Session": world.app.state.rbac.create_session(admin["user_id"])}
    remote = TestClient(world.app, client=("203.0.113.9", 50000))
    refused = remote.post("/api/import/switch", json={"state": "off"}, headers=headers)
    assert refused.status_code == 403 and "only be changed from this machine" in refused.json()["detail"]


def test_with_a_cap_of_zero_every_contradiction_is_unlisted_and_the_slot_is_current(world, tmp_path,
                                                                                    monkeypatch) -> None:
    from superlocalmemory.core.feature_flags import FeatureFlags
    from superlocalmemory.portability import constants as C
    from superlocalmemory.truth.slot_pass import SlotPass
    from tests.test_truth.test_slot_pass import NEW, OLD, _fact

    monkeypatch.setattr(C, "DISPUTES_PER_RUN", 0)
    old = _fact(world.db, OLD, "2026-03-01T00:00:00+00:00")
    run = world.run("m.json", fx.dumps([{"id": "x", "memory": NEW, "created_at": "2026-11-10T00:00:00Z"}]))
    SlotPass(world.db, FeatureFlags(file_values={"truth.slots": "on"}, environ={}),
             state_dir=tmp_path / "slots").run()
    assert [dict(r)["state"] for r in world.db.execute("SELECT state FROM slot_disputes")] == ["dismissed"]
    assert dict(world.db.execute("SELECT state FROM memory_slots")[0])["state"] == "current"
    head = dict(world.db.execute("SELECT r.fact_id FROM memory_slots s JOIN slot_revisions r ON "
                                 "r.revision_id = s.head_revision_id")[0])["fact_id"]
    assert head == old and run["saved_items"] == 1


def test_the_left_out_line_reads_right_for_a_count_of_one() -> None:
    """A reason word is a plural phrase: the count follows it, never "1 copies of ..."."""
    from types import SimpleNamespace

    from superlocalmemory.portability import summary

    parsed = SimpleNamespace(format="chatgpt", items=[], dates=("", ""), conversations=2,
                             skipped={"duplicate": 1, "empty": 2}, notes=[])
    text = summary.describe(parsed, {})["summary"]
    assert "copies of something already in this file: 1" in text
    assert "empty messages or entries: 2" in text
    assert "1 copies" not in text
