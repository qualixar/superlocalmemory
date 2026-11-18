# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm why`, `slm signing` and `slm doctor --truth` (signatures, audit chain): read-only, in ids and words."""

from __future__ import annotations

import json
import sqlite3
import sys
from argparse import Namespace

import pytest

from superlocalmemory.cli import doctor_trust, signing_cmd, why_cmd
from superlocalmemory.cli.doctor_truth import cmd_doctor_truth
from superlocalmemory.compliance.audit import AuditChain
from superlocalmemory.trust import signing_key as sk
from tests.test_trust_signing.conftest import SIGNING, save


@pytest.fixture
def held(monkeypatch, signer):
    """The one keychain read of a command returns this key (never the real keychain)."""
    monkeypatch.setattr(doctor_trust, "read_key", lambda store=None: sk.Loaded(signer, sk.READY))
    return signer


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(doctor_trust, "read_key",
                        lambda store=None: sk.Loaded(None, sk.UNAVAILABLE_STORE))


def _signed_store(db, monkeypatch, signer, text="Signed by the daemon."):
    monkeypatch.setenv(SIGNING, "on")
    sk.install(signer)
    fact = save(db, text)
    sk.install(None)
    return fact


def _json(capsys):
    out = capsys.readouterr().out
    return json.loads(out[out.index("{"):])


# -- slm why --------------------------------------------------------------------------------------

def test_why_prints_a_report_in_words(db, capsys) -> None:
    fact = save(db, "I keep tea in the blue tin.")
    why_cmd.run(Namespace(fact_id=fact.fact_id, json=False, verify=False, deliveries=10,
                          profile="", db=str(db.db_path)))
    out = capsys.readouterr().out
    assert fact.fact_id in out and "Where it came from" in out and "Signature" in out


def test_why_json_is_the_machine_readable_report(db, capsys) -> None:
    fact = save(db, "I keep tea in the blue tin.")
    why_cmd.run(Namespace(fact_id=fact.fact_id, json=True, verify=False, deliveries=3,
                          profile="", db=str(db.db_path)))
    data = _json(capsys)["data"]
    assert data["fact_id"] == fact.fact_id and "origin" in data and "deliveries" in data


def test_why_for_an_unknown_memory_exits_1_with_a_message(db, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        why_cmd.run(Namespace(fact_id="nope", json=False, verify=False, deliveries=10, profile="",
                              db=str(db.db_path)))
    assert exc.value.code == 1 and "No memory nope" in capsys.readouterr().err


def test_why_without_a_store_exits_2(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        why_cmd.run(Namespace(fact_id="x", json=False, verify=False, deliveries=10, profile="",
                              db=str(tmp_path / "missing.db")))
    assert exc.value.code == 2


def test_why_does_not_read_the_keychain_unless_asked(db, monkeypatch, capsys) -> None:
    fact = save(db, "Plain.")
    monkeypatch.setattr(doctor_trust, "read_key",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("keychain")))
    why_cmd.run(Namespace(fact_id=fact.fact_id, json=False, verify=False, deliveries=10,
                          profile="", db=str(db.db_path)))


def test_why_verify_checks_the_signature_with_the_key(db, monkeypatch, signer, held, capsys) -> None:
    fact = _signed_store(db, monkeypatch, signer)
    why_cmd.run(Namespace(fact_id=fact.fact_id, json=True, verify=True, deliveries=10,
                          profile="", db=str(db.db_path)))
    assert _json(capsys)["data"]["signature"]["state"] == "ok"


def test_why_changes_nothing_in_the_store(db, capsys) -> None:
    fact = save(db, "Untouched.")
    before = (db.db_path.stat().st_mtime_ns, db.db_path.stat().st_size)
    why_cmd.run(Namespace(fact_id=fact.fact_id, json=False, verify=False, deliveries=10,
                          profile="", db=str(db.db_path)))
    assert (db.db_path.stat().st_mtime_ns, db.db_path.stat().st_size) == before


# -- slm doctor --truth ---------------------------------------------------------------------------

def _doctor(db, capsys, *, all_facts=False, as_json=True):
    args = Namespace(db=str(db.db_path), json=as_json, signatures_all=all_facts)
    code = 0
    try:
        cmd_doctor_truth(args)
    except SystemExit as exc:
        code = exc.code
    return code, capsys.readouterr().out


def test_doctor_truth_reports_signatures_and_exits_1_on_a_modified_memory(
        db, monkeypatch, signer, held, capsys) -> None:
    fact = _signed_store(db, monkeypatch, signer)
    code, out = _doctor(db, capsys)
    sig = json.loads(out[out.index("{"):])["data"]["signatures"]
    assert code == 0 and sig["signed"] == 1 and sig["modified"] == 0 and sig["key_status"] == "ready"
    db.execute("UPDATE atomic_facts SET content = 'Edited by a copy.' WHERE fact_id = ?",
               (fact.fact_id,))
    code, out = _doctor(db, capsys, all_facts=True)
    sig = json.loads(out[out.index("{"):])["data"]["signatures"]
    assert code == 1 and sig["modified"] == 1 and sig["modified_ids"] == [fact.fact_id]
    assert "Edited by a copy" not in out  # ids, never text


def test_doctor_truth_without_a_key_still_counts_and_says_why(db, monkeypatch, signer, no_key, capsys) -> None:
    _signed_store(db, monkeypatch, signer)
    code, out = _doctor(db, capsys, as_json=False)
    assert code == 0 and "1 of 1 memories are signed" in out
    assert "no supported keychain" in out and "no key file is written" in out


def test_doctor_truth_verifies_the_audit_chain_beside_the_store(db, held, capsys) -> None:
    chain = AuditChain(db.db_path.parent / "audit_chain.db")
    sk.install(held)
    chain.log("store", agent_id="a")
    sk.install(None)
    code, out = _doctor(db, capsys, as_json=False)
    assert code == 0 and "Audit chain: 2 entries" in out and "2 keyed links checked by HMAC" in out
    with sqlite3.connect(db.db_path.parent / "audit_chain.db") as conn:
        conn.execute("UPDATE audit_chain SET agent_id = 'evil' WHERE id = 2")
    code, out = _doctor(db, capsys, as_json=False)
    assert code == 1 and "Audit chain: FAILED" in out


def test_doctor_truth_says_when_there_is_no_audit_file(db, held, capsys) -> None:
    code, out = _doctor(db, capsys, as_json=False)
    assert code == 0 and "Audit chain: no file next to this store" in out


# -- slm signing ----------------------------------------------------------------------------------

def test_signing_status_prints_and_exits_0(db, held, capsys) -> None:
    signing_cmd.run(Namespace(signing_command="status", db=str(db.db_path), json=False, all=False))
    assert "Signatures" in capsys.readouterr().out


def test_signing_status_json(db, held, monkeypatch, signer, capsys) -> None:
    _signed_store(db, monkeypatch, signer)
    signing_cmd.run(Namespace(signing_command=None, db=str(db.db_path), json=True, all=True))
    data = _json(capsys)["data"]
    assert data["signatures"]["signed"] == 1 and data["audit_chain"] == {"present": False}


def test_resign_without_a_terminal_needs_yes_and_never_prompts(db, held, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("an agent run must not prompt"))
    with pytest.raises(SystemExit) as exc:
        signing_cmd.run(Namespace(signing_command="resign", yes=False, db=str(db.db_path)))
    assert exc.value.code == 2 and "pass --yes" in capsys.readouterr().err


def test_an_agent_signs_again_with_yes_and_the_decision_is_recorded(
        db, held, monkeypatch, signer, capsys) -> None:
    """D-49: no terminal is needed; the run is a recorded decision, listed and not undoable."""
    from superlocalmemory.core import decisions

    old = _signed_store(db, monkeypatch, sk.Signer(bytes(reversed(range(32)))), "Signed under the lost key.")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(signing_cmd, "_daemon_running", lambda: False)
    monkeypatch.setenv("SLM_AGENT_ID", "codex-7")
    signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path),
                              profile="default"))
    assert "Signed 1 memories again" in capsys.readouterr().out
    assert db.execute("SELECT key_id FROM fact_signatures WHERE fact_id = ?",
                      (old.fact_id,))[0]["key_id"] == signer.key_id
    conn = sqlite3.connect(str(db.db_path))
    try:
        rows = [r for r in decisions.recent(conn, "default") if r["kind"] == decisions.SIGNING]
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0]["actor"] == "agent:codex-7" and rows[0]["channel"] == "cli"
    assert rows[0]["action"] == "resign" and rows[0]["can_undo"] is False


def _terminal(monkeypatch, *, daemon=False):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(signing_cmd, "_daemon_running", lambda: daemon)


def test_resign_refuses_while_the_daemon_runs(db, held, monkeypatch, capsys) -> None:
    _terminal(monkeypatch, daemon=True)
    with pytest.raises(SystemExit) as exc:
        signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path)))
    assert exc.value.code == 2 and "Stop it first" in capsys.readouterr().err


def test_resign_without_a_key_refuses_and_never_creates_one(db, no_key, monkeypatch, capsys) -> None:
    _terminal(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path)))
    assert exc.value.code == 2 and "no key file is written" in capsys.readouterr().err


def test_resign_signs_again_only_what_a_lost_key_signed(db, held, monkeypatch, signer, capsys) -> None:
    old = _signed_store(db, monkeypatch, sk.Signer(bytes(reversed(range(32)))), "Signed under the lost key.")
    _terminal(monkeypatch)
    signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path)))
    assert "Signed 1 memories again" in capsys.readouterr().out
    row = db.execute("SELECT key_id FROM fact_signatures WHERE fact_id = ?", (old.fact_id,))[0]
    assert row["key_id"] == signer.key_id
    signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path)))
    assert "Nothing to sign again" in capsys.readouterr().out


def test_resign_signs_a_memory_whose_signature_was_removed_after_the_person_says_yes(
        db, held, monkeypatch, capsys) -> None:
    fact = _signed_store(db, monkeypatch, held, "Run the tests before every commit.")
    db.execute("UPDATE atomic_facts SET content = 'Changed outside.' WHERE fact_id = ?", (fact.fact_id,))
    db.execute("DELETE FROM fact_signatures WHERE fact_id = ?", (fact.fact_id,))
    _terminal(monkeypatch)
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda q="": prompts.append(q) or "y")
    signing_cmd.run(Namespace(signing_command="resign", yes=False, db=str(db.db_path)))
    assert "lost their signature" in prompts[0] and "Signed 1 memories again" in capsys.readouterr().out
    signing_cmd.run(Namespace(signing_command="resign", yes=True, db=str(db.db_path)))
    assert "Nothing to sign again" in capsys.readouterr().out


def test_resign_asks_and_a_no_changes_nothing(db, held, monkeypatch, capsys) -> None:
    old = _signed_store(db, monkeypatch, sk.Signer(bytes(reversed(range(32)))), "Signed under the lost key.")
    _terminal(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    with pytest.raises(SystemExit) as exc:
        signing_cmd.run(Namespace(signing_command="resign", yes=False, db=str(db.db_path)))
    assert exc.value.code == 1
    row = db.execute("SELECT key_id FROM fact_signatures WHERE fact_id = ?", (old.fact_id,))[0]
    assert row["key_id"] != held.key_id


def test_the_parsers_register_and_every_command_is_listed_in_help() -> None:
    import argparse

    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    why_cmd.register_why_parsers(sub)
    signing_cmd.register_signing_parsers(sub)
    assert root.parse_args(["why", "abc", "--json", "--verify"]).fact_id == "abc"
    assert root.parse_args(["signing", "resign", "--yes"]).signing_command == "resign"
