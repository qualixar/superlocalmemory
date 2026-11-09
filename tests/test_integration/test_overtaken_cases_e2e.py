# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The user's action wins over a machine proposal, through a REAL daemon (4.1.22).

Varun's decision (2026-10-06): a delete, replace or edit by the user goes
through a correction SLM proposed by itself and nobody reviewed; that case is
closed as overtaken and can be put back. A case a person proposed, or one
already applied, still refuses. Everything here runs over HTTP, the real CLI
(``slm delete``, ``slm corrections``) and the real MCP tools, against a daemon
on a private port and a synthetic store.

Needs ``SLM_TEST_MODEL_CACHE`` (see test_erasure_integrity_e2e.py).
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid

from tests.test_integration.test_erasure_integrity_e2e import (  # noqa: F401 - fixture
    REPO_ROOT,
    RealDaemon,
    _count,
    _machine_case,
    _remember_complete,
    _ro,
    daemon,
    stub_embedder,
)

# One topic per memory: near-duplicates (even with a different tag) are merged
# into the existing fact on save, so every memory here is about something else.
_TOPICS = iter((
    "the blue kayak is stored in shed nine",
    "Orvel prefers jasmine tea after lunch",
    "the quarterly bird census starts in early May",
    "the attic network switch runs firmware seven",
    "Marisol's violin lesson moved to Thursday evenings",
    "the compost bin needs turning every second weekend",
    "invoice batches for Kestrel Ltd are sent on the 28th",
    "the spare car key hangs behind the pantry door",
    "Dr. Pellam recommended a ten minute stretch each morning",
    "the greenhouse thermostat is set to nineteen degrees",
    "Tobin's dentist appointment is on a Tuesday in March",
    "the bicycle chain was oiled after the coastal ride",
    "the library returns box closes at six on Saturdays",
    "Quill the parrot eats sunflower seeds and apple",
    "the basement dehumidifier empties into a blue bucket",
    "the team retrospective uses the sailboat format",
    "Ansel collects postage stamps from Iceland",
    "the solar inverter reports through a green status light",
))


def _facts(daemon: RealDaemon, n: int, word: str) -> list[str]:
    """``n`` memories, each its own fact."""
    tag = uuid.uuid4().hex[:6]
    out = [_remember_complete(daemon, f"Synthetic {word} {tag}: {next(_TOPICS)}", "")[0]
           for _ in range(n)]
    assert len(set(out)) == n, out
    return out


def _retry_503(daemon: RealDaemon, method: str, path: str, body: dict | None = None):
    for _ in range(45):  # 503 = the writer is busy; the route says retry
        code, out = daemon.request(method, path, body)
        if code != 503:
            return code, out
        time.sleep(2)
    return code, out


def _slm(daemon: RealDaemon, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "superlocalmemory.cli.main", *args],
                          env=daemon.env, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=180)


def _pair(daemon: RealDaemon, case_id: str) -> tuple[str, str]:
    """The case's own two facts (SLM may have proposed one by itself already)."""
    [(pred, succ)] = _ro(daemon, "SELECT predecessor_fact_id, successor_fact_id FROM "
                         "correction_cases WHERE case_id = ?", (case_id,))
    return str(pred), str(succ)


def _status(daemon: RealDaemon, case_id: str) -> list[tuple]:
    return _ro(daemon, "SELECT status, proposed_by_actor_kind FROM correction_cases "
               "WHERE case_id = ?", (case_id,))


def _human_case_on(daemon: RealDaemon, fact_id: str) -> tuple[str, int]:
    rows = _ro(daemon, "SELECT case_id, version FROM correction_cases WHERE "
               "predecessor_fact_id = ? AND proposed_by_actor_kind = 'host_authenticated' "
               "AND status = 'proposed'", (fact_id,))
    assert len(rows) == 1, rows
    return str(rows[0][0]), int(rows[0][1])


def test_an_applied_machine_case_still_refuses_the_delete(daemon: RealDaemon) -> None:
    pred, succ = _facts(daemon, 2, "teal")
    case_id = _machine_case(daemon, pred, succ)
    code, body = _retry_503(daemon, "POST", f"/api/corrections/{case_id}/apply",
                            {"expected_version": 0})
    assert code == 200, body
    assert _status(daemon, case_id) == [("applied", "host_attested")]

    for fact in (pred, succ):
        code, body = daemon.request("DELETE", f"/api/memories/{fact}")
        assert code == 409, body
        assert "protected by correction history" in str(body.get("detail")), body
    assert _count(daemon, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id IN (?, ?)",
                  (pred, succ)) == 2
    assert _count(daemon, "SELECT COUNT(*) FROM correction_cases_overtaken WHERE case_id = ?",
                  (case_id,)) == 0


def test_a_person_case_among_machine_cases_refuses_and_moves_nothing(daemon) -> None:
    a, b, c = _facts(daemon, 3, "amber")
    machine = _machine_case(daemon, a, b)          # b is its successor
    code, body = _retry_503(daemon, "PATCH", f"/api/memories/{b}",
                            {"content": "Synthetic amber tally, corrected by a person"})
    assert code in (200, 202), body
    human, _version = _human_case_on(daemon, b)    # b is its predecessor

    code, body = daemon.request("DELETE", f"/api/memories/{b}")

    assert code == 409, body
    assert human in str(body.get("detail")) and machine not in str(body.get("detail")), body
    assert _status(daemon, machine) == [("proposed", "host_attested")]
    assert _status(daemon, human) == [("proposed", "host_authenticated")]
    assert _count(daemon, "SELECT COUNT(*) FROM correction_cases_overtaken WHERE case_id IN "
                  "(?, ?)", (machine, human)) == 0
    del c


def test_edit_reject_then_restore_through_the_cli(daemon: RealDaemon) -> None:
    pred, succ = _facts(daemon, 2, "ochre")
    case_id = _machine_case(daemon, pred, succ)
    code, body = _retry_503(daemon, "PATCH", f"/api/memories/{pred}",
                            {"content": "Synthetic ochre tally, edited by the user"})
    assert code in (200, 202), body
    assert _status(daemon, case_id) == []          # overtaken by the edit

    listed = _slm(daemon, "corrections", "overtaken", "--json")
    assert listed.returncode == 0, (listed.stdout, listed.stderr)
    [row] = [r for r in json.loads(listed.stdout)["data"]["overtaken"]
             if r["case_id"] == case_id]
    assert row["user_action"] == "update" and row["restorable"] is False, row
    assert "another open correction" in row["not_restorable_because"], row

    early = _slm(daemon, "corrections", "restore-overtaken", case_id)
    assert early.returncode == 1 and "Refused:" in early.stderr, early.stderr
    assert _status(daemon, case_id) == []          # nothing changed

    human, version = _human_case_on(daemon, pred)
    code, body = _retry_503(daemon, "POST", f"/api/corrections/{human}/reject",
                            {"expected_version": version})
    assert code == 200, body

    done = _slm(daemon, "corrections", "restore-overtaken", case_id, "--json")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert json.loads(done.stdout)["data"]["restored"] == case_id
    assert _status(daemon, case_id) == [("proposed", "host_attested")]
    again = _slm(daemon, "corrections", "restore-overtaken", case_id, "--json")
    assert again.returncode == 1, again.stdout
    assert json.loads(again.stdout)["error"]["code"] == "CONFLICT"


def test_after_a_delete_the_case_is_listed_but_cannot_come_back(daemon) -> None:
    pred, succ = _facts(daemon, 2, "slate")
    case_id = _machine_case(daemon, pred, succ)
    pred, succ = _pair(daemon, case_id)
    deleted = _slm(daemon, "delete", succ, "--yes", "--json")   # the SUCCESSOR side
    assert deleted.returncode == 0, (deleted.stdout, deleted.stderr)
    assert _count(daemon, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id = ?", (succ,)) == 0
    assert _count(daemon, "SELECT COUNT(*) FROM atomic_facts WHERE fact_id = ?", (pred,)) == 1

    plain = _slm(daemon, "corrections", "overtaken")
    assert plain.returncode == 0 and case_id in plain.stdout, plain.stdout
    restore = _slm(daemon, "corrections", "restore-overtaken", case_id)
    assert restore.returncode == 1, restore.stdout
    assert "no longer exists" in restore.stderr and "Nothing was changed" in restore.stderr


def test_an_explicit_replace_goes_through_a_machine_proposal(daemon) -> None:
    old, other = _facts(daemon, 2, "umber")
    case_id = _machine_case(daemon, old, other)
    key = f"g07-replace-{uuid.uuid4().hex[:8]}"
    body = {"content": "Synthetic umber tally, the replacement the user wrote",
            "idempotency_key": key, "replaces": old}
    for _ in range(30):
        code, out = daemon.request("POST", "/remember", body)
        assert code in (200, 202), out
        if not out.get("replaced", {}).get("pending"):
            break
        time.sleep(2)  # 202: resend the same key for the final receipt
    assert out["replaced"]["ok"] is True, out["replaced"]
    assert out["replaced"]["fact_ids"] == [old]
    assert _ro(daemon, "SELECT user_action FROM correction_cases_overtaken WHERE case_id = ?",
               (case_id,)) == [("replace",)]
    code, listed = daemon.request("GET", "/api/overtaken-corrections")
    assert code == 200, listed
    [row] = [r for r in listed["overtaken"] if r["case_id"] == case_id]
    assert row["restorable"] is False  # the user's replacement now holds the memory


_MCP_CHILD = r"""
import asyncio, json
from superlocalmemory.mcp import tools_core
tools = {}
class S:
    def tool(self, *a, **k):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco
tools_core.register_core_tools(S(), lambda: None)
out = {"delete": asyncio.run(tools["delete_memory"](FACT)),
       "list": asyncio.run(tools["list_corrections"]())}
print(json.dumps(out))
"""


def test_mcp_delete_says_why_and_lists_overtaken_cases(daemon: RealDaemon) -> None:
    pred, succ = _facts(daemon, 2, "russet")
    code, body = _retry_503(daemon, "PATCH", f"/api/memories/{pred}",
                            {"content": "Synthetic russet tally, corrected by a person"})
    assert code in (200, 202), body
    gone, kept = _facts(daemon, 2, "sienna")
    overtaken = _machine_case(daemon, gone, kept)
    assert daemon.request("DELETE", f"/api/memories/{gone}")[0] == 200

    child = subprocess.run([sys.executable, "-c", f"FACT={pred!r}\n" + _MCP_CHILD],
                           env=daemon.env, cwd=str(REPO_ROOT), capture_output=True,
                           text=True, timeout=180)
    assert child.returncode == 0, child.stderr
    out = json.loads(child.stdout.strip().splitlines()[-1])
    refused = out["delete"]
    assert refused["success"] is False and refused["code"] == "CONFLICT", refused
    assert refused["retryable"] is False
    assert "protected by correction history" in refused["error"], refused
    assert overtaken in [r["case_id"] for r in out["list"]["overtaken"]], out["list"]
    del succ
