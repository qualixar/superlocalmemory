"""The direct-database fallback of session start is for callers on this computer only."""

from superlocalmemory.mcp import tools_active
from superlocalmemory.mcp.remote_caller import remote_caller


def test_a_remote_caller_never_gets_the_direct_database_fallback(monkeypatch):
    called = []
    monkeypatch.setattr(tools_active, "_sqlite_emergency_recall",
                        lambda *a, **k: called.append(1) or "LOCAL")
    with remote_caller("key-1"):
        answer = tools_active._emergency_or_nothing("q", 5, "default", 30)
    assert called == [] and list(answer.results) == []


def test_a_local_caller_still_gets_the_fallback(monkeypatch):
    monkeypatch.setattr(tools_active, "_sqlite_emergency_recall", lambda *a, **k: "LOCAL")
    assert tools_active._emergency_or_nothing("q", 5, "default", 30) == "LOCAL"
