"""Local agent sessions register with an app and a name, and come back after a lost row."""

from superlocalmemory.mcp import tools_mesh


def test_registration_stores_app_and_display_name(broker) -> None:
    peer = broker.register_peer("s1", "work", agent_type="claude_code",
                                app_name="claude-code", display_name="Claude Code · proj")["peer_id"]
    got = {p["peer_id"]: p for p in broker.list_peer_overview()}[peer]
    assert got["app"] == "claude-code" and got["display_name"] == "Claude Code · proj"


def test_reregistering_keeps_an_owner_chosen_name(broker) -> None:
    peer = broker.register_peer("s1", app_name="codex", display_name="Codex · a")["peer_id"]
    broker.rename_peer(peer, "My Codex")
    broker.register_peer("s1", app_name="codex", display_name="Codex · a")
    assert broker.list_peer_overview()[0]["display_name"] == "My Codex"


def test_overview_reports_stale_and_hides_expired(broker) -> None:
    fresh = broker.register_peer("fresh", app_name="codex")["peer_id"]
    old = broker.register_peer("old", app_name="codex")["peer_id"]
    gone = broker.register_peer("gone", app_name="codex")["peer_id"]
    from datetime import datetime, timedelta, timezone

    def beat(peer, minutes):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        c = broker._conn()
        c.execute("UPDATE mesh_peers SET last_heartbeat=? WHERE peer_id=?", (ts, peer))
        c.commit()
        c.close()

    beat(old, 10)
    beat(gone, 45)
    status = {p["peer_id"]: p["status"] for p in broker.list_peer_overview()}
    assert status == {fresh: "active", old: "stale"}


def test_heartbeat_for_a_lost_peer_registers_again(monkeypatch) -> None:
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append(path)
        if path == "/heartbeat":
            return None  # the daemon no longer knows this peer
        return {"ok": True, "peer_id": "newpeer"}

    monkeypatch.setattr(tools_mesh, "_mesh_request", fake)
    monkeypatch.setattr(tools_mesh, "_REGISTERED", True)
    tools_mesh._heartbeat_once()
    assert "/register" in calls and tools_mesh._PEER_ID == "newpeer"


def test_registration_payload_names_the_app(monkeypatch) -> None:
    sent = {}
    monkeypatch.setattr(tools_mesh, "_mesh_request",
                        lambda m, p, body=None, **k: sent.update(body or {}) or {"ok": True, "peer_id": "p"})
    monkeypatch.setattr(tools_mesh, "_start_heartbeat", lambda: None)
    monkeypatch.setenv("SLM_MCP_HOST", "codex")
    monkeypatch.setenv("SLM_AGENT_ID", "codex")
    monkeypatch.setattr(tools_mesh, "_PROJECT_PATH", "/work/proj")
    tools_mesh._register_local_session()
    assert sent["app"] == "codex"
    assert sent["display_name"].startswith("Codex")
