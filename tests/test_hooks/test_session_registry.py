"""Privacy-safe recent-client summaries for the Living Brain."""

from __future__ import annotations

import json
import os

from superlocalmemory.hooks import session_registry


def test_active_client_summary_aggregates_hosts_without_session_ids(
    tmp_path, monkeypatch,
) -> None:
    registry = tmp_path / "active-sessions.json"
    now_ns = 1_000_000_000_000
    monkeypatch.setattr(session_registry, "_registry_file", lambda: registry)
    monkeypatch.setattr(session_registry, "_now_ns", lambda: now_ns)
    registry.write_text(json.dumps({
        "1": {
            "session_id": "secret-codex-session", "agent_type": "codex",
            "profile_id": "default", "ts_ns": now_ns - 5_000_000_000,
        },
        "2": {
            "session_id": "secret-claude-session", "agent_type": "claude",
            "profile_id": "default", "ts_ns": now_ns - 10_000_000_000,
        },
        "3": {
            "session_id": "stale", "agent_type": "cursor",
            "profile_id": "default", "ts_ns": now_ns - 400_000_000_000,
        },
    }), encoding="utf-8")

    clients = session_registry.active_client_summary("default", within_seconds=60)

    assert clients == [
        {"kind": "claude_code", "active": True, "last_seen_seconds_ago": 10,
         "source": "session_registry", "is_real": True},
        {"kind": "codex", "active": True, "last_seen_seconds_ago": 5,
         "source": "session_registry", "is_real": True},
    ]
    assert "secret" not in json.dumps(clients)


def test_active_client_summary_maps_unknown_hosts_to_safe_bucket(
    tmp_path, monkeypatch,
) -> None:
    registry = tmp_path / "active-sessions.json"
    now_ns = 1_000_000_000_000
    monkeypatch.setattr(session_registry, "_registry_file", lambda: registry)
    monkeypatch.setattr(session_registry, "_now_ns", lambda: now_ns)
    registry.write_text(json.dumps({
        "1": {
            "session_id": "private", "agent_type": "untrusted-host-name",
            "profile_id": "default", "ts_ns": now_ns,
        },
    }), encoding="utf-8")

    assert session_registry.active_client_summary("default") == [
        {"kind": "other", "active": True, "last_seen_seconds_ago": 0,
         "source": "session_registry", "is_real": True},
    ]


def test_active_client_summary_does_not_cross_profiles(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "active-sessions.json"
    now_ns = 1_000_000_000_000
    monkeypatch.setattr(session_registry, "_registry_file", lambda: registry)
    monkeypatch.setattr(session_registry, "_now_ns", lambda: now_ns)
    registry.write_text(json.dumps({
        "1": {
            "session_id": "a", "agent_type": "codex", "profile_id": "a",
            "ts_ns": now_ns,
        },
        "2": {
            "session_id": "b", "agent_type": "claude", "profile_id": "b",
            "ts_ns": now_ns,
        },
        # Legacy unscoped entries must not be guessed into either profile.
        "3": {"session_id": "old", "agent_type": "cursor", "ts_ns": now_ns},
    }), encoding="utf-8")

    assert [item["kind"] for item in session_registry.active_client_summary("a")] == ["codex"]
    assert [item["kind"] for item in session_registry.active_client_summary("b")] == ["claude_code"]


def test_resolve_active_profile_uses_canonical_profile_state(
    tmp_path, monkeypatch,
) -> None:
    """Hook presence must follow a daemon profile switch without env wiring."""
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"active_profile": "research"}), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)

    assert session_registry.resolve_active_profile() == "research"


def test_resolve_active_profile_supports_legacy_profile_manager_state(
    tmp_path, monkeypatch,
) -> None:
    """The legacy active *name* must resolve to its durable profile ID."""
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "active": "Research workspace",
        "profiles": [
            {"name": "default", "profile_id": "default-id"},
            {"name": "Research workspace", "profile_id": "research-id"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)
    registry = tmp_path / "active-sessions.json"
    now_ns = 1_000_000_000_000
    monkeypatch.setattr(session_registry, "_registry_file", lambda: registry)
    monkeypatch.setattr(session_registry, "_now_ns", lambda: now_ns)

    assert session_registry.resolve_active_profile() == "research-id"
    session_registry.mark_active(
        "legacy-private-session",
        agent_type="codex",
        profile_id=session_registry.resolve_active_profile(),
    )
    assert [item["kind"] for item in session_registry.active_client_summary("research-id")] == ["codex"]
    assert session_registry.active_client_summary("default-id") == []


def test_resolve_active_profile_canonicalizes_converted_legacy_pointer(
    tmp_path, monkeypatch,
) -> None:
    """Compatibility conversion may rename ``active`` but retain list entries."""
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "active_profile": "Research workspace",
        "profiles": [
            {"name": "Research workspace", "profile_id": "research-id"},
        ],
    }), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)

    assert session_registry.resolve_active_profile() == "research-id"


def test_resolve_active_profile_rejects_an_unmapped_pointer(tmp_path, monkeypatch) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "active_profile": "unmapped display name",
        "profiles": {"default-id": {"name": "default"}},
    }), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)

    assert session_registry.resolve_active_profile() is None


def test_resolve_active_profile_rejects_a_malformed_catalog(tmp_path, monkeypatch) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "active_profile": "research-id",
        "profiles": ["not a profile object"],
    }), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)

    assert session_registry.resolve_active_profile() is None


def test_resolve_active_profile_does_not_fall_back_to_stale_legacy_pointer(
    tmp_path, monkeypatch,
) -> None:
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({
        "active_profile": "unmapped modern pointer",
        "active": "Research workspace",
        "profiles": [{"name": "Research workspace", "profile_id": "research-id"}],
    }), encoding="utf-8")
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: profiles)

    assert session_registry.resolve_active_profile() is None


def test_codex_hook_records_the_canonical_active_profile(monkeypatch) -> None:
    from superlocalmemory.hooks import hook_handlers

    captured: dict[str, str] = {}
    # _apply_codex_session updates this process environment for its legacy
    # lifecycle bridge. Register the original through monkeypatch so later
    # MCP tests cannot inherit this synthetic Codex session identity.
    monkeypatch.setenv("CLAUDE_SESSION_ID", "before-profile-test")
    monkeypatch.setattr(session_registry, "resolve_active_profile", lambda: "research")
    monkeypatch.setattr(
        session_registry,
        "mark_active",
        lambda session_id, agent_type, profile_id: captured.update({
            "session_id": session_id,
            "agent_type": agent_type,
            "profile_id": profile_id,
        }),
    )

    hook_handlers._apply_codex_session({
        "cwd": "/tmp/project",
        "session_id": "codex-profile-session",
    })

    assert captured == {
        "session_id": "codex-profile-session",
        "agent_type": "codex",
        "profile_id": "research",
    }


def test_missing_profile_authority_preserves_legacy_attribution_but_hides_presence(
    tmp_path, monkeypatch,
) -> None:
    """An unreadable profile cache must not change existing session attribution."""
    registry = tmp_path / "active-sessions.json"
    monkeypatch.setattr(session_registry, "_registry_file", lambda: registry)
    monkeypatch.setattr(session_registry, "_profiles_file", lambda: tmp_path / "missing.json")

    assert session_registry.resolve_active_profile() is None
    session_registry.mark_active(
        "unknown-profile-session",
        agent_type="codex",
        profile_id=session_registry.resolve_active_profile(),
    )

    row = json.loads(registry.read_text(encoding="utf-8"))[str(os.getpid())]
    assert row["session_id"] == "unknown-profile-session"
    assert "profile_id" not in row
    assert session_registry.active_client_summary("default") == []
