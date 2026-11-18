# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""D-49: locality comes from the connection, the actor from the credential, never a parameter."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core.agent_attribution import CONNECTION_HEADER
from superlocalmemory.server import decision_door as door

CAP = "daemon-capability:abcdef012345"
TOKEN = "local-capability:dashboard:owner:ns:feedfacecafe"
KEY = "api-key:dashboard:0123456789abcdef0123"


def _req(host="127.0.0.1", headers=None, query=None):
    return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers or {},
                           query_params=query or {})


def test_a_loopback_peer_is_local_and_another_machine_is_not() -> None:
    assert door.peer_is_local(_req("127.0.0.1")) and door.peer_is_local(_req("::1"))
    assert not door.peer_is_local(_req("203.0.113.9")) and not door.peer_is_local(_req(""))


def test_a_query_parameter_cannot_make_a_remote_caller_local() -> None:
    remote = _req("203.0.113.9", {door.CHANNEL_HEADER: "dashboard"}, {"caller_view": ""})
    who = door.decider_for(remote, KEY)
    assert who.channel == "http" and who.actor.startswith("api-key:")
    assert door.peer_is_local(_req("203.0.113.9", query={"caller_view": "local"})) is False


def test_a_remote_caller_cannot_claim_the_dashboard_or_an_agent_name() -> None:
    remote = _req("203.0.113.9", {door.CHANNEL_HEADER: "dashboard", CONNECTION_HEADER: "claude"})
    who = door.decider_for(remote, KEY, claimed="the-owner")
    assert who == door.Decider(who.actor, "http") and "claude" not in who.actor
    assert "owner" not in who.actor


def test_the_dashboard_door_is_the_install_token_with_the_page_header() -> None:
    page = _req(headers={door.CHANNEL_HEADER: "dashboard"})
    assert door.decider_for(page, TOKEN) == door.Decider("dashboard", "dashboard")
    # the daemon capability alone, or the header alone, is not the dashboard
    assert door.decider_for(page, CAP).channel != "dashboard"
    assert door.decider_for(_req(), TOKEN).channel == "http"


def test_a_capability_holder_names_its_agent_and_its_channel() -> None:
    mcp = _req(headers={CONNECTION_HEADER: "claude-code", door.CHANNEL_HEADER: "mcp"})
    assert door.decider_for(mcp, CAP) == door.Decider("agent:claude-code", "mcp")
    cli = _req(headers={door.CHANNEL_HEADER: "cli"})
    assert door.decider_for(cli, CAP) == door.Decider("client:local", "cli")
    # the agent header is believed only from a capability holder
    assert door.decider_for(_req(headers={CONNECTION_HEADER: "evil"}), TOKEN).actor == "client:local"


def test_a_client_name_is_cleaned_and_unknown_channels_become_http() -> None:
    who = door.decider_for(_req(headers={door.CHANNEL_HEADER: "person"}), CAP, claimed="me <b>")
    assert who.channel == "http" and "<" not in who.actor and who.actor.startswith("client:")
