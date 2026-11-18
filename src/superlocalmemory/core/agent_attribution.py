# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Which agent saved a memory: the connection's name, or a name it merely claimed.

4.1.25 audit L-15 / G-M2. "Who saved what" was whatever the caller put in
``metadata.agent_id`` (or the ``agent_id`` argument of a tool call), so any
process could claim to be any agent, and the dashboard showed those claims next
to capability digests as if both were facts.

The connection is the better witness, when something vouches for it. The stdio
server's ``SLM_AGENT_ID`` is set by whoever configured the agent host. A name
in an HTTP MCP URL (``/mcp/<agent>``) is chosen by the caller on every request,
so it is only a claim unless that request also presents the daemon capability
(PR 248 review, finding 3: ``POST /mcp/claude-code/`` with no credential was
recorded as the "claude-code" connection). A vouched name rides on the daemon
request in :data:`CONNECTION_HEADER`, and the daemon believes it only from a
caller holding the daemon capability (a file only this computer's user can
read; the MCP server and the command line hold it, a web page does not). The
saved memory then records:

* ``agent_id``: the connection's name when there is one, else the claimed name;
* ``agent_id_source``: ``connection`` or ``claimed``;
* ``claimed_agent_id``: a different name the call asserted, kept, never trusted.
"""

from __future__ import annotations

from typing import Any

CONNECTION_HEADER = "X-SLM-Connection-Agent"
#: Whether the client process sat at an interactive terminal (WS-5, instruction
#: firewall): ``tty`` or ``no_tty``. Believed only from a capability holder, like
#: the connection name, and used only when the save names no agent at all.
SHELL_HEADER = "X-SLM-Client-Shell"
SHELL_TTY = "shell:tty"
SHELL_NO_TTY = "shell:no_tty"
#: Not a connection: the stdio sentinel when no agent was configured.
UNNAMED = "mcp_client"


def connection_agent() -> str:
    """This process's (or this HTTP MCP request's) vouched connection name, or ``""``.

    An unvouched URL name is not returned here; tools still pass it on as the
    claimed ``agent_id``, so it is recorded as ``claimed``.
    """
    from superlocalmemory.mcp.agent_context import (
        current_agent_is_vouched,
        get_current_agent_id,
        sanitize_agent_id,
    )

    name = get_current_agent_id()
    if not name or name == UNNAMED or not current_agent_is_vouched():
        return ""
    return sanitize_agent_id(name)


def forwarded_headers(capability: str | None) -> dict[str, str]:
    """The connection header a daemon request carries (only with the capability)."""
    agent = connection_agent()
    return {CONNECTION_HEADER: agent} if agent and capability is not None else {}


def shell_headers(capability: str | None) -> dict[str, str]:
    """The terminal header a daemon request carries (only with the capability).

    ``tty`` when this process's standard input and output are an interactive
    terminal, ``no_tty`` otherwise (a script, a hook, an agent's shell tool, the
    daemon itself). A closed or odd stream counts as no terminal.
    """
    if capability is None:
        return {}
    import sys

    try:
        tty = bool(sys.stdin is not None and sys.stdin.isatty()
                   and sys.stdout is not None and sys.stdout.isatty())
    except (ValueError, OSError):
        tty = False
    return {SHELL_HEADER: "tty" if tty else "no_tty"}


def shell_signal(value: str, trusted_actor_id: str) -> str | None:
    """``shell:tty`` / ``shell:no_tty`` from a request's header, or ``None``.

    Believed only from a daemon-capability caller; anything else (the dashboard,
    a remote app, a forged header) says nothing, so the save is judged as before.
    """
    if not str(trusted_actor_id or "").startswith("daemon-capability:"):
        return None
    return {"tty": SHELL_TTY, "no_tty": SHELL_NO_TTY}.get((value or "").strip().lower())


def bind(metadata: dict[str, Any], connection: str, shell: str | None = None) -> dict[str, Any]:
    """``metadata`` with the agent fields set from ``connection`` (may be ``""``).

    ``shell`` (``shell_signal``) is recorded as ``agent_id_source`` only when the
    save names no agent at all: it is the one thing left to say how the saver
    identified itself (``trust/origin_class``, row ``remember.shell_no_tty``).
    """
    out = dict(metadata)
    # A caller cannot set these itself: they say how far the name is believed.
    out.pop("agent_id_source", None)
    out.pop("claimed_agent_id", None)
    claimed = str(out.get("agent_id") or "").strip()
    if connection:
        out["agent_id"] = connection
        out["agent_id_source"] = "connection"
        if claimed and claimed not in (connection, UNNAMED):
            out["claimed_agent_id"] = claimed
    elif claimed:
        out["agent_id_source"] = "claimed"
    elif shell:
        out["agent_id_source"] = shell
    return out


def audit_fields(metadata: dict[str, Any] | None) -> dict[str, str]:
    """The agent fields an audit entry records for a save with ``metadata``."""
    meta = metadata or {}
    agent = str(meta.get("agent_id") or "")
    if not agent:
        return {}
    if meta.get("agent_id_source") == "connection":
        fields = {"connection_agent": agent}
        if meta.get("claimed_agent_id"):
            fields["claimed_agent_id"] = str(meta["claimed_agent_id"])
        return fields
    return {"claimed_agent_id": agent}


#: Explicit speaker values that mean the human, not an agent.
_HUMAN = frozenset({"user", "me", "the user", "human"})


def write_speaker(explicit: str, metadata: dict[str, Any] | None) -> tuple[str, str]:
    """``(speaker, role)`` for a save: who the first person ("I", "my") is.

    Issue #265: every daemon save used to default to ``role="user"``, so an
    agent writing "I prefer short answers" produced the derived fact "the user
    prefers short answers". A save that carries an ``agent_id`` (every tool
    call does; the command line, hooks and the dashboard do not) is the
    agent's own words unless the caller names another speaker. The stored
    memory is unchanged either way; this only steers fact extraction.
    """
    name = " ".join(str(explicit or "").split())[:64]
    if name.lower() in _HUMAN:
        return "", "user"
    if name:
        return name, "assistant"
    agent = str((metadata or {}).get("agent_id") or "").strip()
    if not agent:
        return "", "user"
    return ("" if agent == UNNAMED else agent), "assistant"


__all__ = [
    "CONNECTION_HEADER", "SHELL_HEADER", "SHELL_NO_TTY", "SHELL_TTY", "UNNAMED", "audit_fields",
    "bind", "connection_agent", "forwarded_headers", "shell_headers", "shell_signal",
    "write_speaker",
]
