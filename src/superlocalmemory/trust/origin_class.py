# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Where a memory came from, as one of ten classes, and how classes compare (M-15).

``storage/fact_origin`` (M074, D-21) records raw signals once, at admission:
the channel a fact arrived by, how its text relates to its input, the source
type, the agent attribution and the speaker and role. Nothing there says how
far to trust it. This module is the one place those signals become a class, so
a change to the mapping is one reviewed table, never data nobody kept.

The classes, best first (the lattice order of M-15)::

    user_direct  >=  user_confirmed  >=  system  >=  agent_inferred  >=
    tool_output  >=  mesh_peer  >=  web  >=  imported_file  >=  model_generated

``unknown`` is outside the order: it is not above or below anything, so a
comparison with it is never "at least" (it always disputes), and a join with it
stays unknown.

``user_confirmed`` is not read from any signal. It is what a person's approval
makes of a memory (the approval record is the instruction firewall's, package
B); ``fact_trust.mark_confirmed`` writes it, and no mapping row yields it.

How a class is decided (:func:`classify`): the first matching row of
:data:`ROWS` gives the base class from ``channel`` and the speaker/role
signals; then the ``transform`` may only lower it (a model that rewrote the
text can no longer vouch for who said it). Every row says why in its
``why``. Nothing here reads a database or memory text.

Fails closed (D-49): no signal is never the trusted case. A save is ``user_direct`` only
with positive evidence, the dashboard's own door (the raw signal ``agent_id_source`` =
``dashboard``, recorded by the daemon when the page's install-token request names the
dashboard channel), and then only with no agent id and role ``user``. A save from the command
line, a script, a hook or an agent's shell has no such signal and is ``agent_inferred``. The
old terminal check (``shell:tty``) was removed rather than patched: any local process can send
or omit a header, so it proved nothing. The dashboard signal is a recorded label of the door
used, not a claim that a person is present. What a promotion changes is not here: an explicit
promotion call (``trust/firewall``) is what lifts a memory above its class, and it is recorded.
Facts classified under mapping ``v2`` are recomputed under ``v3`` by the background batch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

#: The raw signal the dashboard's own door records (``agent_id_source``), see the module text.
DASHBOARD_SOURCE = "dashboard"

USER_DIRECT = "user_direct"
USER_CONFIRMED = "user_confirmed"
SYSTEM = "system"
AGENT_INFERRED = "agent_inferred"
TOOL_OUTPUT = "tool_output"
MESH_PEER = "mesh_peer"
WEB = "web"
IMPORTED_FILE = "imported_file"
MODEL_GENERATED = "model_generated"
UNKNOWN = "unknown"
#: Saved before this store took 4.2.30 (no origin row, created before ``fact_origin`` existed
#: here). Varun, 19 Nov: memories from before the upgrade keep working exactly as in 4.2.19, so
#: they rank with ``system`` (at the firewall's floor). Decided by date, not by the missing row:
#: a fact saved after the upgrade with no origin row is still ``unknown`` and fails closed.
LEGACY = "legacy"

#: Best first. The position is the rank: a smaller rank is more trusted.
ORDER: tuple[str, ...] = (USER_DIRECT, USER_CONFIRMED, SYSTEM, AGENT_INFERRED, TOOL_OUTPUT,
                          MESH_PEER, WEB, IMPORTED_FILE, MODEL_GENERATED)
CLASSES: frozenset[str] = frozenset(ORDER) | {UNKNOWN, LEGACY}
_RANK: Mapping[str, int] = {**{name: i for i, name in enumerate(ORDER)},
                            LEGACY: ORDER.index(SYSTEM)}

#: What people read (dashboard "Where this came from", ``slm why``).
LABELS: Mapping[str, str] = {
    USER_DIRECT: "you said this",
    USER_CONFIRMED: "you confirmed this",
    SYSTEM: "built by SuperLocalMemory",
    AGENT_INFERRED: "an agent's own words",
    TOOL_OUTPUT: "a tool's output",
    MESH_PEER: "another agent's message",
    WEB: "a web page",
    IMPORTED_FILE: "an imported file",
    MODEL_GENERATED: "written by a model",
    UNKNOWN: "unknown (its source was not recorded)",
    LEGACY: "saved before 4.2.30 (trusted as it was before)",
}

#: The mapping version, the prefix of every ``basis``. A change to :data:`ROWS`
#: or to the caps bumps it, and the background batch recomputes older rows.
VERSION = "v3"


def rank_of(name: str) -> int | None:
    """The position in the order (0 = most trusted), ``None`` for ``unknown``."""
    return _RANK.get(name)


def at_least(a: str, b: str) -> bool:
    """``a`` is at least as trusted as ``b`` (reflexive). Never true with ``unknown``."""
    ra, rb = _RANK.get(a), _RANK.get(b)
    return ra is not None and rb is not None and ra <= rb


def join(a: str, b: str) -> str:
    """The taint of two things mixed: the lower class; ``unknown`` if either is."""
    ra, rb = _RANK.get(a), _RANK.get(b)
    if ra is None or rb is None:
        return UNKNOWN
    if ra == rb and a != b:  # ``legacy`` shares ``system``'s rank: the mix reads as ``system``
        return SYSTEM
    return a if ra >= rb else b


def join_all(classes) -> str:
    """The join of any number of classes (``user_direct`` for none)."""
    out = USER_DIRECT
    for name in classes:
        out = join(out, name)
    return out


def label(name: str) -> str:
    return LABELS.get(name, LABELS[UNKNOWN])


@dataclass(frozen=True)
class Signals:
    """The raw evidence ``fact_origin`` keeps for one fact."""

    channel: str = "unknown"
    transform: str = "unknown"
    source_type: str | None = None
    agent_id: str | None = None
    speaker: str | None = None
    role: str | None = None
    #: How the saver identified itself (``fact_origin.agent_id_source``): ``connection``,
    #: ``claimed``, or, for a save with no agent at all, ``dashboard`` (the page's own door).
    agent_id_source: str | None = None


@dataclass(frozen=True)
class Classified:
    origin_class: str
    #: ``<VERSION>:<row id>`` (plus ``+<cap>`` when the transform lowered it).
    basis: str


@dataclass(frozen=True)
class Row:
    row_id: str
    applies: Callable[[Signals], bool]
    origin_class: str
    why: str


def _in(*channels: str) -> Callable[[Signals], bool]:
    return lambda s: s.channel in channels


def _from_the_dashboard(s: Signals) -> bool:
    """A save with no agent behind it, role user, through the dashboard's own door (D-49)."""
    return (not s.agent_id and not s.speaker and (s.role or "").lower() == "user"
            and (s.agent_id_source or "") == DASHBOARD_SOURCE)


def _source_has(*words: str) -> Callable[[Signals], bool]:
    """Channel not named by the save path (unknown or derived), source type says what it is."""
    return lambda s: (s.channel in ("unknown", "derived")
                      and any(w in (s.source_type or "").lower() for w in words))


#: First match wins. Order matters only where two rows could both match.
ROWS: tuple[Row, ...] = (
    Row("source.mesh", _source_has("mesh"), MESH_PEER,
        "a message another agent sent over the mesh is that agent's claim, not ours"),
    Row("source.web", _source_has("web", "browser", "url", "crawl", "scrape", "fetch"), WEB,
        "text taken from a web page is whatever the page said, however it got here"),
    Row("source.tool", _source_has("tool", "bash", "shell", "command", "exec"), TOOL_OUTPUT,
        "the output of a command or tool is data the tool returned, not a statement"),
    Row("remember.dashboard", lambda s: s.channel in ("remember_http", "python_api")
        and _from_the_dashboard(s), USER_DIRECT,
        "a save typed into the dashboard: no agent id, role user and the dashboard's own door "
        "(the one positive signal of a user's statement; it records the door, it does not "
        "prove a person)"),
    Row("remember.agent", _in("remember_http", "python_api"), AGENT_INFERRED,
        "every other save through the shared route (an agent's call, the command line, a "
        "script, a hook) is the caller's own words: it may restate the user, or a tool, or "
        "guess, and no signal says otherwise, so it fails closed"),
    Row("hook.file_change", _in("hook_file_change"), TOOL_OUTPUT,
        "built from a tool event: the file path and edit come from the tool, not the user"),
    Row("hook.session_summary", _in("hook_session_summary"), AGENT_INFERRED,
        "a summary of what an agent session did; its content is the session's, not ours"),
    Row("observe.capture", _in("observe", "cli_observe"), AGENT_INFERRED,
        "auto-captured from conversation text: a hook chose it, nobody said 'remember this'"),
    Row("mcp.offline", _in("mcp_offline"), AGENT_INFERRED,
        "an MCP save queued while the daemon was away: an agent's call, replayed later"),
    Row("adapter.service", _in("ingest_adapter"), WEB,
        "mail, calendar and transcript text comes from other people over a service, like a "
        "page: it can carry instructions"),
    Row("import.file", _in("import", "folder", "document", "bundle"), IMPORTED_FILE,
        "text read from a file or an export: the file's author is unknown (D-19: the bottom "
        "of the lattice)"),
    Row("media.model", _in("media"), MODEL_GENERATED,
        "captions and OCR text are a model's reading of a picture or sound (D-19)"),
    Row("loop.ledger", _in("loop_ledger"), SYSTEM,
        "written by SuperLocalMemory's own bounded-loop ledger from its own records"),
    Row("correction.proposal", _in("correction"), AGENT_INFERRED,
        "a corrected statement is written by whoever called remember with replaces: a "
        "person or an agent, not told apart here (it also inherits its parent's taint)"),
    Row("edit.external", _in("external_edit"), AGENT_INFERRED,
        "text read back from a Markdown file of the living wiki (D-17). A person editing a file on "
        "their own disk is close to user_direct, but nothing tells that person from a tool or an "
        "agent that wrote the same file (no terminal, no agent id, a watcher reads it later), so "
        "the base class is the one a save with no proof of a person gets. A person's approval of "
        "the proposal that admits it is what makes it user_confirmed (trust/fact_trust.mark_confirmed)"),
    Row("derived.mechanical", _in("derived"), SYSTEM,
        "cut or built by SuperLocalMemory from other memories: the class comes from what "
        "it was cut from (taint), and a model's rewrite is capped below"),
    Row("origin.lost", _in("legacy_replay", "restore", "repair", "unknown"), UNKNOWN,
        "the signals were not kept or were rewritten (a replay, a restore, a repair, a "
        "save from before 4.2.30): nothing honest can be said"),
)

#: What the transform may do to the base class: only lower it.
_TRANSFORM_CAP: Mapping[str, str] = {
    # A model wrote this text (mode B or C extraction, a gist): it can no longer vouch
    # for who said it, and a prompt could have steered it.
    "model_rewritten": MODEL_GENERATED,
    # The text's relation to its input is not known: no better than an agent's guess.
    "unknown": AGENT_INFERRED,
}


def _lower(base: str, cap: str) -> str:
    rank_b, rank_c = _RANK.get(base), _RANK.get(cap)
    if rank_b is None:
        return UNKNOWN
    return cap if rank_c is not None and rank_c > rank_b else base


def classify(signals: Signals) -> Classified:
    """The class for one fact's own signals (before taint joins its parents)."""
    row = next((r for r in ROWS if r.applies(signals)), None)
    if row is None:  # every channel has a row; a channel added later lands here, unknown
        return Classified(UNKNOWN, f"{VERSION}:no_row")
    cap = _TRANSFORM_CAP.get(signals.transform)
    final = _lower(row.origin_class, cap) if cap else row.origin_class
    suffix = f"+{signals.transform}" if final != row.origin_class else ""
    return Classified(final, f"{VERSION}:{row.row_id}{suffix}")


__all__ = ["AGENT_INFERRED", "CLASSES", "Classified", "DASHBOARD_SOURCE", "IMPORTED_FILE", "LABELS", "MESH_PEER",
           "MODEL_GENERATED", "ORDER", "ROWS", "Row", "SYSTEM",
           "Signals", "TOOL_OUTPUT",
           "UNKNOWN", "USER_CONFIRMED", "USER_DIRECT", "VERSION", "WEB", "at_least", "classify",
           "join", "join_all", "label", "rank_of"]
