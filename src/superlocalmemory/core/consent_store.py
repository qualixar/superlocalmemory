# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Consents to send memory text off this machine: kept apart from feature flags.

A feature flag says whether a subsystem runs; a consent says whether a person
agreed that their memories may leave the machine for a stated purpose. They
must not share a mechanism: flags can be switched by an environment variable
for a test run or turned on by a release scorecard, and neither may ever give
consent (4.2.30 blueprint D-20).

``<SLM_HOME>/consents.json`` holds one entry per purpose, each the literal
``true`` plus when and through which path it was given, and the provider it
names. Reading fails closed: a missing, unreadable or hand-damaged file, an
unknown purpose, any value other than ``true``, or a provider other than the
one consented to means no consent. There is no environment override.

People give or withdraw consent through ``change``: the dashboard (owner or
admin), or ``slm consent give`` at a terminal (never an agent: there is no MCP
tool). Each change first writes a receipt to the audit chain (who, when, which
notice, its hash); if that fails nothing changes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

CONSENTS_FILE = "consents.json"

#: purpose -> (notice version, what the person agrees to, as shown when asking).
NOTICES: Mapping[str, tuple[str, str]] = {
    "judge_ladder_hosted": (
        "jev-ladder/2026-11",
        "In the background, SuperLocalMemory sends pairs of your memories to {provider}, one "
        "pair per request, to ask whether the newer one replaces the older one. Passwords and "
        "keys are removed first. Don't use this for client, confidential or personal material."),
}
PURPOSES: Mapping[str, str] = {k: v[1] for k, v in NOTICES.items()}
_PROVIDER_LABELS = {"typesafe": "TypeSafe", "openrouter": "OpenRouter"}


class ConsentNotRecorded(RuntimeError):
    """The audit receipt could not be written, so the consent was not changed."""


def notice(purpose: str, provider: str) -> str:
    return NOTICES[purpose][1].replace("{provider}", _PROVIDER_LABELS.get(provider, provider))


def _path(slm_home: Path) -> Path:
    return Path(slm_home) / CONSENTS_FILE


def _read(slm_home: Path) -> dict[str, object]:
    try:
        data = json.loads(_path(slm_home).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("consents.json unreadable (%s); treated as no consent", exc)
        return {}
    return dict(data) if isinstance(data, dict) else {}


def given(slm_home: Path, purpose: str, *, provider: str | None = None) -> bool:
    """True only when this purpose has an explicit, well-formed consent on file,
    and, when ``provider`` is named, one given for that provider."""
    if purpose not in PURPOSES:
        return False
    entry = _read(slm_home).get(purpose)
    if not (isinstance(entry, dict) and entry.get("consent") is True):
        return False
    return provider is None or entry.get("provider") in (None, provider)


def state(slm_home: Path, purpose: str) -> dict[str, object]:
    """What the dashboard and ``slm consent`` show: given, when, how, for which provider."""
    entry = _read(slm_home).get(purpose)
    entry = entry if isinstance(entry, dict) else {}
    return {"purpose": purpose, "given": given(slm_home, purpose),
            "at": entry.get("at") if isinstance(entry.get("at"), str) else None,
            "via": entry.get("via") if isinstance(entry.get("via"), str) else None,
            "provider": entry.get("provider") if isinstance(entry.get("provider"), str) else None}


def reference(slm_home: Path, purpose: str, *, provider: str | None = None) -> str | None:
    """``"<purpose>@<when given>"`` for a consent on file, else None.

    Stamped on every receipt of a hosted ask (I-15), so a receipt names the
    record it relied on even after that consent is withdrawn. With ``provider``
    named, a consent given for another provider is none.
    """
    if not given(slm_home, purpose, provider=provider):
        return None
    at = _read(slm_home).get(purpose, {}).get("at")
    return f"{purpose}@{at if isinstance(at, str) and at else 'unknown'}"


def record(slm_home: Path, purpose: str, consent: bool, *, via: str,
           provider: str | None = None) -> None:
    """Give or withdraw one consent; ``via`` names the path (CLI, dashboard, setup)."""
    if purpose not in PURPOSES:
        raise KeyError(f"unknown consent purpose: {purpose}")
    if not isinstance(consent, bool) or not via:
        raise ValueError("consent must be a bool and via must name the path it came from")
    values = _read(slm_home)
    values[purpose] = {"consent": consent, "at": datetime.now(UTC).isoformat(), "via": via,
                       **({"provider": provider} if provider else {})}
    path = _path(slm_home)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def change(slm_home: Path, purpose: str, consent: bool, *, via: str, actor: str,
           provider: str, audit_path: Path | None = None) -> bool:
    """A person gives or withdraws ``purpose``: receipt first, then the change.

    Returns whether anything changed (giving what is already given for the same
    provider, or withdrawing what is not given, changes nothing). Raises
    ``ConsentNotRecorded`` when the receipt cannot be written; then nothing changed.
    """
    import hashlib

    if purpose not in PURPOSES:
        raise KeyError(f"unknown consent purpose: {purpose}")
    now = state(slm_home, purpose)
    if consent == now["given"] and (not consent or now["provider"] == provider):
        return False
    text = notice(purpose, provider)
    try:
        from superlocalmemory.compliance.audit import AuditChain
        from superlocalmemory.infra.data_root import state_path

        AuditChain(str(audit_path or state_path("audit_chain.db"))).log(
            operation=f"consent.{purpose}", agent_id=actor or "unknown",
            metadata={"decision": "granted" if consent else "withdrawn", "purpose": purpose,
                      "provider": provider, "via": via, "notice_version": NOTICES[purpose][0],
                      "notice_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    except Exception as exc:  # noqa: BLE001 - no receipt, no change
        raise ConsentNotRecorded(str(exc)) from exc
    record(slm_home, purpose, consent, via=via, provider=provider if consent else None)
    return True


__all__ = ["CONSENTS_FILE", "ConsentNotRecorded", "NOTICES", "PURPOSES", "change", "given",
           "notice", "record", "reference", "state"]
