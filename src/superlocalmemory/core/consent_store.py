# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Consents to send memory text off this machine: kept apart from feature flags.

A feature flag says whether a subsystem runs; a consent says whether a person
agreed that their memories may leave the machine for a stated purpose. They
must not share a mechanism: flags can be switched by an environment variable
for a test run or turned on by a release scorecard, and neither may ever give
consent (4.2.30 blueprint D-20).

``<SLM_HOME>/consents.json`` holds one entry per purpose, each the literal
``true`` plus when and through which path it was given. Reading fails closed:
a missing, unreadable or hand-damaged file, an unknown purpose, or any value
other than ``true`` means no consent. There is no environment override.
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

#: purpose -> what the person agrees to, as shown when asking.
PURPOSES: Mapping[str, str] = {
    "judge_ladder_hosted": (
        "Let the judge ladder send pairs of your memories to the hosted Jev judge (TypeSafe) "
        "to decide whether one replaces the other. Texts are redacted for credentials first."),
}


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


def given(slm_home: Path, purpose: str) -> bool:
    """True only when this purpose has an explicit, well-formed consent on file."""
    if purpose not in PURPOSES:
        return False
    entry = _read(slm_home).get(purpose)
    return isinstance(entry, dict) and entry.get("consent") is True


def record(slm_home: Path, purpose: str, consent: bool, *, via: str) -> None:
    """Give or withdraw one consent; ``via`` names the path (CLI, dashboard, setup)."""
    if purpose not in PURPOSES:
        raise KeyError(f"unknown consent purpose: {purpose}")
    if not isinstance(consent, bool) or not via:
        raise ValueError("consent must be a bool and via must name the path it came from")
    values = _read(slm_home)
    values[purpose] = {"consent": consent, "at": datetime.now(UTC).isoformat(), "via": via}
    path = _path(slm_home)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


__all__ = ["CONSENTS_FILE", "PURPOSES", "given", "record"]
