# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Feature flags for the 4.2.30 subsystems: off, shadow or on.

Every new subsystem in 4.2.30 (slots, leases, the judge ladder, receipts, the
need gate, the taint firewall...) ships behind one flag so it can be turned off
on its own, and so it can run in *shadow*: compute and record its decision
without acting on it. Shadow readings are how a subsystem earns its default.

Flags live in their own file, ``<SLM_HOME>/features.json``, not in
``config.json``. ``config.json`` is rewritten whole by mode switches and the
dashboard; a flag kept there would be lost the first time someone switched
mode. An environment variable ``SLM_FEATURE_<ID>`` (dots become underscores,
upper case) overrides the file, for tests and one-off runs.

Unknown flag ids and unreadable values fall back to the flag's default, and a
missing or corrupt file means "all defaults": a flag lookup never raises.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

FEATURES_FILE = "features.json"


class FlagState(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ON = "on"


@dataclass(frozen=True)
class Flag:
    flag_id: str
    workstream: str
    default: FlagState
    summary: str
    #: False for flags whose subsystem has no meaningful shadow (it either acts
    #: or does nothing), so "shadow" is read as "off" rather than half-acting.
    shadowable: bool = True


#: The registry. Adding a subsystem means adding its flag here first.
FLAGS: Mapping[str, Flag] = {f.flag_id: f for f in (
    Flag("truth.slots", "WS-1", FlagState.OFF,
         "Bind facts to slots and decide supersede / compatible / dispute"),
    Flag("truth.slots.auto_apply", "WS-1", FlagState.OFF,
         "Apply deterministic slot supersessions through the correction ledger",
         shadowable=False),
    Flag("truth.leases", "WS-1", FlagState.OFF,
         "Record source fingerprints and flag facts whose source changed"),
    Flag("judge.ladder", "WS-2", FlagState.OFF,
         "Route typed memory questions through the judge ladder"),
    Flag("outcomes.receipts", "WS-3", FlagState.OFF,
         "Write an injection receipt for every delivery"),
    Flag("outcomes.usage", "WS-3", FlagState.OFF,
         "Detect whether delivered memories were used (local, ids only)"),
    Flag("outcomes.holdout", "WS-3", FlagState.OFF,
         "Hold out a deterministic share of sessions from injection",
         shadowable=False),
    Flag("delivery.need_gate", "WS-4", FlagState.OFF,
         "Decide per prompt whether memory is needed before recalling"),
    Flag("delivery.cues", "WS-4", FlagState.OFF,
         "Fire prospective memories on cues (file, command, topic)"),
    Flag("delivery.harm_gate", "WS-4", FlagState.OFF,
         "Withhold memories whose expected value is negative"),
    Flag("trust.taint", "WS-5", FlagState.OFF,
         "Track origin class and keep untrusted text out of instructions"),
    Flag("trust.secret_screen", "WS-5", FlagState.OFF,
         "Screen saves for credentials before they are stored"),
    Flag("trust.signing", "WS-5", FlagState.OFF,
         "Sign facts and key the audit chain"),
    Flag("compiler.auto_apply", "WS-6", FlagState.OFF,
         "Apply deterministic-safe compiler proposals without review",
         shadowable=False),
    Flag("compiler.wiki", "WS-6", FlagState.OFF,
         "Project current-state pages to a Markdown wiki"),
    Flag("portability.importers", "WS-9", FlagState.OFF,
         "Offer importers for other memory products",
         shadowable=False),
)}


def env_name(flag_id: str) -> str:
    return "SLM_FEATURE_" + flag_id.replace(".", "_").upper()


def _parse(value: object) -> FlagState | None:
    if isinstance(value, bool):
        return FlagState.ON if value else FlagState.OFF
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    aliases = {"1": "on", "true": "on", "yes": "on", "0": "off", "false": "off", "no": "off"}
    try:
        return FlagState(aliases.get(text, text))
    except ValueError:
        return None


def read_features_file(slm_home: Path) -> dict[str, object]:
    """The ``features`` map from ``features.json``; empty when absent or unreadable."""
    path = Path(slm_home) / FEATURES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("features.json unreadable (%s); using flag defaults", exc)
        return {}
    features = data.get("features") if isinstance(data, dict) else None
    return dict(features) if isinstance(features, dict) else {}


def resolve(
    flag_id: str,
    *,
    file_values: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> FlagState:
    """The effective state of one flag: environment, then file, then default."""
    flag = FLAGS.get(flag_id)
    if flag is None:
        logger.warning("unknown feature flag %r read as off", flag_id)
        return FlagState.OFF
    env = os.environ if environ is None else environ
    state = _parse(env.get(env_name(flag_id), "")) or _parse(
        (file_values or {}).get(flag_id)) or flag.default
    if state is FlagState.SHADOW and not flag.shadowable:
        return FlagState.OFF
    return state


class FeatureFlags:
    """A snapshot of every flag for one SLM home, read once."""

    def __init__(self, slm_home: Path | None = None, *,
                 environ: Mapping[str, str] | None = None,
                 file_values: Mapping[str, object] | None = None) -> None:
        if file_values is None:
            file_values = read_features_file(slm_home) if slm_home is not None else {}
        self._states = {
            fid: resolve(fid, file_values=file_values, environ=environ) for fid in FLAGS}

    def state(self, flag_id: str) -> FlagState:
        if flag_id not in self._states:
            logger.warning("unknown feature flag %r read as off", flag_id)
        return self._states.get(flag_id, FlagState.OFF)

    def enabled(self, flag_id: str) -> bool:
        """True when the subsystem should act (``on``)."""
        return self.state(flag_id) is FlagState.ON

    def observing(self, flag_id: str) -> bool:
        """True when the subsystem should compute and record (``shadow`` or ``on``)."""
        return self.state(flag_id) is not FlagState.OFF

    def as_dict(self) -> dict[str, str]:
        return {fid: state.value for fid, state in self._states.items()}


def write_flag(slm_home: Path, flag_id: str, state: FlagState) -> None:
    """Persist one flag atomically, keeping the others as they are."""
    if flag_id not in FLAGS:
        raise KeyError(f"unknown feature flag: {flag_id}")
    path = Path(slm_home) / FEATURES_FILE
    values = read_features_file(slm_home)
    values[flag_id] = FlagState(state).value
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"features": values}, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


__all__ = ["FEATURES_FILE", "FLAGS", "FeatureFlags", "Flag", "FlagState", "env_name",
           "read_features_file", "resolve", "write_flag"]
