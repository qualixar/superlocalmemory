# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Feature flags for the 4.2.30 subsystems: off, shadow or on.

Every new subsystem in 4.2.30 (slots, leases, the judge ladder, receipts, the
need gate, the taint firewall...) sits behind one flag so it can be turned off
on its own, and so it can run in *shadow*: compute and record its decision
without acting on it. Every flag is **on by default** (Varun, 19 Nov: "every
feature we are enabling should be on by default for my end users"); a gate that
needs a statistical basis it does not have yet (labels, certificates, an anytime
bound) acts exactly as before until the basis exists. A person's explicit off
(``features.json``, ``SLM_FEATURE_<ID>=off``) and the kill switch still win.

Flags live in their own file, ``<SLM_HOME>/features.json``, not in
``config.json``. ``config.json`` is rewritten whole by mode switches and the
dashboard; a flag kept there would be lost the first time someone switched
mode. An environment variable ``SLM_FEATURE_<ID>`` (dots become underscores,
upper case) overrides the file, for tests and one-off runs.

Unknown flag ids and unreadable values fall back to the flag's default, and a
missing or corrupt file means "all defaults": a flag lookup never raises.

The kill switch (blueprint D-23) turns every 4.2.30 subsystem off at once,
whatever each flag says: ``"kill_switch": true`` at the top of
``features.json`` (``set_kill_switch``) or ``SLM_4230_OFF=1`` in the
environment. Reads then behave as before 4.2.30, with no downgrade. It does
not undo retirements already applied; that is a separate, explicit ledger
command (``truth.slot_resolver.rollback_slot_cases``).
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
KILL_ENV = "SLM_4230_OFF"
KILL_KEY = "kill_switch"


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
    Flag("truth.slots", "WS-1", FlagState.ON,
         "Bind facts to slots and decide supersede / compatible / dispute"),
    Flag("truth.slots.auto_apply", "WS-1", FlagState.ON,
         "Apply deterministic slot supersessions through the correction ledger",
         shadowable=False),
    Flag("truth.leases", "WS-1", FlagState.ON,
         "Record source fingerprints and flag facts whose source changed"),
    Flag("judge.ladder", "WS-2", FlagState.ON,
         "Route typed memory questions through the judge ladder"),
    # D-22: measurement starts at W0, on by default (ids only, local). Since 19 Nov (Varun: "every
    # feature we are enabling should be on by default for my end users") every flag defaults to on;
    # a gate that needs a statistical basis it does not have yet acts exactly as before until the
    # basis exists, and the kill switch (``SLM_4230_OFF``) or a person's own off still wins.
    Flag("outcomes.receipts", "WS-3", FlagState.ON,
         "Write an injection receipt for every delivery (ids only, local)"),
    # D-49 (Varun, 18 Nov): usage detection is on by default. Without it withholding and
    # swaps can only ever measure ``reask`` and the position curve (which needs ``used`` in
    # all four cells) cannot be measured. It compares the agent's reply with the memories
    # delivered, locally, and keeps ids and scores only.
    Flag("outcomes.usage", "WS-3", FlagState.ON,
         "Detect whether delivered memories were used (local, ids and scores only)"),
    # D-28 + D-34 (Varun, 15-16 Nov): withholding is on by default for every
    # user, with a clear switch to turn it off. The default flipped in the commit
    # that landed D-28's shipping conditions: the install/upgrade disclosure
    # (outcomes.disclosure), the one off switch every surface calls
    # (outcomes.switch; slm measurement, the dashboard, plugin settings) and usage
    # detection (outcomes.usage). An RC gate test holds this.
    Flag("outcomes.holdout", "WS-3", FlagState.ON,
         "Withhold each eligible delivered memory with a small logged probability "
         "(never pinned rules, explicit recall, cues or lease warnings)",
         shadowable=False),
    # D-46 (Varun, 18 Nov): two neighbouring results of an agent's recall trade
    # places in 5% of recalls, logged, to measure position attention for the
    # ranker's offline gate. Same shipping rule as withholding (D-34): on by
    # default with the same disclosure and off switch.
    Flag("outcomes.swaps", "WS-3", FlagState.ON,
         "Swap two neighbouring results in a small logged share of agents' recalls "
         "(nothing withheld) to measure how much attention each position gets",
         shadowable=False),
    Flag("delivery.need_gate", "WS-4", FlagState.ON,
         "Decide per prompt whether memory is needed before recalling (shadow: the fixed "
         "rules still skip, the score never skips and the threshold is only logged, and "
         "memories are still injected on the other prompts, because labels need deliveries)"),
    Flag("delivery.cues", "WS-4", FlagState.ON,
         "Fire prospective memories on cues (file path, tool, error, repository, date)"),
    Flag("delivery.harm_gate", "WS-4", FlagState.ON,
         "Withhold memories whose expected value is negative"),
    Flag("trust.taint", "WS-5", FlagState.ON,
         "Track origin class and keep untrusted text out of instructions"),
    Flag("trust.secret_screen", "WS-5", FlagState.ON,
         "Screen saves for credentials before they are stored"),
    Flag("trust.signing", "WS-5", FlagState.ON,
         "Sign facts and key the audit chain"),
    Flag("compiler.auto_apply", "WS-6", FlagState.ON,
         "Apply deterministic-safe compiler proposals without review",
         shadowable=False),
    Flag("compiler.dream", "WS-6", FlagState.ON,
         "Dream cycle: in the background, lint the store and propose extractive pages and "
         "duplicate merges to the review queue (nothing is applied)",
         shadowable=False),
    Flag("compiler.wiki", "WS-6", FlagState.ON,
         "Project the compiled pages to a Markdown folder and read edits back as proposals "
         "(shadow: show what would be written, write nothing)"),
    # D-19, then "on by default" (Varun, 19 Nov): a caller may preview, run and undo an import of a
    # ChatGPT, Claude, Mem0, Letta, OKF or Markdown export (WS-9). Nothing runs on its own; an
    # import happens only when someone starts one. Nothing runs in shadow either.
    Flag("portability.import", "WS-9", FlagState.ON,
         "Import memories and conversations from ChatGPT, Claude, Mem0, Letta, OKF and "
         "Markdown files (preview, run in the background, undo the whole run)",
         shadowable=False),
    # D-19, then "on by default" (Varun, 19 Nov): the multimodal pack (WS-8). The flag is on, but a
    # part only acts once its model is on this computer. The running daemon downloads the models by
    # itself on a computer with 16 GB or more (``runtimes/media_auto``); on a smaller one a person
    # taps one button (``slm media pack enable``). Nothing downloads from a hook or a recall, and a
    # part someone turned off stays off.
    Flag("media.captions", "WS-8", FlagState.ON,
         "Describe each saved picture in a few words with a small local model, on every platform "
         "(the description is marked as written by a model; a failure never blocks the save)",
         shadowable=False),
    Flag("media.document_pack", "WS-8", FlagState.ON,
         "Read scanned document pages with the PaddleOCR-VL model instead of the default reader "
         "(the default reader stays as the fallback)",
         shadowable=False),
    Flag("media.regions", "WS-8", FlagState.ON,
         "Keep the page and the box each picture text came from, and show it in why "
         "(\"from page 3, top left\")",
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


def _read_file(slm_home: Path) -> dict[str, object]:
    path = Path(slm_home) / FEATURES_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("features.json unreadable (%s); using flag defaults", exc)
        return {}
    return dict(data) if isinstance(data, dict) else {}


def read_features_file(slm_home: Path) -> dict[str, object]:
    """The ``features`` map from ``features.json``; empty when absent or unreadable."""
    features = _read_file(slm_home).get("features")
    return dict(features) if isinstance(features, dict) else {}


def kill_switch_on(slm_home: Path | None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether the D-23 kill switch is set (environment or file); never raises."""
    env = os.environ if environ is None else environ
    if _parse(env.get(KILL_ENV, "")) is FlagState.ON:
        return True
    # Any truthy spelling counts (W0-19): a hand-written "on" or 1 must never fail open.
    if slm_home is None:
        return False
    value = _read_file(slm_home).get(KILL_KEY)
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)  # a hand-written 1
    return _parse(value) is FlagState.ON


def _write_file(slm_home: Path, data: dict[str, object]) -> None:
    path = Path(slm_home) / FEATURES_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def set_kill_switch(slm_home: Path, on: bool) -> None:
    """Set or clear the kill switch, keeping every flag as it is."""
    data = _read_file(slm_home)
    if on:
        data[KILL_KEY] = True
    else:
        data.pop(KILL_KEY, None)
    _write_file(slm_home, data)


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
                 file_values: Mapping[str, object] | None = None,
                 killed: bool | None = None) -> None:
        if file_values is None:
            file_values = read_features_file(slm_home) if slm_home is not None else {}
        self.killed = kill_switch_on(slm_home, environ) if killed is None else killed
        self._states = {
            fid: FlagState.OFF if self.killed else resolve(
                fid, file_values=file_values, environ=environ) for fid in FLAGS}

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
    data = _read_file(slm_home)  # keep the kill switch and anything else at the top
    values = read_features_file(slm_home)
    values[flag_id] = FlagState(state).value
    data["features"] = values
    _write_file(slm_home, data)


__all__ = ["FEATURES_FILE", "FLAGS", "FeatureFlags", "Flag", "FlagState", "KILL_ENV", "KILL_KEY",
           "env_name", "kill_switch_on", "read_features_file", "resolve", "set_kill_switch",
           "write_flag"]
