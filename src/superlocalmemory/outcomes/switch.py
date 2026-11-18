# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The one off switch for what SLM measures (4.2.30 D-28).

``slm measurement``, the dashboard card and the plugin skills all call this
module, so there is one definition of "measuring" and one place that changes it.
The interventions, randomised withholding (``outcomes.holdout``), logged swaps
(``outcomes.swaps``) and the usage check (``outcomes.usage``, on by default since D-49: the
other two cannot observe a ``used`` outcome without it), go on and off together. Receipts
(``outcomes.receipts``: ids and numbers only) are left alone unless asked for.
The need gate is a separate choice and is never touched here. The kill switch
(``slm flags kill``) still turns everything off.

Every change is written to the history in ``disclosures.json`` with its time and
surface; a failure to write that history never stops a switch being turned off.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.core.feature_flags import FeatureFlags, FlagState, write_flag
from superlocalmemory.outcomes import disclosure

logger = logging.getLogger(__name__)

INTERVENTIONS = ("outcomes.holdout", "outcomes.swaps")
USAGE = "outcomes.usage"
#: What ``slm measurement off`` / ``on`` changes (receipts only when asked).
SWITCHED = (*INTERVENTIONS, USAGE)
RECEIPTS = "outcomes.receipts"

#: (flag, short label, what it does in plain words)
_FLAG_WORDS = (
    ("outcomes.receipts", "Receipts",
     "Writes a receipt each time memories are given to an agent: ids, rank, score and the "
     "logged chance of each choice. This computer only."),
    ("outcomes.holdout", "Withholding",
     "In a small share of session starts, leaves out one eligible memory or switches off one "
     "way of searching, by a logged random draw. Never pinned rules, recall, cues or lease "
     "warnings."),
    ("outcomes.swaps", "Swaps",
     "In a small share of an agent's recalls, two neighbouring results trade places. Nothing "
     "is left out or added."),
    ("outcomes.usage", "Usage check",
     "Compares an agent's last reply with the memories it was given to see which were used. "
     "Ids and scores only; this computer only. Switched off with the withholding and swaps."),
    ("delivery.need_gate", "Need check",
     "Decides for each prompt whether memory is needed, and recalls a few of the prompts it "
     "would skip to measure its own misses. Never stores the prompt text. Off unless you "
     "turn it on."),
)


def _check_channel(channel: str) -> None:
    if channel not in disclosure.CHANNELS:
        raise ValueError(f"channel must be one of {', '.join(disclosure.CHANNELS)}")


def status(slm_home: Path) -> dict[str, object]:
    """Each flag's effective state (environment, file, default), the kill switch,
    whether the interventions are running, and whether the notice is pending."""
    flags = FeatureFlags(slm_home)
    rows = [{"id": fid, "label": label, "what": what,
             "state": "on" if flags.enabled(fid) else "off"}
            for fid, label, what in _FLAG_WORDS]
    return {
        "kill_switch": flags.killed,
        "measuring": any(flags.enabled(f) for f in INTERVENTIONS),
        "flags": rows,
        "notice_version": disclosure.NOTICE_VERSION,
        "pending": disclosure.pending(slm_home),
    }


def set_measurement(slm_home: Path, on: bool, *, channel: str,
                    include_receipts: bool = False) -> dict[str, object]:
    """Turn the interventions (and, if asked, the receipts) on or off.

    Returns ``{"changed": {flag: state}}`` listing only what actually changed.
    """
    _check_channel(channel)
    target = FlagState.ON if on else FlagState.OFF
    wanted = (*SWITCHED, RECEIPTS) if include_receipts else SWITCHED
    changed: dict[str, str] = {}
    for flag_id in wanted:
        # Compare the stored choice, not an environment override or the kill
        # switch: the person's file is what this command owns.
        stored = FeatureFlags(slm_home, environ={}, killed=False).state(flag_id)
        if stored is not target:
            write_flag(slm_home, flag_id, target)
            changed[flag_id] = target.value
    if changed:
        _record(slm_home, on, channel, changed)
    return {"changed": changed}


def _record(slm_home: Path, on: bool, channel: str, changed: dict[str, str]) -> None:
    try:
        disclosure.record_change(slm_home, {
            "at": datetime.now(UTC).isoformat(), "action": "on" if on else "off",
            "channel": channel, "flags": changed})
    except Exception as exc:  # noqa: BLE001 - the history never blocks the switch
        logger.warning("measurement change not recorded in disclosures.json: %s", exc)


__all__ = ["INTERVENTIONS", "RECEIPTS", "SWITCHED", "USAGE", "set_measurement", "status"]
