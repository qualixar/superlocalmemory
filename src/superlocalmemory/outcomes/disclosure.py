# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The measurement notice (4.2.30 D-28, D-49): what SLM measures, said once in plain words.

Receipts, randomised withholding, logged swaps and usage detection are on by default for
everyone, with a clear switch (``outcomes.switch``). The notice is **informational and gates
nothing** (D-49): install and upgrade print it, the first ``slm`` run on a terminal after an
upgrade prints it, the dashboard shows a banner until it has been seen, and ``slm measurement
status`` prints it, but no measure waits for anyone to read it.

``<SLM_HOME>/disclosures.json`` records which version of the notice has been
shown, when, and through which surface (``cli``, ``dashboard``, ``install``),
plus a short history of on/off changes (``record_change``). A change to the
notice's wording bumps ``NOTICE_VERSION`` and shows it again. Reading never
raises: a missing or damaged file means "not shown yet".
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

NOTICE_VERSION = "4.2.30-2"
DISCLOSURES_FILE = "disclosures.json"
CHANNELS = ("cli", "dashboard", "install")
#: Most entries kept in the on/off history.
HISTORY_MAX = 50

_KEY = "measurement"
_HISTORY = "changes"


def _one_in(share: float) -> str:
    return f"1 in {max(1, round(1 / share))}" if share > 0 else "none"


def text() -> str:
    """The notice, in plain words. Numbers come from the design constants."""
    from superlocalmemory.outcomes import randomiser, swaps

    withhold = _one_in(randomiser.EPSILON)
    swap = _one_in(swaps.SWAP_SHARE)
    return (
        "SuperLocalMemory now measures whether memory helps your agents.\n"
        "\n"
        "Every time it gives memories to an agent, it writes a receipt: which "
        "memories, their rank and score, and the chance each one had of being "
        "chosen. Receipts hold ids and numbers only, never your text, and they "
        "stay on this computer. Nothing is sent anywhere.\n"
        "\n"
        f"To find out whether memory really helps, about {withhold} session "
        "starts change one thing, picked by a logged random draw: one memory is "
        "left out, or (when an agent opens a session through MCP) one way of "
        "searching is switched off for that one fetch. Pinned rules, standing "
        "rules, core blocks, anything an agent asked for with a recall, cues "
        "and lease warnings are never left out.\n"
        "\n"
        f"In about {swap} of an agent's own recalls, two neighbouring results "
        "trade places. Nothing is left out and nothing is added. This shows how "
        "much attention each position gets, so SuperLocalMemory does not just "
        "favour whatever is on top.\n"
        "\n"
        "Usage detection is on too. When an agent finishes a reply, SuperLocalMemory "
        "compares that reply with the memories it was given, to see which ones were "
        "used. It keeps ids and scores only, never the reply, and the comparison "
        "happens on this computer. Without it, leaving a memory out could never show "
        "whether it was missed.\n"
        "\n"
        "One more measure stays off unless you turn it on: deciding for each prompt "
        "whether memory is needed.\n"
        "\n"
        "To turn the withholding, swapping and usage detection off, run: slm measurement off\n"
        "Or open the dashboard, go to Settings, and switch off \"Help SLM "
        "measure what works\". To stop the receipts too, run: slm measurement "
        "off --receipts\n"
        "To see where things stand: slm measurement status"
    )


def _path(slm_home: Path) -> Path:
    return Path(slm_home) / DISCLOSURES_FILE


def _read(slm_home: Path) -> dict[str, object]:
    try:
        data = json.loads(_path(slm_home).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("disclosures.json unreadable (%s); notice treated as not shown", exc)
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _write(slm_home: Path, data: dict[str, object]) -> None:
    path = _path(slm_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def state(slm_home: Path) -> dict[str, object]:
    """What was acknowledged: ``{"version", "at", "channel"}`` or ``{}``."""
    entry = _read(slm_home).get(_KEY)
    return dict(entry) if isinstance(entry, dict) else {}


def pending(slm_home: Path) -> bool:
    """True until this version of the notice was acknowledged. Never raises."""
    try:
        return state(slm_home).get("version") != NOTICE_VERSION
    except Exception:  # noqa: BLE001 - a notice that cannot be checked is shown
        return True


def acknowledge(slm_home: Path, channel: str) -> dict[str, object]:
    """Record that a person was shown the notice, through ``channel``."""
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {', '.join(CHANNELS)}")
    data = _read(slm_home)
    entry = {"version": NOTICE_VERSION, "at": datetime.now(UTC).isoformat(), "channel": channel}
    data[_KEY] = entry
    _write(slm_home, data)
    return entry


def record_change(slm_home: Path, entry: dict[str, object]) -> None:
    """Append one on/off change to the history (newest last, capped)."""
    data = _read(slm_home)
    history = data.get(_HISTORY)
    history = list(history) if isinstance(history, list) else []
    history.append(dict(entry))
    data[_HISTORY] = history[-HISTORY_MAX:]
    _write(slm_home, data)


def announce(slm_home: Path, *, stream=None, channel: str = "install") -> bool:
    """Print the notice to ``stream`` (stderr) when it is pending; never prompts, never raises.

    Used by install and upgrade paths. It counts as seen only when the stream is
    a terminal: a log or a pipe is not a person reading it, and the dashboard
    and ``slm measurement status`` will show it again. Returns whether it printed.
    """
    import sys

    try:
        if not pending(slm_home):
            return False
        out = stream if stream is not None else sys.stderr
        out.write("\n" + text() + "\n\n")
        if getattr(out, "isatty", lambda: False)():
            acknowledge(slm_home, channel)
        return True
    except Exception as exc:  # noqa: BLE001 - a notice problem never breaks an install
        logger.warning("measurement notice not shown: %s", exc)
        return False


__all__ = ["CHANNELS", "DISCLOSURES_FILE", "HISTORY_MAX", "NOTICE_VERSION", "acknowledge",
           "announce", "pending", "record_change", "state", "text"]
