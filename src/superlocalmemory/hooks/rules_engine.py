# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Rules engine for configurable auto-capture and auto-recall behavior.

Rules live under ``"rules"`` in config.json. ``auto_recall.on_every_prompt``
switches the per-prompt recall hooks and ``auto_capture.capture_session_summary``
the end-of-session summary (both wired in 4.1.25; turning a whole section off
turns its hooks off too). Before 4.1.25 every dashboard save wrote
``on_every_prompt: false``, the unused default of the time; a store saved
before ``rules_version`` 2 drops that one value rather than end per-prompt
recall nobody chose to end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RULES = {
    "auto_recall": {
        "enabled": True,
        "on_session_start": True,
        "on_every_prompt": True,
        "max_memories_injected": 10,
        "relevance_threshold": 0.3,
    },
    "auto_capture": {
        "enabled": True,
        "capture_decisions": True,
        "capture_bugs": True,
        "capture_preferences": True,
        "capture_session_summary": True,
        "min_confidence": 0.5,
    },
}


#: 2 = on_every_prompt is wired (4.1.25). Stored alongside "rules".
RULES_VERSION = 2


def _merged(stored: dict) -> dict:
    """Each section's defaults overlaid by what was stored (new dicts)."""
    out = {name: dict(section) for name, section in DEFAULT_RULES.items()}
    for name, value in stored.items():
        out[name] = {**out.get(name, {}), **value} if isinstance(value, dict) else value
    return out


def _upgraded(stored: dict, version: int) -> dict:
    if version >= RULES_VERSION:
        return stored
    recall = stored.get("auto_recall")
    if not isinstance(recall, dict) or "on_every_prompt" not in recall:
        return stored
    kept = {k: v for k, v in recall.items() if k != "on_every_prompt"}
    return {**stored, "auto_recall": kept}


class RulesEngine:
    """Manage configurable rules for auto-capture and auto-recall."""

    def __init__(self, config: dict | None = None, config_path: Path | None = None,
                 *, injection_on: bool = True):
        # ``injection.enabled`` is the master switch for everything SLM adds to an agent's
        # context on its own (InjectionConfig); off, no automatic recall fires anywhere.
        self._injection_on = injection_on
        if config:
            self._rules = _merged(config)
        elif config_path and config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            version = data.get("rules_version", 1)
            stored = data.get("rules", {})
            self._rules = _merged(_upgraded(stored, version if isinstance(version, int) else 1))
            injection = data.get("injection")
            if isinstance(injection, dict) and injection.get("enabled") is False:
                self._injection_on = False
        else:
            self._rules = _merged({})

    def should_capture(self, category: str, confidence: float) -> bool:
        """Check if a category should be captured based on rules."""
        capture_rules = self._rules.get("auto_capture", {})
        if not capture_rules.get("enabled", True):
            return False

        category_key = (f"capture_{category}s" if not category.endswith("s")
                        else f"capture_{category}")
        if not capture_rules.get(category_key, True):
            return False

        min_conf = capture_rules.get("min_confidence", 0.5)
        return confidence >= min_conf

    def should_recall(self, trigger: str) -> bool:
        """Check if auto-recall should fire for a trigger."""
        recall_rules = self._rules.get("auto_recall", {})
        if not self._injection_on or not recall_rules.get("enabled", True):
            return False

        if trigger == "session_start":
            return recall_rules.get("on_session_start", True)
        if trigger == "every_prompt":
            return recall_rules.get("on_every_prompt", True)

        return True

    def per_prompt_recall_on(self) -> bool:
        """The per-prompt recall hooks (UserPromptSubmit and Codex's prompt)."""
        return bool(self.should_recall("every_prompt"))

    def captures_session_summary(self) -> bool:
        capture = self._rules.get("auto_capture", {})
        return bool(capture.get("enabled", True)
                    and capture.get("capture_session_summary", True))

    def get_recall_config(self) -> dict:
        """Get auto-recall configuration."""
        return dict(self._rules.get("auto_recall", DEFAULT_RULES["auto_recall"]))

    def get_capture_config(self) -> dict:
        """Get auto-capture configuration."""
        return dict(self._rules.get("auto_capture", DEFAULT_RULES["auto_capture"]))

    def update_rule(self, section: str, key: str, value: Any) -> None:
        """Update a specific rule."""
        if section not in self._rules:
            self._rules[section] = {}
        self._rules[section][key] = value

    def save(self, config_path: Path) -> None:
        """Save rules to config file."""
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            data = {}
        data["rules"] = self._rules
        data["rules_version"] = RULES_VERSION
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_dict(self) -> dict:
        """Export rules as dict."""
        return dict(self._rules)


def _config_path() -> Path:
    from superlocalmemory.infra.data_root import state_path

    return state_path("config.json")


def load(config_path: Path | None = None) -> RulesEngine:
    """The stored rules, or the defaults when they cannot be read. Never raises:
    hooks call this on every prompt and must fail open."""
    try:
        return RulesEngine(config_path=config_path or _config_path())
    except Exception as exc:  # noqa: BLE001 — a bad config never breaks a hook
        logger.debug("rules unreadable, using defaults: %s", exc)
        return RulesEngine()
