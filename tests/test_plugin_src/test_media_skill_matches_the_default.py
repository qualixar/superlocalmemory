# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""The slm-media skill and the pictures doc must say what the code does (4.2.30).

The skill's front matter said "Off by default" and "16 GB of memory needed" while
4.2.30 turned pictures and documents on by default, set them up by itself from
15 GiB of memory, and made them one click (never refused) below that. An agent
reading the description would tell a user on a 16 GB laptop that the feature is
off and cannot run. The code is the source of truth; this reads it.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GIB = 1024 ** 3
SKILL_COPIES = [
    "plugin-src/skills/slm-media/SKILL.md",
    "plugin/skills/slm-media/SKILL.md",
]
# Wording that says the feature is off, or that a small computer is refused.
STALE = [
    r"Off by default",
    r"16 GB of memory needed",
    r"each \*\*off\*\* until",
    r"switch is refused",
    r"computer that qualifies",
]


def _media_env():
    # The package also exports a function called media_env, so load the module by name.
    return importlib.import_module("superlocalmemory.runtimes.media_env")


def test_the_code_sets_up_on_16_gb_and_never_refuses_a_smaller_computer():
    env = _media_env()
    assert env.media_policy(16 * GIB, env={}).mode == env.AUTO
    assert env.media_policy(int(15.6 * GIB), env={}).mode == env.AUTO  # a "16 GB" laptop reports less
    small = env.media_policy(8 * GIB, env={})
    assert small.mode == env.ONE_CLICK and small.warning
    assert env.media_ram_refusal(8 * GIB) == ""


def test_the_picture_and_document_models_are_on_by_default_in_the_code():
    from superlocalmemory.core.feature_flags import FLAGS, FlagState

    for flag in ("media.captions", "media.document_pack", "media.regions"):
        assert FLAGS[flag].default is FlagState.ON, flag


@pytest.mark.parametrize("rel", SKILL_COPIES)
def test_the_media_skill_says_on_by_default_and_the_16_gb_rule(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    front = text.split("\n---\n", 1)[0]
    assert "On by default." in front, f"{rel}: front matter must say the feature is on by default"
    assert "16 GB" in front and "one click" in front, f"{rel}: front matter must state the 16 GB rule"
    assert "**on by default**" in text
    stale = [p for p in STALE if re.search(p, text)]
    assert stale == [], f"{rel} still says: {stale}"


@pytest.mark.parametrize("rel", [
    "docs/pictures-and-documents.md",
    "plugin-src/skills/slm-getting-started-bot/SKILL.md",
    "plugin/skills/slm-getting-started-bot/SKILL.md",
])
def test_other_texts_do_not_say_a_small_computer_is_refused(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    stale = [p for p in STALE if re.search(p, text)]
    assert stale == [], f"{rel} still says: {stale}"


# Hand-written texts outside the generated skill trees that also describe the feature.
HAND_WRITTEN = [
    "codex-plugin/AGENTS.md",
    "plugin-src/rules/AGENTS.md",
    "ide/configs/continue-skills.yaml",
    "scripts/get-slm.sh",
    "scripts/get-slm.ps1",
    "wiki-content/Privacy-and-Data-Flow.md",
]
_FEATURE = re.compile(r"picture|document|media|image", re.I)
# A line about the feature must not call it off by default or say 16 GB is required.
_STALE_LINE = re.compile(r"off by default|needs 16 GB|needs 16 GiB|requires 16 GB|optional and off", re.I)


@pytest.mark.parametrize("rel", HAND_WRITTEN)
def test_hand_written_texts_do_not_call_pictures_off_by_default_or_16_gb_only(rel):
    lines = (REPO / rel).read_text(encoding="utf-8").splitlines()
    stale = [f"{n}: {ln.strip()[:110]}" for n, ln in enumerate(lines, 1)
             if _FEATURE.search(ln) and _STALE_LINE.search(ln)]
    assert stale == [], f"{rel} still says the feature is off or needs 16 GB:\n" + "\n".join(stale)


@pytest.mark.parametrize("rel", ["codex-plugin/AGENTS.md", "plugin-src/rules/AGENTS.md"])
def test_agent_rules_say_on_by_default_and_the_16_gb_rule(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    section = text.split("## Pictures and documents", 1)[1].split("\n---\n", 1)[0]
    assert section.lstrip().startswith("On by default."), rel
    assert "automatic on a computer with 16 GB" in section and "one click below that" in section, rel
