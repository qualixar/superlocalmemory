# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""TDD — assertions on the slm-compress SKILL.md artifact (v3.6.14 replacement for slm-optimize).

slm-optimize was retired in v3.6.14 (Group-A deletions). Its functionality is now
covered by slm-compress (reversible compression) + slm-cache (KV caching).
These tests verify slm-compress, the direct replacement for slm-optimize's compress
surface. slm-compress lives in plugin/skills/slm-compress/SKILL.md (DOC-CORRECT layout).

Per original LLD-02 §7 acceptance criteria (adapted for replacement skill):
  1. Frontmatter validity
  2. name field == "slm-compress"
  3. Token budget < 2500 words
  4. Core tool names present (slm_compress, slm_retrieve)
  5. Anti-overclaim check
  6. Fail-open rule present
  7. Never-compress exclusions present
  8. File exists at plugin/skills/slm-compress/SKILL.md
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ─── Paths — DOC-CORRECT layout (v3.6.14) ────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent.parent
# slm-compress replaced slm-optimize; lives in plugin/skills/
_SKILL_FILE = _REPO_ROOT / "plugin" / "skills" / "slm-compress" / "SKILL.md"
# Source in plugin-src (single source of truth)
_SRC_SKILL_FILE = _REPO_ROOT / "plugin-src" / "skills" / "slm-compress" / "SKILL.md"

# slm-compress tools (subset of original slm-optimize tools)
_REQUIRED_TOOLS = (
    "slm_compress",
    "slm_retrieve",
)

# slm-cache tools (complementary skill, not checked here)
_CACHE_TOOLS = (
    "slm_cache_set",
    "slm_cache_get",
)

_BANNED_CLAIMS = (
    "90% on every",
    "full-turn caching",
    "saves 90%",
    "100% on every",
)


# ─── Helper ──────────────────────────────────────────────────────────────────


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from Markdown body."""
    if not text.startswith("---\n"):
        return {}, text
    try:
        end = text.index("\n---\n", 4)
    except ValueError:
        return {}, text
    fm = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5:]
    return fm, body


def _read_skill() -> tuple[dict, str, str]:
    """Return (frontmatter_dict, body_text, full_text) for the primary SKILL.md."""
    text = _SKILL_FILE.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    return fm, body, text


# ─── Test 1: file exists at plugin/skills/slm-compress/ ──────────────────────


def test_skill_files_exist_at_both_paths():
    """plugin/skills/slm-compress/SKILL.md exists (DOC-CORRECT layout).
    Also verifies plugin-src source exists (single source of truth).
    NOTE: slm-optimize was retired in v3.6.14; this test now covers slm-compress.
    """
    assert _SKILL_FILE.exists(), (
        f"Missing plugin skill: {_SKILL_FILE.relative_to(_REPO_ROOT)}\n"
        "slm-compress is the replacement for retired slm-optimize in v3.6.14.\n"
        "Run: node scripts/build-plugin.mjs to regenerate plugin/"
    )
    assert _SRC_SKILL_FILE.exists(), (
        f"Missing source skill: {_SRC_SKILL_FILE.relative_to(_REPO_ROOT)}"
    )


# ─── Test 2: valid YAML frontmatter ──────────────────────────────────────────


def test_frontmatter_is_valid_yaml():
    """SKILL.md has parseable YAML frontmatter between --- delimiters."""
    text = _SKILL_FILE.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with '---' frontmatter"
    fm, _ = _parse_frontmatter(text)
    assert isinstance(fm, dict), "Frontmatter must parse to a dict"
    assert fm, "Frontmatter must not be empty"


# ─── Test 3: name field matches directory ────────────────────────────────────


def test_name_field_matches_directory():
    """Frontmatter 'name' == 'slm-compress' (matches directory name)."""
    fm, _, _ = _read_skill()
    assert "name" in fm, "Frontmatter must have a 'name' field"
    assert fm["name"] == "slm-compress", (
        f"name must be 'slm-compress', got '{fm.get('name')}'"
    )


# ─── Test 4: token budget < 2500 words ───────────────────────────────────────


def test_token_budget_under_2500_words():
    """Skill file word count < 2500 (LLD-02 token budget target)."""
    _, _, text = _read_skill()
    word_count = len(text.split())
    assert word_count < 2500, (
        f"SKILL.md exceeds 2500-word budget: {word_count} words. "
        "Trim content to keep it injectable at skill-invocation time."
    )


# ─── Test 5: core tool names present ─────────────────────────────────────────


def test_all_five_tool_names_present():
    """Skill body references core compression tool names (slm_compress, slm_retrieve).
    Note: slm_cache_* tools are in slm-cache skill (separate in v3.6.14).
    """
    _, _, text = _read_skill()
    missing = [t for t in _REQUIRED_TOOLS if t not in text]
    assert not missing, (
        f"These tool names are missing from SKILL.md: {missing}. "
        "Agents must know the exact tool names to call them."
    )


# ─── Test 6: anti-overclaim check ────────────────────────────────────────────


def test_no_banned_overclaims():
    """SKILL.md must not contain banned overclaim phrases."""
    _, _, text = _read_skill()
    found = [phrase for phrase in _BANNED_CLAIMS if phrase.lower() in text.lower()]
    assert not found, (
        f"SKILL.md contains banned overclaims: {found}. "
        "Per LLD-02 §3: never claim full-turn caching or universal 90% savings."
    )


# ─── Test 7: fail-open rule present ──────────────────────────────────────────


def test_fail_open_rule_present():
    """Skill must instruct the agent to continue with original on ok:False."""
    _, _, text = _read_skill()
    has_fail_open = (
        "ok:false" in text.lower()
        or "ok: false" in text.lower()
        or "fail-open" in text.lower()
        or "fail open" in text.lower()
    )
    assert has_fail_open, (
        "SKILL.md must mention fail-open (ok:false / fail-open) so agents know when a tool failed"
    )
    assert "continue" in text.lower(), (
        "SKILL.md must tell agents to continue (fail-open) when tools return errors"
    )


# ─── Test 8: never-compress exclusions present ───────────────────────────────


def test_never_compress_exclusions_present():
    """Skill must list what NOT to compress (code, JSON, secrets)."""
    _, _, text = _read_skill()
    lower = text.lower()
    assert "secret" in lower or "credential" in lower or "key" in lower, (
        "SKILL.md must warn agents not to cache/compress secrets or credentials"
    )
    assert "json" in lower, (
        "SKILL.md must exclude JSON from compression (structured data must be parseable)"
    )
    assert "code" in lower or "edit" in lower or "write" in lower, (
        "SKILL.md must exclude code/file content being sent to Edit/Write tools"
    )
