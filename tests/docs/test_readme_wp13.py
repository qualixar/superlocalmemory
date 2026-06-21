"""
WP-13 README integrity tests.

LLD §6 blocking CI job. Every assertion corresponds to an AC in the LLD.
These tests run against README.md in the repo root.
"""

from __future__ import annotations

import re
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
DOCS_DIR = REPO_ROOT / "docs"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _readme_lines() -> list[str]:
    return _readme_text().splitlines()


# GitHub slugger: lower-case, strip everything except alphanum + dash, spaces→dash.
# Does NOT strip emoji characters fully — LLD CRIT-1 says strip emoji for slugger.
def _gh_slug(heading: str) -> str:
    # Strip leading # chars and whitespace
    text = re.sub(r"^#+\s*", "", heading)
    # Strip emoji (any non-ASCII that isn't a combining char we care about)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Lower-case
    text = text.lower()
    # Replace spaces and remaining special chars with dash, keep alphanum and dash
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text


BANNED_SLOP = [
    "delve",
    "tapestry",
    "multifaceted",
    "paradigm",
    "foster",
    "cutting-edge",
    "holistic",
    "synergy",
    "resonate",
    "pivotal",
    "game-changer",
    "transformative",
    "embark",
    "unravel",
    "beacon",
]

# "landscape" is only banned when figurative — detect obvious figurative patterns.
LANDSCAPE_FIGURATIVE_PATTERNS = [
    r"\blandscape\s+of\b",
    r"\blandscape\s+for\b",
    r"\bai\s+landscape\b",
    r"\blandscape\s+has\b",
    r"\blandscape\s+is\b",
    r"\bbroader\s+landscape\b",
    r"\bevolving\s+landscape\b",
    r"\bchanging\s+landscape\b",
    r"\bcurrent\s+landscape\b",
    r"\bmarket\s+landscape\b",
    r"\bcompetitive\s+landscape\b",
]


# ---------------------------------------------------------------------------
# AC1: line count
# ---------------------------------------------------------------------------


def test_line_count_le_720():
    """AC1: README must be ≤720 lines (was 953)."""
    lines = _readme_lines()
    assert len(lines) <= 720, (
        f"README has {len(lines)} lines — hard ceiling is 720 (LLD AC1)."
    )


# ---------------------------------------------------------------------------
# AC2: exactly one ## Quick Start, zero ### Quick Start
# ---------------------------------------------------------------------------


def test_single_quick_start():
    """AC2: Exactly one '## Quick Start', zero '### Quick Start'."""
    text = _readme_text()
    h2_matches = re.findall(r"^## Quick Start", text, re.MULTILINE)
    h3_matches = re.findall(r"^### Quick Start", text, re.MULTILINE)
    assert len(h2_matches) == 1, (
        f"Expected exactly 1 '## Quick Start', found {len(h2_matches)}."
    )
    assert len(h3_matches) == 0, (
        f"Expected zero '### Quick Start', found {len(h3_matches)} "
        "(duplicate accordion Quick Start must be deleted — LLD AC2)."
    )


# ---------------------------------------------------------------------------
# AC3: current hero version is 3.6.17
# ---------------------------------------------------------------------------


def test_version_3616_only():
    """AC3: no pre-current stragglers (3.6.10-13) in hero; at least one 3.6.17;
    h1 contains V3.6.17. The release-history table may still list older versions
    (e.g. the v3.6.14 row) as legitimate history."""
    text = _readme_text()
    stale = re.findall(r"3\.6\.1[0-3]", text)
    assert not stale, (
        f"Found stale version strings: {stale}. Hero version refs must be 3.6.17 (LLD AC3)."
    )
    assert "3.6.17" in text, "README must contain at least one '3.6.17' (LLD AC3)."
    # h1 check — first heading
    lines = _readme_lines()
    h1_lines = [l for l in lines if l.startswith("# ") or l.startswith("<h1")]
    assert any("3.6.17" in l or "V3.6.17" in l for l in h1_lines[:5]), (
        "h1 / hero must reference V3.6.17 (LLD AC3)."
    )


# ---------------------------------------------------------------------------
# AC4: zero banned slop
# ---------------------------------------------------------------------------


def test_no_banned_slop():
    """AC4: Zero banned slop words (case-insensitive)."""
    text = _readme_text().lower()
    hits: list[str] = []
    for word in BANNED_SLOP:
        pattern = r"\b" + re.escape(word.replace("-", r"[\-\s]")) + r"\b"
        if re.search(pattern, text):
            hits.append(word)
    # Also check figurative "landscape" patterns
    for pat in LANDSCAPE_FIGURATIVE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append(f"landscape[figurative]: {pat}")
    assert not hits, f"Banned slop words found: {hits} (LLD AC4 anti-slop)."


# ---------------------------------------------------------------------------
# AC5: internal links resolve
# ---------------------------------------------------------------------------


def test_internal_links_resolve():
    """AC5: Every internal markdown link [text](#anchor) or [text](docs/file) resolves."""
    text = _readme_text()

    # Collect all headings → slugs for anchor resolution
    heading_slugs: set[str] = set()
    for line in _readme_lines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            heading_slugs.add(_gh_slug(m.group(2)))
    # Also collect <a id="..."> anchors
    for anchor_id in re.findall(r'<a\s+id=["\']([^"\']+)["\']', text):
        heading_slugs.add(anchor_id)

    # Extract all internal links
    broken: list[str] = []

    # [text](#anchor)
    for anchor in re.findall(r"\[(?:[^\]]*)\]\(#([^)]+)\)", text):
        if anchor not in heading_slugs:
            broken.append(f"#{anchor}")

    # [text](docs/...) or [text](CONTRIBUTING.md) etc.
    for rel_path in re.findall(r"\[(?:[^\]]*)\]\(([^)#]+\.md[^)]*)\)", text):
        # Strip query/fragment
        path_only = rel_path.split("#")[0].split("?")[0]
        full_path = REPO_ROOT / path_only
        if not full_path.exists():
            broken.append(rel_path)

    assert not broken, (
        f"Broken internal links found: {broken} (LLD AC5 — resolve or remove)."
    )


# ---------------------------------------------------------------------------
# AC6: no "Save up to 90%" + "without a proxy" present
# ---------------------------------------------------------------------------


def test_no_90pct_overclaim():
    """AC6a: 'Save up to 90%' overclaim must be absent."""
    text = _readme_text()
    assert "Save up to 90%" not in text, (
        "'Save up to 90%' overclaim must be removed (LLD §4 DROP, AC6)."
    )


def test_without_a_proxy_present():
    """AC6b: 'without a proxy' hard constraint must appear once."""
    text = _readme_text()
    assert "without a proxy" in text.lower(), (
        "The 'without a proxy' hard-constraint sentence must be present (LLD AC6)."
    )


# ---------------------------------------------------------------------------
# AC7: moat in hero (first 30 lines)
# ---------------------------------------------------------------------------


def test_moat_in_hero():
    """AC7: First 30 lines must contain zero-cloud, Mem0, and 'best of our knowledge' signals."""
    lines = _readme_lines()
    hero = "\n".join(lines[:30]).lower()
    assert "zero" in hero or "zero-cloud" in hero or "zero-llm" in hero, (
        "Hero (first 30 lines) must mention zero-cloud/zero-LLM moat (LLD AC7)."
    )
    assert "mem0" in hero, (
        "Hero (first 30 lines) must reference Mem0 comparison (LLD AC7)."
    )
    assert "best of our knowledge" in hero or "best-of-knowledge" in hero or "knowledge" in hero, (
        "Hero (first 30 lines) must include a hedged 'best of our knowledge' claim (LLD AC7)."
    )


# ---------------------------------------------------------------------------
# AC8: four install paths + slm wrap claude
# ---------------------------------------------------------------------------


def test_four_install_paths():
    """AC8: npm / pip / /plugin install / slm connect all present + slm wrap claude."""
    text = _readme_text()
    checks = {
        "npm i -g superlocalmemory or npm install -g": (
            "npm i -g superlocalmemory" in text or "npm install -g superlocalmemory" in text
        ),
        "pip install superlocalmemory": "pip install superlocalmemory" in text,
        "/plugin install superlocalmemory@qualixar": "/plugin install superlocalmemory@qualixar" in text,
        "slm connect": "slm connect" in text,
        "slm wrap claude": "slm wrap claude" in text,
    }
    missing = [k for k, v in checks.items() if not v]
    assert not missing, (
        f"Missing install path commands: {missing} (LLD AC8 — 4 paths + slm wrap claude)."
    )


# ---------------------------------------------------------------------------
# AC9 / CLAIM-AUDIT: dropped overclaims absent
# ---------------------------------------------------------------------------


def test_claim_audit_dropped_absent():
    """AC9: '2,900+' and '1,300+' overclaims must not appear."""
    text = _readme_text()
    assert "2,900+" not in text and "2900+" not in text, (
        "'2,900+ tests' overclaim must be removed (LLD §4 CLAIM-AUDIT DROP)."
    )
    assert "1,300+" not in text and "1300+" not in text, (
        "'1,300+ entities' overclaim must be removed (LLD §4 CLAIM-AUDIT DROP)."
    )


# ---------------------------------------------------------------------------
# Section order: Why < Quick Start < Three Pillars < Papers
# ---------------------------------------------------------------------------


def test_section_order():
    """LLD §6: Why SLM < Quick Start < Three Pillars < Papers (no Support-before-Why regressions)."""
    text = _readme_text()
    positions: dict[str, int] = {}
    # Use first match position for each section
    patterns = {
        "Why": r"^## Why",
        "Quick Start": r"^## Quick Start",
        "Three Pillars": r"^## Three Pillars",
        "Papers": r"^## Research Papers|^## Papers",
    }
    for name, pat in patterns.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            positions[name] = m.start()

    missing = [k for k in patterns if k not in positions]
    assert not missing, f"Section(s) not found in README: {missing}"

    assert positions["Why"] < positions["Quick Start"], (
        "'## Why' must come before '## Quick Start'."
    )
    assert positions["Quick Start"] < positions["Three Pillars"], (
        "'## Quick Start' must come before '## Three Pillars'."
    )
    assert positions["Three Pillars"] < positions["Papers"], (
        "'## Three Pillars' must come before '## Research Papers'."
    )


# ---------------------------------------------------------------------------
# CAVEAT-1: dead docs/benchmarks repro pointer must be absent
# ---------------------------------------------------------------------------


def test_no_dead_repro_script_pointer():
    """CAVEAT-1: 'repro script in docs/benchmarks/' must not appear (it's a 404)."""
    text = _readme_text()
    assert "repro script in docs/benchmarks" not in text.lower(), (
        "Dead repro-script pointer in docs/benchmarks/ must be removed (LLD CAVEAT-1 / D-2)."
    )
