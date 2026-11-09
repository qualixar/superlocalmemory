"""
README integrity tests.

LLD §6 blocking CI job. Every assertion corresponds to an AC in the LLD.
These tests run against README.md in the repo root.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

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


def _project_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


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
# AC2: one install block, at the top. The 4.1.20 README rewrite (e48b83b2)
# replaced "## Quick Start" with a three-line install block before the
# "## 30-second example". Feature announcements can precede installation;
# the walkthrough must not. Keep one place to start, with no duplicate copy.
# ---------------------------------------------------------------------------


def test_single_quick_start():
    """AC2: exactly one install block, near the top; no duplicate Quick Start."""
    text = _readme_text()
    blocks = re.findall(r"```(?:bash|console|sh)?\n(.*?)```", text, re.DOTALL)
    install_blocks = [b for b in blocks if "pipx install superlocalmemory" in b]
    assert len(install_blocks) == 1, (
        f"Expected exactly 1 install block, found {len(install_blocks)} (LLD AC2)."
    )
    assert "slm setup" in install_blocks[0]
    assert text.index("pipx install superlocalmemory") < text.index("\n## 30-second example"), (
        "The install block must come before the runnable example (LLD AC2)."
    )
    assert len(re.findall(r"^#{2,3} Quick Start", text, re.MULTILINE)) <= 1


# ---------------------------------------------------------------------------
# AC3: the README states no release number that can go stale.
# Since 4.1.20 the version comes only from the live PyPI and npm badges, and
# scripts/bump_version.py no longer rewrites README (a stamp it could not find
# made every bump fail half way). A hard-coded number here is now drift.
# ---------------------------------------------------------------------------


def test_current_version_in_hero():
    """AC3: no stale or hard-coded release in the hero; live version badges."""
    text = _readme_text()
    stale = re.findall(r"3\.6\.1[0-3]", text)
    assert not stale, (
        f"Found stale version strings: {stale}. Hero version refs must not "
        "point at pre-3.6.14 releases (LLD AC3)."
    )
    hero = text[: text.index("\n## ")]
    assert "img.shields.io/pypi/v/superlocalmemory" in hero
    assert "img.shields.io/npm/v/superlocalmemory" in hero
    assert _project_version() not in hero, (
        "The hero must not hard-code the release; the live badges show it (LLD AC3)."
    )
    assert "Current_Release" not in text


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
    # The README uses absolute GitHub URLs so it renders on PyPI and npm; a
    # link into this repository's main branch must still name a real file.
    repo_prefix = "https://github.com/qualixar/superlocalmemory/blob/main/"
    for rel_path in re.findall(r"\[(?:[^\]]*)\]\(([^)#]+\.md[^)]*)\)", text):
        # Strip query/fragment
        path_only = rel_path.split("#")[0].split("?")[0]
        if path_only.startswith(repo_prefix):
            path_only = path_only[len(repo_prefix):]
        elif path_only.startswith(("http://", "https://")):
            continue
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
    """AC6b: the proxy is one way to optimize, never the only one."""
    text = _readme_text().lower()
    assert "through a proxy (`slm wrap claude`), mcp tools or a skill" in text, (
        "The README must say optimize works without a proxy too (LLD AC6)."
    )


# ---------------------------------------------------------------------------
# AC7: evidence-safe positioning near the hero
# ---------------------------------------------------------------------------


def test_evidence_safe_positioning_near_hero():
    """AC7: the opening states the local contract; benchmarks state their source."""
    text = _readme_text()
    opening = "\n".join(_readme_lines()[:60]).lower()
    assert "lives on your machine" in opening, (
        "Opening copy must describe the local runtime contract (LLD AC7)."
    )
    assert "in mode a, core remember and recall make no model-provider call" in opening
    assert "anything that sends data out is a choice you make" in opening, (
        "Opening copy must disclose optional provider/network choices (LLD AC7)."
    )
    bench = text.split("## Benchmarks", 1)
    assert len(bench) == 2, "README must keep its Benchmarks section"
    bench_text = bench[1].split("\n## ", 1)[0]
    # The README carries no release numbers, so the evidence is named by its paper.
    assert "published architecture paper ([arXiv:2603.14588]" in bench_text, (
        "Benchmarks must be identified as the published paper's evidence (LLD AC7)."
    )
    assert "They are not a fresh run of the current package." in bench_text, (
        "Benchmarks must keep the protocol boundary of the evidence (LLD AC7)."
    )
    assert "comparable only when the subset, answer model and judge match" in bench_text


# ---------------------------------------------------------------------------
# AC8: four install paths + slm wrap claude
# ---------------------------------------------------------------------------


def test_four_install_paths():
    """AC8: pipx / npm / Claude Code plugin / slm connect, plus slm wrap claude."""
    text = _readme_text()
    checks = {
        "pipx install superlocalmemory": "pipx install superlocalmemory" in text,
        "npm install -g superlocalmemory": (
            "npm i -g superlocalmemory" in text or "npm install -g superlocalmemory" in text
        ),
        "claude plugin install superlocalmemory@qualixar": (
            "claude plugin install superlocalmemory@qualixar" in text
        ),
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
# Section order (4.1.20 README): example, why, integrations, features, then
# evidence and research, then upgrade. Selling before showing, or evidence
# before the reader knows what the product is, is the regression guarded.
# ---------------------------------------------------------------------------


def test_section_order():
    """Example < Why < Works with < Everything < Benchmarks < Research < Upgrade."""
    text = _readme_text()
    order = [
        ("Example", r"^## 30-second example"),
        ("Why", r"^## Why"),
        ("Works with", r"^## Works with your agents"),
        ("Everything", r"^## Everything SLM does"),
        ("Benchmarks", r"^## Benchmarks"),
        ("Research", r"^## Research"),
        ("Upgrade", r"^## Upgrade"),
    ]
    positions: dict[str, int] = {}
    for name, pat in order:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            positions[name] = m.start()

    missing = [name for name, _ in order if name not in positions]
    assert not missing, f"Section(s) not found in README: {missing}"
    found = [positions[name] for name, _ in order]
    assert found == sorted(found), (
        f"README sections out of order: {sorted(positions, key=positions.get)}"
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
