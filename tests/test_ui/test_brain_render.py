# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-04 §6.2

"""Static-scan tests for ``ui/js/brain.js`` — XSS-safe rendering (LLD-04 §4.3).

We don't run a headless browser here; we statically scan the JS source
to assert the absence of DOM-injection sinks. This is the CI-reliable
lower bound; a Playwright suite can layer on top without changing this.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_UI_JS = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui" / "js" / "brain.js"
_UI_CSS = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui" / "css" / "brain.css"
_INDEX_HTML = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "ui" / "index.html"


def test_brain_js_file_exists() -> None:
    assert _UI_JS.exists(), f"missing UI JS file: {_UI_JS}"


def test_brain_css_file_exists() -> None:
    assert _UI_CSS.exists(), f"missing UI CSS file: {_UI_CSS}"


def test_brain_js_uses_strict_mode() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "'use strict'" in js or '"use strict"' in js, \
        "brain.js must declare strict mode"


def test_brain_js_no_inner_html_assignment() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert not re.search(r"\binnerHTML\s*=", js), \
        "innerHTML assignment banned in brain.js"


def test_brain_js_no_insert_adjacent_html() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "insertAdjacentHTML" not in js, \
        "insertAdjacentHTML banned in brain.js"


def test_brain_js_no_dangerously_set_inner_html() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "dangerouslySetInnerHTML" not in js


def test_brain_js_defines_safe_el_helper() -> None:
    """EL helper or equivalent safe constructor must exist."""
    js = _UI_JS.read_text(encoding="utf-8")
    # Either `const EL =` or `function EL(` — either is acceptable.
    assert re.search(r"\bEL\s*=\s*\(|\bfunction\s+EL\s*\(", js), \
        "brain.js missing safe DOM helper EL"


def test_brain_js_uses_textContent() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "textContent" in js, "brain.js must use textContent"


def test_brain_js_uses_setAttribute_for_tooltips() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    # title attribute is set via setAttribute — auto-escaped by browser.
    assert "setAttribute" in js


def test_brain_js_sends_install_token_header() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    # Case-insensitive — header name may be lower-cased.
    assert "X-Install-Token" in js or "x-install-token" in js


def test_brain_js_fetches_correct_endpoint() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "/api/v3/brain" in js


def test_brain_js_has_toggle_function() -> None:
    """toggleBrainView retained as a no-op backward-compat shim.

    Developer view was removed (April 18, 2026 UX fix). The symbol is
    kept on ``window`` so any out-of-tree script that still calls it
    does not throw; it now simply triggers a re-load of the full view.
    """
    js = _UI_JS.read_text(encoding="utf-8")
    assert "toggleBrainView" in js


def test_brain_js_renders_all_sections() -> None:
    """Post-v3.4.22: all ML / behavioral / adapter state on one view."""
    js = _UI_JS.read_text(encoding="utf-8")
    for card in (
        "cardLearning",
        "cardLegacyMigration",
        "cardBandit",
        "cardUsage",
        "cardPreferences",
        "cardBehavioralOutcomes",
        "cardReportOutcome",
        "cardCrossPlatform",
        "cardCache",
        "cardEvolution",
        "cardDangerZone",
    ):
        assert card in js, f"brain.js must render {card} section"


def test_brain_js_fetches_behavioral_status() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "/api/behavioral/status" in js


def test_brain_js_has_report_outcome_and_reset_endpoints() -> None:
    js = _UI_JS.read_text(encoding="utf-8")
    assert "/api/behavioral/report-outcome" in js
    assert "/api/learning/reset" in js
    assert "/api/learning/retrain" in js
    assert "/api/learning/migrate-legacy" in js


def test_brain_js_no_developer_view_render_path() -> None:
    """renderDeveloper / kvTable removed — one honest view only."""
    js = _UI_JS.read_text(encoding="utf-8")
    assert "renderDeveloper" not in js
    assert "kvTable" not in js


def test_index_html_removes_patterns_and_behavioral_tabs() -> None:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="patterns-tab"' not in html
    assert 'id="behavioral-tab"' not in html
    assert 'id="patterns-pane"' not in html
    assert 'id="behavioral-pane"' not in html


def test_index_html_removes_legacy_script_tags() -> None:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'js/patterns.js' not in html
    assert 'js/behavioral.js' not in html
    assert 'js/learning.js' not in html


def test_index_html_adds_brain_tab_and_script() -> None:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="brain-tab"' in html
    assert 'id="brain-pane"' in html
    assert 'js/brain.js' in html


def test_brain_css_uses_css_variables() -> None:
    css = _UI_CSS.read_text(encoding="utf-8")
    # LLD-04 §4.5 — CSS must consume design-system variables.
    assert "var(--" in css
