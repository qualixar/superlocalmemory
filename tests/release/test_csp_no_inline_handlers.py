# B2 (3.7.9): script-src is locked to 'self' — no inline event handlers may
# exist in the dashboard UI, and every data-act-* action key used in markup
# must resolve to a registry entry in event-delegation.js. These guard against
# a regression that would either break the dashboard (missing key) or force
# 'unsafe-inline' back onto script-src (a new inline on*= handler).
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UI = REPO / "src" / "superlocalmemory" / "ui"
MIDDLEWARE = REPO / "src" / "superlocalmemory" / "server" / "security_middleware.py"

_INLINE = re.compile(r'\bon[a-z]+\s*=\s*"')


def _ui_files():
    files = list(UI.glob("*.html"))
    files += [p for p in (UI / "js").glob("*.js") if p.name != "event-delegation.js"]
    return files


def test_no_inline_event_handlers_in_ui():
    offenders = {}
    for f in _ui_files():
        hits = _INLINE.findall(f.read_text())
        if hits:
            offenders[f.name] = len(hits)
    assert not offenders, f"inline on*= handlers force script-src unsafe-inline: {offenders}"


def test_script_src_has_no_unsafe_inline():
    txt = MIDDLEWARE.read_text()
    m = re.search(r'"script-src[^"]*"', txt)
    assert m, "script-src directive not found in security_middleware"
    assert "unsafe-inline" not in m.group(0), f"script-src must not allow unsafe-inline: {m.group(0)}"


def test_every_action_key_has_a_registry_entry():
    used = set()
    for f in _ui_files():
        used |= set(re.findall(r'data-act-(?:click|change|input|keydown)="([a-z0-9-]+)"', f.read_text()))
    reg = (UI / "js" / "event-delegation.js").read_text()
    keys = set(re.findall(r"^\s*'([a-z0-9-]+)':\s*\(", reg, re.M))
    missing = used - keys
    assert not missing, f"data-act keys used in markup but missing from registry: {sorted(missing)}"
    assert used, "no data-act-* keys found — did the markup change?"
