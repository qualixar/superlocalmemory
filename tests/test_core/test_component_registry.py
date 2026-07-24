"""Component registry + healer (v3.8.2 "whole self-healer").

Locks the contract for the central capability registry and its repair layer:

1. Component is immutable (frozen) and round-trips to a dict.
2. Probes never raise and produce honest statuses; the transient overlay
   shows live repair progress but never masks a confirmed-present component.
3. snapshot() summary counts + "healthy" verdict are correct.
4. auto_fixable classification honours the safety rules:
   - torch/search deps are NEVER background auto-installed (too big),
   - the 560MB compressor is NEVER auto-downloaded (lazy on first use),
   - sqlite-vec auto-installs only when the interpreter is user-writable.
5. Model repo ids stay in sync with the setup wizard.
6. heal_missing() retries a flaky action, records the fix command on
   permanent failure, filters by key, and cleans up the transient overlay.
"""
from __future__ import annotations

import pytest

from superlocalmemory.core import component_healer as ch
from superlocalmemory.core import component_registry as cr


# --------------------------------------------------------------------------
# Component value object
# --------------------------------------------------------------------------

def test_component_is_frozen():
    c = cr.Component(key="x", label="X", category=cr.CATEGORY_OPTIONAL,
                     status=cr.STATUS_OK)
    with pytest.raises(Exception):
        c.status = cr.STATUS_MISSING  # frozen dataclass rejects mutation


def test_component_as_dict_round_trips():
    c = cr.Component(key="k", label="L", category=cr.CATEGORY_REQUIRED,
                     status=cr.STATUS_MISSING, detail="d", fix_cmd="f",
                     auto_fixable=True, last_checked=1.0)
    d = c.as_dict()
    assert d == {
        "key": "k", "label": "L", "category": cr.CATEGORY_REQUIRED,
        "status": cr.STATUS_MISSING, "detail": "d", "fix_cmd": "f",
        "auto_fixable": True, "last_checked": 1.0,
    }


# --------------------------------------------------------------------------
# Probes + snapshot
# --------------------------------------------------------------------------

def test_probe_all_never_raises_and_covers_expected_keys():
    comps = cr.probe_all(None)
    keys = {c.key for c in comps}
    assert {"python", "search_deps", "embedder_model", "reranker_model",
            "compressor_model", "sqlite_vec", "ollama"} <= keys
    for c in comps:
        assert c.status in (cr.STATUS_OK, cr.STATUS_MISSING,
                            cr.STATUS_DEGRADED, cr.STATUS_RETRYING)


def test_snapshot_structure_and_counts():
    snap = cr.snapshot(None)
    assert set(snap) == {"components", "summary", "generated_at"}
    s = snap["summary"]
    assert set(s) == {"ok", "missing", "degraded", "retrying",
                      "auto_fixable_missing", "healthy"}
    assert s["ok"] + s["missing"] + s["degraded"] + s["retrying"] \
        == len(snap["components"])
    assert isinstance(s["healthy"], bool)


def test_ollama_not_in_use_without_mode_b():
    # No config → Ollama is reported OK/"not in use", never a false alarm.
    ollama = next(c for c in cr.probe_all(None) if c.key == "ollama")
    assert ollama.status == cr.STATUS_OK
    assert "not in use" in ollama.detail


def test_pip_writable_returns_bool():
    assert isinstance(cr.pip_is_user_writable(), bool)


# --------------------------------------------------------------------------
# Safety-critical auto_fixable classification
# --------------------------------------------------------------------------

def test_search_deps_never_background_auto_installed(monkeypatch):
    # torch missing must NOT be flagged auto_fixable (multi-GB surprise).
    monkeypatch.setattr(cr, "_module_present", lambda m: False)
    comp = cr.probe_search_deps()
    assert comp.status == cr.STATUS_MISSING
    assert comp.auto_fixable is False
    assert "superlocalmemory[search]" in comp.fix_cmd


def test_compressor_never_auto_downloaded(monkeypatch):
    monkeypatch.setattr(cr, "_hf_model_cached", lambda repo: False)
    comp = cr.probe_compressor_model()
    assert comp.status == cr.STATUS_MISSING
    assert comp.auto_fixable is False  # 560MB lazy model, never auto


def test_sqlite_vec_auto_only_when_writable(monkeypatch):
    monkeypatch.setattr(cr, "_module_present", lambda m: False)
    monkeypatch.setattr(cr, "pip_is_user_writable", lambda: True)
    assert cr.probe_sqlite_vec().auto_fixable is True
    monkeypatch.setattr(cr, "pip_is_user_writable", lambda: False)
    comp = cr.probe_sqlite_vec()
    assert comp.auto_fixable is False
    assert "pipx" in comp.fix_cmd or "uv" in comp.fix_cmd


def test_embedder_auto_fix_requires_torch(monkeypatch):
    # Model missing but torch absent → cannot help by downloading → not auto.
    monkeypatch.setattr(cr, "_hf_model_cached", lambda repo: False)
    monkeypatch.setattr(cr, "_module_present",
                        lambda m: m != "sentence_transformers")
    assert cr.probe_embedder_model(None).auto_fixable is False
    # torch present → downloading the model is the right auto-fix.
    monkeypatch.setattr(cr, "_module_present", lambda m: True)
    assert cr.probe_embedder_model(None).auto_fixable is True


def test_reranker_absent_but_disabled_is_ok():
    class _RT:
        use_cross_encoder = False

    class _Cfg:
        retrieval = _RT()

    comp = cr.probe_reranker_model(_Cfg())
    # Disabled → its absence is expected, reported OK not missing.
    assert comp.status == cr.STATUS_OK
    assert "disabled" in comp.detail


# --------------------------------------------------------------------------
# Transient overlay
# --------------------------------------------------------------------------

def test_transient_overlay_shows_and_clears():
    cr.mark_transient("sqlite_vec", cr.STATUS_RETRYING, "installing…")
    try:
        # Force the base probe to report missing so the overlay is applied
        # (a confirmed-OK component must supersede a stale retry marker).
        missing = cr.Component(key="sqlite_vec", label="v", category="x",
                               status=cr.STATUS_MISSING)
        overlaid = cr._apply_transient(missing)
        assert overlaid.status == cr.STATUS_RETRYING
        ok = cr.Component(key="sqlite_vec", label="v", category="x",
                          status=cr.STATUS_OK)
        assert cr._apply_transient(ok).status == cr.STATUS_OK
    finally:
        cr.clear_transient("sqlite_vec")
    assert cr._transient.get("sqlite_vec") is None


# --------------------------------------------------------------------------
# Model repo-id sync with the wizard
# --------------------------------------------------------------------------

def test_model_ids_match_wizard():
    from superlocalmemory.cli import setup_wizard as w

    assert cr._EMBED_MODEL == w._EMBED_MODEL
    assert cr._RERANKER_MODEL == w._RERANKER_MODEL
    assert cr._COMPRESSOR_MODEL == w._COMPRESSOR_MODEL


# --------------------------------------------------------------------------
# Healer
# --------------------------------------------------------------------------

def test_heal_missing_retries_then_succeeds(monkeypatch):
    fake = cr.Component(key="embedder_model", label="Embedding model",
                        category=cr.CATEGORY_REQUIRED, status=cr.STATUS_MISSING,
                        fix_cmd="slm doctor --fix", auto_fixable=True)
    monkeypatch.setattr(cr, "auto_fixable_missing", lambda config=None: [fake])
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return (False, "blip") if calls["n"] < 2 else (True, "ready")

    monkeypatch.setitem(ch._ACTIONS, "embedder_model", flaky)
    res = ch.heal_missing(config=None)
    assert res == {"attempted": 1, "healed": 1, "failed": 0,
                   "results": [{"key": "embedder_model", "success": True,
                                "detail": "ready"}]}
    assert calls["n"] == 2
    assert cr._transient.get("embedder_model") is None  # cleaned up


def test_heal_missing_records_fix_cmd_on_permanent_failure(monkeypatch):
    fake = cr.Component(key="sqlite_vec", label="Vector index",
                        category=cr.CATEGORY_RECOMMENDED,
                        status=cr.STATUS_MISSING, fix_cmd="pip install sqlite-vec",
                        auto_fixable=True)
    monkeypatch.setattr(cr, "auto_fixable_missing", lambda config=None: [fake])
    monkeypatch.setitem(ch._ACTIONS, "sqlite_vec",
                        lambda: (False, "network down"))
    msgs: list[str] = []
    res = ch.heal_missing(config=None, on_progress=lambda k, m: msgs.append(m))
    assert res["healed"] == 0 and res["failed"] == 1
    assert res["results"][0]["success"] is False
    # The manual fix command is surfaced for the non-technical user.
    assert any("pip install sqlite-vec" in m for m in msgs)


def test_heal_missing_filters_by_key(monkeypatch):
    a = cr.Component(key="embedder_model", label="A", category="x",
                     status=cr.STATUS_MISSING, auto_fixable=True)
    b = cr.Component(key="sqlite_vec", label="B", category="x",
                     status=cr.STATUS_MISSING, auto_fixable=True)
    monkeypatch.setattr(cr, "auto_fixable_missing", lambda config=None: [a, b])
    monkeypatch.setitem(ch._ACTIONS, "embedder_model", lambda: (True, "ok"))
    monkeypatch.setitem(ch._ACTIONS, "sqlite_vec", lambda: (True, "ok"))
    res = ch.heal_missing(config=None, keys=["sqlite_vec"])
    assert res["attempted"] == 1
    assert res["results"][0]["key"] == "sqlite_vec"
