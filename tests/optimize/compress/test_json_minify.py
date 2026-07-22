# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for lossless JSON minification (P4a optimize-default lossless tier).

Covers the pure ``_json_minify`` helper (losslessness guards) and its wiring
into ``CompressRouter._compress_text`` / ``compress_text`` (strategy label,
config gating, code protection). Config is always mocked in-memory — these
tests never read or write the machine's optimize.json.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from superlocalmemory.optimize.compress.router import (
    CompressRouter,
    _MIN_CHARS_FOR_COMPRESSION,
    _json_minify,
)
from superlocalmemory.optimize.config.schema import OptimizeConfig


# A pretty-printed payload comfortably over the min-compression threshold.
def _big_pretty() -> str:
    return json.dumps(
        [{"id": i, "name": f"item-{i}", "active": True, "score": i * 1.5}
         for i in range(40)],
        indent=4,
    )


def _router_with(compress_enabled: bool, mode: str = "safe") -> CompressRouter:
    """Fresh router whose config is a pure in-memory OptimizeConfig (no disk)."""
    r = CompressRouter()
    cfg = dataclasses.replace(
        OptimizeConfig(), compress_enabled=compress_enabled, compress_mode=mode,
    )
    r._get_config = lambda: cfg  # type: ignore[method-assign]
    return r


# ---- _json_minify: lossless wins ----

def test_minify_shrinks_pretty_json_value_equal() -> None:
    pretty = _big_pretty()
    compact = _json_minify(pretty)
    assert compact is not None
    assert len(compact) < len(pretty)
    assert json.loads(compact) == json.loads(pretty)


def test_minify_preserves_unicode() -> None:
    pretty = json.dumps({"city": "café ☕ Zürich", "emoji": "🚀"}, indent=2)
    compact = _json_minify(pretty)
    assert compact is not None
    assert "café ☕ Zürich" in compact and "🚀" in compact  # ensure_ascii=False
    assert json.loads(compact) == json.loads(pretty)


def test_minify_preserves_float_int_distinction() -> None:
    compact = _json_minify('{\n  "a": 1.0,\n  "b": 2,\n  "big": 100000000000000000000\n}')
    assert compact is not None
    parsed = json.loads(compact)
    assert parsed["a"] == 1.0 and isinstance(parsed["a"], float)
    assert parsed["b"] == 2 and isinstance(parsed["b"], int)
    assert parsed["big"] == 100000000000000000000


def test_minify_nested_structures_roundtrip() -> None:
    pretty = json.dumps({"o": {"a": [1, [2, {"deep": True}]], "n": None}}, indent=4)
    compact = _json_minify(pretty)
    assert compact is not None
    assert json.loads(compact) == json.loads(pretty)


# ---- _json_minify: losslessness guards (return None → caller passes original) ----

def test_minify_already_compact_returns_none() -> None:
    assert _json_minify('{"a":1,"b":2}') is None


def test_minify_duplicate_keys_returns_none() -> None:
    # Reserialization would drop the first value — refuse it.
    assert _json_minify('{"a":1,"a":2}') is None


def test_minify_nested_duplicate_keys_returns_none() -> None:
    assert _json_minify('{"outer":{"k":1,"k":2}}') is None


@pytest.mark.parametrize("payload", ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'])
def test_minify_non_finite_returns_none(payload: str) -> None:
    assert _json_minify(payload) is None


@pytest.mark.parametrize("payload", ["just prose", "[not, valid, json]", "", "   ", "def f(): pass"])
def test_minify_non_json_returns_none(payload: str) -> None:
    assert _json_minify(payload) is None


def test_minify_does_not_mutate_input() -> None:
    pretty = _big_pretty()
    snapshot = str(pretty)
    _json_minify(pretty)
    assert pretty == snapshot


# ---- router _compress_text integration (worker; config-independent) ----

def test_compress_text_worker_labels_json_minify() -> None:
    r = CompressRouter()
    pretty = _big_pretty()
    compressed, tb, ta, strat = r._compress_text(
        pretty, aggressive=False, request_id="t", model="", tenant_id="default",
    )
    assert strat == "json_minify"
    assert json.loads(compressed) == json.loads(pretty)
    assert ta <= tb


def test_compress_text_worker_code_never_json_minified() -> None:
    r = CompressRouter()
    code = "import os\n\n\n\ndef handler(x):\n    return x + 1\n" * 40
    assert len(code) > _MIN_CHARS_FOR_COMPRESSION
    _, _, _, strat = r._compress_text(
        code, aggressive=False, request_id="t", model="", tenant_id="default",
    )
    assert strat != "json_minify"


def test_compress_text_worker_small_json_below_threshold_passthrough() -> None:
    r = CompressRouter()
    small = json.dumps({"k": "v"}, indent=4)  # < 500 chars
    assert len(small) < _MIN_CHARS_FOR_COMPRESSION
    text, tb, ta, strat = r._compress_text(
        small, aggressive=False, request_id="t", model="", tenant_id="default",
    )
    assert strat == "none" and text == small


# ---- compress_text config gating (P4a honesty) ----

def test_compress_text_enabled_minifies() -> None:
    res = _router_with(compress_enabled=True).compress_text(_big_pretty())
    assert res.strategy == "json_minify"
    assert res.lossy is False


def test_compress_text_disabled_passes_through() -> None:
    pretty = _big_pretty()
    res = _router_with(compress_enabled=False).compress_text(pretty)
    assert res.strategy == "none"
    assert res.compressed_text == pretty
    assert res.lossy is False


def test_compress_text_default_config_is_enabled() -> None:
    # P4a: safe compression is ON by default (in-memory OptimizeConfig, no disk).
    assert OptimizeConfig().compress_enabled is True
