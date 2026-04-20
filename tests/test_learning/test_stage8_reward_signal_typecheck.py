# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — Stage 8 F5 (Mediums/Lows)

"""Stage 8 F5 regressions — reward signal strict typing + ISO-8601.

Covers:
  - SEC-M1: ``_coerce_signal_value('dwell_ms', …)`` rejects bool,
    float, str, bytes, bytearray.
  - SEC-GTH-01 / S-G-02: ``_iso_from_ms`` uses strict ISO-8601 with
    the ``T`` separator and ``Z`` suffix.
"""

from __future__ import annotations

import re

from superlocalmemory.learning.reward import (
    _coerce_signal_value,
    _iso_from_ms,
)


# ---------------------------------------------------------------------------
# SEC-M1 — dwell_ms strict-int contract.
# ---------------------------------------------------------------------------


def test_sec_m1_coerce_dwell_accepts_plain_int() -> None:
    assert _coerce_signal_value("dwell_ms", 3000) == 3000


def test_sec_m1_coerce_dwell_clamps_negative_to_zero() -> None:
    assert _coerce_signal_value("dwell_ms", -500) == 0


def test_sec_m1_coerce_dwell_clamps_above_max() -> None:
    # _DWELL_MAX_MS == 1h == 3_600_000 ms
    assert _coerce_signal_value("dwell_ms", 10**9) == 3_600_000


def test_sec_m1_coerce_dwell_rejects_bool() -> None:
    # bool is an int subclass in Python — must be rejected so a hook
    # that leaks a True/False cannot be attributed as "1 ms of dwell".
    assert _coerce_signal_value("dwell_ms", True) is None
    assert _coerce_signal_value("dwell_ms", False) is None


def test_sec_m1_coerce_dwell_rejects_float() -> None:
    assert _coerce_signal_value("dwell_ms", 2500.5) is None
    assert _coerce_signal_value("dwell_ms", -0.5) is None


def test_sec_m1_coerce_dwell_rejects_str() -> None:
    assert _coerce_signal_value("dwell_ms", "3000") is None


def test_sec_m1_coerce_dwell_rejects_bytes() -> None:
    assert _coerce_signal_value("dwell_ms", b"3000") is None


def test_sec_m1_coerce_dwell_rejects_bytearray() -> None:
    assert _coerce_signal_value("dwell_ms", bytearray(b"3000")) is None


def test_sec_m1_coerce_dwell_rejects_none() -> None:
    assert _coerce_signal_value("dwell_ms", None) is None


def test_sec_m1_coerce_bool_signals_still_coerce() -> None:
    # cite / edit / requery remain permissive bool coercion — the
    # contract tightening is scoped to dwell_ms only.
    assert _coerce_signal_value("cite", 1) is True
    assert _coerce_signal_value("edit", 0) is False
    assert _coerce_signal_value("requery", "anything") is True


# ---------------------------------------------------------------------------
# SEC-GTH-01 / S-G-02 — ISO-8601 strict.
# ---------------------------------------------------------------------------


def test_sec_gth_01_iso_from_ms_uses_t_and_z() -> None:
    s = _iso_from_ms(0)
    # Must have the 'T' separator and trailing 'Z'.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s), s
    assert s.endswith("Z")


def test_sec_gth_01_iso_from_ms_deterministic_for_known_epoch() -> None:
    # 2026-04-19 15:30:00 UTC
    ms = 1776612600_000
    assert _iso_from_ms(ms) == "2026-04-19T15:30:00Z"
