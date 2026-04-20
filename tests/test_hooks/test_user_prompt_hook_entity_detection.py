# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-13 Track C.1

"""UserPromptSubmit hook — inline entity detection integration tests.

Two tests per Manifest Track C.1:
  1. ``test_user_prompt_hook_enriches_topic_signature`` — hook enriches
     the topic signature with entity IDs when trigram lookup hits.
  2. ``test_user_prompt_hook_budget_unchanged`` — entire hook stays under
     the 50 ms p95 overall budget (I1) with entity detection added.

These tests exercise the additive extension to ``user_prompt_hook.py``
(must not break the existing context-cache hit/miss flow) and to
``topic_signature.py`` (backward-compatible ``entity_hits`` kwarg).
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# 1. Hook enriches topic signature with entity hits
# --------------------------------------------------------------------------


def test_user_prompt_hook_enriches_topic_signature(monkeypatch):
    """When trigram lookup yields hits for the prompt, the signature used
    to probe the context cache must differ from the no-hits signature.

    Strategy:
      - Capture `read_entry_fast(session_id, signature)` calls.
      - Case A: lookup stubbed to return ``[]``  → sig_A passed in.
      - Case B: lookup stubbed to return ``[("e001", 3)]`` → sig_B passed in.
      - Assert sig_A != sig_B  AND  both contain the base topic signature.
    """
    from superlocalmemory.core import topic_signature as ts

    # Baseline signature (no entity enrichment).
    base_sig = ts.compute_topic_signature("what is SuperLocalMemory")

    captured: dict[str, str] = {}

    # Stub context_cache read — always miss so the hook exits cleanly.
    class _StubEntry:
        content = "test"

    def _fake_read_entry_fast(session_id: str, topic_sig: str):
        captured["sig"] = topic_sig
        return None  # miss

    # Stub TrigramIndex.lookup to return deterministic hits.
    def _lookup_hits(self, text: str):
        return [("e001", 3), ("e007", 2)]

    def _lookup_empty(self, text: str):
        return []

    from superlocalmemory.learning import trigram_index as ti
    from superlocalmemory.core import context_cache as cc

    # Force construction of a singleton index inside the hook to be a no-op.
    def _fake_get_or_none(*args, **kwargs):
        return _FakeIdx()

    class _FakeIdx:
        def lookup(self, text: str):
            return self._hits

    # --- Case A: no hits ---
    idx_a = _FakeIdx()
    idx_a._hits = []
    monkeypatch.setattr(ti, "get_or_none", lambda: idx_a, raising=False)
    monkeypatch.setattr(cc, "read_entry_fast", _fake_read_entry_fast)

    payload = {"session_id": "sess_test", "prompt": "what is SuperLocalMemory"}
    _invoke_hook(monkeypatch, payload)
    sig_a = captured.get("sig")

    # --- Case B: with hits ---
    captured.clear()
    idx_b = _FakeIdx()
    idx_b._hits = [("e001", 3), ("e007", 2)]
    monkeypatch.setattr(ti, "get_or_none", lambda: idx_b, raising=False)

    _invoke_hook(monkeypatch, payload)
    sig_b = captured.get("sig")

    assert sig_a is not None and sig_b is not None
    assert sig_a != sig_b, (
        f"signatures must differ when entity hits change; both = {sig_a}"
    )
    # Both signatures remain 16-char hex (compute_topic_signature contract
    # preserved through the entity_hits kwarg).
    assert len(sig_a) == 16 and len(sig_b) == 16
    assert all(c in "0123456789abcdef" for c in sig_a + sig_b)
    # And base_sig (no entity enrichment) matches sig_a (no hits case).
    assert sig_a == base_sig


# --------------------------------------------------------------------------
# 2. Hook overall budget unchanged (still <50 ms p95)
# --------------------------------------------------------------------------


def test_user_prompt_hook_budget_unchanged(monkeypatch):
    """1000 hook invocations with entity detection in the critical path
    must still finish under the I1 50 ms p95 budget (we assert 50 ms hard,
    far larger than the expected <10 ms on a dev box).
    """
    from superlocalmemory.learning import trigram_index as ti
    from superlocalmemory.core import context_cache as cc

    class _FakeIdx:
        def lookup(self, text: str):
            return [("e001", 2)]

    monkeypatch.setattr(ti, "get_or_none", lambda: _FakeIdx(), raising=False)
    monkeypatch.setattr(cc, "read_entry_fast", lambda s, t: None)

    payload = {"session_id": "sess_bench",
               "prompt": "budget check for SuperLocalMemory hook path"}

    N = 1000
    timings: list[float] = []
    for _ in range(N):
        t0 = time.perf_counter_ns()
        _invoke_hook(monkeypatch, payload)
        timings.append((time.perf_counter_ns() - t0) / 1_000_000.0)

    timings.sort()
    p95 = timings[int(N * 0.95)]
    assert p95 < 50.0, f"hook p95 {p95:.2f} ms exceeds 50 ms I1 budget"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _invoke_hook(monkeypatch, payload: dict) -> dict:
    """Run the hook's ``main()`` with payload piped on stdin; return parsed
    stdout envelope."""
    from superlocalmemory.hooks import user_prompt_hook as uph

    raw = json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(raw))
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    rc = uph.main()
    assert rc == 0
    body = buf.getvalue() or "{}"
    try:
        return json.loads(body)
    except Exception:
        return {}
