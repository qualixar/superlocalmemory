# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""v3.8.2 UX-3: unified `slm setup` improvements.

Covers the side-effect-free decision points added to the wizard:
- Ollama pre-detection (used to recommend Mode B with one keypress),
- the multi-IDE connect step is a strict no-op when non-interactive.
Full flow is verified by a non-interactive wizard run; these lock the units.
"""
from __future__ import annotations

import superlocalmemory.cli.setup_wizard as sw


def test_ollama_available_when_server_responds(monkeypatch):
    import httpx

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert sw._ollama_available() is True


def test_ollama_available_when_only_binary_present(monkeypatch):
    import httpx

    def _boom(*a, **k):
        raise RuntimeError("no ollama server")

    monkeypatch.setattr(httpx, "get", _boom)
    monkeypatch.setattr(sw.shutil, "which", lambda name: "/opt/homebrew/bin/ollama")
    assert sw._ollama_available() is True


def test_ollama_available_false_when_absent(monkeypatch):
    import httpx

    def _boom(*a, **k):
        raise RuntimeError("no ollama server")

    monkeypatch.setattr(httpx, "get", _boom)
    monkeypatch.setattr(sw.shutil, "which", lambda name: None)
    assert sw._ollama_available() is False


def test_configure_other_ides_noop_when_noninteractive():
    # Editing IDE configs crosses an ownership boundary — never in a
    # non-interactive run, even best-effort.
    assert sw._configure_other_ides(interactive=False) is False
