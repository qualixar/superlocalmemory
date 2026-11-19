"""No test and no CI run may ever download a model by itself (4.2.30)."""

from __future__ import annotations

import os
from pathlib import Path

from superlocalmemory.runtimes import media_auto


def test_the_isolation_plugin_sets_the_no_auto_download_guard():
    assert os.environ.get("SLM_NO_AUTO_DOWNLOAD") == "1"  # set by tests/_isolation_plugin.py at import
    assert media_auto.auto_disabled() is True
    plugin = (Path(__file__).resolve().parents[1] / "_isolation_plugin.py").read_text(encoding="utf-8")
    assert 'os.environ["SLM_NO_AUTO_DOWNLOAD"] = "1"' in plugin


def test_the_look_does_nothing_here_even_on_a_big_machine_with_nothing_faked(tmp_path):
    out = media_auto.tick(tmp_path, ram_bytes=64 * 1024 ** 3)  # the real environment: the guard is set
    assert out == media_auto.Outcome("disabled", None)
    assert list(tmp_path.iterdir()) == []


def test_the_daemon_loop_does_nothing_here():
    import asyncio

    out = asyncio.run(media_auto.run(None, first_delay_s=0.0))
    assert out.label == "disabled"
