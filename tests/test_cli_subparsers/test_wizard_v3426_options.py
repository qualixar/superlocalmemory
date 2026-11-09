"""Setup-wizard extensions shipped with v3.4.26.

End-user contract:
- The wizard validates the chosen data directory at install time (iCloud,
  Dropbox, OneDrive, etc. are rejected BEFORE the user finishes setup, not
  at first recall failure).
- The queue is on by default. The wizard exposes a single "Enable
  concurrent-recall queue? [Y/n]" prompt; everything else (rate limits,
  priorities) has sane defaults.
- Non-interactive installs get defaults silently.
- Chosen options land in ``<data_dir>/v3426_options.json``.
"""
from __future__ import annotations

import json

import pytest

from superlocalmemory.cli.wizard_v3426_options import (
    V3426Options,
    persist_v3426_options,
    prompt_v3426_options,
    validate_install_data_dir,
)


@pytest.fixture
def slm_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    return tmp_path


class TestDefaults:
    def test_defaults_non_interactive(self, slm_home):
        opts = prompt_v3426_options(interactive=False)
        assert opts.queue_enabled is True
        assert opts.rate_limit_per_pid > 0
        assert opts.rate_limit_per_agent > 0
        assert opts.rate_limit_global > opts.rate_limit_per_pid

    def test_defaults_stable(self, slm_home):
        """Defaults must not drift release-over-release without intent."""
        opts = prompt_v3426_options(interactive=False)
        assert opts.rate_limit_per_pid == 30
        assert opts.rate_limit_per_agent == 10
        assert opts.rate_limit_global == 100


class TestDataDirValidation:
    def test_local_path_accepted(self, tmp_path):
        ok, msg = validate_install_data_dir(tmp_path / "slm-local")
        assert ok is True
        assert msg == ""

    def test_icloud_rejected(self, tmp_path):
        faux = tmp_path / "Mobile Documents" / "com~apple~CloudDocs" / "slm"
        faux.mkdir(parents=True)
        ok, msg = validate_install_data_dir(faux)
        assert ok is False
        assert "cloud" in msg.lower() or "mobile documents" in msg.lower()

    def test_dropbox_rejected(self, tmp_path):
        faux = tmp_path / "Dropbox" / "slm"
        faux.mkdir(parents=True)
        ok, msg = validate_install_data_dir(faux)
        assert ok is False
        assert "dropbox" in msg.lower()

    def test_onedrive_rejected(self, tmp_path):
        faux = tmp_path / "OneDrive" / "slm"
        faux.mkdir(parents=True)
        ok, msg = validate_install_data_dir(faux)
        assert ok is False
        assert "onedrive" in msg.lower()


class TestPersistence:
    def test_persist_then_reload_json(self, slm_home):
        opts = V3426Options(
            queue_enabled=True,
            rate_limit_per_pid=30,
            rate_limit_per_agent=10,
            rate_limit_global=100,
        )
        persist_v3426_options(opts, slm_home)

        out = json.loads((slm_home / "v3426_options.json").read_text(encoding="utf-8"))
        assert out == {
            "queue_enabled": True,
            "rate_limit_per_pid": 30,
            "rate_limit_per_agent": 10,
            "rate_limit_global": 100,
        }

    def test_persist_creates_parent_dir(self, tmp_path):
        target = tmp_path / "fresh"
        opts = V3426Options(True, 30, 10, 100)
        persist_v3426_options(opts, target)
        assert (target / "v3426_options.json").exists()

    def test_persist_overwrites(self, slm_home):
        persist_v3426_options(V3426Options(True, 30, 10, 100), slm_home)
        persist_v3426_options(V3426Options(False, 5, 2, 20), slm_home)
        out = json.loads((slm_home / "v3426_options.json").read_text(encoding="utf-8"))
        assert out["queue_enabled"] is False
        assert out["rate_limit_per_pid"] == 5


class TestSecurityHardening:
    def test_options_written_at_0600(self, slm_home):
        import os
        import sys
        if sys.platform == "win32":
            pytest.skip("POSIX perm check")
        persist_v3426_options(V3426Options(True, 30, 10, 100), slm_home)
        target = slm_home / "v3426_options.json"
        mode = os.stat(target).st_mode & 0o777
        assert mode == 0o600

    def test_parent_dir_tightened_to_0700(self, tmp_path):
        import os
        import sys
        if sys.platform == "win32":
            pytest.skip("POSIX perm check")
        loose = tmp_path / "loose"
        loose.mkdir(mode=0o755)
        persist_v3426_options(V3426Options(True, 30, 10, 100), loose)
        mode = os.stat(loose).st_mode & 0o777
        assert mode == 0o700

    def test_cloudstorage_path_rejected(self, tmp_path):
        faux = tmp_path / "Library" / "CloudStorage" / "Dropbox-Personal" / "slm"
        faux.mkdir(parents=True)
        ok, reason = validate_install_data_dir(faux)
        assert ok is False
        assert "cloudstorage" in reason.lower() or "cloud" in reason.lower()


class TestInteractivePrompt:
    def test_queue_toggle_default_yes(self, slm_home, monkeypatch):
        # Empty input → accept default (Y)
        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        opts = prompt_v3426_options(interactive=True)
        assert opts.queue_enabled is True

    def test_queue_toggle_explicit_no(self, slm_home, monkeypatch):
        inputs = iter(["n"])
        monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
        opts = prompt_v3426_options(interactive=True)
        assert opts.queue_enabled is False
