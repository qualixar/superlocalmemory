# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for CLI — Task 15 of V3 build.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from superlocalmemory.cli.main import main


def test_main_no_args(capsys):
    """No arguments prints help and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        with patch("sys.argv", ["slm"]):
            main()
    assert exc_info.value.code == 0


def test_main_status(capsys):
    """Status command prints product name."""
    with patch("sys.argv", ["slm", "status"]):
        main()
    captured = capsys.readouterr()
    assert "SuperLocalMemory V3" in captured.out


def test_main_mode_get(capsys):
    """Mode command without value prints current mode."""
    with patch("sys.argv", ["slm", "mode"]):
        main()
    captured = capsys.readouterr()
    assert "mode" in captured.out.lower()


def test_setup_wizard_importable():
    """Setup wizard function is importable."""
    from superlocalmemory.cli.setup_wizard import run_wizard

    assert callable(run_wizard)


def test_commands_dispatch_importable():
    """Dispatch function is importable."""
    from superlocalmemory.cli.commands import dispatch

    assert callable(dispatch)


def test_provider_presets_available():
    """Provider presets contain at least 4 providers."""
    from superlocalmemory.core.config import SLMConfig

    presets = SLMConfig.provider_presets()
    assert len(presets) >= 4
