# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Dashboard write surfaces must present the local install credential."""

from pathlib import Path


def test_quick_store_fetches_and_sends_install_token() -> None:
    source = Path("src/superlocalmemory/ui/js/dashboard.js").read_text(
        encoding="utf-8"
    )

    assert "/internal/token" in source
    assert "X-Install-Token" in source
    assert "slm_install_token" in source
