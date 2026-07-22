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
    # The slm_install_token sessionStorage key was removed in 3.7.9 (B2/M-06 XSS
    # fix). The token is fetched from /internal/token and sent as X-Install-Token
    # (asserted above) — no longer parked in client-readable storage.


def test_all_same_origin_browser_mutations_receive_install_token() -> None:
    source = Path("src/superlocalmemory/ui/js/core.js").read_text(
        encoding="utf-8"
    )

    assert "/internal/token" in source
    assert "X-Install-Token" in source
    # The slm_install_token sessionStorage key was removed in 3.7.9 (B2/M-06 XSS
    # fix). The token is fetched from /internal/token and sent as X-Install-Token
    # (asserted above) — no longer parked in client-readable storage.
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert method in source


def test_memory_mutation_routes_require_authenticated_actor() -> None:
    source = Path("src/superlocalmemory/server/routes/memories.py").read_text(
        encoding="utf-8"
    )

    for function_name in (
        "delete_memory",
        "forget_memory",
        "merge_memory",
        "edit_memory",
    ):
        start = source.index(f"async def {function_name}")
        next_route = source.find("\n@router.", start + 1)
        block = source[start: next_route if next_route != -1 else None]
        assert "_authorize_memory_mutation" in block, function_name

    assert "require_write_actor" in source
