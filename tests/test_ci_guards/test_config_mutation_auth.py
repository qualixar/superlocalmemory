"""Configuration writes must cross the explicit MANAGE authorization boundary."""

from pathlib import Path


def _handler_block(source: str, name: str) -> str:
    async_marker = f"async def {name}"
    sync_marker = f"def {name}"
    start = (
        source.index(async_marker)
        if async_marker in source
        else source.index(sync_marker)
    )
    next_handler = source.find("\n@router.", start + 1)
    return source[start:next_handler if next_handler != -1 else None]


def test_dashboard_config_mutations_require_manage() -> None:
    source = Path("src/superlocalmemory/server/routes/v3_api.py").read_text(
        encoding="utf-8"
    )

    for handler in (
        "set_auto_capture_config",
        "set_auto_recall_config",
        "set_auto_invoke_config",
    ):
        assert "_require_manage(request)" in _handler_block(source, handler)


def test_evolution_config_mutations_require_manage() -> None:
    source = Path("src/superlocalmemory/server/routes/evolution.py").read_text(
        encoding="utf-8"
    )

    for handler in (
        "evolution_enable",
        "evolution_disable",
        "evolution_run",
        "evolution_config",
    ):
        assert "_require_manage(request)" in _handler_block(source, handler)
