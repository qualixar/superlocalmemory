"""Regression guard for the MCP startup reaper's ownership boundary."""

from superlocalmemory.infra.process_reaper import SlmProcessInfo, is_mcp_server_process


def _process(command: str) -> SlmProcessInfo:
    return SlmProcessInfo(
        pid=12345,
        ppid=1,
        start_time=0.0,
        command=command,
        is_orphan=True,
        parent_name="init",
        age_hours=1.0,
    )


def test_mcp_startup_reaper_matches_only_mcp_servers() -> None:
    assert is_mcp_server_process(_process("/venv/bin/slm mcp"))
    assert is_mcp_server_process(
        _process("/venv/bin/python -m superlocalmemory.cli.main mcp")
    )
    assert not is_mcp_server_process(
        _process("/venv/bin/python -m superlocalmemory.server.unified_daemon --start")
    )
