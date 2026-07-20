"""OS service definitions must preserve the selected SLM namespace."""

from __future__ import annotations

import plistlib
from pathlib import Path


def test_macos_service_propagates_canonical_root_and_port(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.cli import service_installer

    selected = tmp_path / "root with spaces"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SL_MEMORY_PATH", str(tmp_path / "wrong"))
    monkeypatch.setenv("SLM_DAEMON_PORT", "19123")

    payload = plistlib.loads(service_installer._macos_plist_content().encode())

    assert payload["EnvironmentVariables"]["SLM_DATA_DIR"] == str(selected.resolve())
    assert payload["EnvironmentVariables"]["SLM_DAEMON_PORT"] == "19123"
    assert "--port=19123" in payload["ProgramArguments"]
    assert Path(payload["StandardOutPath"]).is_relative_to(selected.resolve())
    assert Path(payload["StandardErrorPath"]).is_relative_to(selected.resolve())


def test_linux_service_propagates_canonical_root_and_port(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.cli import service_installer

    selected = tmp_path / "root with spaces"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SLM_DAEMON_PORT", "19124")

    content = service_installer._linux_service_content()

    assert f'Environment="SLM_DATA_DIR={selected.resolve()}"' in content
    assert 'Environment="SLM_DAEMON_PORT=19124"' in content
    assert "--port=19124" in content
    assert str(selected.resolve() / "logs" / "daemon.log") in content


def test_windows_wrapper_propagates_canonical_root_and_port(
    monkeypatch, tmp_path: Path,
) -> None:
    from superlocalmemory.cli import service_installer

    selected = tmp_path / "root with spaces"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))
    monkeypatch.setenv("SLM_DAEMON_PORT", "19125")

    content = service_installer._windows_vbs_content()

    assert f'SLM_DATA_DIR") = "{selected.resolve()}"' in content
    assert 'SLM_DAEMON_PORT") = "19125"' in content
    assert "--port=19125" in content


def test_service_logs_are_always_under_selected_root(monkeypatch, tmp_path: Path) -> None:
    from superlocalmemory.cli import service_installer

    selected = tmp_path / "service-root"
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    assert service_installer.get_log_path() == selected.resolve() / "logs" / "daemon.log"
    assert service_installer.get_error_log_path() == (
        selected.resolve() / "logs" / "daemon-error.log"
    )
