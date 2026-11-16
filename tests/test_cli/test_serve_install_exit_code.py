"""Issue #319: `slm serve install` printed "Failed" but exited 0."""

from argparse import Namespace

import pytest

from superlocalmemory.cli import commands

#: Patched by dotted path, so the module ``cmd_serve`` imports at call time is
#: the one patched: a test that re-imports the CLI under another platform
#: (``test_platform_import_safety``) leaves a stale module object behind, and
#: patching that one let the real installer run ``launchctl load`` (F-38).
_INSTALLER = "superlocalmemory.cli.service_installer"


@pytest.mark.parametrize(("action", "attr"), [("install", "install_service"),
                                              ("uninstall", "uninstall_service")])
def test_failed_service_change_exits_nonzero(action, attr, monkeypatch, capsys):
    monkeypatch.setattr(f"{_INSTALLER}.{attr}", lambda: False)
    with pytest.raises(SystemExit) as stopped:
        commands.cmd_serve(Namespace(action=action))
    assert stopped.value.code == 1
    assert "Failed" in capsys.readouterr().err


def test_successful_install_still_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(f"{_INSTALLER}.install_service", lambda: True)
    commands.cmd_serve(Namespace(action="install"))
    assert "Service installed" in capsys.readouterr().out


def test_write_failure_is_logged_with_its_reason(monkeypatch, tmp_path, caplog):
    from superlocalmemory.cli import service_inventory

    class Refusing:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            raise OSError("Verified service definition changed")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(service_inventory, "ServiceFiles", Refusing)
    assert service_inventory.write(tmp_path / "start.vbs", "x") is False
    assert "Verified service definition changed" in caplog.text
