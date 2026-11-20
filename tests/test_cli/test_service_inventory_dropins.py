"""#339: a user-wide systemd drop-in that cannot change the command stays harmless."""

from types import SimpleNamespace

import pytest

from superlocalmemory.cli import service_inventory as inventory

FEDORA = "[Service]\nTimeoutStopFailureMode=abort\n"


@pytest.fixture
def unit(tmp_path, monkeypatch):
    path = tmp_path / "home" / ".config" / "systemd" / "user" / "superlocalmemory.service"
    path.parent.mkdir(parents=True)
    path.write_text("[Service]\nExecStart=/bin/true\n")
    state = SimpleNamespace(path=path, dropins="")

    def fake_run(argv):
        assert argv[:3] == ["systemctl", "--user", "show"]
        return SimpleNamespace(
            returncode=0, stderr="",
            stdout=f"FragmentPath={path}\nDropInPaths={state.dropins}\n")

    monkeypatch.setattr(inventory, "_run", fake_run)
    return state


def make_dropin(tmp_path, content, *, directory="service.d", name="10-timeout-abort.conf"):
    folder = tmp_path / "usr-lib" / directory
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(content)
    return path


def test_empty_dropin_paths_is_accepted(unit):
    assert inventory._linux_effective(unit.path) is True


def test_fedora_userwide_timeout_dropin_is_accepted(unit, tmp_path):
    unit.dropins = str(make_dropin(tmp_path, FEDORA))
    assert inventory._linux_effective(unit.path) is True


@pytest.mark.parametrize("content", [
    "[Service]\nExecStart=/bin/evil\n",
    "[Service]\nExecStart=\nExecStart=/bin/evil\n",
    "[Service]\nExecStartPre=/bin/evil\n",
    "[Service]\nExecStartPost=/bin/evil\n",
    "[Service]\nExecReload=/bin/evil\n",
    "[Service]\nEnvironment=SLM_DATA_DIR=/elsewhere\n",
    "[Service]\nEnvironmentFile=/etc/evil.env\n",
    "[Service]\nWorkingDirectory=/elsewhere\n",
    "[Service]\nUser=root\n",
    "[Service]\nGroup=root\n",
    "[Service]\nRootDirectory=/jail\n",
    "[Service]\nPassEnvironment=SLM_DATA_DIR\n",
    "[Service]\nUnsetEnvironment=SLM_DATA_DIR\n",
    "[Service]\nProtectHome=tmpfs\n",
    "[Service]\nInaccessiblePaths=%h/.superlocalmemory\n",
    "[Service]\nBindPaths=/elsewhere:%h/.superlocalmemory\n",
    "[Service]\nTemporaryFileSystem=%h\n",
    "[Service]\nTimeoutStopFailureMode=abort\nEnvironment=A=B\n",
    "[Service]\nEnvironment=A=\\\nB\n",
    "[Unit]\nDescription=ok\n[Service]\nExecStart=/bin/evil\n",
])
def test_userwide_dropin_that_changes_the_command_is_refused(unit, tmp_path, content):
    unit.dropins = str(make_dropin(tmp_path, content))
    assert inventory._linux_effective(unit.path) is False


def test_unit_keys_and_comments_are_harmless(unit, tmp_path):
    content = ("# ExecStart=/bin/commented\n; Environment=A=B\n"
               "[Unit]\nEnvironment=ignored\nExecStart=ignored\nDescription=x\n"
               "[Service]\n# User=root\nTimeoutStopFailureMode=abort\n")
    unit.dropins = str(make_dropin(tmp_path, content))
    assert inventory._linux_effective(unit.path) is True


def test_key_match_is_case_sensitive_like_systemd(unit, tmp_path):
    unit.dropins = str(make_dropin(tmp_path, "[Service]\nenvironment=A=B\n"))
    assert inventory._linux_effective(unit.path) is True


def test_continuation_line_hides_no_key(unit, tmp_path):
    content = "[Service]\nTimeoutStopSec=1 \\\n  2\nUser=root\n"
    unit.dropins = str(make_dropin(tmp_path, content))
    assert inventory._linux_effective(unit.path) is False


def test_harmless_dropin_in_the_units_own_directory_is_refused(unit, tmp_path):
    unit.dropins = str(make_dropin(tmp_path, FEDORA, directory="superlocalmemory.service.d"))
    assert inventory._linux_effective(unit.path) is False


def test_prefix_named_directory_is_not_the_units_own(unit, tmp_path):
    unit.dropins = str(make_dropin(tmp_path, FEDORA, directory="other.service.d"))
    assert inventory._linux_effective(unit.path) is True


def test_unreadable_dropin_is_refused(unit, tmp_path):
    unit.dropins = str(tmp_path / "usr-lib" / "service.d" / "missing.conf")
    assert inventory._linux_effective(unit.path) is False


def test_undecodable_dropin_is_refused(unit, tmp_path):
    path = make_dropin(tmp_path, "")
    path.write_bytes(b"[Service]\n\xff\xfe=\x00\n")
    unit.dropins = str(path)
    assert inventory._linux_effective(unit.path) is False


@pytest.mark.parametrize("content", ["[Service\nTimeoutStopSec=1\n", "TimeoutStopSec=1\n",
                                     "[Service]\nnot an assignment\n"])
def test_unparseable_dropin_is_refused(unit, tmp_path, content):
    unit.dropins = str(make_dropin(tmp_path, content))
    assert inventory._linux_effective(unit.path) is False


def test_one_bad_dropin_among_several_refuses_all(unit, tmp_path):
    good = make_dropin(tmp_path, FEDORA)
    bad = make_dropin(tmp_path, "[Service]\nExecStart=/bin/evil\n", name="90-bad.conf")
    unit.dropins = f"{good} {bad}"
    assert inventory._linux_effective(unit.path) is False
    unit.dropins = f"{good} {good}"
    assert inventory._linux_effective(unit.path) is True


def test_dropin_directory_next_to_unit_file_is_still_refused(unit):
    (unit.path.parent / f"{unit.path.name}.d").mkdir()
    assert inventory._linux_effective(unit.path) is False
