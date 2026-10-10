"""Folders that must never be picked as a source."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from superlocalmemory.sources.roots import RootRefused, check_root, is_inside


@pytest.fixture(autouse=True)
def _data_root(tmp_path, monkeypatch):
    """The data folder is somewhere else, so these folders are judged on their own merits."""
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path.parent / (tmp_path.name + "-slm-data")))


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home" / "me"
    h.mkdir(parents=True)
    return h


def refused(path, home, **kw) -> str:
    with pytest.raises(RootRefused) as exc:
        check_root(path, home=home, environ=kw.pop("environ", {}), mounts=kw.pop("mounts", ""), **kw)
    assert str(exc.value)
    return exc.value.code


def test_ok_folder_returns_resolved_path(home):
    notes = home / "notes"
    notes.mkdir()
    assert check_root(notes, home=home, environ={}, mounts="") == notes.resolve()


def test_home_itself(home):
    assert refused(home, home) == "home_directory"


def test_filesystem_root(home):
    assert refused(Path("/"), home) == "filesystem_root"


def test_drive_root_windows_style(home):
    assert refused("C:\\", home, windows=True) == "filesystem_root"
    assert refused("\\\\server\\share\\x", home, windows=True) == "network_share"


def test_library_folder(home):
    lib = home / "Library"
    lib.mkdir()
    assert refused(lib, home) == "system_folder"
    (lib / "Application Support").mkdir()
    assert refused(lib / "Application Support", home) == "system_folder"


def test_library_cloud_folders_are_allowed(home):
    vault = home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    vault.mkdir(parents=True)
    assert check_root(vault, home=home, environ={}, mounts="") == vault.resolve()


def test_appdata(home, tmp_path):
    appdata = tmp_path / "AppData" / "Roaming"
    appdata.mkdir(parents=True)
    assert refused(appdata, home, environ={"APPDATA": str(appdata)}) == "system_folder"


@pytest.mark.parametrize("name", [".ssh", ".aws", ".gnupg"])
def test_credential_folder_at_top_level(home, name):
    proj = home / "proj"
    (proj / name).mkdir(parents=True)
    assert refused(proj, home) == "holds_credentials"


def test_credential_folder_deeper_is_not_a_refusal(home):
    proj = home / "proj"
    (proj / "sub" / ".ssh").mkdir(parents=True)
    assert check_root(proj, home=home, environ={}, mounts="") == proj.resolve()


def test_symlink_to_home_is_resolved_then_refused(home, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(home, target_is_directory=True)
    assert refused(link, home) == "home_directory"


def test_symlink_root_pointing_elsewhere_is_refused(home, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (home / "inbox").mkdir()
    link = home / "inbox" / "shortcut"
    link.symlink_to(target, target_is_directory=True)
    assert refused(link, home) == "symlink_escape"


def test_missing_and_file_roots(home):
    assert refused(home / "nope", home) == "not_a_folder"
    f = home / "a.txt"
    f.write_text("x")
    assert refused(f, home) == "not_a_folder"


MOUNTS = (
    "sysfs /sys sysfs rw 0 0\n"
    "/dev/sda1 / ext4 rw 0 0\n"
    "srv:/export {nfs} nfs4 rw 0 0\n"
    "//nas/share {smb} cifs rw 0 0\n"
    "me@host:/ {ssh} fuse.sshfs rw 0 0\n"
    "//nas/space\\040dir {sp} smb3 rw 0 0\n"
)


@pytest.mark.parametrize("key", ["nfs", "smb", "ssh", "sp"])
def test_network_mounts(home, tmp_path, key):
    mount = tmp_path / ("mnt " + key if key == "sp" else "mnt_" + key)
    inner = mount / "notes"
    inner.mkdir(parents=True)
    text = MOUNTS.format(nfs=tmp_path / "mnt_nfs", smb=tmp_path / "mnt_smb",
                         ssh=tmp_path / "mnt_ssh", sp=str(mount).replace(" ", "\\040"))
    assert refused(inner, home, mounts=text) == "network_share"


def test_local_mount_is_fine(home):
    d = home / "notes"
    d.mkdir()
    text = f"/dev/sda1 / ext4 rw 0 0\n/dev/sdb1 {home} ext4 rw 0 0\n"
    assert check_root(d, home=home, environ={}, mounts=text) == d.resolve()


def test_is_inside(tmp_path):
    root = tmp_path / "r"
    (root / "a").mkdir(parents=True)
    out = tmp_path / "out"
    out.mkdir()
    (root / "a" / "esc").symlink_to(out, target_is_directory=True)
    assert is_inside(root, root / "a")
    assert not is_inside(root, root / "a" / "esc")
    assert not is_inside(root, tmp_path)
    assert is_inside(root, root)
