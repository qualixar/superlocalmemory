"""The data-folder refusal compares folders by identity, not by spelling."""

import os
from pathlib import Path

from superlocalmemory.infra import data_root as dr


def test_an_alias_spelled_differently_is_still_the_data_folder(tmp_path, monkeypatch):
    data = tmp_path / "slm-data"
    (data / "media").mkdir(parents=True)
    alias = tmp_path / "SLM-DATA-ALIAS"
    # A hard-to-spot second name for the same folder (bind mounts and
    # case-insensitive disks behave alike): realpath keeps the alias spelling.
    os.symlink(data, alias)
    monkeypatch.setattr(dr, "canonical_data_root", lambda: data)
    real_realpath = os.path.realpath
    monkeypatch.setattr(dr.os.path, "realpath",
                        lambda p: p if str(p).startswith(str(alias)) else real_realpath(p))
    assert dr.overlaps_data_root(alias / "media")
    assert dr.overlaps_data_root(alias)


def test_an_unrelated_folder_is_not_the_data_folder(tmp_path, monkeypatch):
    data = tmp_path / "slm-data"
    data.mkdir()
    other = tmp_path / "photos"
    other.mkdir()
    monkeypatch.setattr(dr, "canonical_data_root", lambda: data)
    assert not dr.overlaps_data_root(other)
    assert not dr.overlaps_data_root(Path(str(other) + "-missing"))
