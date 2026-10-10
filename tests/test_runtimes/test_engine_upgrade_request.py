# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The recorded "upgrade the memory engine" request in features.json."""

from __future__ import annotations

import json

from superlocalmemory.runtimes import features as feat


def _write(root, data):
    (root / "features.json").write_text(json.dumps(data), encoding="utf-8")


def test_a_fresh_store_has_no_request_and_no_key_in_the_defaults(tmp_path):
    assert feat.engine_upgrade_requested(tmp_path) is False
    assert "engine_upgrade" not in feat.read_features(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_recorded_request_is_read_back(tmp_path):
    _write(tmp_path, {"schema": 1, "engine_upgrade": {"requested": True, "choice_source": "npm"}})
    assert feat.engine_upgrade_requested(tmp_path) is True
    assert feat.read_features(tmp_path)["engine_upgrade"]["choice_source"] == "npm"


def test_turning_images_on_keeps_a_recorded_upgrade_request(tmp_path):
    _write(tmp_path, {"schema": 1, "media": {"requested": True},
                      "engine_upgrade": {"requested": True, "requested_at": "t"}})
    data = feat.read_features(tmp_path)
    feat._write_features(tmp_path, data)
    assert json.loads((tmp_path / "features.json").read_text())["engine_upgrade"]["requested"] is True


def test_clearing_removes_only_the_request(tmp_path):
    _write(tmp_path, {"schema": 1, "media": {"enabled": True, "choice_source": "cli"},
                      "engine_upgrade": {"requested": True}})
    feat.clear_engine_upgrade_request(tmp_path)
    saved = json.loads((tmp_path / "features.json").read_text())
    assert "engine_upgrade" not in saved and saved["media"]["enabled"] is True


def test_clearing_with_no_file_creates_nothing(tmp_path):
    feat.clear_engine_upgrade_request(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_unreadable_value_is_not_a_request(tmp_path):
    _write(tmp_path, {"schema": 1, "engine_upgrade": "yes please"})
    assert feat.engine_upgrade_requested(tmp_path) is False
