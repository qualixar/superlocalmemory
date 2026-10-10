"""``sources.enabled`` is its own switch and leaves the images switch alone."""

from __future__ import annotations

import json

from superlocalmemory.runtimes import features


def test_default_is_off_and_reading_creates_nothing(tmp_path):
    assert features.sources_enabled(tmp_path) is False
    assert not (tmp_path / "features.json").exists()


def test_enable_creates_media_db_and_keeps_the_media_switch(tmp_path):
    assert features.enable_sources(source="api", data_root=tmp_path) is True
    assert features.sources_enabled(tmp_path) and not features.media_enabled(tmp_path)
    assert (tmp_path / "media.db").exists()
    saved = json.loads((tmp_path / "features.json").read_text())
    assert saved["sources"]["enabled"] is True and saved["media"]["enabled"] is False


def test_turning_images_on_does_not_drop_the_sources_choice(tmp_path):
    features.enable_sources(source="api", data_root=tmp_path)
    data = features.read_features(tmp_path)
    data["media"]["enabled"] = True
    features._write_features(tmp_path, data)
    assert features.sources_enabled(tmp_path) and features.media_enabled(tmp_path)


def test_disable_never_creates_the_file(tmp_path):
    features.disable_sources(data_root=tmp_path)
    assert not (tmp_path / "features.json").exists()
    features.enable_sources(source="api", data_root=tmp_path)
    features.disable_sources(data_root=tmp_path)
    assert features.sources_enabled(tmp_path) is False


def test_bad_source_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        features.enable_sources(source="nope", data_root=tmp_path)
