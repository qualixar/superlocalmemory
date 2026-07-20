# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CodeGraph configuration."""

from pathlib import Path

from superlocalmemory.code_graph.config import (
    CodeGraphConfig,
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXTENSION_MAP,
    DEFAULT_LANGUAGES,
)


class TestCodeGraphConfig:
    def test_defaults(self):
        cfg = CodeGraphConfig()
        assert cfg.enabled is False
        assert cfg.parallel_workers == 4
        assert cfg.batch_size == 450
        assert cfg.max_depth_blast_radius == 2
        assert cfg.max_nodes_blast_radius == 500
        assert cfg.rrf_k == 60
        assert cfg.heuristic_confidence == 0.6
        assert cfg.bridge_enabled is False
        assert cfg.watch_debounce_ms == 300

    def test_frozen(self):
        cfg = CodeGraphConfig()
        try:
            cfg.enabled = True  # type: ignore
            assert False, "Should have raised"
        except AttributeError:
            pass

    def test_languages_default(self):
        cfg = CodeGraphConfig()
        assert "python" in cfg.languages
        assert "typescript" in cfg.languages
        assert "tsx" in cfg.languages

    def test_extension_map_default(self):
        cfg = CodeGraphConfig()
        assert cfg.extension_map[".py"] == "python"
        assert cfg.extension_map[".ts"] == "typescript"
        assert cfg.extension_map[".tsx"] == "tsx"

    def test_exclude_dirs_default(self):
        cfg = CodeGraphConfig()
        assert "node_modules" in cfg.exclude_dirs
        assert ".git" in cfg.exclude_dirs
        assert "__pycache__" in cfg.exclude_dirs

    def test_db_path_explicit(self, tmp_path: Path):
        db = tmp_path / "my.db"
        cfg = CodeGraphConfig(db_path=db)
        assert cfg.get_db_path() == db

    def test_db_path_from_slm_base(self, tmp_path: Path):
        cfg = CodeGraphConfig()
        result = cfg.get_db_path(slm_base_dir=tmp_path)
        assert result == tmp_path / "code_graph.db"

    def test_db_path_fallback(self, monkeypatch, tmp_path):
        selected = tmp_path / "selected"
        monkeypatch.setenv("SLM_DATA_DIR", str(selected))
        cfg = CodeGraphConfig()
        result = cfg.get_db_path()
        assert result == selected.resolve() / "code_graph.db"

    def test_custom_config(self, tmp_path: Path):
        cfg = CodeGraphConfig(
            enabled=True,
            repo_root=tmp_path,
            parallel_workers=8,
            heuristic_confidence=0.8,
        )
        assert cfg.enabled is True
        assert cfg.parallel_workers == 8
        assert cfg.heuristic_confidence == 0.8

    def test_max_file_size(self):
        cfg = CodeGraphConfig()
        assert cfg.max_file_size_bytes == 1_000_000

    def test_parse_timeout(self):
        cfg = CodeGraphConfig()
        assert cfg.parse_timeout_seconds == 30.0
