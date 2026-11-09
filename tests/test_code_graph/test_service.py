# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for CodeGraphService."""

from pathlib import Path

import pytest

from superlocalmemory.code_graph.config import CodeGraphConfig
from superlocalmemory.code_graph.service import (
    CodeGraphNotEnabledError,
    CodeGraphService,
)


class TestServiceInit:
    def test_lazy_db(self, config: CodeGraphConfig):
        svc = CodeGraphService(config)
        # DB not created until first access
        assert svc._db is None

    def test_db_created_on_access(self, config: CodeGraphConfig):
        svc = CodeGraphService(config)
        _ = svc.db
        assert svc._db is not None
        assert config.get_db_path().exists()

    def test_config_exposed(self, config: CodeGraphConfig):
        svc = CodeGraphService(config)
        assert svc.config is config
        assert svc.config.enabled is True


class TestEnsureEnabled:
    def test_raises_when_disabled(self, tmp_path: Path):
        cfg = CodeGraphConfig(enabled=False, db_path=tmp_path / "test.db")
        svc = CodeGraphService(cfg)
        with pytest.raises(CodeGraphNotEnabledError):
            svc.ensure_enabled()

    def test_passes_when_enabled(self, config: CodeGraphConfig):
        svc = CodeGraphService(config)
        svc.ensure_enabled()  # Should not raise


class TestGetStats:
    def test_stats_before_db(self, tmp_path: Path):
        cfg = CodeGraphConfig(
            enabled=True,
            db_path=tmp_path / "nonexistent" / "code_graph.db",
        )
        svc = CodeGraphService(cfg)
        stats = svc.get_stats()
        assert stats["nodes"] == 0
        assert stats["built"] is False

    def test_stats_empty_db(self, service: CodeGraphService):
        stats = service.get_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["files"] == 0
        assert stats["built"] is False

    def test_stats_with_data(self, service: CodeGraphService):
        import time

        from superlocalmemory.code_graph.models import GraphNode, NodeKind

        node = GraphNode(
            node_id="n1", kind=NodeKind.FUNCTION, name="test",
            qualified_name="test.py::test", file_path="test.py",
            language="python", created_at=time.time(), updated_at=time.time(),
        )
        service.db.upsert_node(node)
        stats = service.get_stats()
        assert stats["nodes"] == 1
        assert stats["built"] is True
