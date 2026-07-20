"""Auxiliary graph, learning, math, MCP, and route state stays root-local."""

from pathlib import Path
import re

import pytest


_RUNTIME_FILES = (
    "code_graph/config.py",
    "core/consolidation_engine.py",
    "core/graph_analyzer.py",
    "evolution/skill_evolver.py",
    "infra/event_bus.py",
    "learning/trigram_index.py",
    "math/polar_quant.py",
    "math/turbo_quant.py",
    "mcp/tools_v28.py",
    "server/api.py",
    "server/routes/agents.py",
    "server/routes/brain.py",
    "server/routes/v3_api.py",
)
_DIRECT_HOME_STATE = re.compile(
    r"(?:Path|_Path|_P)\.home\(\)\s*/\s*[\"']\.superlocalmemory[\"']",
)


def test_auxiliary_runtime_files_have_no_direct_home_state_constructor() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory"
    violations = []
    for relative in _RUNTIME_FILES:
        text = (source_root / relative).read_text(encoding="utf-8")
        if _DIRECT_HOME_STATE.search(text):
            violations.append(relative)
    assert violations == []


def test_auxiliary_default_paths_follow_selected_root_after_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from superlocalmemory.code_graph.config import CodeGraphConfig
    from superlocalmemory.learning.trigram_index import TrigramIndex
    from superlocalmemory.server import api
    from superlocalmemory.server.routes import agents, brain

    selected = (tmp_path / "selected").resolve()
    monkeypatch.setenv("SLM_DATA_DIR", str(selected))

    assert CodeGraphConfig().get_db_path() == selected / "code_graph.db"
    assert Path(TrigramIndex.CACHE_DB_PATH) == selected / "active_brain_cache.db"
    assert Path(api.MEMORY_DIR) == selected
    assert Path(api.DB_PATH) == selected / "memory.db"
    assert agents._registry_path() == selected / "agents.json"
    assert brain._memory_dir() == selected
