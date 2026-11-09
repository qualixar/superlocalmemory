# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""4.1.21 #16: ``search`` and ``list_recent`` already read the CONFIGURED
``memory_kinds.display_min_confidence`` to apply a ``kind=`` filter — this
pins that the same configured value also LABELS each returned item
(``memory_kind_state``), not a 0.20 hard-coded at the display call
regardless of what the filter call was given. Mirrors
``tests/test_cli/test_list_kind_filter.py``'s CLI-side regression.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from superlocalmemory.mcp import tools_core
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

_PROFILE = "default"


class _CapturingServer:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *args, **kwargs):
        def decorate(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorate


@dataclass
class _MemoryKindsCfg:
    display_min_confidence: float = 0.50


@dataclass
class _Cfg:
    memory_kinds: _MemoryKindsCfg


class _Engine:
    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self.profile_id = _PROFILE
        self._config = _Cfg(memory_kinds=_MemoryKindsCfg())


def _save(db, content, *, kind, source, confidence) -> str:
    memory_id = db.store_memory(MemoryRecord(profile_id=_PROFILE, content=content))
    fact = AtomicFact(profile_id=_PROFILE, memory_id=memory_id, content=content,
                      fact_type=FactType.SEMANTIC, memory_kind=kind,
                      memory_kind_source=source, memory_kind_confidence=confidence)
    return db.store_fact(fact)


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    engine = _Engine(db)
    # 0.30 clears the module default (0.20) but not the configured 0.50.
    _save(db, "Ship on Fridays only with sign-off.",
         kind="rule", source="model:llm", confidence=0.30)

    async def _profile(*args, **kwargs) -> str:
        return _PROFILE
    monkeypatch.setattr(tools_core, "_runtime_profile", _profile)

    server = _CapturingServer()
    tools_core.register_core_tools(server, lambda: engine)
    return server.tools


def test_list_recent_labels_at_the_configured_threshold(tools) -> None:
    out = asyncio.run(tools["list_recent"]())
    assert out["success"] is True and out["count"] == 1
    item = out["results"][0]
    assert item["memory_kind_state"] == "legacy", (
        "0.30 confidence must be demoted at the configured 0.50 threshold, "
        "not shown as a suggestion because this tool kept its own 0.20")
    assert item["memory_kind"] == "semantic"


def test_search_labels_at_the_configured_threshold(tools) -> None:
    out = asyncio.run(tools["search"]("Fridays"))
    assert out["success"] is True and out["count"] == 1
    item = out["results"][0]
    assert item["memory_kind_state"] == "legacy"
    assert item["memory_kind"] == "semantic"
