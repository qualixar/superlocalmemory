# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W3-6f (D-45): co-retrieval learns from every agent recall; workflow mining
learns from the profile's own memories, in shadow."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from superlocalmemory.learning.signals import LearningSignals
from superlocalmemory.outcomes import receipts


def _edges(path, profile="default") -> int:
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT COUNT(*) FROM co_retrieval_edges WHERE profile_id = ?",
                            (profile,)).fetchone()[0]
    finally:
        conn.close()


def test_an_agents_answer_links_its_memories_together(tmp_path) -> None:
    LearningSignals(tmp_path / "learning.db")
    pull = receipts.Pull(tmp_path, tmp_path / "memory.db", "work", None, "mcp_recall", "q",
                         tuple((f"f{i}", 0.5) for i in range(12)))
    assert receipts.learn_together(pull) > 0
    assert _edges(tmp_path / "learning.db", "work") > 0
    conn = sqlite3.connect(str(tmp_path / "learning.db"))
    linked = {f for row in conn.execute("SELECT fact_id_a, fact_id_b FROM co_retrieval_edges")
              for f in row}
    conn.close()
    assert linked <= {f"f{i}" for i in range(receipts.CO_RETRIEVAL_TOP)}   # first ten only


def test_one_memory_or_no_learning_store_links_nothing(tmp_path) -> None:
    one = receipts.Pull(tmp_path, tmp_path / "m.db", "d", None, "mcp_recall", "q", (("a", 1.0),))
    assert receipts.learn_together(one) == 0
    two = receipts.Pull(tmp_path, tmp_path / "m.db", "d", None, "mcp_recall", "q",
                        (("a", 1.0), ("b", 0.5)))
    assert receipts.learn_together(two) == 0 and not (tmp_path / "learning.db").exists()


def test_the_writer_links_only_after_storing_the_receipt(tmp_path, monkeypatch) -> None:
    linked = []
    monkeypatch.setattr(receipts, "learn_together", linked.append)
    monkeypatch.setattr(receipts, "record_pull", lambda p: p.query == "stored")
    for q in ("stored", "refused"):
        receipts.submit_pull(receipts.Pull(tmp_path, tmp_path / "m.db", "d", None,
                                           "mcp_recall", q, (("a", 1.0), ("b", 0.5))))
    assert receipts.PULL_WRITER.drain()
    assert [p.query for p in linked] == ["stored"]


from tests.test_modes.test_mode_a_never_fails import mode_a_engine  # noqa: E402,F401


def test_a_real_hook_recall_teaches_co_retrieval(mode_a_engine) -> None:  # noqa: F811
    from superlocalmemory.server.unified_daemon import EngineRecallAdapter

    home = mode_a_engine._config.base_dir
    LearningSignals(home / "learning.db")
    for text in ("The invoice run happens on the last working day of the month.",
                 "Invoices are approved by the finance lead before the run.",
                 "The invoice archive lives in the shared finance folder."):
        mode_a_engine.store(text)
    out = EngineRecallAdapter(mode_a_engine).recall("invoice run", 5, "conv-c")
    assert len(out["results"]) >= 2 and receipts.PULL_WRITER.drain()
    assert _edges(home / "learning.db", mode_a_engine.profile_id) > 0


# -- workflow mining, in shadow ---------------------------------------------------------------------

def _memory_store(path, rows, profile="default") -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE atomic_facts (profile_id TEXT, content TEXT, created_at TEXT)")
    conn.executemany("INSERT INTO atomic_facts VALUES (?, ?, ?)",
                     [(profile, c, t) for c, t in rows])
    conn.commit()
    conn.close()


def test_the_workflow_step_learns_from_the_profiles_memories(tmp_path) -> None:
    from superlocalmemory.learning.workflows import WorkflowMiner
    from superlocalmemory.server.consolidation_runner import workflow_step

    WorkflowMiner(tmp_path / "learning.db")
    texts = ["fix the bug in the parser", "deploy the release to staging",
             "write the readme documentation", "fix the error in the cache",
             "deploy the build pipeline", "update the readme"] * 4
    rows = [(t, f"2026-11-{1 + i // 6:02d}T{9 + i % 6:02d}:00:00+00:00")
            for i, t in enumerate(texts)]
    _memory_store(tmp_path / "memory.db", rows)
    other = sqlite3.connect(str(tmp_path / "memory.db"))   # another workspace's habits
    other.executemany("INSERT INTO atomic_facts VALUES ('work', ?, ?)",
                      [(t, f"2026-11-01T{8 + i % 12:02d}:30:00+00:00") for i, t in
                       enumerate(["run the pytest coverage", "edit the config settings"] * 15)])
    other.commit()
    other.close()
    config = SimpleNamespace(base_dir=tmp_path, db_path=tmp_path / "memory.db")
    got = workflow_step("default", config=config)
    assert got["sequences"] > 0 and got["time_of_day"] >= 1
    conn = sqlite3.connect(str(tmp_path / "learning.db"))
    keys = {r[0] for r in conn.execute("SELECT pattern_key FROM workflow_patterns"
                                       " WHERE pattern_type = 'sequence'")}
    assert "debug -> deploy" in keys
    assert not any("test" in k or "config" in k for k in keys)    # never the other workspace
    assert workflow_step("default", config=config) == got          # replaced, not doubled
    assert conn.execute("SELECT COUNT(*) FROM workflow_patterns").fetchone()[0] == \
        got["sequences"] + got["time_of_day"]
    conn.close()


def test_the_workflow_step_never_creates_a_store_or_raises(tmp_path) -> None:
    from superlocalmemory.server.consolidation_runner import workflow_step

    config = SimpleNamespace(base_dir=tmp_path, db_path=tmp_path / "memory.db")
    assert workflow_step("default", config=config) == {"skipped": "no_store"}
    assert not (tmp_path / "learning.db").exists()


def test_shadow_workflows_never_reach_soft_prompts(tmp_path) -> None:
    """The extractor still reads the action table nothing fills; the mined
    patterns live only in workflow_patterns until evidence says they help."""
    import inspect

    from superlocalmemory.parameterization import pattern_extractor

    source = inspect.getsource(pattern_extractor.PatternExtractor._extract_from_workflows)
    assert "workflow_patterns" not in source and "learn_from_memories" not in source


def test_a_full_consolidation_runs_the_workflow_step(monkeypatch) -> None:
    import asyncio

    from superlocalmemory.server import consolidation_runner as cr

    monkeypatch.setattr(cr, "_consolidate_blocking", lambda *a: {})
    monkeypatch.setattr(cr, "ranker_step", lambda pid: {})
    ran = []
    monkeypatch.setattr(cr, "workflow_step", lambda pid: ran.append(pid) or {"sequences": 1})
    assert asyncio.run(cr.run_full_consolidation(None, "default"))["workflows"] == {
        "sequences": 1}
    assert "workflows" not in asyncio.run(cr.run_full_consolidation(None, "d", lightweight=True))
    assert ran == ["default"]
