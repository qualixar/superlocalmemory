"""`slm tool` quiets SLM's loggers for its own process; a test must not inherit that.

CI (PR #343, shard 1) failed a migration test that reads an INFO line through
``caplog``: an earlier test in the same worker had run ``slm tool`` in-process,
which set the ``superlocalmemory`` logger to WARNING, and the level stayed for
every later test. The two tests below run in file order in one worker: the
first quiets the loggers, the second proves an INFO record still reaches
``caplog``.
"""

from __future__ import annotations

import logging

from superlocalmemory.cli import tool_cmd


def test_slm_tool_quiets_the_superlocalmemory_logger() -> None:
    tool_cmd._quiet_logs()
    assert logging.getLogger("superlocalmemory").level == logging.WARNING


def test_the_next_test_still_sees_info_records(caplog) -> None:
    with caplog.at_level(logging.INFO):
        logging.getLogger("superlocalmemory.storage.migrations.probe").info(
            "probe line",
        )
    assert "probe line" in caplog.text
