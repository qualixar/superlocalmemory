"""Public benchmark artifacts must retain metrics without leaking host paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_bench_perf():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "slm_bench_perf", root / "benchmark" / "bench_perf.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_redacts_database_path_but_keeps_size(tmp_path: Path) -> None:
    bench_perf = _load_bench_perf()
    database = tmp_path / "private-machine-layout" / "memory.db"
    database.parent.mkdir()
    database.write_bytes(b"SQLite fixture bytes")
    config = SimpleNamespace(
        mode=SimpleNamespace(value="b"), active_profile="default"
    )

    environment = bench_perf._environment(database, {}, config)

    assert environment["db_path"] == "<redacted: benchmark database path>"
    assert environment["db_size_bytes"] == database.stat().st_size
    assert str(tmp_path) not in repr(environment)
