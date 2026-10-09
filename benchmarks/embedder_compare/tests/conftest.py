"""Shared fixtures: put the harness on sys.path and build a tiny dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

FIXTURE = HARNESS / "tests" / "fixtures" / "fake_locomo.json"
TINY_QUOTAS = {
    "text_single_hop": 3, "text_multi_hop": 2, "entity": 2,
    "temporal": 2, "unanswerable": 2,
}


@pytest.fixture(scope="session")
def tiny_home(tmp_path_factory) -> Path:
    """A built text-only dataset from the invented fixture."""
    import build_dataset as bd

    home = tmp_path_factory.mktemp("bench_home")
    data = json.loads(FIXTURE.read_text())
    selection = bd.select_locomo(data, TINY_QUOTAS, n_convs=2)
    sel_path = home / "selection.json"
    sel_path.write_text(json.dumps(selection))
    bd.build(FIXTURE, sel_path, home / "dataset", media=False)
    return home
