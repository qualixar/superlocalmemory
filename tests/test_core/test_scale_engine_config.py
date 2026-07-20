"""Scale Engine configuration must persist without silently activating it."""

from __future__ import annotations

from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


def test_scale_engine_selection_round_trips_without_promotion(tmp_path) -> None:
    config = SLMConfig.for_mode(Mode.A, base_dir=tmp_path)
    config.graph_backend = "cozo"
    config.vector_backend = "lancedb"
    config.scale_engine_state = "verified"
    config.save()

    loaded = SLMConfig.load(tmp_path / "config.json")
    assert loaded.graph_backend == "cozo"
    assert loaded.vector_backend == "lancedb"
    assert loaded.scale_engine_state == "verified"


def test_invalid_scale_engine_state_fails_closed_to_local_core(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"mode":"a", "scale_engine_state":"active_everywhere"}')

    assert SLMConfig.load(path).scale_engine_state == "local_core"
