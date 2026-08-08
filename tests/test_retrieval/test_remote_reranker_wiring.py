# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Issue #105 — config plumbing and wiring for the remote reranker.

The wiring tests patch ``CrossEncoderReranker`` ONLY to stop the local worker
subprocess from spawning; the routing decision under test is never mocked.
"""

from __future__ import annotations

import json
import logging
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from superlocalmemory.core.config import RetrievalConfig, SLMConfig
from superlocalmemory.core.engine_wiring import init_reranker
from superlocalmemory.retrieval.remote_reranker import RemoteReranker
from superlocalmemory.storage.models import Mode

ENDPOINT = "https://reranker.example.test/v1/rerank"
MODEL = "/root/model/reranker.gguf"


# ---------------------------------------------------------------------------
# RetrievalConfig now READS the key it used to swallow
# ---------------------------------------------------------------------------

class TestRetrievalConfigKeys:

    def test_defaults_are_local(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.cross_encoder_endpoint == ""
        assert cfg.cross_encoder_backend == ""
        assert cfg.is_remote_cross_encoder is False

    def test_remote_pair_is_detected(self) -> None:
        cfg = RetrievalConfig(
            cross_encoder_backend="openai", cross_encoder_endpoint=ENDPOINT,
        )
        assert cfg.is_remote_cross_encoder is True

    def test_endpoint_alone_is_not_a_remote_config(self) -> None:
        cfg = RetrievalConfig(cross_encoder_endpoint=ENDPOINT)
        assert cfg.is_remote_cross_encoder is False

    def test_backend_alone_is_not_a_remote_config(self) -> None:
        cfg = RetrievalConfig(cross_encoder_backend="openai")
        assert cfg.is_remote_cross_encoder is False

    def test_config_json_endpoint_is_no_longer_discarded(
        self, tmp_path: Path,
    ) -> None:
        """The #103 leftover, pinned.

        ``cross_encoder_endpoint`` was not a ``RetrievalConfig`` field, so
        ``SLMConfig.load`` filtered it out of the user's own config.json
        without a word. The reporter's exact block must now survive the load.
        """
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "mode": "b",
            "retrieval": {
                "use_cross_encoder": True,
                "cross_encoder_endpoint": ENDPOINT,
                "cross_encoder_model": MODEL,
                "cross_encoder_backend": "openai",
            },
        }))

        loaded = SLMConfig.load(path)

        assert loaded.retrieval.cross_encoder_endpoint == ENDPOINT
        assert loaded.retrieval.cross_encoder_model == MODEL
        assert loaded.retrieval.cross_encoder_backend == "openai"
        assert loaded.retrieval.is_remote_cross_encoder is True

    def test_round_trip_preserves_the_remote_reranker_block(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "config.json"
        cfg = SLMConfig.for_mode(Mode.B)
        cfg.retrieval.cross_encoder_backend = "openai"
        cfg.retrieval.cross_encoder_endpoint = ENDPOINT
        cfg.retrieval.cross_encoder_model = MODEL
        cfg.retrieval.cross_encoder_timeout_seconds = 8.0
        cfg.save(path)

        on_disk = json.loads(path.read_text())["retrieval"]
        assert on_disk["cross_encoder_endpoint"] == ENDPOINT

        loaded = SLMConfig.load(path)
        assert loaded.retrieval.cross_encoder_endpoint == ENDPOINT
        assert loaded.retrieval.cross_encoder_timeout_seconds == 8.0
        assert loaded.retrieval.is_remote_cross_encoder is True

    def test_saved_reranker_secret_is_owner_only_and_round_trips(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "config.json"
        cfg = SLMConfig.for_mode(Mode.B)
        cfg.retrieval.cross_encoder_api_key = "reranker-secret"

        cfg.save(path)

        on_disk = json.loads(path.read_text())
        assert on_disk["retrieval"]["cross_encoder_api_key"] == "reranker-secret"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert SLMConfig.load(path).retrieval.cross_encoder_api_key == (
            "reranker-secret"
        )

    def test_legacy_config_without_the_key_still_loads_local(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({
            "mode": "a",
            "retrieval": {"use_cross_encoder": True, "cross_encoder_backend": "onnx"},
        }))
        loaded = SLMConfig.load(path)
        assert loaded.retrieval.cross_encoder_endpoint == ""
        assert loaded.retrieval.is_remote_cross_encoder is False


# ---------------------------------------------------------------------------
# init_reranker routing
# ---------------------------------------------------------------------------

class TestInitRerankerRouting:

    def test_remote_config_builds_a_remote_reranker_and_no_local_worker(
        self,
    ) -> None:
        cfg = RetrievalConfig(
            cross_encoder_backend="openai",
            cross_encoder_endpoint=ENDPOINT,
            cross_encoder_model=MODEL,
        )
        with patch(
            "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
        ) as local:
            reranker = init_reranker(cfg)

        assert isinstance(reranker, RemoteReranker)
        assert reranker.safe_endpoint == ENDPOINT
        local.assert_not_called()

    def test_remote_reranker_receives_model_and_timeout(self) -> None:
        cfg = RetrievalConfig(
            cross_encoder_backend="remote",
            cross_encoder_endpoint="https://reranker.example.test/v1",
            cross_encoder_model=MODEL,
            cross_encoder_timeout_seconds=4.0,
        )
        reranker = init_reranker(cfg)
        assert isinstance(reranker, RemoteReranker)
        assert reranker._model_name == MODEL
        assert reranker._read_timeout == 4.0
        assert reranker.safe_endpoint == "https://reranker.example.test/v1/rerank"

    def test_default_config_builds_the_local_reranker(self) -> None:
        with patch(
            "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
        ) as local:
            reranker = init_reranker(RetrievalConfig())
        local.assert_called_once()
        assert reranker is local.return_value

    def test_remote_backend_without_endpoint_disables_reranking_loudly(
        self, caplog,
    ) -> None:
        """Falling back to the local English model here would be the bug."""
        cfg = RetrievalConfig(cross_encoder_backend="openai")
        with caplog.at_level(logging.ERROR):
            with patch(
                "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
            ) as local:
                reranker = init_reranker(cfg)

        assert reranker is None
        local.assert_not_called()
        assert "cross_encoder_endpoint" in caplog.text
        assert "DISABLED" in caplog.text

    def test_invalid_scheme_disables_reranking_loudly(self, caplog) -> None:
        cfg = RetrievalConfig(
            cross_encoder_backend="openai",
            cross_encoder_endpoint="file:///etc/passwd",
        )
        with caplog.at_level(logging.ERROR):
            with patch(
                "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
            ) as local:
                reranker = init_reranker(cfg)

        assert reranker is None
        local.assert_not_called()
        assert "http" in caplog.text

    def test_endpoint_against_local_backend_errors_then_runs_local(
        self, caplog,
    ) -> None:
        """The #103 leftover at the wiring layer: no longer silent."""
        cfg = RetrievalConfig(cross_encoder_endpoint=ENDPOINT)
        with caplog.at_level(logging.ERROR):
            with patch(
                "superlocalmemory.retrieval.reranker.CrossEncoderReranker",
            ) as local:
                reranker = init_reranker(cfg)

        assert reranker is local.return_value
        assert "cross_encoder_endpoint" in caplog.text
        assert "cross_encoder_backend" in caplog.text

    def test_local_config_logs_nothing_alarming(self, caplog) -> None:
        with caplog.at_level(logging.ERROR):
            with patch("superlocalmemory.retrieval.reranker.CrossEncoderReranker"):
                init_reranker(RetrievalConfig(cross_encoder_backend="onnx"))
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# Warmup logging must not describe a local worker for a remote reranker
# ---------------------------------------------------------------------------

class TestWarmupStatusLogging:

    def test_remote_reranker_warmup_uses_its_own_probe(self, caplog) -> None:
        from superlocalmemory.core.engine_wiring import (
            _log_reranker_warmup_status,
        )

        rr = RemoteReranker(MODEL, ENDPOINT)
        with patch.object(
            RemoteReranker, "warmup_sync", autospec=True, return_value=True,
        ) as probe:
            with caplog.at_level(logging.INFO):
                _log_reranker_warmup_status(rr)

        # No 180s local-worker budget handed to a short HTTP probe, and no
        # local-worker vocabulary in the log.
        probe.assert_called_once_with(rr)
        assert "warm and ready" not in caplog.text
        assert "another process" not in caplog.text

    def test_mocked_reranker_is_not_mistaken_for_a_remote_one(self) -> None:
        """MagicMock fabricates any attribute — the branch must use the type.

        A ``getattr(reranker, "is_remote", False)`` probe sent every mocked
        reranker in the suite down the remote branch and silenced the
        local-worker diagnostics that v3.4.42 exists to produce.
        """
        from unittest.mock import MagicMock

        from superlocalmemory.core.engine_wiring import (
            _log_reranker_warmup_status,
        )

        fake = MagicMock()
        fake.warmup_sync.return_value = True
        _log_reranker_warmup_status(fake)
        fake.warmup_sync.assert_called_once_with(timeout=180)


# ---------------------------------------------------------------------------
# Diagnostics: doctor / wizard must not demand a local model that is unused
# ---------------------------------------------------------------------------

def _cfg(backend: str, endpoint: str) -> SimpleNamespace:
    return SimpleNamespace(retrieval=SimpleNamespace(
        use_cross_encoder=True,
        cross_encoder_backend=backend,
        cross_encoder_endpoint=endpoint,
        cross_encoder_model=MODEL,
    ))


class TestRemoteRerankerDetection:

    def test_wizard_detects_a_remote_reranker(self) -> None:
        from superlocalmemory.cli.setup_wizard import _reranker_is_remote

        assert _reranker_is_remote(_cfg("openai", ENDPOINT)) is True
        assert _reranker_is_remote(_cfg("remote", ENDPOINT)) is True

    def test_stray_endpoint_alone_is_not_treated_as_remote(self) -> None:
        """A misconfiguration must not silently skip the local download."""
        from superlocalmemory.cli.setup_wizard import _reranker_is_remote

        assert _reranker_is_remote(_cfg("", ENDPOINT)) is False
        assert _reranker_is_remote(_cfg("onnx", ENDPOINT)) is False

    def test_local_config_is_not_remote(self) -> None:
        from superlocalmemory.cli.setup_wizard import _reranker_is_remote

        assert _reranker_is_remote(_cfg("", "")) is False
        assert _reranker_is_remote(SimpleNamespace()) is False

    def test_doctor_reports_remote_endpoint_instead_of_a_missing_model(
        self,
    ) -> None:
        from superlocalmemory.core.component_registry import (
            probe_reranker_model,
        )

        comp = probe_reranker_model(_cfg("openai", ENDPOINT))
        assert comp.status == "ok"
        assert "remote" in comp.detail
        assert comp.auto_fixable is False

    def test_doctor_still_probes_the_local_model_when_local(self) -> None:
        from superlocalmemory.core.component_registry import (
            probe_reranker_model,
        )

        comp = probe_reranker_model(_cfg("", ""))
        assert "remote" not in comp.detail


class TestWizardPreservesRemoteReranker:

    def test_rerunning_setup_does_not_erase_the_remote_endpoint(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """``retrieval`` is a mode-owned block the wizard resets wholesale.

        Without the carve-out, ``slm setup`` silently reverted a working
        multilingual endpoint to the bundled English cross-encoder.
        """
        monkeypatch.setenv("SLM_BASE_DIR", str(tmp_path))
        from superlocalmemory.cli.setup_wizard import _build_wizard_config

        existing = SLMConfig.for_mode(Mode.C)
        existing.retrieval.cross_encoder_backend = "openai"
        existing.retrieval.cross_encoder_endpoint = ENDPOINT
        existing.retrieval.cross_encoder_model = MODEL
        existing.retrieval.cross_encoder_api_key = "token"
        existing.save(mode_change=True)

        rebuilt = _build_wizard_config(Mode.A)

        assert rebuilt.mode is Mode.A
        assert rebuilt.retrieval.cross_encoder_endpoint == ENDPOINT
        assert rebuilt.retrieval.cross_encoder_backend == "openai"
        assert rebuilt.retrieval.cross_encoder_model == MODEL
        assert rebuilt.retrieval.cross_encoder_api_key == "token"
