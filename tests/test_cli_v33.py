# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under the MIT License - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for V3.3 CLI commands.

Covers:
    - slm decay (Ebbinghaus forgetting)
    - slm quantize (EAP quantization)
    - slm consolidate --cognitive (CCQ pipeline)
    - slm soft-prompts (list active soft prompts)
    - slm reap (orphaned process cleanup)
    - Each command's --json output
    - Dispatch routing for all 5 new commands

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: mock objects
# ---------------------------------------------------------------------------


def _mock_engine(profile_id: str = "test-profile"):
    """Create mock engine with common attributes."""
    engine = MagicMock()
    engine.profile_id = profile_id
    engine._db.execute.return_value = []
    engine._db.db_path = MagicMock()
    return engine


def _mock_config(profile_id: str = "test-profile"):
    """Create a mock SLMConfig."""
    config = MagicMock()
    config.active_profile = profile_id
    config.forgetting = MagicMock()
    config.quantization = MagicMock()
    return config


@dataclass
class _MockCCQResult:
    clusters_found: int = 3
    clusters_processed: int = 3
    blocks_created: int = 2
    facts_archived: int = 15
    compression_ratio: float = 0.45


# ---------------------------------------------------------------------------
# Dispatch routing tests
# ---------------------------------------------------------------------------


class TestV33Dispatch:
    """Verify V3.3 commands are registered in dispatch."""

    def test_decay_in_dispatch(self):
        """'decay' maps to cmd_decay in dispatch."""
        from superlocalmemory.cli.commands import cmd_decay
        assert callable(cmd_decay)

    def test_quantize_in_dispatch(self):
        """'quantize' maps to cmd_quantize in dispatch."""
        from superlocalmemory.cli.commands import cmd_quantize
        assert callable(cmd_quantize)

    def test_consolidate_in_dispatch(self):
        """'consolidate' maps to cmd_consolidate in dispatch."""
        from superlocalmemory.cli.commands import cmd_consolidate
        assert callable(cmd_consolidate)

    def test_soft_prompts_in_dispatch(self):
        """'soft-prompts' maps to cmd_soft_prompts in dispatch."""
        from superlocalmemory.cli.commands import cmd_soft_prompts
        assert callable(cmd_soft_prompts)

    def test_reap_in_dispatch(self):
        """'reap' maps to cmd_reap in dispatch."""
        from superlocalmemory.cli.commands import cmd_reap
        assert callable(cmd_reap)

    def test_dispatch_routes_decay(self):
        """dispatch correctly routes 'decay' command."""
        from superlocalmemory.cli.commands import dispatch

        args = Namespace(command="decay", dry_run=True, profile="", json=False)
        with patch("superlocalmemory.cli.commands.cmd_decay") as mock_cmd:
            dispatch(args)
            mock_cmd.assert_called_once_with(args)

    def test_dispatch_routes_quantize(self):
        """dispatch correctly routes 'quantize' command."""
        from superlocalmemory.cli.commands import dispatch

        args = Namespace(command="quantize", dry_run=True, profile="", json=False)
        with patch("superlocalmemory.cli.commands.cmd_quantize") as mock_cmd:
            dispatch(args)
            mock_cmd.assert_called_once_with(args)

    def test_dispatch_routes_consolidate(self):
        """dispatch correctly routes 'consolidate' command."""
        from superlocalmemory.cli.commands import dispatch

        args = Namespace(command="consolidate", cognitive=True, profile="", json=False)
        with patch("superlocalmemory.cli.commands.cmd_consolidate") as mock_cmd:
            dispatch(args)
            mock_cmd.assert_called_once_with(args)

    def test_dispatch_routes_soft_prompts(self):
        """dispatch correctly routes 'soft-prompts' command."""
        from superlocalmemory.cli.commands import dispatch

        args = Namespace(command="soft-prompts", profile="", json=False)
        with patch("superlocalmemory.cli.commands.cmd_soft_prompts") as mock_cmd:
            dispatch(args)
            mock_cmd.assert_called_once_with(args)

    def test_dispatch_routes_reap(self):
        """dispatch correctly routes 'reap' command."""
        from superlocalmemory.cli.commands import dispatch

        args = Namespace(command="reap", force=False, json=False)
        with patch("superlocalmemory.cli.commands.cmd_reap") as mock_cmd:
            dispatch(args)
            mock_cmd.assert_called_once_with(args)


# ---------------------------------------------------------------------------
# cmd_decay tests
# ---------------------------------------------------------------------------


class TestCmdDecay:
    """Tests for the 'slm decay' CLI command."""

    def test_decay_prints_stats(self, capsys):
        """decay command prints zone distribution table."""
        config = _mock_config()
        engine = _mock_engine()

        mock_result = {
            "total": 100, "active": 50, "warm": 20,
            "cold": 15, "archive": 10, "forgotten": 5,
            "transitions": 8,
        }

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
        ) as MockSched, patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve"
        ):
            MockSched.return_value.run_decay_cycle.return_value = mock_result
            from superlocalmemory.cli.commands import cmd_decay
            cmd_decay(Namespace(dry_run=True, profile="", json=False))

        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "Active" in captured.out
        assert "Transitions" in captured.out

    def test_decay_json_output(self, capsys):
        """decay with --json produces valid JSON envelope."""
        config = _mock_config()
        engine = _mock_engine()

        mock_result = {"total": 50, "active": 30, "warm": 10,
                       "cold": 5, "archive": 3, "forgotten": 2,
                       "transitions": 4}

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
        ) as MockSched, patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve"
        ):
            MockSched.return_value.run_decay_cycle.return_value = mock_result
            from superlocalmemory.cli.commands import cmd_decay
            cmd_decay(Namespace(dry_run=True, profile="", json=True))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is True
        assert envelope["command"] == "decay"
        assert envelope["data"]["total"] == 50

    def test_decay_skipped_result(self, capsys):
        """decay handles skipped result (within interval)."""
        config = _mock_config()
        engine = _mock_engine()

        mock_result = {"skipped": True, "reason": "within_interval"}

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.learning.forgetting_scheduler.ForgettingScheduler"
        ) as MockSched, patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve"
        ):
            MockSched.return_value.run_decay_cycle.return_value = mock_result
            from superlocalmemory.cli.commands import cmd_decay
            cmd_decay(Namespace(dry_run=True, profile="", json=False))

        captured = capsys.readouterr()
        assert "Skipped" in captured.out


# ---------------------------------------------------------------------------
# cmd_quantize tests
# ---------------------------------------------------------------------------


class TestCmdQuantize:
    """Tests for the 'slm quantize' CLI command."""

    def test_quantize_prints_stats(self, capsys):
        """quantize command prints compression stats."""
        config = _mock_config()
        engine = _mock_engine()

        mock_result = {
            "total": 50, "downgrades": 10, "upgrades": 3,
            "skipped": 35, "deleted": 2, "errors": 0,
        }

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.dynamics.eap_scheduler.EAPScheduler"
        ) as MockEAP, patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve"
        ), patch(
            "superlocalmemory.storage.quantized_store.QuantizedEmbeddingStore"
        ), patch(
            "superlocalmemory.math.polar_quant.PolarQuantEncoder"
        ), patch(
            "superlocalmemory.math.qjl.QJLEncoder"
        ):
            MockEAP.return_value.run_eap_cycle.return_value = mock_result
            from superlocalmemory.cli.commands import cmd_quantize
            cmd_quantize(Namespace(dry_run=True, profile="", json=False))

        captured = capsys.readouterr()
        assert "50" in captured.out
        assert "Downgrades" in captured.out

    def test_quantize_json_output(self, capsys):
        """quantize with --json produces valid JSON envelope."""
        config = _mock_config()
        engine = _mock_engine()

        mock_result = {"total": 50, "downgrades": 10, "upgrades": 3,
                       "skipped": 35, "deleted": 2, "errors": 0}

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.dynamics.eap_scheduler.EAPScheduler"
        ) as MockEAP, patch(
            "superlocalmemory.math.ebbinghaus.EbbinghausCurve"
        ), patch(
            "superlocalmemory.storage.quantized_store.QuantizedEmbeddingStore"
        ), patch(
            "superlocalmemory.math.polar_quant.PolarQuantEncoder"
        ), patch(
            "superlocalmemory.math.qjl.QJLEncoder"
        ):
            MockEAP.return_value.run_eap_cycle.return_value = mock_result
            from superlocalmemory.cli.commands import cmd_quantize
            cmd_quantize(Namespace(dry_run=True, profile="", json=True))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is True
        assert envelope["data"]["downgrades"] == 10


# ---------------------------------------------------------------------------
# cmd_consolidate tests
# ---------------------------------------------------------------------------


class TestCmdConsolidate:
    """Tests for the 'slm consolidate' CLI command."""

    def test_consolidate_without_cognitive_flag(self, capsys):
        """consolidate without --cognitive prints usage hint."""
        from superlocalmemory.cli.commands import cmd_consolidate
        cmd_consolidate(Namespace(cognitive=False, profile="", json=False))

        captured = capsys.readouterr()
        assert "--cognitive" in captured.out

    def test_consolidate_without_cognitive_json(self, capsys):
        """consolidate without --cognitive in JSON mode returns error."""
        from superlocalmemory.cli.commands import cmd_consolidate

        with pytest.raises(SystemExit) as exc_info:
            cmd_consolidate(Namespace(cognitive=False, profile="", json=True))

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is False

    def test_consolidate_cognitive_prints_results(self, capsys):
        """consolidate --cognitive prints pipeline results."""
        config = _mock_config()
        engine = _mock_engine()

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.encoding.cognitive_consolidator.CognitiveConsolidator"
        ) as MockCCQ:
            MockCCQ.return_value.run_pipeline.return_value = _MockCCQResult()
            from superlocalmemory.cli.commands import cmd_consolidate
            cmd_consolidate(Namespace(cognitive=True, profile="", json=False))

        captured = capsys.readouterr()
        assert "Clusters processed" in captured.out
        assert "3" in captured.out

    def test_consolidate_cognitive_json(self, capsys):
        """consolidate --cognitive --json produces valid envelope."""
        config = _mock_config()
        engine = _mock_engine()

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ), patch(
            "superlocalmemory.encoding.cognitive_consolidator.CognitiveConsolidator"
        ) as MockCCQ:
            MockCCQ.return_value.run_pipeline.return_value = _MockCCQResult()
            from superlocalmemory.cli.commands import cmd_consolidate
            cmd_consolidate(Namespace(cognitive=True, profile="", json=True))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is True
        assert envelope["data"]["clusters_processed"] == 3


# ---------------------------------------------------------------------------
# cmd_soft_prompts tests
# ---------------------------------------------------------------------------


class TestCmdSoftPrompts:
    """Tests for the 'slm soft-prompts' CLI command."""

    def test_no_prompts(self, capsys):
        """soft-prompts with no data prints 'No active soft prompts'."""
        config = _mock_config()
        engine = _mock_engine()
        engine._db.execute.return_value = []

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ):
            from superlocalmemory.cli.commands import cmd_soft_prompts
            cmd_soft_prompts(Namespace(profile="", json=False))

        captured = capsys.readouterr()
        assert "No active soft prompts" in captured.out

    def test_with_prompts(self, capsys):
        """soft-prompts with data prints prompt categories."""
        config = _mock_config()
        engine = _mock_engine()
        engine._db.execute.return_value = [{
            "prompt_id": "sp-001",
            "category": "tech_preference",
            "content": "Prefers Python",
            "confidence": 0.9,
            "effectiveness": 0.8,
            "token_count": 5,
            "version": 1,
            "created_at": "2026-03-30",
        }]

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ):
            from superlocalmemory.cli.commands import cmd_soft_prompts
            cmd_soft_prompts(Namespace(profile="", json=False))

        captured = capsys.readouterr()
        assert "tech_preference" in captured.out
        assert "0.90" in captured.out

    def test_soft_prompts_json(self, capsys):
        """soft-prompts --json produces valid JSON envelope."""
        config = _mock_config()
        engine = _mock_engine()
        engine._db.execute.return_value = []

        with patch(
            "superlocalmemory.core.engine.MemoryEngine", return_value=engine,
        ), patch(
            "superlocalmemory.core.config.SLMConfig.load", return_value=config,
        ):
            from superlocalmemory.cli.commands import cmd_soft_prompts
            cmd_soft_prompts(Namespace(profile="", json=True))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is True
        assert envelope["data"]["count"] == 0


# ---------------------------------------------------------------------------
# cmd_reap tests
# ---------------------------------------------------------------------------


class TestCmdReap:
    """Tests for the 'slm reap' CLI command."""

    def test_reap_dry_run_prints_stats(self, capsys):
        """reap in dry-run mode prints process stats."""
        mock_result = {
            "total_found": 5, "orphans_found": 2,
            "killed": 0, "skipped": 2,
            "errors": [], "processes": [],
        }

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            return_value=mock_result,
        ), patch(
            "superlocalmemory.infra.process_reaper.ReaperConfig",
        ):
            from superlocalmemory.cli.commands import cmd_reap
            cmd_reap(Namespace(force=False, json=False))

        captured = capsys.readouterr()
        assert "dry run" in captured.out
        assert "5" in captured.out
        assert "2" in captured.out
        assert "--force" in captured.out

    def test_reap_force_kills(self, capsys):
        """reap with --force kills orphans."""
        mock_result = {
            "total_found": 5, "orphans_found": 2,
            "killed": 2, "skipped": 0,
            "errors": [], "processes": [],
        }

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            return_value=mock_result,
        ), patch(
            "superlocalmemory.infra.process_reaper.ReaperConfig",
        ):
            from superlocalmemory.cli.commands import cmd_reap
            cmd_reap(Namespace(force=True, json=False))

        captured = capsys.readouterr()
        assert "Killed" in captured.out
        assert "2" in captured.out

    def test_reap_json_output(self, capsys):
        """reap --json produces valid JSON envelope."""
        mock_result = {
            "total_found": 3, "orphans_found": 1,
            "killed": 0, "skipped": 1,
            "errors": [], "processes": [],
        }

        with patch(
            "superlocalmemory.infra.process_reaper.cleanup_all_orphans",
            return_value=mock_result,
        ), patch(
            "superlocalmemory.infra.process_reaper.ReaperConfig",
        ):
            from superlocalmemory.cli.commands import cmd_reap
            cmd_reap(Namespace(force=False, json=True))

        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["success"] is True
        assert envelope["data"]["total_found"] == 3


# ---------------------------------------------------------------------------
# Argparse integration tests
# ---------------------------------------------------------------------------


class TestArgparseRegistration:
    """Verify V3.3 commands are registered in argparse."""

    def test_decay_recognized(self):
        """argparse recognizes 'decay' subcommand."""
        with patch("sys.argv", ["slm", "decay"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_decay") as mock:
                main()
                mock.assert_called_once()

    def test_quantize_recognized(self):
        """argparse recognizes 'quantize' subcommand."""
        with patch("sys.argv", ["slm", "quantize"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_quantize") as mock:
                main()
                mock.assert_called_once()

    def test_consolidate_recognized(self):
        """argparse recognizes 'consolidate' subcommand."""
        with patch("sys.argv", ["slm", "consolidate", "--cognitive"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_consolidate") as mock:
                main()
                mock.assert_called_once()

    def test_soft_prompts_recognized(self):
        """argparse recognizes 'soft-prompts' subcommand."""
        with patch("sys.argv", ["slm", "soft-prompts"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_soft_prompts") as mock:
                main()
                mock.assert_called_once()

    def test_reap_recognized(self):
        """argparse recognizes 'reap' subcommand."""
        with patch("sys.argv", ["slm", "reap"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_reap") as mock:
                main()
                mock.assert_called_once()

    def test_decay_dry_run_default(self):
        """decay defaults to dry_run=True."""
        with patch("sys.argv", ["slm", "decay"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_decay") as mock:
                main()
                args = mock.call_args[0][0]
                assert args.dry_run is True

    def test_decay_execute_flag(self):
        """decay --execute sets dry_run=False."""
        with patch("sys.argv", ["slm", "decay", "--execute"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_decay") as mock:
                main()
                args = mock.call_args[0][0]
                assert args.dry_run is False

    def test_reap_force_flag(self):
        """reap --force sets force=True."""
        with patch("sys.argv", ["slm", "reap", "--force"]):
            from superlocalmemory.cli.main import main
            with patch("superlocalmemory.cli.commands.cmd_reap") as mock:
                main()
                args = mock.call_args[0][0]
                assert args.force is True
