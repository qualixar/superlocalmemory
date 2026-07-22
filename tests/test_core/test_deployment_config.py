# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | TDD tests for DeploymentConfig

"""TDD tests for DeploymentConfig + load_deployment_config.

Coverage:
  (a) DeploymentConfig parses [deployment] with correct presets
  (b) Defaults to Personal when [deployment] is absent
  (c) Enterprise preset: require_login=True, pii=True, retention=True, audit=True
  (d) Personal preset: require_login=False, pii=False, retention=False, audit=True
  (e) Unknown mode falls back to personal
  (f) Missing config.toml returns Personal defaults
  (g) load_deployment_config reads from explicit path
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from superlocalmemory.core.config import (  # noqa: E402
    DeploymentConfig,
    DEPLOYMENT_PERSONAL,
    DEPLOYMENT_ENTERPRISE,
    load_deployment_config,
)


# ---------------------------------------------------------------------------
# DeploymentConfig dataclass
# ---------------------------------------------------------------------------


class TestDeploymentConfig:
    def test_personal_defaults(self) -> None:
        dc = DeploymentConfig()
        assert dc.mode == "personal"
        assert dc.require_login is False
        assert dc.pii_redaction is False
        assert dc.retention_enabled is False
        assert dc.audit is True

    def test_enterprise_preset_fields(self) -> None:
        dc = DEPLOYMENT_ENTERPRISE
        assert dc.mode == "enterprise"
        assert dc.require_login is True
        assert dc.pii_redaction is True
        assert dc.retention_enabled is True
        assert dc.audit is True

    def test_personal_preset_matches_defaults(self) -> None:
        assert DEPLOYMENT_PERSONAL == DeploymentConfig()

    def test_is_personal(self) -> None:
        assert DEPLOYMENT_PERSONAL.is_personal is True
        assert DEPLOYMENT_ENTERPRISE.is_personal is False

    def test_is_enterprise(self) -> None:
        assert DEPLOYMENT_ENTERPRISE.is_enterprise is True
        assert DEPLOYMENT_PERSONAL.is_enterprise is False

    def test_as_dict_personal(self) -> None:
        d = DEPLOYMENT_PERSONAL.as_dict()
        assert d["mode"] == "personal"
        assert d["require_login"] is False
        assert d["pii_redaction"] is False
        assert d["retention_enabled"] is False
        assert d["audit"] is True

    def test_as_dict_enterprise(self) -> None:
        d = DEPLOYMENT_ENTERPRISE.as_dict()
        assert d["mode"] == "enterprise"
        assert d["require_login"] is True
        assert d["pii_redaction"] is True
        assert d["retention_enabled"] is True
        assert d["audit"] is True

    def test_frozen_immutable(self) -> None:
        dc = DeploymentConfig()
        with pytest.raises((AttributeError, TypeError)):
            dc.mode = "enterprise"  # type: ignore[misc]

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="personal.*enterprise|enterprise.*personal"):
            DeploymentConfig(mode="superuser")


# ---------------------------------------------------------------------------
# load_deployment_config
# ---------------------------------------------------------------------------


class TestLoadDeploymentConfig:
    def test_returns_personal_when_file_absent(self, tmp_path: Path) -> None:
        result = load_deployment_config(tmp_path / "nonexistent.toml")
        assert result == DEPLOYMENT_PERSONAL

    def test_returns_personal_when_no_deployment_section(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "profile = \"balanced\"\nschema_version = \"3.4.21\"\n"
            "[runtime]\nram_ceiling_mb = 1200\n",
            encoding="utf-8",
        )
        result = load_deployment_config(cfg)
        assert result == DEPLOYMENT_PERSONAL

    def test_parses_personal_section(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[deployment]\nmode = \"personal\"\nrequire_login = false\n"
            "pii_redaction = false\nretention_enabled = false\naudit = true\n",
            encoding="utf-8",
        )
        result = load_deployment_config(cfg)
        assert result.mode == "personal"
        assert result.require_login is False
        assert result.pii_redaction is False
        assert result.retention_enabled is False
        assert result.audit is True

    def test_parses_enterprise_section(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[deployment]\nmode = \"enterprise\"\nrequire_login = true\n"
            "pii_redaction = true\nretention_enabled = true\naudit = true\n",
            encoding="utf-8",
        )
        result = load_deployment_config(cfg)
        assert result.mode == "enterprise"
        assert result.require_login is True
        assert result.pii_redaction is True
        assert result.retention_enabled is True
        assert result.audit is True

    def test_enterprise_mode_uses_enterprise_preset_as_base(
        self, tmp_path: Path
    ) -> None:
        """mode=enterprise with no explicit fields → enterprise preset defaults."""
        cfg = tmp_path / "config.toml"
        cfg.write_text("[deployment]\nmode = \"enterprise\"\n", encoding="utf-8")
        result = load_deployment_config(cfg)
        assert result.require_login is True
        assert result.pii_redaction is True
        assert result.retention_enabled is True
        assert result.audit is True

    def test_unknown_mode_falls_back_to_personal(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[deployment]\nmode = \"superuser\"\n", encoding="utf-8"
        )
        result = load_deployment_config(cfg)
        assert result == DEPLOYMENT_PERSONAL

    def test_malformed_toml_returns_personal(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text("{{{ definitely not toml }}}", encoding="utf-8")
        result = load_deployment_config(cfg)
        assert result == DEPLOYMENT_PERSONAL

    def test_partial_enterprise_override(self, tmp_path: Path) -> None:
        """Enterprise mode with audit=false override."""
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "[deployment]\nmode = \"enterprise\"\naudit = false\n",
            encoding="utf-8",
        )
        result = load_deployment_config(cfg)
        assert result.mode == "enterprise"
        assert result.require_login is True  # from enterprise preset
        assert result.audit is False          # explicit override

    def test_personal_install_no_deployment_section_is_identical_to_defaults(
        self, tmp_path: Path
    ) -> None:
        """Critical: personal install (no [deployment]) MUST behave exactly as before."""
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            "profile = \"balanced\"\n[runtime]\nram_ceiling_mb = 1200\n",
            encoding="utf-8",
        )
        result = load_deployment_config(cfg)
        assert result == DEPLOYMENT_PERSONAL
        assert result.require_login is False
        assert result.pii_redaction is False
        assert result.retention_enabled is False
