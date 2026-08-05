from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from superlocalmemory.core.config import (
    DEPLOYMENT_ENTERPRISE,
    DEPLOYMENT_PERSONAL,
    load_deployment_config,
)
from superlocalmemory.server import unified_daemon


ROOT = Path(__file__).resolve().parents[2]


def test_enterprise_runtime_enables_pii_without_personal_downgrade() -> None:
    apply_runtime = getattr(unified_daemon, "_apply_deployment_runtime", None)
    assert callable(apply_runtime)

    config = SimpleNamespace(pii_redaction=False)
    apply_runtime(config, DEPLOYMENT_ENTERPRISE)
    assert config.pii_redaction is True

    apply_runtime(config, DEPLOYMENT_PERSONAL)
    assert config.pii_redaction is True


def test_enterprise_runtime_starts_and_personal_skips_retention(tmp_path: Path) -> None:
    start_retention = getattr(unified_daemon, "_start_deployment_retention", None)
    assert callable(start_retention)

    app = SimpleNamespace(state=SimpleNamespace())
    config = SimpleNamespace(db_path=tmp_path / "memory.db")
    start_retention(app, config, DEPLOYMENT_PERSONAL)
    assert app.state.retention_scheduler is None

    start_retention(app, config, DEPLOYMENT_ENTERPRISE)
    try:
        assert app.state.retention_scheduler.is_running is True
        assert app.state.retention_connection is None
    finally:
        app.state.retention_scheduler.stop()


def test_dashboard_does_not_claim_unconditional_locality_or_legal_compliance() -> None:
    index = (ROOT / "src/superlocalmemory/ui/index.html").read_text(encoding="utf-8")
    settings = (
        ROOT / "src/superlocalmemory/ui/js/auto-settings.js"
    ).read_text(encoding="utf-8")

    assert "LOCAL ONLY" not in index
    assert "EU AI Act compliant" not in settings


def test_operations_resolution_uses_shared_destructive_confirmation() -> None:
    source = (
        ROOT / "src/superlocalmemory/ui/js/od-ops-health.js"
    ).read_text(encoding="utf-8")
    assert "window.confirm(" not in source
    assert "confirmDestructive" in source


def test_shared_destructive_modal_requires_typed_target() -> None:
    source = (
        ROOT / "src/superlocalmemory/ui/js/modal.js"
    ).read_text(encoding="utf-8")
    assert "slm-cd-challenge" in source
    assert "confirmationText" in source
    assert "confirmBtn.disabled" in source


def test_shared_destructive_modal_cancels_an_existing_session() -> None:
    source = (
        ROOT / "src/superlocalmemory/ui/js/modal.js"
    ).read_text(encoding="utf-8")
    assert "activeDestructiveConfirmation" in source
    assert "activeDestructiveConfirmation.cancel()" in source


def test_present_corrupt_deployment_config_fails_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[deployment\nmode = 'enterprise'", encoding="utf-8")
    assert load_deployment_config(config_path) == DEPLOYMENT_ENTERPRISE


def test_unknown_deployment_mode_fails_closed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[deployment]\nmode = 'enterprize'\n", encoding="utf-8")
    assert load_deployment_config(config_path) == DEPLOYMENT_ENTERPRISE
