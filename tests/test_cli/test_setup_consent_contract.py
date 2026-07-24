"""Consent boundary for setup-time external configuration mutations."""

from __future__ import annotations

import inspect
from unittest.mock import Mock

from superlocalmemory.cli import setup_wizard


def test_first_use_never_installs_hooks_plugins_or_services() -> None:
    source = inspect.getsource(setup_wizard.check_first_use)
    assert "_maybe_install_hooks_on_first_use" not in source
    assert "_try_install_claude_plugin" not in source
    assert "install_service" not in source


def test_noninteractive_setup_never_mutates_external_configuration(
    monkeypatch,
) -> None:
    integrations = Mock()
    service = Mock()
    monkeypatch.setattr(setup_wizard, "_install_external_integrations", integrations)
    monkeypatch.setattr(setup_wizard, "_install_autostart_service", service)

    assert setup_wizard._configure_external_integrations(interactive=False) is False
    assert setup_wizard._configure_autostart(interactive=False) is False
    integrations.assert_not_called()
    service.assert_not_called()


def test_interactive_external_mutations_require_explicit_yes(monkeypatch) -> None:
    integrations = Mock(return_value=True)
    service = Mock(return_value=True)
    monkeypatch.setattr(setup_wizard, "_install_external_integrations", integrations)
    monkeypatch.setattr(setup_wizard, "_install_autostart_service", service)
    # v3.8.2 added an optional "connect other detected IDEs" step inside
    # _configure_external_integrations; it prompts only when IDEs are present
    # on the host. Neutralize it here so this consent test drives a
    # deterministic one-prompt-per-call flow regardless of the machine.
    monkeypatch.setattr(setup_wizard, "_configure_other_ides", lambda **_k: False)

    replies = iter(("n", "n", "y", "yes"))
    monkeypatch.setattr(setup_wizard, "_prompt", lambda *_args: next(replies))

    assert setup_wizard._configure_external_integrations(interactive=True) is False
    assert setup_wizard._configure_autostart(interactive=True) is False
    integrations.assert_not_called()
    service.assert_not_called()

    assert setup_wizard._configure_external_integrations(interactive=True) is True
    assert setup_wizard._configure_autostart(interactive=True) is True
    integrations.assert_called_once_with()
    service.assert_called_once_with()
