"""The daemon wiring: the service registers idle, the host carries the real remote check."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory import sources
from superlocalmemory.sources import host as _host  # noqa: F401
from superlocalmemory.daemon.services import ServiceRegistry
from superlocalmemory.server import sources_wiring
from superlocalmemory.server.remote_access_state import remote_access_configured


def app(runtime=None):
    state = SimpleNamespace(engine=SimpleNamespace(_profile_id="p", _config=None, _db=None),
                            canonical_remember_runtime=runtime)
    return SimpleNamespace(state=state)


def test_host_uses_the_real_remote_check_and_waits_for_the_writer():
    host = sources_wiring.build_host(app())
    assert host.remote_check is remote_access_configured
    assert host.runtime() is None
    ready = SimpleNamespace(ready=True)
    assert sources_wiring.build_host(app(ready)).runtime() is ready
    assert host.profile() == "p"


def test_start_registers_an_idle_service_and_stop_unconfigures(tmp_path, monkeypatch):
    monkeypatch.setattr("superlocalmemory.runtimes.features.sources_enabled", lambda root=None: False)
    registry = ServiceRegistry()
    sources_wiring.start_source_scanner(app(), registry)
    try:
        assert registry.get("source-scan") is not None
        assert registry.get("source-scan").health()["state"] == "stopped"
        assert _host.current_host().profile() == "p"
    finally:
        assert sources_wiring.stop_source_scanner(registry) is True
    sources_wiring.start_source_scanner(app(), registry)  # second start reuses the service
    sources_wiring.stop_source_scanner(registry)
