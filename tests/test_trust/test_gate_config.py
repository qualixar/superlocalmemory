# M-02/M-03 (3.7.9): trust gate thresholds are env-configurable and reads can
# be gated opt-in. Defaults must stay non-breaking.
from unittest.mock import MagicMock

import pytest

from superlocalmemory.trust.gate import TrustError, TrustGate, _env_threshold


def _gate(score, **kw):
    scorer = MagicMock()
    scorer.get_agent_trust.return_value = score
    return TrustGate(scorer, **kw)


def test_read_gate_off_by_default_always_passes():
    _gate(0.0).check_read("a", "p")  # must not raise


def test_read_gate_on_blocks_low_trust():
    g = _gate(0.05, read_threshold=0.1, read_gate_enabled=True)
    with pytest.raises(TrustError):
        g.check_read("a", "p")


def test_read_gate_on_allows_sufficient_trust():
    _gate(0.5, read_threshold=0.1, read_gate_enabled=True).check_read("a", "p")


def test_write_threshold_is_configurable():
    g = _gate(0.4, write_threshold=0.5)
    with pytest.raises(TrustError):
        g.check_write("a", "p")


def test_default_new_agent_passes_write_gate():
    # A new agent scores 0.5; the default 0.3 write threshold must admit it.
    _gate(0.5).check_write("a", "p")


def test_env_threshold_parsing(monkeypatch):
    monkeypatch.setenv("SLM_X", "0.7")
    assert _env_threshold("SLM_X", 0.3) == 0.7
    monkeypatch.setenv("SLM_X", "bad")
    assert _env_threshold("SLM_X", 0.3) == 0.3
    monkeypatch.setenv("SLM_X", "1.5")  # out of [0,1]
    assert _env_threshold("SLM_X", 0.3) == 0.3
    monkeypatch.delenv("SLM_X", raising=False)
    assert _env_threshold("SLM_X", 0.3) == 0.3
