# H-01 (3.7.9): SLM_REQUIRE_CREDENTIALS opt-in suppresses the uncredentialed
# loopback trusted-actor bypass. Default is OFF (behaviour unchanged).
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from superlocalmemory.server import write_identity


def _loopback_request():
    req = MagicMock()
    req.headers = {}
    req.client.host = "127.0.0.1"
    return req


def test_default_does_not_require_credentials():
    assert write_identity._REQUIRE_CREDENTIALS is False


def test_require_credentials_blocks_uncredentialed_loopback(monkeypatch):
    monkeypatch.setattr(write_identity, "_REQUIRE_CREDENTIALS", True)
    with pytest.raises(HTTPException) as excinfo:
        write_identity.require_http_mutation_actor(_loopback_request(), None)
    assert excinfo.value.status_code == 403
