"""``daemon_request`` can tell a daemon that answered 5xx from a daemon that is down."""

from __future__ import annotations

import io
import json
import urllib.error
from types import SimpleNamespace

import pytest

from tests._urlopen_fake import patch_urlopen


def _http(status: int, body) -> urllib.error.HTTPError:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError("http://127.0.0.1:1/x", status, "err", {}, io.BytesIO(raw))


def _post(error, **flags):
    from superlocalmemory.cli.daemon import daemon_request

    descriptor = SimpleNamespace(port=1, capability="cap", instance_id="inst")
    with patch_urlopen(side_effect=error):
        return daemon_request("GET", "/api/v3/sources", None, expected_descriptor=descriptor,
                              verify_health=False, **flags)


def test_a_503_with_a_dict_detail_carries_code_and_message() -> None:
    from superlocalmemory.cli.daemon import DaemonServerError

    body = {"detail": {"code": "writer_not_ready", "message": "The memory writer is not ready."}}
    with pytest.raises(DaemonServerError) as err:
        _post(_http(503, body), preserve_server_error=True)
    assert (err.value.status, err.value.code, err.value.message) == (
        503, "writer_not_ready", "The memory writer is not ready.")
    assert isinstance(err.value, RuntimeError)


def test_a_500_with_a_string_detail() -> None:
    from superlocalmemory.cli.daemon import DaemonServerError

    with pytest.raises(DaemonServerError) as err:
        _post(_http(500, {"detail": "Could not save."}), preserve_server_error=True)
    assert (err.value.status, err.value.code, err.value.message) == (500, "", "Could not save.")


def test_an_unreadable_body_falls_back_to_the_generic_message() -> None:
    from superlocalmemory.cli.daemon import DaemonServerError

    with pytest.raises(DaemonServerError) as err:
        _post(_http(502, b"<html>"), preserve_server_error=True)
    assert err.value.code == ""
    assert err.value.message == "The SLM daemon could not finish the request."


def test_the_flag_off_keeps_the_old_none() -> None:
    assert _post(_http(503, {"detail": {"code": "writer_not_ready", "message": "m"}})) is None


def test_a_4xx_is_not_a_server_error() -> None:
    assert _post(_http(400, {"detail": "x"}), preserve_server_error=True) is None
