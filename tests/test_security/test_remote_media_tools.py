# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Image and document tools for a remote app: only with the signed grant AND the key's opt-in."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.mcp import remote_caller
from superlocalmemory.mcp.remote_caller import remote_grant
from superlocalmemory.server import remote_profile_binding as binding
from superlocalmemory.server import remote_tool_policy as policy
from superlocalmemory.server.remote_access import RemotePrincipal
from tests.test_security.test_remote_tool_policy import (
    CID,
    WEB_READ,
    WEB_WRITE,
    _grant,
    _Keys,
    _run,
    _StubMcp,
)

MEDIA = sorted(policy.MEDIA_TOOLS)
SAVE = ("remember_media", "remember_document")


class _Probe(_StubMcp):
    """Also records what the tool layer would see for this request."""

    def __init__(self) -> None:
        super().__init__(MEDIA)
        self.flags: list[bool] = []
        self.args: list[dict] = []

    async def __call__(self, scope, receive, send):
        self.flags.append(remote_caller.current_remote_media_allowed())
        await super().__call__(scope, receive, send)
        self.args.append(self.reached[-1].get("params", {}).get("arguments"))


def _call(tool: str, **arguments):
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": tool, "arguments": arguments}}).encode()


def test_the_flag_is_off_outside_a_remote_media_call() -> None:
    assert remote_caller.current_remote_media_allowed() is False
    with remote_caller.remote_media_allowed(True):
        assert remote_caller.current_remote_media_allowed() is True
    assert remote_caller.current_remote_media_allowed() is False


@pytest.mark.parametrize("scopes,extras,expected", [
    (("slm:media",), ("media",), True),
    (("slm:media", "slm:write"), ("media", "mesh"), True),
    (("slm:media",), (), False),
    ((), ("media",), False),
    (("slm:mesh",), ("media",), False),
    (("slm:media",), ("mesh",), False),
])
def test_the_flag_is_set_only_for_grant_scope_plus_key_opt_in(scopes, extras, expected) -> None:
    probe = _Probe()
    with remote_grant(_grant(*scopes)):
        _run(_call("recall"), WEB_WRITE, stub=probe, key_store=_Keys(WEB_WRITE, extras))
    assert probe.flags == [expected]


def test_the_flag_is_off_without_a_grant_and_for_a_foreign_grant() -> None:
    probe = _Probe()
    _run(_call("recall"), WEB_WRITE, stub=probe, key_store=_Keys(WEB_WRITE, ("media",)))
    with remote_grant(_grant("slm:media", cid="b" * 32)):
        _run(_call("recall"), WEB_WRITE, stub=probe, key_store=_Keys(WEB_WRITE, ("media",)))
    assert probe.flags == [False, False]


@pytest.mark.parametrize("tool", MEDIA)
def test_the_four_tools_are_not_host_only_and_each_is_classified_once(tool) -> None:
    assert tool in policy.MEDIA_TOOLS and tool not in policy.HOST_ONLY_TOOLS
    assert not hasattr(policy, "REMOTE_MEDIA_TOOLS_ENABLED")


def test_a_read_key_lists_and_reads_but_cannot_save() -> None:
    keys = _Keys(WEB_READ, ("media",))
    with remote_grant(_grant("slm:media", "slm:write")):
        for tool in MEDIA:
            probe = _Probe()
            _run(_call(tool, media_id="a" * 32, job_id="a" * 32), WEB_READ, stub=probe,
                 key_store=keys)
            assert bool(probe.reached) == (tool not in SAVE), tool


@pytest.mark.parametrize("tool,args", [
    ("remember_media", {"download_url": "https://h.example/a.png", "content": "x", "tags": "t"}),
    ("remember_media", {"base64": "QUJD", "idempotency_key": "k"}),
    ("remember_document", {"base64": "QUJD", "file_name": "a.pdf"}),
    ("get_media", {"media_id": "a" * 32, "variant": "thumb"}),
    ("media_status", {"job_id": "a" * 32}),
])
def test_media_arguments_are_accepted_and_the_key_profile_is_forced(tool, args) -> None:
    probe = _Probe()
    with remote_grant(_grant("slm:media", "slm:write")):
        _run(_call(tool, **args), WEB_WRITE, stub=probe, key_store=_Keys(WEB_WRITE, ("media",)))
    assert len(probe.reached) == 1, tool
    sent = probe.args[0]
    assert sent["profile_id"] == "default"
    assert {k: v for k, v in sent.items() if k != "profile_id"} == args


def test_a_media_call_naming_another_profile_is_refused() -> None:
    probe = _Probe()
    with remote_grant(_grant("slm:media")):
        _, body, _ = _run(_call("get_media", media_id="a" * 32, profile_id="other"), WEB_WRITE,
                          stub=probe, key_store=_Keys(WEB_WRITE, ("media",)))
    assert probe.reached == [] and body["result"]["isError"] is True


@pytest.mark.parametrize("tool", MEDIA)
@pytest.mark.parametrize("arguments", [{"profile_id": "other"}, {"payload": {"profile_id": "other"}}])
def test_every_media_tool_refuses_another_profile(tool, arguments) -> None:
    probe = _Probe()
    with remote_grant(_grant("slm:media", "slm:write")):
        _, body, _ = _run(_call(tool, **arguments), WEB_WRITE, stub=probe,
                          key_store=_Keys(WEB_WRITE, ("media",)))
    assert probe.reached == [] and body["result"]["structuredContent"]["error"] == binding.PROFILE_DENIAL


def test_every_media_tool_argument_is_classified() -> None:
    for name in ("path", "download_url", "base64", "file_name", "media_id", "variant", "job_id"):
        assert name in binding.CLASSIFIED_ARGUMENTS, name
    assert policy.MEDIA_TOOLS <= binding.ROUTED_TOOLS


def test_the_registered_media_tools_take_only_classified_arguments() -> None:
    from superlocalmemory.mcp import tools_media
    from superlocalmemory.mcp.http_transport import SLMFastMCP

    server = SLMFastMCP("t")
    tools_media.register_media_tools(server)
    tools_media.register_document_tools(server)
    import asyncio

    for tool in asyncio.run(server.list_tools()):
        assert set(tool.input_schema.get("properties", {})) <= binding.CLASSIFIED_ARGUMENTS, tool.name
