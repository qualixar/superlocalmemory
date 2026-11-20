# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``slm tool``: the MCP tools from a shell, with the MCP server's own registration.

In-process checks of the pieces (arguments, results, channel, tool sets, the
generated skill). The registry parity with ``mcp/server.py`` runs in fresh
processes in ``test_slm_tool_parity.py``; calls against a real daemon are in
``tests/test_integration/test_slm_tool_real_daemon.py``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from mcp_types import CallToolResult, ImageContent, TextContent

from superlocalmemory.cli import tool_args, tool_cmd, tool_host

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def host():
    import os

    before = os.environ.get("SLM_CLIENT_CHANNEL")
    try:
        yield tool_host.build_host()
    finally:  # build_host claims the cli channel for the process; give it back
        if before is None:
            os.environ.pop("SLM_CLIENT_CHANNEL", None)
        else:
            os.environ["SLM_CLIENT_CHANNEL"] = before


@pytest.fixture(scope="module")
def tools(host):
    return {t.name: t for t in tool_host.registered_tools(host)}


# -- the registry -----------------------------------------------------------------------------

def test_every_registered_tool_is_reachable_by_default(host, tools):
    """No tool set named: every registered tool can be called (none is filtered out)."""
    assert len(tools) == 112
    assert getattr(host, "_slm_allowed", None) is None


def test_no_tool_is_named_like_a_subcommand(tools):
    assert not {"list", "describe"} & set(tools)


def test_no_tool_argument_collides_with_an_option_of_slm_tool(tools):
    for tool in tools.values():
        taken = set((tool.parameters or {}).get("properties") or {})
        assert not taken & tool_cmd.RESERVED_OPTIONS, (tool.name, taken & tool_cmd.RESERVED_OPTIONS)


def test_a_tool_set_narrows_with_the_servers_own_filter():
    from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

    assert tool_host.allowed_tools("core") == _PROFILE_DEFINITIONS["core"]
    assert tool_host.allowed_tools("whole") is None
    assert tool_host.allowed_tools(None) is None
    with pytest.raises(tool_host.UnknownToolSet):
        tool_host.allowed_tools("banana")


def test_mcp_tool_set_is_what_a_connection_sees_now(monkeypatch):
    from superlocalmemory.mcp.profiles import _PROFILE_DEFINITIONS

    monkeypatch.setenv("SLM_MCP_PROFILE", "code")
    assert tool_host.allowed_tools("mcp") == _PROFILE_DEFINITIONS["code"]
    monkeypatch.delenv("SLM_MCP_PROFILE")
    monkeypatch.setenv("SLM_MCP_TOOLS", "recall,remember")
    assert tool_host.allowed_tools("mcp") == frozenset({"recall", "remember"})


# -- the channel ------------------------------------------------------------------------------

def test_the_channel_is_cli_whatever_the_environment_says(monkeypatch):
    from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, channel_headers

    monkeypatch.setenv("SLM_CLIENT_CHANNEL", "mcp")
    tool_host.claim_cli_channel()
    assert channel_headers("capability") == {CHANNEL_HEADER: "cli"}


# -- arguments --------------------------------------------------------------------------------

def test_flags_become_typed_arguments(tools):
    schema = tools["recall"].parameters
    pairs = tool_args.flag_pairs(["--query", "auth flow", "--limit=5", "--fast", "false",
                                  "--profile-id", "work"])
    args = tool_args.gather(json_text=None, args_file=None, use_stdin=False,
                            pairs=pairs, schema=schema)
    assert args == {"query": "auth flow", "limit": 5, "fast": False, "profile_id": "work"}
    tool_args.validate(args, schema)


def test_json_file_stdin_and_flags_merge_but_never_twice(tmp_path, tools):
    schema = tools["recall"].parameters
    (tmp_path / "a.json").write_text('{"limit": 3}', encoding="utf-8")
    args = tool_args.gather(json_text='{"query": "q"}', args_file=str(tmp_path / "a.json"),
                            use_stdin=True, pairs=[("fast", "true")], schema=schema,
                            stdin=io.StringIO('{"profile_id": "p"}'))
    assert args == {"query": "q", "limit": 3, "profile_id": "p", "fast": True}
    with pytest.raises(tool_args.ArgumentProblem, match="more than once"):
        tool_args.gather(json_text='{"query": "q"}', args_file=None, use_stdin=False,
                         pairs=[("query", "again")], schema=schema)


@pytest.mark.parametrize("arguments, expected", [
    ({"query": "q", "channel": "dashboard"}, "unknown argument 'channel'"),
    ({"query": "q", "limit": "lots"}, "limit:"),
    ({}, "'query' is a required property"),
])
def test_arguments_the_schema_refuses_are_refused(tools, arguments, expected):
    with pytest.raises(tool_args.ArgumentProblem) as caught:
        tool_args.validate(arguments, tools["recall"].parameters)
    assert any(expected in line for line in caught.value.details), caught.value.details


def test_bad_json_and_stray_tokens_say_what_is_wrong():
    with pytest.raises(tool_args.ArgumentProblem, match="not valid JSON"):
        tool_args.gather(json_text="{oops", args_file=None, use_stdin=False, pairs=[], schema={})
    with pytest.raises(tool_args.ArgumentProblem, match="must be a JSON object"):
        tool_args.gather(json_text="[1]", args_file=None, use_stdin=False, pairs=[], schema={})
    with pytest.raises(tool_args.ArgumentProblem, match="unexpected argument"):
        tool_args.flag_pairs(["query"])
    with pytest.raises(tool_args.ArgumentProblem, match="needs a value"):
        tool_args.flag_pairs(["--query"])


def test_list_arguments_take_json_from_a_flag(tools):
    schema = tools["mesh_send"].parameters
    args = tool_args.gather(json_text=None, args_file=None, use_stdin=False, schema=schema,
                            pairs=tool_args.flag_pairs(["--to", "broadcast", "--message", "hi",
                                                        "--refs", '["fact:1"]']))
    assert args["refs"] == ["fact:1"]
    tool_args.validate(args, schema)


# -- results ----------------------------------------------------------------------------------

def test_structured_content_is_the_result():
    result = CallToolResult(content=[TextContent(type="text", text='{"a":1}')],
                            structured_content={"a": 1})
    assert tool_host.result_payload(result) == ({"a": 1}, False)


def test_a_plain_text_refusal_becomes_a_structured_error():
    result = CallToolResult(content=[TextContent(type="text", text="Image not found.")],
                            is_error=True)
    assert tool_host.result_payload(result) == (
        {"success": False, "error": "Image not found."}, True)


def test_images_go_to_files_when_asked(tmp_path):
    raw = b"\x89PNG\r\n\x1a\nfake"
    result = CallToolResult(content=[
        TextContent(type="text", text="Thumbnail of image x."),
        ImageContent(type="image", data=base64.b64encode(raw).decode(), mime_type="image/webp"),
    ])
    inline, _ = tool_host.result_payload(result)
    assert inline["text"] == "Thumbnail of image x."
    assert inline["images"][0]["data"] == base64.b64encode(raw).decode()
    saved, is_error = tool_host.result_payload(result, tmp_path / "img")
    path = Path(saved["images"][0]["path"])
    assert not is_error and path.read_bytes() == raw and path.suffix == ".webp"


def test_a_json_result_with_thumbnails_keeps_its_keys(tmp_path):
    result = CallToolResult(content=[
        TextContent(type="text", text='{"success": true, "results": []}'),
        ImageContent(type="image", data=base64.b64encode(b"x").decode(), mime_type="image/webp"),
    ])
    payload, _ = tool_host.result_payload(result, tmp_path)
    assert payload["success"] is True and payload["results"] == []
    assert Path(payload["images"][0]["path"]).read_bytes() == b"x"


# -- the mesh session -------------------------------------------------------------------------

def test_mesh_session_is_named_stable_or_given(monkeypatch, tmp_path):
    for var in ("SLM_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLM_AGENT_ID", "alpha")
    monkeypatch.chdir(tmp_path)
    first = tool_host.bind_mesh_session(None)
    assert first == tool_host.default_mesh_session("alpha", str(tmp_path))
    assert first == tool_host.bind_mesh_session(None)  # the same agent and folder: same peer
    assert tool_host.default_mesh_session("beta", str(tmp_path)) != first
    assert tool_host.bind_mesh_session("mine") == "mine"


# -- the CLI form -----------------------------------------------------------------------------

def test_the_cli_form_names_the_required_arguments(tools):
    assert tool_host.cli_form("recall", tools["recall"].parameters) == \
        "slm tool recall --query <query>"
    assert tool_host.cli_form("get_status", tools["get_status"].parameters) == \
        "slm tool get_status"


# -- the generated skill ----------------------------------------------------------------------

def test_the_slm_cli_skill_matches_the_registry(tools):
    from superlocalmemory.cli import cli_skill

    committed = (REPO / cli_skill.SKILL_PATH).read_text(encoding="utf-8")
    assert committed == cli_skill.render(tools.values()), (
        f"{cli_skill.SKILL_PATH} has drifted from the tool registry; run {cli_skill.REGENERATE}")
    for name in tools:
        assert f"| `{name}` | `slm tool {name}" in committed, name
    assert f"{len(tools)} tools." in committed


def test_every_tool_lands_in_a_named_family(tools):
    from superlocalmemory.cli import cli_skill

    assert {row[0] for row in cli_skill.rows(tools.values())} <= set(cli_skill._ORDER[:-1])


def test_the_skill_ships_to_every_host():
    import json

    manifest = json.loads((REPO / "plugin-src" / "manifest.json").read_text(encoding="utf-8"))
    assert "slm-cli" in {s["name"] for s in manifest["skills"]}
    source = (REPO / "plugin-src" / "skills" / "slm-cli" / "SKILL.md").read_text(encoding="utf-8")
    for built in ("plugin/skills", "codex-plugin/skills", "antigravity-plugin/skills",
                  "hermes-plugin/skills", "muse-plugin/skills", "vscode-agent-plugin/skills",
                  "cursor-plugin/cursor-skills"):
        shipped = REPO / built / "slm-cli" / "SKILL.md"
        assert shipped.is_file(), built
        assert "slm tool remember" in shipped.read_text(encoding="utf-8"), built
    assert "slm tool list" in source
    assert (REPO / "copilot-plugin/.github/prompts/slm-cli.prompt.md").is_file()


def test_a_lone_dash_means_stdin_but_a_dash_value_is_a_value():
    assert tool_cmd._split_stdin(["-"]) == ([], True)
    assert tool_cmd._split_stdin(["--content", "-"]) == (["--content", "-"], False)
    assert tool_cmd._split_stdin(["--pretty", "-"]) == (["--pretty"], True)
    assert tool_cmd._split_stdin(["--limit=3", "-"]) == (["--limit=3"], True)
