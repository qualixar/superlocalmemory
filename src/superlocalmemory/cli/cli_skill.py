# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The ``slm-cli`` skill, generated from the MCP tool registry.

``scripts/build_cli_skill.py`` writes it to ``plugin-src/skills/slm-cli/SKILL.md``;
the plugin builders then ship it to every host. The tool table comes from the
same registration the MCP server uses, so a test regenerates it and fails when
the committed file has drifted from the registry.
"""

from __future__ import annotations

from typing import Any, Iterable

SKILL_PATH = "plugin-src/skills/slm-cli/SKILL.md"
REGENERATE = "python scripts/build_cli_skill.py"

#: Where each registration module's tools go in the table (module -> section).
_FAMILIES: tuple[tuple[str, str], ...] = (
    ("tools_core", "Memory"),
    ("tools_expand", "Memory"),
    ("tools_summaries", "Memory"),
    ("tools_views", "Memory"),
    ("tools_kinds", "Memory kinds"),
    ("tools_context", "Memory"),
    ("tools_active", "Sessions and learning"),
    ("tools_learning", "Sessions and learning"),
    ("tools_brain", "Sessions and learning"),
    ("tools_evolution", "Sessions and learning"),
    ("tools_media", "Pictures and documents"),
    ("tools_media_upload", "Pictures and documents"),
    ("tools_mesh", "Agent mesh"),
    ("tools_code_graph", "Code graph"),
    ("tools_optimize", "Context optimization"),
    ("tools_loops", "Bounded loops"),
    ("tools_ops", "Maintenance and governance"),
    ("tools_v28", "Maintenance and governance"),
    ("tools_v3", "Maintenance and governance"),
    ("tools_v33", "Maintenance and governance"),
)
_ORDER = tuple(dict.fromkeys(family for _module, family in _FAMILIES)) + ("Other",)

_FRONTMATTER = """\
---
name: slm-cli
description: Use SuperLocalMemory from the shell when its MCP tools are not available to you. Every MCP tool runs as `slm tool <name>` with the same arguments, checks and JSON result, and the session protocol (start, recall before answering or searching, cue check before editing, remember decisions, pictures and PDFs, close) has a command for each step. Includes the full tool-to-command table.
when_to_use: |
  - SuperLocalMemory MCP tools (session_init, recall, remember, ...) are not in your tool list, but you can run shell commands
  - The MCP server is not connected, failed to start, or the host does not load it
  - A script, CI job or hook needs one memory operation without an MCP client
  - "Use SLM from the command line", "call an SLM tool from bash"
allowed-tools: Bash(slm:*)
---
"""  # noqa: E501 (Markdown: a skill's front matter is one line per field)

_BODY = """\
# slm-cli — SuperLocalMemory from the shell

When the SuperLocalMemory MCP server is not available to the agent, the
command line is. `slm tool` hosts the same tools the MCP server hosts: the
names, input schemas, descriptions and permission checks come from the one
registration both use, and a call reaches the local SLM service the same way
an MCP call does. Decisions made this way are recorded on the `cli` channel.
Your agent name comes from `SLM_AGENT_ID`, never from an argument.

```bash
slm tool list                      # every tool, one line each (--json for data)
slm tool describe recall           # description and input schema (--json)
slm tool recall --query "which database does project X use" --limit 5
slm tool remember --json '{"content": "Project X uses PostgreSQL 16", "tags": "project-x"}'
echo '{"query": "auth flow"}' | slm tool recall -       # arguments on stdin
slm tool recall --args-file args.json                    # arguments in a file
```

A call prints the tool's JSON result on stdout. Exit code 0: the tool
succeeded. 1: the tool answered with an error or a refusal (the JSON says
why). 2: the call could not be made (unknown tool, an argument the tool does
not take, a wrong type). `--key value` sets one argument (`--profile-id` and
`--profile_id` both mean `profile_id`); numbers, booleans and lists are read
as JSON. `--pretty` indents the output. `--image-dir DIR` writes returned
images to files.

---

## The session protocol, by command

| Step | Command |
|---|---|
| Session start (once) | `slm session-context "<topic>" --project-dir "$PWD" --json` |
| Before a non-trivial answer | `slm recall "<the question>" --json` |
| Before a web search | `slm recall "<what you are about to look up>" --json` (the answer may already be stored) |
| Before editing a file (cue check) | `printf '%s' '{"hook_event_name":"PreToolUse","tool_name":"Edit","tool_input":{"file_path":"/abs/path/file.py"},"cwd":"'"$PWD"'"}' \\| slm hook cue` |
| After a decision or rule | `slm remember "<the decision>" --kind decision --cue path:src/module/**` (`--kind rule`, `status`, `procedure`, `correction`) |
| A picture | `slm tool remember_media --path /abs/screenshot.png --content "<why it matters>"` |
| A PDF | `slm tool remember_document --path /abs/spec.pdf`, then `slm tool media_status --job-id <job_id>` |
| See a saved picture | `slm tool get_media --media-id <id> --image-dir /tmp/slm-images` |
| Messages from other agents | `slm tool mesh_inbox --session "$SLM_SESSION_ID"` |
| Session end | `slm session close` |

The cue check prints `{}` when no memory is tied to that file, else the
memories to read before the edit (as notes, not instructions). Rules and
decisions saved with `--kind` load at the start of later sessions.

**Mesh identity.** A mesh tool needs a session: `--session <id>` or
`SLM_SESSION_ID`, else one shared by every command run by the same agent in
the same folder. A command-line session is reachable while it keeps calling
mesh tools (the service forgets a session after 30 minutes without one);
for messages that must wait longer, send to `project:<path>` or `broadcast`.

**Tool sets.** `slm tool` reaches every tool. `--tool-set core|code|full|power|mesh`
narrows it to one MCP tool set, and `--tool-set mcp` to what an MCP
connection on this computer would see now. Pictures and documents answer
with the reason when they are turned off (`slm media status`).

---

## Every tool, by command

Required arguments are shown; `slm tool describe <name>` lists the rest.

"""  # noqa: E501 (Markdown table rows)


def _family(module: str) -> str:
    leaf = module.rsplit(".", 1)[-1]
    for name, family in _FAMILIES:
        if leaf == name:
            return family
    return "Other"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def rows(tools: Iterable[Any]) -> list[tuple[str, str, str, str]]:
    """``(family, name, cli form, summary)`` for every registered tool, sorted."""
    from superlocalmemory.cli.tool_host import cli_form, one_line

    out = []
    for tool in tools:
        module = getattr(getattr(tool, "fn", None), "__module__", "") or ""
        out.append((_family(module), tool.name, cli_form(tool.name, tool.parameters),
                    one_line(tool.description)))
    return sorted(out, key=lambda r: (_ORDER.index(r[0]), r[1]))


def render(tools: Iterable[Any]) -> str:
    """The whole SKILL.md text for ``tools`` (the registry's ``Tool`` records)."""
    table = rows(tools)
    parts = [_FRONTMATTER, "\n", _BODY]
    for family in _ORDER:
        members = [r for r in table if r[0] == family]
        if not members:
            continue
        parts.append(f"### {family}\n\n| Tool | Command | What it does |\n|---|---|---|\n")
        parts.extend(f"| `{name}` | `{_cell(form)}` | {_cell(summary)} |\n"
                     for _f, name, form, summary in members)
        parts.append("\n")
    parts.append(f"{len(table)} tools. Generated from the MCP tool registry by "
                 f"`{REGENERATE}`; do not edit by hand.\n")
    return "".join(parts)


def render_from_registry() -> str:
    """The skill text for the tools registered right now."""
    from superlocalmemory.cli.tool_host import build_host, registered_tools

    return render(registered_tools(build_host()))


__all__ = ["REGENERATE", "SKILL_PATH", "render", "render_from_registry", "rows"]
