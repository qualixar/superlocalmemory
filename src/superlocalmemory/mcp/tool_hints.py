# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Behaviour hints every MCP tool declares (issue #300).

Hosted directories read four hints per tool, as explicit booleans: Anthropic's
Connectors Directory wants ``readOnlyHint`` / ``destructiveHint`` plus a title,
and OpenAI's app review wants all four and checks them against what the tool
does. An unset hint is not "false": the MCP default for a tool with no hints is
``destructiveHint=true``, so a missing annotation makes a harmless read look
like a deletion.

Each tool registers as ``@server.tool(title="...", annotations=READ_ONLY)``.
The constants below name the combinations in use; ``hints`` builds the rest.

* ``readOnlyHint``    - the call changes nothing a person stored.
* ``destructiveHint`` - the call can delete, erase, overwrite or rewrite stored
  data, or stop a process. Where this was a judgment call the tool says True.
* ``idempotentHint``  - repeating the same call has no further effect.
* ``openWorldHint``   - the call reaches outside this machine (a web link, a
  hosted model). Local daemon and relay calls do not count.
"""

from __future__ import annotations

from mcp.types import ToolAnnotations


def hints(
    *,
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
) -> ToolAnnotations:
    """Build a ToolAnnotations with all four hints set to explicit booleans."""
    if read_only and destructive:
        raise ValueError("a read-only tool cannot also be destructive")
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


#: Looks things up; changes nothing, so repeating it is always safe.
READ_ONLY = hints(read_only=True, idempotent=True)

#: Adds a record (a memory, an event, a message). Repeating adds another.
ADDS = hints()

#: Writes a setting or derived value without removing data; same input, same state.
SETS = hints(idempotent=True)

#: Deletes, archives, overwrites or rewrites stored data; repeating may do more.
DESTROYS = hints(destructive=True)

#: Destructive, but a second identical call has nothing left to change.
DESTROYS_ONCE = hints(destructive=True, idempotent=True)
