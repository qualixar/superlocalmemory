# OpenCode Integration

[OpenCode](https://opencode.ai) can connect to SuperLocalMemory through its
standard local MCP server entry. Install SLM first (`npm install -g
superlocalmemory` or `pipx install superlocalmemory`, see
[getting started](getting-started.md)).

## Quick setup

```bash
slm connect opencode
```

This writes the MCP entry into OpenCode's own config for you. To do it by hand,
add the entry below to `opencode.json` in your project root, or to your global
OpenCode config:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "superlocalmemory": {
      "type": "local",
      "command": ["slm", "mcp"],
      "enabled": true
    }
  }
}
```

If you did not install SLM on your PATH, `uvx` runs it without a prior install:

```json
"command": ["uvx", "--from", "superlocalmemory", "slm", "mcp"]
```

The first `uvx` start downloads the package, so raise OpenCode's `timeout` for
that server (milliseconds, default 5000) if the first connection times out.

## Check that it connected

```bash
opencode mcp list
```

You should see `superlocalmemory` with a `connected` status. Start OpenCode and
the SLM tools (`remember`, `recall`, `search` and the rest) appear in its tool
list. Try a prompt such as "Remember that this project uses pytest", then
"What test runner does this project use?" in a new session.

If the server shows `failed`, run `slm doctor --quick` in the same shell that
OpenCode uses, then confirm `slm mcp` starts without errors.

## Next

OpenCode reads MCP tools only. For skills, hooks and sub-agents on hosts that
support them, see [Editor and agent plugins](plugins.md).
