# pi.dev Integration
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

pi.dev can connect to SuperLocalMemory through the standard stdio MCP
transport. Install SLM first, then add this MCP server entry:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"],
      "directTools": true
    }
  }
}
```

Save it at:

```text
~/.pi/agent/mcp.json
```

Then open pi.dev's MCP menu:

```text
/mcp
```

Enable the `superlocalmemory` server with space, then save with `Ctrl-S`.
The SLM tools should appear in pi.dev's tool list after the MCP server
restarts.

Verify with a direct memory prompt, for example:

```text
Can you store this to memory?
```

If the server does not appear, run `slm doctor --quick` in the same shell
environment pi.dev uses, then confirm `slm mcp` starts without errors.
