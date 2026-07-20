# IDE Setup

SuperLocalMemory provides documented MCP configurations for named clients. Run `slm connect --list` to see the current release surface; other MCP-compatible clients can be configured manually but are not automatically release-verified.

## Auto-Detection (Recommended)

```bash
slm connect        # Auto-detect and configure all installed IDEs
slm connect --list # See which IDEs are configured
```

After running, restart your IDE to activate the connection.

`slm connect` writes only product-authored connection and memory protocol to
IDE instruction files. It does not copy recalled topics, decisions, entities,
or memory text into those high-trust files. Dynamic context is fetched through
MCP at runtime and is delivered as bounded, untrusted evidence.

## Manual Setup by IDE

If auto-detection does not find your IDE, configure it manually. All IDEs use the same MCP server command:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

Below are the config file paths for each IDE.

### Claude Code

Config file: `~/.claude.json` (or project-level `.mcp.json`)

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

### Cursor

Config file: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

### VS Code (with Copilot MCP extension)

Config file: `~/.vscode/mcp.json`

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

### Windsurf

Config file: `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

### Gemini CLI

Config file: `~/.gemini/settings.json`

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

### JetBrains IDEs (IntelliJ, PyCharm, WebStorm, etc.)

Settings > Tools > AI Assistant > MCP Servers:
- Name: `superlocalmemory`
- Command: `slm`
- Args: `mcp`

### Continue.dev

Config file: `~/.continue/config.json`

```json
{
  "mcpServers": [
    {
      "name": "superlocalmemory",
      "command": "slm",
      "args": ["mcp"]
    }
  ]
}
```

### Zed

Config file: `~/.config/zed/settings.json`

```json
{
  "language_models": {
    "mcp_servers": {
      "superlocalmemory": {
        "command": "slm",
        "args": ["mcp"]
      }
    }
  }
}
```

## Verifying the Connection

After configuring your IDE:

```bash
slm status        # Check SLM is running
slm connect --list # See which IDEs are configured
```

In your IDE, try asking your AI: "What do you know about my recent work?" If SuperLocalMemory is connected and has stored memories, the AI will reference them.

## Troubleshooting

**IDE does not detect SuperLocalMemory:**
- Verify installation: `which slm` should return a path
- Verify the MCP server starts: `slm mcp` (should hang waiting for stdio input — Ctrl+C to stop)
- Check your IDE's MCP config file path is correct

**Memories not appearing in AI responses:**
- Check that you have stored memories: `slm recall "test"`
- Check the active profile: `slm profile list`
- Restart your IDE after config changes

**Multiple IDEs:**
All IDEs share the same memory database. A memory stored via Claude Code is available in Cursor, VS Code, and every other connected IDE.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
