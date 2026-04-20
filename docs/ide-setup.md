# IDE Setup
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Connect SuperLocalMemory to your AI coding tool. Once connected, memories are captured and recalled automatically.

---

## Auto-Detection

The fastest way to connect all your IDEs:

```bash
slm connect
```

This scans your system for installed IDEs, configures each one, and verifies the connection. Run it once after installing SLM.

To connect a specific IDE:

```bash
slm connect claude
slm connect cursor
slm connect vscode
```

## Claude Code

**Auto:**

```bash
slm connect claude
```

**Manual:** Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "npx",
      "args": ["-y", "superlocalmemory", "mcp"],
      "env": {}
    }
  }
}
```

Restart Claude Code. Verify with: `slm status` in a Claude Code session.

## Cursor

**Auto:**

```bash
slm connect cursor
```

**Manual:** Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "npx",
      "args": ["-y", "superlocalmemory", "mcp"],
      "env": {}
    }
  }
}
```

Restart Cursor. The memory tools appear in the tool list automatically.

## VS Code / GitHub Copilot

**Auto:**

```bash
slm connect vscode
```

**Manual:** Add to VS Code settings (JSON):

```json
{
  "mcp": {
    "servers": {
      "superlocalmemory": {
        "command": "npx",
        "args": ["-y", "superlocalmemory", "mcp"]
      }
    }
  }
}
```

## Windsurf

**Auto:**

```bash
slm connect windsurf
```

**Manual:** Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "npx",
      "args": ["-y", "superlocalmemory", "mcp"],
      "env": {}
    }
  }
}
```

## Gemini CLI

**Auto:**

```bash
slm connect gemini
```

**Manual:** Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "command": "npx",
      "args": ["-y", "superlocalmemory", "mcp"]
    }
  }
}
```

## JetBrains (IntelliJ, PyCharm, WebStorm, etc.)

**Auto:**

```bash
slm connect jetbrains
```

**Manual:** Open **Settings > Tools > AI Assistant > MCP Servers** and add:

| Field | Value |
|-------|-------|
| Name | `superlocalmemory` |
| Command | `npx` |
| Arguments | `-y superlocalmemory mcp` |

Restart the IDE after adding the server.

## Continue.dev

**Auto:**

```bash
slm connect continue
```

**Manual:** Add to `~/.continue/config.json`:

```json
{
  "mcpServers": [
    {
      "name": "superlocalmemory",
      "command": "npx",
      "args": ["-y", "superlocalmemory", "mcp"]
    }
  ]
}
```

## Zed

**Auto:**

```bash
slm connect zed
```

**Manual:** Add to `~/.config/zed/settings.json`:

```json
{
  "context_servers": {
    "superlocalmemory": {
      "command": {
        "path": "npx",
        "args": ["-y", "superlocalmemory", "mcp"]
      }
    }
  }
}
```

## Verifying the Connection

After connecting any IDE, verify it works:

1. Open a chat/prompt session in your IDE
2. Ask: "What do you know about my preferences?"
3. If SLM is connected, the AI will check your memories before responding

Or run from the terminal:

```bash
slm status
```

Look for `Connected IDEs: claude, cursor, ...` in the output.

## Troubleshooting

### "slm command not found"

The npm global bin directory is not in your PATH.

```bash
# Find where npm installs global packages
npm root -g

# Add the bin directory to your PATH
# For zsh (~/.zshrc):
export PATH="$(npm root -g)/../bin:$PATH"

# For bash (~/.bashrc):
export PATH="$(npm root -g)/../bin:$PATH"
```

### IDE does not detect SLM tools

1. Ensure SLM is installed globally: `npm list -g superlocalmemory`
2. Restart the IDE completely (not just reload)
3. Check the MCP config file has correct JSON syntax
4. Run `slm connect <ide>` to regenerate the config

### "Connection refused" or timeout errors

```bash
# Test the MCP server directly
npx superlocalmemory mcp --test

# Check for port conflicts
slm status --verbose
```

### Multiple IDE configs conflicting

Each IDE has its own config file. They do not conflict. All IDEs share the same memory database at `~/.superlocalmemory/memory.db`.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
