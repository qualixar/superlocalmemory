# IDE Setup
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

Connect SuperLocalMemory to your AI coding tool. Once connected, memories are captured and recalled automatically.

---

## Transport Options (v3.6.7+)

SLM ships with **two MCP transports**. Pick the one that fits your tool.

| Transport | How it works | RAM cost | Requirement |
|-----------|-------------|----------|-------------|
| **HTTP** (recommended) | All clients share one daemon process | ~2 GB once, flat forever | `slm serve` must be running |
| **stdio** (universal) | One `slm mcp` subprocess per connection | ~90–110 MB × connections | None — works offline |
| **`mcp-remote` bridge** | stdio wrapper that tunnels to HTTP | ~50 MB bridge + HTTP pool | npm `@modelcontextprotocol/client-cli` |

**Quick rule:** use HTTP if your tool supports it. Use `mcp-remote` if your tool is stdio-only but you want shared RAM. Fall back to pure stdio if you are offline or running daemon-free.

The `mcp-remote` bridge package:
```bash
npm install -g @modelcontextprotocol/client-cli
```

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

---

## Claude Code

**Auto:**

```bash
slm connect claude
```

**Manual — HTTP (recommended, v3.6.7+):**

```bash
claude mcp add --transport http superlocalmemory http://127.0.0.1:8765/mcp/
```

Or edit `~/.claude.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**Manual — stdio (fallback):**

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

Restart Claude Code. Verify with: `slm status` in a Claude Code session.

---

## Claude Desktop

Claude Desktop uses a separate config file from Claude Code CLI.

**Manual — HTTP (recommended, v3.6.7+):**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**Manual — stdio (fallback):**

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

Restart the Claude Desktop app after editing. Config is not hot-reloaded.

---

## Cursor

**Auto:**

```bash
slm connect cursor
```

**Manual — HTTP (recommended, v3.6.7+):**

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**Manual — stdio (fallback):**

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

---

## VS Code / GitHub Copilot

**Auto:**

```bash
slm connect vscode
```

**Manual — HTTP (recommended, v3.6.7+):**

Add to VS Code settings (JSON):

```json
{
  "mcp": {
    "servers": {
      "superlocalmemory": {
        "type": "http",
        "url": "http://127.0.0.1:8765/mcp/"
      }
    }
  }
}
```

**Manual — stdio (fallback):**

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

---

## Windsurf

**Auto:**

```bash
slm connect windsurf
```

**Manual — HTTP (recommended, v3.6.7+):**

Add to `~/.windsurf/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**Manual — stdio (fallback):**

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

---

## Gemini CLI

**Auto:**

```bash
slm connect gemini
```

**Manual — HTTP (recommended, v3.6.7+):**

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**Manual — stdio (fallback):**

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

---

## Antigravity IDE / agy CLI

Antigravity IDE and the `agy` CLI both read from `~/.gemini/antigravity/mcp_config.json`.

**HTTP (recommended, v3.6.7+):**

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**stdio (fallback):**

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

Restart Antigravity IDE after editing. The `agy` CLI picks up changes on next invocation.

---

## Grok CLI

Grok CLI supports stdio only. Use the `mcp-remote` bridge to connect via HTTP:

```bash
npm install -g @modelcontextprotocol/client-cli
```

Add to `~/.grok/config.toml`:

```toml
[mcp_servers.superlocalmemory]
command = "mcp-remote"
args = [
    "http://127.0.0.1:8765/mcp/",
    "--allow-http",
    "--transport",
    "http-only",
]
```

**stdio (if mcp-remote is not available):**

```toml
[mcp_servers.superlocalmemory]
command = "slm"
args = ["mcp"]

[mcp_servers.superlocalmemory.env]
SLM_AGENT_ID = "grok"
```

---

## Hermes Agent

**HTTP (recommended, v3.6.7+):**

Edit `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  superlocalmemory:
    type: http
    url: http://127.0.0.1:8765/mcp/
```

**stdio (fallback):**

```yaml
mcp_servers:
  superlocalmemory:
    command: /opt/homebrew/bin/slm
    args:
      - mcp
    env:
      SLM_AGENT_ID: hermes
```

Both Hermes CLI and Hermes Desktop read the same `~/.hermes/config.yaml`. Restart Hermes after editing.

---

## Kimi

**HTTP (recommended, v3.6.7+):**

Add to `~/.kimi/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**stdio (fallback):**

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

---

## CommandCode AICLI

**HTTP (recommended, v3.6.7+):**

Add to `~/.commandcode/mcp.json`:

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "transport": "http",
      "enabled": true,
      "url": "http://127.0.0.1:8765/mcp/"
    }
  }
}
```

**stdio (fallback):**

```json
{
  "mcpServers": {
    "superlocalmemory": {
      "transport": "stdio",
      "enabled": true,
      "command": "slm",
      "args": ["mcp"]
    }
  }
}
```

---

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

HTTP transport support varies by JetBrains version. Use stdio if HTTP is not available in your version.

Restart the IDE after adding the server.

---

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

---

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

---

## Framework Adapters (v3.8.0)

Nine Python packages back their framework's native memory interface with the local SLM data root. Install alongside the framework; no running daemon is required for core usage.

| Framework | Install name | Implements |
|-----------|-------------|------------|
| LangGraph | `langgraph-superlocalmemory` | `BaseStore` |
| Semantic Kernel | `semantic-kernel-superlocalmemory` | `VectorStore` |
| Microsoft Agent Framework | `agent-framework-superlocalmemory` | `ContextProvider` / `HistoryProvider` |
| LangChain | `langchain-superlocalmemory` | `BaseChatMessageHistory` |
| LlamaIndex | `llama-index-storage-chat-store-superlocalmemory` | `BaseChatStore` |
| CrewAI | `crewai-superlocalmemory` | `StorageBackend` |
| AutoGen | `autogen-superlocalmemory` | `Memory` |
| Google ADK | `google-adk-superlocalmemory` | `BaseMemoryService` |
| OpenAI Agents | `openai-agents-superlocalmemory` | `SessionABC` |

All adapters write to the same SLM data root that the CLI, MCP, and dashboard surface. Optional SLM providers, connectors, and backup have separate network behavior.

Full usage examples and prerequisites: [Framework Adapters →](framework-adapters.md)

---

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

For HTTP transport specifically:

```bash
# Confirm the daemon is running and /mcp is live
curl -s http://127.0.0.1:8765/mcp/ \
  -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}' \
  | grep '"name":"SuperLocalMemory'
```

A response containing `"name":"SuperLocalMemory V3"` confirms HTTP MCP is live.

---

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

### HTTP transport returns 404 or connection refused

The SLM daemon is not running. Start it:

```bash
slm serve
# or
slm restart
```

Then re-test with the curl command above.

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

When using HTTP transport, all IDEs connect to the same daemon process at `http://127.0.0.1:8765/mcp/`. This is intentional — it is what keeps RAM flat.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
