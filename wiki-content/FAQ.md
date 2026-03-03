# Frequently Asked Questions

Common questions about SuperLocalMemory, answered.

---

## General Questions

### What is SuperLocalMemory?

SuperLocalMemory is an **intelligent local memory system** for AI assistants. It stores your conversations, code decisions, and project context locally, so AI assistants like Claude can remember everything about you and your projects.

### Is it really free?

**Yes, 100% free.** MIT license. No usage limits. No credit systems. No "free tier" restrictions. Use it commercially if you want.

### Does it only work with Claude?

No! SuperLocalMemory v2.1.0 is **universal** and works with 17+ IDEs:
- **MCP Integration:** Cursor, Windsurf, Claude Desktop, Continue.dev, ChatGPT Desktop, Perplexity, Zed, OpenCode, Antigravity
- **Skills:** Claude Code, Continue.dev, Cody
- **CLI:** Aider, any terminal
- **Python API:** Custom integrations

[[See all supported IDEs →|MCP-Integration]]

### Where is my data stored?

**100% on your local machine** at `~/.claude-memory/`. Nothing is ever sent to any cloud service. No telemetry, no analytics, no data collection.

---

## Privacy & Security

### Is my data safe?

Yes. Your data:
- Never leaves your computer
- Is stored in a local SQLite database
- Has no network connectivity
- Requires no API keys or accounts

### Is it GDPR/HIPAA compliant?

**Yes, by default.** Since no data leaves your machine, there's no third-party data processing to worry about. You have complete control over your data.

### Can my employer use this for sensitive projects?

Yes. SuperLocalMemory is ideal for:
- Enterprise environments
- Classified projects
- Healthcare (HIPAA)
- Financial services
- Air-gapped systems

### What about the pattern learning? Does it phone home?

No. Pattern learning happens entirely locally using basic frequency analysis. No external AI services, no API calls, no internet required.

---

## Technical Questions

### What's the universal architecture?

SuperLocalMemory v2.2.0 has a **9-layer universal architecture**:

1. **Layer 9: Visualization** — Interactive web dashboard (NEW v2.2.0)
2. **Layer 8: Hybrid Search** — Multi-strategy retrieval (NEW v2.2.0)

3. **Layer 7: Universal Access** — MCP + Skills + CLI (works everywhere)
4. **Layer 6: MCP Integration** — Model Context Protocol for 17+ IDEs
5. **Layer 5: Skills Layer** — 7 universal slash-commands
6. **Layer 4: Pattern Learning** — Learns your preferences
7. **Layer 3: Knowledge Graph** — Auto-discovers relationships
8. **Layer 2: Hierarchical Index** — Tree structure for navigation
9. **Layer 1: Raw Storage** — SQLite + full-text search

All layers share the **same local database** - no duplication, no sync issues.

[[Full architecture explanation →|Universal-Architecture]]

### How does the knowledge graph work?

1. Extracts key terms from your memories (TF-IDF)
2. Calculates similarity between memories
3. Groups similar memories into clusters (Leiden algorithm)
4. Auto-names clusters based on content

Example: It discovers "JWT", "OAuth", and "session tokens" are all related to "Authentication" — even if you never tagged them.

### What's pattern learning?

Pattern learning analyzes your memories to detect:
- Framework preferences ("React: 73% confidence")
- Coding style ("Performance over readability: 58%")
- Testing approaches ("Jest preferred: 65%")

You can feed this to Claude to get personalized suggestions.

### Can I use multiple profiles?

Yes! Create isolated contexts:

```bash
superlocalmemoryv2-profile create work
superlocalmemoryv2-profile create personal
superlocalmemoryv2-profile create client-acme
superlocalmemoryv2-profile switch work
```

Each profile has completely separate memories, graphs, and patterns.

### What databases does it support?

SQLite only (by design). Benefits:
- Zero configuration
- No server to run
- Portable (single file)
- Reliable and fast
- Works everywhere

### Can I export my data?

Yes. Your data is in `~/.claude-memory/memory.db`. You can:
- Copy the SQLite file
- Query it with any SQLite tool
- Write custom export scripts

---

## Comparison Questions

### How is this different from Mem0?

| Aspect | Mem0 | SuperLocalMemory |
|--------|------|------------------|
| **Price** | Usage-based | Free forever |
| **Data location** | Cloud | 100% local |
| **Pattern learning** | No | Yes |
| **Setup** | API keys, accounts | `./install.sh` |

### How is this different from Zep?

| Aspect | Zep | SuperLocalMemory v2.1.0 |
|--------|-----|------------------|
| **Price** | $50/month | Free forever |
| **Data location** | Cloud | 100% local |
| **IDE Support** | 1-2 | 17+ IDEs |
| **Universal Architecture** | No | Yes (9 layers) |
| **MCP Integration** | No | Yes |
| **Credit limits** | Yes | No limits |

### Why not just use ChatGPT memory?

ChatGPT memory:
- Is cloud-based (privacy concerns)
- Has limited capacity
- Doesn't work with Claude
- No knowledge graphs
- No pattern learning
- No multi-profile

### Is this like Obsidian or Notion?

No. Those are note-taking apps. SuperLocalMemory is specifically designed for:
- AI assistant context
- Automatic relationship discovery
- Pattern learning
- Code/development workflows

---

## V2.2.0 Visualization & Search

### What is the Visualization Dashboard?

**NEW in v2.2.0:** An **interactive web-based dashboard** for exploring your memories visually.

**Features:**
- **📈 Timeline View** - See all memories chronologically with importance color-coding
- **🔍 Search Explorer** - Real-time semantic search with visual score bars
- **🕸️ Graph Visualization** - Interactive knowledge graph with zoom/pan
- **📊 Statistics Dashboard** - Memory trends, tag clouds, pattern insights

**Launch:**
```bash
python ~/.claude-memory/ui_server.py
# Opens at http://localhost:8765
```

[[Complete guide →|Visualization-Dashboard]]

### What is Hybrid Search?

**NEW in v2.2.0:** A **multi-strategy search system** that combines three methods for maximum accuracy:

1. **Semantic Search (TF-IDF)** - Finds conceptually similar content
2. **Full-Text Search (FTS5)** - Exact phrase and keyword matching
3. **Graph-Enhanced Search** - Traverses knowledge graph for related memories

**Why use it?**
- **Better accuracy:** 89% precision vs 78% for semantic-only
- **Maximum recall:** Finds 91% of relevant results
- **Best F1 score:** 0.90 (balanced precision and recall)
- **Minimal overhead:** ~80ms vs ~45ms for single strategy

**Usage:**
```bash
# Hybrid (default)
slm recall "authentication patterns"

# Semantic only
slm recall "auth" --strategy semantic

# Full-text only
slm recall "JWT tokens" --strategy fts

# Graph only
slm recall "security" --strategy graph
```

### How do I use the Timeline View?

**Timeline View** shows all memories chronologically with visual indicators.

**Steps:**
1. Start dashboard: `python ~/.claude-memory/ui_server.py`
2. Navigate to Timeline tab
3. See all memories sorted by date
4. Filter by date range (last 7/30/90 days, custom)
5. Click memories to expand details

**Features:**
- **Color-coded importance:** Red (critical), Orange (high), Yellow (medium), Green (low)
- **Cluster badges:** Shows which cluster each memory belongs to
- **Hover tooltips:** Preview full content
- **Quick actions:** Edit, delete, export

**Use case:** "What did I work on last week?"
```
Timeline → Filter: Last 7 days → Scan chronologically
```

### Can I visualize the Knowledge Graph?

**Yes!** The dashboard includes an **interactive graph visualization**.

**Steps:**
1. Start dashboard
2. Navigate to Graph tab
3. Interact with the graph:
   - **Zoom:** Mouse wheel or pinch
   - **Pan:** Click and drag background
   - **Move nodes:** Click and drag nodes
   - **Explore:** Click clusters to see members

**Features:**
- **Cluster coloring:** Each cluster has unique color
- **Edge weights:** Thicker edges = stronger relationships
- **Node sizing:** Larger nodes = more connections
- **Layout options:** Force-directed, circular, hierarchical

**Use case:** "How are my authentication memories related?"
```
Graph → Click "Authentication & Security" cluster → See all 12 connected memories
```

### What's the difference between Search Explorer and CLI search?

**Both use the same search engine** but Search Explorer adds **visual features**:

| Feature | CLI | Search Explorer |
|---------|-----|-----------------|
| **Results** | Text list | Visual cards with scores |
| **Scores** | Numeric (0-1) | Visual bars (0-100%) |
| **Filters** | Command flags | Interactive UI |
| **Live search** | No | Yes (updates as you type) |
| **Strategy toggle** | Flag `--strategy` | Dropdown menu |
| **Export** | Copy/paste | JSON/CSV/PDF buttons |

**When to use each:**

**CLI:** Quick searches, scripting, automation
```bash
slm recall "authentication" --limit 5
```

**Search Explorer:** Visual exploration, comparing results, filtering
```
Dashboard → Search tab → Type "authentication" → See visual scores → Filter → Export
```

---

## V2.1.0 New Features

### What's new in v2.1.0?

**Universal integration across 17+ IDEs:**
- **MCP Server** - Native integration with Cursor, Windsurf, Claude Desktop, Continue.dev, and 7+ more
- **7 Universal Skills** - Slash-commands for Claude Code, Continue.dev, Cody
- **Universal CLI** - `slm` command works in any terminal
- **Auto-Configuration** - Zero manual setup for major IDEs

### What is MCP?

**MCP (Model Context Protocol)** is Anthropic's protocol for connecting AI assistants to external tools. SuperLocalMemory's MCP server lets AI naturally use your memory without slash commands.

Example:
```
You: "Remember that we use FastAPI for APIs"
Claude: [Uses remember tool automatically] ✓ Saved
```

[[Learn more →|MCP-Integration]]

### What are universal skills?

**Skills** are slash-commands that work across multiple IDEs:
- `/slm-remember` - Save memory
- `/slm-recall` - Search
- `/slm-status` - System health
- `/slm-build-graph` - Rebuild graph
- `/slm-list-recent` - Show recent
- `/slm-switch-profile` - Change profile
- `/slm-show-patterns` - Show learned patterns

All skills use the **same local database** as MCP and CLI.

[[Learn more →|Universal-Skills]]

### Which IDEs are supported?

**Auto-configured (run install.sh):**
- ✅ Claude Desktop
- ✅ Cursor
- ✅ Windsurf
- ✅ Continue.dev (VS Code)

**Manual setup available:**
- ChatGPT Desktop, Perplexity AI, Zed, OpenCode, Antigravity, Cody, Aider

[[Full setup guide →|MCP-Integration]]

### Do MCP, Skills, and CLI share data?

**Yes!** All three methods use the **same SQLite database** at `~/.claude-memory/memory.db`. No duplication, no sync issues.

Save with MCP → Search with CLI → View with Skills → All work on the same memories.

### Will v2.1.0 break my existing setup?

**No, 100% backward compatible.** All v2.0 commands still work. Your existing memories are preserved. Nothing breaks.

---

## Usage Questions

### How many memories can it store?

**Unlimited.** Tested up to 5,000+ memories with no issues. The graph engine has a configurable limit (default 5,000) for performance.

### How fast is search?

- **Full-text search:** ~45ms (3.3x faster than v1)
- **Graph queries:** <100ms
- **Pattern lookup:** <50ms

### Do I need to manually tag everything?

No! The knowledge graph **automatically discovers relationships**. Tags are optional for additional organization.

### How often should I rebuild the graph?

- **After adding 10+ new memories** — rebuild for best results
- **Weekly** — if you add memories regularly
- **It's fast** — <2 seconds for 100 memories

```bash
python ~/.claude-memory/graph_engine.py build
```

### Can I use this offline?

**Yes, 100% offline.** No internet connection required for any feature.

---

## Troubleshooting

### "command not found: superlocalmemoryv2"

Add to PATH:
```bash
export PATH="${HOME}/.claude-memory/bin:${PATH}"
```

### Graph build fails with "sklearn not found"

Install optional dependency:
```bash
pip install scikit-learn
```

### Memories not showing up in search

1. Check if memory was added: `superlocalmemoryv2-list`
2. Try exact phrase search
3. Rebuild FTS index (rare): restart the app

[[More troubleshooting →|Troubleshooting]]

---

## Contributing

### How can I contribute?

See [Contributing](https://github.com/varun369/SuperLocalMemoryV2/blob/main/CONTRIBUTING.md) for guidelines. Areas we need help:
- Performance optimizations
- Graph visualization UI
- Additional pattern categories
- Documentation improvements

### Is there a roadmap?

Yes! See [[Roadmap]] for upcoming features.

---

## Support

### Where do I report bugs?

[GitHub Issues](https://github.com/varun369/SuperLocalMemoryV2/issues)

### Where can I ask questions?

[GitHub Discussions](https://github.com/varun369/SuperLocalMemoryV2/discussions)

### How can I support the project?

- ⭐ [Star on GitHub](https://github.com/varun369/SuperLocalMemoryV2)
- ☕ [Buy Me a Coffee](https://buymeacoffee.com/varunpratah)
- 💸 [PayPal](https://paypal.me/varunpratapbhardwaj)
- 💖 [GitHub Sponsors](https://github.com/sponsors/varun369)

---

## Creator

### Who created SuperLocalMemory?

**Varun Pratap Bhardwaj** - Solution Architect and Original Creator

SuperLocalMemory is built as an open-source alternative to expensive cloud-based memory services like Mem0 and Zep.

- **GitHub:** [github.com/varun369](https://github.com/varun369)
- **License:** MIT (free for commercial use)
- **Support:** [Buy me a coffee](https://buymeacoffee.com/varunpratah)

---

[[← Back to Home|Home]]

---

**Created by Varun Pratap Bhardwaj**
