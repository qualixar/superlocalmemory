<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory</h1>
<p align="center"><strong>Your AI Finally Remembers You</strong></p>

<p align="center">
  <strong>Stop re-explaining your codebase every session. 100% local. Zero setup. Completely free.</strong>
</p>

<p align="center">
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/🌐_Website-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Official Website"/></a>
</p>

<p align="center">
  <a href="https://zenodo.org/records/18709670"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18709670-blue?style=for-the-badge&logo=doi&logoColor=white" alt="DOI"/></a>
  <a href="https://zenodo.org/records/18709670"><img src="https://img.shields.io/badge/Zenodo-Research_Paper-1682D4?style=for-the-badge&logo=zenodo&logoColor=white" alt="Zenodo"/></a>
  <a href="https://www.researchgate.net/publication/400976053"><img src="https://img.shields.io/badge/ResearchGate-Paper-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white" alt="ResearchGate"/></a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/local--first-100%25-brightgreen?style=flat-square" alt="100% Local"/></a>
  <a href="#"><img src="https://img.shields.io/badge/setup-5%20min-orange?style=flat-square" alt="5 Min Setup"/></a>
  <a href="#"><img src="https://img.shields.io/badge/platform-Mac%20%7C%20Linux%20%7C%20Windows-blue?style=flat-square" alt="Cross Platform"/></a>
  <a href="https://github.com/varun369/SuperLocalMemoryV2/wiki"><img src="https://img.shields.io/badge/📚_Wiki-Documentation-blue?style=flat-square" alt="Wiki"/></a>
</p>

<p align="center">
  <a href="https://superlocalmemory.com"><strong>superlocalmemory.com</strong></a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-why-superlocalmemory">Why This?</a> •
  <a href="#-features">Features</a> •
  <a href="#-documentation">Docs</a> •
  <a href="https://github.com/varun369/SuperLocalMemoryV2/issues">Issues</a>
</p>

<p align="center">
  <b>A <a href="https://github.com/varun369">Qualixar</a> Product</b> · Created by <a href="https://github.com/varun369">Varun Pratap Bhardwaj</a> •
  <a href="https://github.com/sponsors/varun369">💖 Sponsor</a> •
  <a href="LICENSE">📜 MIT License</a>
</p>

---

## Research Paper

**SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning**

*Varun Pratap Bhardwaj, 2026*

The paper presents SuperLocalMemory's architecture for defending against OWASP ASI06 memory poisoning through local-first design, Bayesian trust scoring, and adaptive learning-to-rank — all without cloud dependencies or LLM inference calls.

| Platform | Link |
|----------|------|
| **Zenodo** (CERN) | [DOI: 10.5281/zenodo.18709670](https://zenodo.org/records/18709670) |
| **ResearchGate** | [Publication Page](https://www.researchgate.net/publication/400976053) |
| **arXiv** | Submission under review |
| **Research Portfolio** | [superlocalmemory.com/research](https://superlocalmemory.com/research) |

If you use SuperLocalMemory in your research, please cite:

```bibtex
@misc{bhardwaj2026superlocalmemory,
  title={SuperLocalMemory: Privacy-Preserving Multi-Agent Memory with Bayesian Trust Defense Against Memory Poisoning},
  author={Bhardwaj, Varun Pratap},
  year={2026},
  doi={10.5281/zenodo.18709670},
  url={https://zenodo.org/records/18709670},
  note={Preprint}
}
```

---

## What's New in v2.8 — "Memory That Manages Itself"

**SuperLocalMemory now manages its own memory lifecycle, learns from action outcomes, and provides enterprise-grade compliance — all 100% locally on your machine.**

### Memory Lifecycle Management (v2.8)
Memories automatically transition through lifecycle states based on usage patterns:
- **Active** — Frequently used, instantly available
- **Warm** — Recently used, included in searches
- **Cold** — Older, retrievable on demand
- **Archived** — Compressed, restorable when needed

Configure bounds to keep your memory system fast:
```bash
# Check lifecycle status
slm lifecycle-status

# Compact stale memories
slm compact --dry-run
```

### Behavioral Learning (v2.8)
The system learns from what works:
- Report outcomes: `slm report-outcome --memory-ids 1,5 --outcome success`
- View patterns: `slm behavioral-patterns`
- Knowledge transfers across projects automatically

### Enterprise Compliance (v2.8)
Built for regulated environments:
- **Access Control** — Attribute-based policies (ABAC)
- **Audit Trail** — Tamper-evident event logging
- **Retention Policies** — GDPR erasure, HIPAA retention, EU AI Act compliance

### New MCP Tools (v2.8)

| Tool | Purpose |
|------|---------|
| `report_outcome` | Record action outcomes for behavioral learning |
| `get_lifecycle_status` | View memory lifecycle states |
| `set_retention_policy` | Configure retention policies |
| `compact_memories` | Trigger lifecycle transitions |
| `get_behavioral_patterns` | View learned behavioral patterns |
| `audit_trail` | Query compliance audit trail |

### Performance

| Operation | Latency |
|-----------|---------|
| Lifecycle evaluation | Sub-2ms |
| Access control check | Sub-1ms |
| Feature vector (20-dim) | Sub-5ms |

**Upgrade:** `npm install -g superlocalmemory@latest` — All v2.7 behavior preserved, zero breaking changes.

[Upgrading to v2.8](https://github.com/varun369/SuperLocalMemoryV2/wiki/Upgrading-to-v2.8) | [Full Changelog](CHANGELOG.md)

---

<details>
<summary><strong>Previous: v2.7 — "Your AI Learns You"</strong></summary>

SuperLocalMemory learns your patterns, adapts to your workflow, and personalizes recall — all 100% locally. No cloud. No LLM. Your behavioral data never leaves your device.

- **Adaptive Learning** — Learns tech preferences, project context, and workflow patterns
- **Three-Phase Ranking** — Baseline → Rule-Based → ML Ranking (gets smarter over time)
- **Privacy by Design** — Learning data stored separately, one-command GDPR erasure
- **3 New MCP Tools** — Feedback signal, pattern transparency, and user correction

</details>

<details>
<summary><strong>Previous: v2.6.5 — Interactive Knowledge Graph</strong></summary>

- Fully interactive visualization with zoom, pan, and click-to-explore
- 6 layout algorithms, smart cluster filtering, 10,000+ node performance
- Mobile & accessibility support: touch gestures, keyboard nav, screen reader

</details>

<details>
<summary><strong>Previous: v2.6 — Security & Scale</strong></summary>

## What's New in v2.6

SuperLocalMemory is now **production-hardened** with security, performance, and scale improvements:

- **Trust Enforcement** — Bayesian scoring actively protects your memory. Agents with trust below 0.3 are blocked from write/delete operations.
- **Profile Isolation** — Memory profiles fully sandboxed. Zero cross-profile data leakage.
- **Rate Limiting** — Protects against memory flooding from misbehaving agents.
- **HNSW-Accelerated Graphs** — Knowledge graph edge building uses HNSW index for faster construction at scale.
- **Hybrid Search Engine** — Combined semantic + FTS5 + graph retrieval for maximum accuracy.

**v2.5 highlights (included):** Real-time event stream, WAL-mode concurrent writes, agent tracking, memory provenance, 28 API endpoints.

**Upgrade:** `npm install -g superlocalmemory@latest`

[Interactive Architecture Diagram](https://superlocalmemory.com/architecture.html) | [Architecture Doc](docs/ARCHITECTURE.md) | [Full Changelog](CHANGELOG.md)

</details>

---

## The Problem

Every time you start a new Claude session:

```
You: "Remember that authentication bug we fixed last week?"
Claude: "I don't have access to previous conversations..."
You: *sighs and explains everything again*
```

**AI assistants forget everything between sessions.** You waste time re-explaining your:
- Project architecture
- Coding preferences
- Previous decisions
- Debugging history

---

## The Solution

```bash
# Install in one command
npm install -g superlocalmemory

# Save a memory
superlocalmemoryv2-remember "Fixed auth bug - JWT tokens were expiring too fast, increased to 24h"

# Later, in a new session...
superlocalmemoryv2-recall "auth bug"
# ✓ Found: "Fixed auth bug - JWT tokens were expiring too fast, increased to 24h"
```

**Your AI now remembers everything.** Forever. Locally. For free.

---

## 🚀 Quick Start

### Install (One Command)

```bash
npm install -g superlocalmemory
```

Or clone manually:
```bash
git clone https://github.com/varun369/SuperLocalMemoryV2.git && cd SuperLocalMemoryV2 && ./install.sh
```

Both methods auto-detect and configure **17+ IDEs and AI tools** — Cursor, VS Code/Copilot, Codex, Claude, Windsurf, Gemini CLI, JetBrains, and more.

### Verify Installation

```bash
superlocalmemoryv2-status
# ✓ Database: OK (0 memories)
# ✓ Graph: Ready
# ✓ Patterns: Ready
```

**That's it.** No Docker. No API keys. No cloud accounts. No configuration.

### Launch Dashboard

```bash
# Start the interactive web UI
python3 ~/.claude-memory/ui_server.py

# Opens at http://localhost:8765
# Features: Timeline, search, interactive graph, statistics
```

---

## 💡 Why SuperLocalMemory?

### For Developers Who Use AI Daily

| Scenario | Without Memory | With SuperLocalMemory |
|----------|---------------|----------------------|
| New Claude session | Re-explain entire project | `recall "project context"` → instant context |
| Debugging | "We tried X last week..." starts over | Knowledge graph shows related past fixes |
| Code preferences | "I prefer React..." every time | Pattern learning knows your style |
| Multi-project | Context constantly bleeds | Separate profiles per project |

### Built on Peer-Reviewed Research

Not another simple key-value store. SuperLocalMemory implements **cutting-edge memory architecture** backed by peer-reviewed research — hierarchical organization, knowledge graph clustering, identity pattern learning, multi-level retrieval, adaptive re-ranking, workflow sequence mining, temporal confidence scoring, and cold-start mitigation.

**The only open-source implementation combining all these approaches — entirely locally.**

[Read the paper →](https://zenodo.org/records/18709670)

---

## ✨ Features

### Multi-Layer Memory Architecture

**[View Interactive Architecture Diagram](https://superlocalmemory.com/architecture.html)** — Click any layer for details, research references, and file paths.

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 9: VISUALIZATION (v2.2+)                             │
│  Interactive dashboard: timeline, graph explorer, analytics │
├─────────────────────────────────────────────────────────────┤
│  Layer 8: HYBRID SEARCH (v2.2+)                             │
│  Combines: Semantic + FTS5 + Graph traversal                │
├─────────────────────────────────────────────────────────────┤
│  Layer 7: UNIVERSAL ACCESS                                  │
│  MCP + Skills + CLI (works everywhere)                      │
│  17+ IDEs with single database                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 6: MCP INTEGRATION                                   │
│  Model Context Protocol: 18 tools, 6 resources, 2 prompts   │
│  Auto-configured for Cursor, Windsurf, Claude               │
├─────────────────────────────────────────────────────────────┤
│  Layer 5½: ADAPTIVE LEARNING (v2.7 — NEW)                   │
│  Three-layer learning: tech prefs + project context + flow  │
│  Local ML re-ranking — no cloud, no telemetry               │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: SKILLS LAYER                                      │
│  7 universal slash-commands for AI assistants               │
│  Compatible with Claude Code, Continue, Cody                │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: PATTERN LEARNING                                  │
│  Confidence-scored preference detection                     │
│  "You prefer React over Vue" (73% confidence)               │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: KNOWLEDGE GRAPH + HIERARCHICAL CLUSTERING         │
│  Auto-clustering: "Python" → "Web API" → "Auth"            │
│  Community summaries with auto-generated labels             │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: HIERARCHICAL INDEX                                │
│  Tree structure for fast navigation                         │
│  O(log n) lookups instead of O(n) scans                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: RAW STORAGE                                       │
│  SQLite + Full-text search + vector search                  │
│  Compression: 60-96% space savings                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Capabilities

- **[Adaptive Learning System](https://github.com/varun369/SuperLocalMemoryV2/wiki/Learning-System)** — Learns your tech preferences, workflow patterns, and project context. Personalizes recall ranking using local ML. Zero cloud dependency. *New in v2.7*
- **[Knowledge Graphs](https://github.com/varun369/SuperLocalMemoryV2/wiki/Knowledge-Graph-Guide)** — Automatic relationship discovery. Interactive visualization with zoom, pan, click.
- **[Pattern Learning](https://github.com/varun369/SuperLocalMemoryV2/wiki/Pattern-Learning-Explained)** — Learns your coding preferences and style automatically.
- **[Multi-Profile Support](https://github.com/varun369/SuperLocalMemoryV2/wiki/Multi-Profile-Workflows)** — Isolated contexts for work, personal, clients. Zero context bleeding.
- **[Hybrid Search](https://github.com/varun369/SuperLocalMemoryV2/wiki/Advanced-Search)** — Semantic + FTS5 + Graph retrieval combined for maximum accuracy.
- **[Visualization Dashboard](https://github.com/varun369/SuperLocalMemoryV2/wiki/Visualization-Dashboard)** — Web UI for timeline, search, graph exploration, analytics.
- **[Framework Integrations](docs/FRAMEWORK-INTEGRATIONS.md)** — Use with LangChain and LlamaIndex applications.
- **[Real-Time Events](https://github.com/varun369/SuperLocalMemoryV2/wiki/Real-Time-Event-System)** — Live notifications via SSE/WebSocket/Webhooks when memories change.
- **[Memory Lifecycle](https://github.com/varun369/SuperLocalMemoryV2/wiki/Memory-Lifecycle)** — Automatic state transitions (Active → Warm → Cold → Archived) with bounded growth guarantees. *New in v2.8*
- **[Behavioral Learning](https://github.com/varun369/SuperLocalMemoryV2/wiki/Behavioral-Learning)** — Learns from action outcomes, extracts success/failure patterns, transfers knowledge across projects. *New in v2.8*
- **[Enterprise Compliance](https://github.com/varun369/SuperLocalMemoryV2/wiki/Enterprise-Compliance)** — ABAC access control, tamper-evident audit trail, GDPR/HIPAA/EU AI Act retention policies. *New in v2.8*

---

## 🌐 Works Everywhere

**SuperLocalMemory is the ONLY memory system that works across ALL your tools:**

### Supported IDEs & Tools

| Tool | Integration | How It Works |
|------|-------------|--------------|
| **Claude Code** | ✅ Skills + MCP | `/superlocalmemoryv2-remember` |
| **Cursor** | ✅ MCP + Skills | AI uses memory tools natively |
| **Windsurf** | ✅ MCP + Skills | Native memory access |
| **Claude Desktop** | ✅ MCP | Built-in support |
| **OpenAI Codex** | ✅ MCP + Skills | Auto-configured (TOML) |
| **VS Code / Copilot** | ✅ MCP + Skills | `.vscode/mcp.json` |
| **Continue.dev** | ✅ MCP + Skills | `/slm-remember` |
| **Cody** | ✅ Custom Commands | `/slm-remember` |
| **Gemini CLI** | ✅ MCP + Skills | Native MCP + skills |
| **JetBrains IDEs** | ✅ MCP | Via AI Assistant settings |
| **Zed Editor** | ✅ MCP | Native MCP tools |
| **Aider** | ✅ Smart Wrapper | `aider-smart` with context |
| **Any Terminal** | ✅ Universal CLI | `slm remember "content"` |

### Three Ways to Access

1. **MCP (Model Context Protocol)** — Auto-configured for Cursor, Windsurf, Claude Desktop
   - AI assistants get natural access to your memory
   - No manual commands needed
   - "Remember that we use this framework" just works

2. **Skills & Commands** — For Claude Code, Continue.dev, Cody
   - `/superlocalmemoryv2-remember` in Claude Code
   - `/slm-remember` in Continue.dev and Cody
   - Familiar slash command interface

3. **Universal CLI** — Works in any terminal or script
   - `slm remember "content"` - Simple, clean syntax
   - `slm recall "query"` - Search from anywhere
   - `aider-smart` - Aider with auto-context injection

**All three methods use the SAME local database.** No data duplication, no conflicts.

[Complete setup guide for all tools →](docs/MCP-MANUAL-SETUP.md)

---

## 🆚 vs Alternatives

### The Hard Truth About "Free" Tiers

| Solution | Free Tier Limits | Paid Price | What's Missing |
|----------|-----------------|------------|----------------|
| **Mem0** | 10K memories, limited API | Usage-based | No pattern learning, not local |
| **Zep** | Limited credits | $50/month | Credit system, cloud-only |
| **Supermemory** | 1M tokens, 10K queries | $19-399/mo | Not local, no graphs |
| **Personal.AI** | ❌ No free tier | $33/month | Cloud-only, closed ecosystem |
| **Letta/MemGPT** | Self-hosted (complex) | TBD | Requires significant setup |
| **SuperLocalMemory** | **Unlimited** | **$0 forever** | **Nothing.** |

### What Actually Matters

| Feature | Mem0 | Zep | Khoj | Letta | **SuperLocalMemory** |
|---------|------|-----|------|-------|------------------------|
| **Works in Cursor** | Cloud Only | ❌ | ❌ | ❌ | ✅ **Local** |
| **Works in Windsurf** | Cloud Only | ❌ | ❌ | ❌ | ✅ **Local** |
| **Works in VS Code** | 3rd Party | ❌ | Partial | ❌ | ✅ **Native** |
| **Universal CLI** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Multi-Layer Architecture** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Pattern Learning** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Adaptive ML Ranking** | Cloud LLM | ❌ | ❌ | ❌ | ✅ **Local ML** |
| **Knowledge Graphs** | ✅ | ✅ | ❌ | ❌ | ✅ |
| **100% Local** | ❌ | ❌ | Partial | Partial | ✅ |
| **GDPR by Design** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Zero Setup** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Completely Free** | Limited | Limited | Partial | ✅ | ✅ |

**SuperLocalMemory is the ONLY solution that:**
- ✅ **Learns and adapts** locally — no cloud LLM needed for personalization
- ✅ Works across 17+ IDEs and CLI tools
- ✅ Remains 100% local (no cloud dependencies)
- ✅ GDPR Article 17 compliant — one-command data erasure
- ✅ Completely free with unlimited memories

[See full competitive analysis →](docs/COMPETITIVE-ANALYSIS.md)

---

## ⚡ Measured Performance

All numbers measured on real hardware (Apple M4 Pro, 24GB RAM). No estimates — real benchmarks.

### Search Speed

| Database Size | Median Latency | P95 Latency |
|---------------|----------------|-------------|
| 100 memories | **10.6ms** | 14.9ms |
| 500 memories | **65.2ms** | 101.7ms |
| 1,000 memories | **124.3ms** | 190.1ms |

For typical personal use (under 500 memories), search results return faster than you blink.

### Concurrent Writes — Zero Errors

| Scenario | Writes/sec | Errors |
|----------|------------|--------|
| 1 AI tool writing | **204/sec** | 0 |
| 2 AI tools simultaneously | **220/sec** | 0 |
| 5 AI tools simultaneously | **130/sec** | 0 |

Concurrent-safe architecture = zero "database is locked" errors, ever.

### Storage

10,000 memories = **13.6 MB** on disk (~1.4 KB per memory). Your entire AI memory history takes less space than a photo.

### Graph Construction

| Memories | Build Time |
|----------|-----------|
| 100 | 0.28s |
| 1,000 | 10.6s |

Auto-clustering discovers 6-7 natural topic communities from your memories.

[Full benchmark details →](https://github.com/varun369/SuperLocalMemoryV2/wiki/Performance-Benchmarks)

---

## 🔧 CLI Commands

```bash
# Memory Operations
superlocalmemoryv2-remember "content" --tags tag1,tag2  # Save memory
superlocalmemoryv2-recall "search query"                 # Search
superlocalmemoryv2-list                                  # Recent memories
superlocalmemoryv2-status                                # System health

# Profile Management
superlocalmemoryv2-profile list                          # Show all profiles
superlocalmemoryv2-profile create <name>                 # New profile
superlocalmemoryv2-profile switch <name>                 # Switch context

# Knowledge Graph
python ~/.claude-memory/graph_engine.py build            # Build graph
python ~/.claude-memory/graph_engine.py stats            # View clusters

# Pattern Learning
python ~/.claude-memory/pattern_learner.py update        # Learn patterns
python ~/.claude-memory/pattern_learner.py context 0.5   # Get identity

# Visualization Dashboard
python ~/.claude-memory/ui_server.py                     # Launch web UI
```

[Complete CLI reference →](docs/CLI-COMMANDS-REFERENCE.md)

---

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](https://github.com/varun369/SuperLocalMemoryV2/wiki/Quick-Start-Tutorial) | Get running in 5 minutes |
| [Installation](https://github.com/varun369/SuperLocalMemoryV2/wiki/Installation) | Detailed setup instructions |
| [Visualization Dashboard](https://github.com/varun369/SuperLocalMemoryV2/wiki/Visualization-Dashboard) | Interactive web UI guide |
| [Interactive Graph](https://github.com/varun369/SuperLocalMemoryV2/wiki/Using-Interactive-Graph) | Graph exploration guide (NEW v2.6.5) |
| [Framework Integrations](docs/FRAMEWORK-INTEGRATIONS.md) | LangChain & LlamaIndex setup |
| [Knowledge Graph](https://github.com/varun369/SuperLocalMemoryV2/wiki/Knowledge-Graph-Guide) | How clustering works |
| [Pattern Learning](docs/PATTERN-LEARNING.md) | Identity extraction |
| [Memory Lifecycle](https://github.com/varun369/SuperLocalMemoryV2/wiki/Memory-Lifecycle) | Lifecycle states, compaction, bounded growth (v2.8) |
| [Behavioral Learning](https://github.com/varun369/SuperLocalMemoryV2/wiki/Behavioral-Learning) | Action outcomes, pattern extraction (v2.8) |
| [Enterprise Compliance](https://github.com/varun369/SuperLocalMemoryV2/wiki/Enterprise-Compliance) | ABAC, audit trail, retention policies (v2.8) |
| [Upgrading to v2.8](https://github.com/varun369/SuperLocalMemoryV2/wiki/Upgrading-to-v2.8) | Migration guide from v2.7 |
| [API Reference](docs/API-REFERENCE.md) | Python API documentation |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas for contribution:**
- Additional pattern categories
- Performance optimizations
- Integration with more AI assistants
- Documentation improvements

---

## 💖 Support This Project

If SuperLocalMemory saves you time, consider supporting its development:

- ⭐ **Star this repo** — helps others discover it
- 🐛 **Report bugs** — [open an issue](https://github.com/varun369/SuperLocalMemoryV2/issues)
- 💡 **Suggest features** — [start a discussion](https://github.com/varun369/SuperLocalMemoryV2/discussions)
- ☕ **Buy me a coffee** — [buymeacoffee.com/varunpratah](https://buymeacoffee.com/varunpratah)
- 💸 **PayPal** — [paypal.me/varunpratapbhardwaj](https://paypal.me/varunpratapbhardwaj)
- 💖 **Sponsor** — [GitHub Sponsors](https://github.com/sponsors/varun369)

---

## 📜 License

MIT License — use freely, even commercially. Just include the license.

---

## 👨‍💻 Author

**Varun Pratap Bhardwaj** — Founder, Qualixar · Solution Architect

[![GitHub](https://img.shields.io/badge/GitHub-@varun369-181717?style=flat-square&logo=github)](https://github.com/varun369)

Building the complete agent development platform at Qualixar — memory, testing, contracts, and security for AI agents.

### Part of the Qualixar Agent Development Platform

SuperLocalMemory is part of **Qualixar**, a suite of open-source tools for building reliable AI agents:

| Product | What It Does |
|---------|-------------|
| **SuperLocalMemory** | Local-first AI agent memory |
| **SkillFortify** | Agent skill supply chain security |

---

<p align="center">
  <b>100% local. 100% private. 100% yours.</b>
</p>

<p align="center">
  <a href="https://github.com/varun369/SuperLocalMemoryV2">
    <img src="https://img.shields.io/badge/⭐_Star_on_GitHub-black?style=for-the-badge&logo=github" alt="Star on GitHub"/>
  </a>
</p>
