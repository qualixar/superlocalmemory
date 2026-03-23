# SuperLocalMemory V3

**The first local-only AI memory to break 74% retrieval on LoCoMo.** No cloud. No APIs. No data leaves your machine.

SuperLocalMemory gives AI assistants persistent memory across sessions. Install once, and your AI remembers your projects, preferences, decisions, and debugging history — forever.

### V3.1: Active Memory — Memory That Learns
SLM now **learns from your usage patterns** and gets smarter over time — at zero token cost. Every recall generates learning signals. After 20+ signals, the system starts optimizing retrieval for YOUR specific patterns. After 200+, a full ML model trains on your data. No other memory system learns without spending LLM tokens. [Read more →](Active-Memory)

## Quick Start

```bash
npm install -g superlocalmemory    # or: pip install superlocalmemory
slm setup                          # Choose mode A/B/C
slm warmup                        # Pre-download embedding model (optional)
```

That's it. Your AI now remembers you.

## Three Operating Modes

| Mode | What It Does | Cloud Required |
|:----:|:------------|:-:|
| **A: Local Guardian** | Zero cloud. Your data never leaves your machine. EU AI Act compliant. 74.8% on LoCoMo. | No |
| **B: Smart Local** | Local LLM via Ollama for answer synthesis. Still fully private. | No |
| **C: Full Power** | Cloud LLM for maximum accuracy (87.7% on LoCoMo). | Yes |

## Dashboard

<details open>
<summary><strong>V3 Dashboard Screenshots</strong></summary>

![Dashboard](https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/01-dashboard-main.png)

<table><tr>
<td><img src="https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/02-knowledge-graph.png" width="200"/></td>
<td><img src="https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/03-math-health.png" width="200"/></td>
<td><img src="https://raw.githubusercontent.com/qualixar/superlocalmemory/main/docs/screenshots/05-trust-dashboard.png" width="200"/></td>
</tr></table>

</details>

## Key Features

- **Works in 17+ IDEs** — Claude Code, Cursor, VS Code, Windsurf, Gemini CLI, JetBrains, and more
- **Dual Interface: MCP + CLI** — MCP for IDEs, agent-native CLI (`--json`) for scripts, CI/CD, agent frameworks
- **4-channel retrieval** — Semantic + keyword + entity graph + temporal for maximum accuracy
- **Mathematical foundations** — Fisher-Rao similarity, sheaf consistency, Langevin lifecycle
- **Trust scoring** — Bayesian trust per agent and per fact
- **EU AI Act compliant** — Mode A satisfies data sovereignty by architecture
- **85% open-domain** — highest of any system evaluated, including cloud-powered ones
- **1400+ tests** — production-grade reliability
- **Multi-profile** — isolated memory contexts for work, personal, clients

## Documentation

| Page | What You'll Learn |
|------|-------------------|
| [Installation](Installation) | Full install guide — npm, pip, git clone |
| [Quick Start Tutorial](Quick-Start-Tutorial) | Step-by-step for new users and V2 upgraders |
| [Getting Started](Getting-Started) | Install + first memory in 5 minutes |
| [Modes Explained](Modes-Explained) | A vs B vs C — which is right for you |
| [CLI Reference](CLI-Reference) | All 18 `slm` commands with `--json` docs |
| [MCP Tools](MCP-Tools) | All 24 MCP tools for IDE integration |
| [IDE Setup](IDE-Setup) | Per-IDE configuration guide |
| [Migration from V2](Migration-from-V2) | Upgrade guide for existing users |
| [Auto-Memory](Auto-Memory) | How auto-capture and auto-recall work |
| [Architecture Overview](V3-Architecture) | How the system works |
| [Mathematical Foundations](V3-Mathematical-Foundations) | The math behind the memory |
| [Compliance](Compliance) | EU AI Act, GDPR, retention policies |
| [FAQ](FAQ) | Common questions answered |

## Research Papers

- **V3 Paper:** [Information-Geometric Foundations for Agent Memory](https://arxiv.org/abs/2603.14588) (arXiv) | [Zenodo](https://zenodo.org/records/19038659)
- **V2 Paper:** [Privacy-Preserving Multi-Agent Memory](https://arxiv.org/abs/2603.02240) (arXiv)

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
