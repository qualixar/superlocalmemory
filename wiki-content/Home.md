<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="96"/>
</p>

# SuperLocalMemory

<p align="center">
  <strong>Your AI Finally Remembers You</strong><br/>
  <em>The only free, local-first AI memory system with universal architecture</em>
</p>

<p align="center">
  <a href="https://superlocalmemory.com/">🌐 Official Website</a> •
  <a href="https://github.com/varun369/SuperLocalMemoryV2">⭐ Star on GitHub</a> •
  <a href="https://buymeacoffee.com/varunpratah">☕ Buy Me a Coffee</a> •
  <a href="https://paypal.me/varunpratapbhardwaj">💸 PayPal</a>
</p>

---

## Visual Overview

![SuperLocalMemory Features](https://superlocalmemory.com/assets/contact-sheet.png)
*All SuperLocalMemory features at a glance: Dashboard, CLI, Live Events, Agents, Knowledge Graph*

---

## 🆕 What's New in v2.8

- **[[Memory-Lifecycle|Memory Lifecycle Management]]** — Memories automatically manage themselves
- **[[Behavioral-Learning|Behavioral Learning]]** — Your AI learns from action outcomes
- **[[Enterprise-Compliance|Enterprise Compliance]]** — Access control, audit trails, retention policies
- **[[Upgrading-to-v2.8|Upgrade Guide]]** — Drop-in upgrade, full backward compatibility

---

## NEW: v2.7.4 — Your AI Learns You

> SuperLocalMemory now **learns your patterns** and personalizes recall — all 100% locally. Adaptive ML ranking, workflow detection, and GDPR-compliant behavioral learning.
>
> **12 MCP tools** | **7 skills** | **6 resources** | **2 prompts**
>
> - [[Learning System]] — How the three-layer learning architecture works
> - [[Upgrading to v2.7|Upgrading-to-v2.7]] — Step-by-step upgrade guide

---

## v2.6.0 — Security Hardening & Performance

> **SuperLocalMemory is now production-hardened with trust enforcement, rate limiting, and accelerated graph building.**
>
> | Feature | Benefit |
> |---------|---------|
> | **Trust Enforcement** | Agents with trust < 0.3 blocked from write/delete operations |
> | **Profile Isolation** | Full sandboxing — no cross-profile data leakage |
> | **Rate Limiting** | Protects against memory flooding from misbehaving agents |
> | **SSRF Protection** | Webhook dispatcher validates URLs to prevent SSRF attacks |
> | **HNSW-Accelerated Graphs** | Faster knowledge graph construction at scale |
> | **Hybrid Search Engine** | Combined semantic + FTS5 + graph retrieval |
>
> **v2.5 included:** Real-time event stream, WAL-mode concurrent writes, agent tracking, memory provenance, 28 API endpoints.
>
> **Upgrade:** `npm install -g superlocalmemory@latest`
>
> [[Universal Architecture|Universal-Architecture]] | [[Roadmap|Roadmap]] | [Full Changelog](https://github.com/varun369/SuperLocalMemoryV2/blob/main/CHANGELOG.md)

---

## 🎯 What is SuperLocalMemory?

SuperLocalMemory is an **intelligent ai memory system** that makes AI assistants like Claude, Cursor, and other mcp-server compatible tools remember everything about you and your projects. Created by **Varun Pratap Bhardwaj**, this local-first memory solution is the perfect alternative to Mem0 and Zep. Unlike cloud-based alternatives like Mem0 ($50+/mo) or Zep ($50/mo), SuperLocalMemory is:

- **100% Local** — Your data never leaves your machine
- **100% Free** — No usage limits, no credit systems, forever
- **100% Private** — GDPR/HIPAA compliant by default
- **Works Everywhere** — 17+ IDEs including Claude Desktop, Cursor IDE, Windsurf, VS Code, and more
- **MCP Protocol** — Native Model Context Protocol integration for 17+ IDEs

---

## Key Features in Action

### 🎯 Real-Time Memory Operations
![Live Events](https://superlocalmemory.com/assets/gifs/event-stream.gif)

### 🔍 Intelligent Search
![Hybrid Search](https://superlocalmemory.com/assets/gifs/dashboard-search.gif)

### 🕸️ Knowledge Graph
![Interactive Graph](https://superlocalmemory.com/assets/gifs/graph-interaction.gif)

### 💻 Simple CLI
![CLI Demo](https://superlocalmemory.com/assets/gifs/cli-demo.gif)

---

## Video Walkthroughs

- [Installation (1 min)](https://superlocalmemory.com/assets/videos/installation-walkthrough.mp4)
- [Quick Start (2 min)](https://superlocalmemory.com/assets/videos/quick-start.mp4)
- [Dashboard Tour (2 min)](https://superlocalmemory.com/assets/videos/dashboard-tour.mp4)

---

## 📚 Documentation

### Getting Started
| Guide | Description |
|-------|-------------|
| [[Installation]] | 5-minute setup for Mac, Linux, Windows |
| [[Quick-Start-Tutorial]] | Your first memory in 2 minutes |
| [[CLI-Cheatsheet]] | Copy-paste command reference |

### Core Concepts
| Guide | Description |
|-------|-------------|
| [[Universal-Architecture]] | Universal architecture with MCP integration |
| [[MCP-Integration]] | Model Context Protocol support for 17+ IDEs |
| [[Universal-Skills]] | 7 agent skills and slash-commands system |
| [[Knowledge-Graph-Guide]] | Auto-discovery of relationships |
| [[Pattern-Learning-Explained]] | How it learns your coding style |
| [[Multi-Profile-Workflows]] | Separate contexts for work/personal/clients |

### Reference
| Guide | Description |
|-------|-------------|
| [[CLI-Cheatsheet]] | Quick command reference |
| [[Python-API]] | Programmatic access |
| [[Configuration]] | Customization options |
| [CHANGELOG](https://github.com/varun369/SuperLocalMemoryV2/blob/main/CHANGELOG.md) | Version history and release notes |

### Comparisons
| Guide | Description |
|-------|-------------|
| [[Comparison-Deep-Dive]] | Detailed comparison with Mem0, Zep, Personal.AI |
| [[Why-Local-Matters]] | Privacy, GDPR, and local-first benefits |

### Community
| Guide | Description |
|-------|-------------|
| [[FAQ]] | Frequently asked questions |
| [[Roadmap]] | Version history and planned features |
| [Troubleshooting](https://github.com/varun369/SuperLocalMemoryV2/blob/main/docs/MCP-TROUBLESHOOTING.md) | Common issues and solutions |
| [Contributing](https://github.com/varun369/SuperLocalMemoryV2/blob/main/CONTRIBUTING.md) | How to contribute |
| [Issues](https://github.com/varun369/SuperLocalMemoryV2/issues) | Report bugs or request features |

---

## 🏆 Why SuperLocalMemory?

### The Problem

Every Claude session:
```
You: "Remember that auth bug we fixed?"
Claude: "I don't have access to previous conversations..."
You: *sighs and re-explains everything*
```

### The Solution

```bash
# Save once
superlocalmemoryv2-remember "Fixed auth bug - JWT tokens expiring too fast"

# Recall forever
superlocalmemoryv2-recall "auth bug"
# ✓ Found: "Fixed auth bug - JWT tokens expiring too fast"
```

---

## 🆚 Alternative to Mem0 and Zep

| Feature | Mem0 | Zep | Personal.AI | **SuperLocalMemory** |
|---------|------|-----|-------------|---------------------|
| **Price** | Usage-based | $50/mo | $33/mo | **$0 forever** |
| **Local-First** | ❌ Cloud | ❌ Cloud | ❌ Cloud | **✅ 100%** |
| **IDE Support** | Limited | 1-2 | None | **✅ 17+ IDEs** |
| **Universal Architecture** | ❌ | ❌ | ❌ | **✅ MCP + Skills + CLI** |
| **MCP Integration** | ❌ | ❌ | ❌ | **✅ Native** |
| **Pattern Learning** | ❌ | ❌ | Partial | **✅ Full** |
| **Knowledge Graphs** | ✅ | ✅ | ❌ | **✅ Leiden Clustering** |
| **Zero Setup** | ❌ | ❌ | ❌ | **✅ 5-min install** |

**SuperLocalMemory is the ONLY solution with universal IDE support, full local operation, and zero cost.** Created by **Varun Pratap Bhardwaj** as an open-source alternative to expensive cloud services.

---

## 🚀 Quick Install

```bash
# Mac/Linux
git clone https://github.com/varun369/SuperLocalMemoryV2.git
cd SuperLocalMemoryV2 && ./install.sh

# Windows
git clone https://github.com/varun369/SuperLocalMemoryV2.git
cd SuperLocalMemoryV2; .\install.ps1
```

**That's it.** No Docker. No API keys. No cloud accounts.

[[Full installation guide →|Installation]]

---

## 🔬 Built on Research

SuperLocalMemory implements cutting-edge 2026 research:

| Research | Source | Implementation |
|----------|--------|----------------|
| **PageIndex** | VectifyAI (Zhang et al., Sep 2025) | Hierarchical memory indexing |
| **GraphRAG** | Microsoft (Edge et al., 2024, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) | Knowledge graph clustering |
| **MemoryBank** | Zhong et al., AAAI 2024 ([arXiv:2305.10250](https://arxiv.org/abs/2305.10250)) | Long-term memory for LLM agents |
| **MACLA** | Forouzandeh et al., Dec 2025 ([arXiv:2512.18950](https://arxiv.org/abs/2512.18950)) | Multi-agent collaborative learning |
| **Hindsight** | Latimer et al., Dec 2025 ([arXiv:2512.12818](https://arxiv.org/abs/2512.12818)) | Retrospective identity pattern learning |
| **A-RAG** | [arXiv:2602.03442](https://arxiv.org/abs/2602.03442) | Multi-level retrieval |

> **Note:** PageIndex attribution corrected from "Meta AI" to VectifyAI (Mingtian Zhang et al., Sep 2025). "xMemory (Stanford)" was a non-existent paper reference and has been replaced with three published research papers: MemoryBank, MACLA, and Hindsight.

**The only open-source implementation combining all these research approaches.**

---

## 👨‍💻 Creator

**Created by Varun Pratap Bhardwaj** — Solution Architect & Original Creator

[![GitHub](https://img.shields.io/badge/GitHub-@varun369-181717?style=flat-square&logo=github)](https://github.com/varun369)

Building open-source tools that make AI assistants actually useful for developers. SuperLocalMemory v2.3.0 brings universal integration to 17+ IDEs while maintaining 100% local-first privacy.

---

## 💖 Support the Project

If SuperLocalMemory saves you time:

- ⭐ [Star on GitHub](https://github.com/varun369/SuperLocalMemoryV2) — helps others discover it
- ☕ [Buy Me a Coffee](https://buymeacoffee.com/varunpratah) — fuel development
- 💸 [PayPal](https://paypal.me/varunpratapbhardwaj) — direct support
- 💖 [GitHub Sponsors](https://github.com/sponsors/varun369) — recurring support

---

<p align="center">
  <strong>100% local. 100% private. 100% yours.</strong><br/>
  <em>Created by Varun Pratap Bhardwaj</em>
</p>

---

**🌐 Official Website:** [superlocalmemory.com](https://superlocalmemory.com/)
**📦 NPM Package:** [npmjs.com/package/superlocalmemory](https://www.npmjs.com/package/superlocalmemory)
**📖 Documentation:** [GitHub Wiki](https://github.com/varun369/SuperLocalMemoryV2/wiki)
**💬 Support:** [GitHub Issues](https://github.com/varun369/SuperLocalMemoryV2/issues)
