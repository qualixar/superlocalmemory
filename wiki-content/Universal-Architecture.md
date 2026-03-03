# Universal Architecture

**SuperLocalMemory is built in 10 layers.** Each layer adds a specific capability, and each builds on the one below it. You can use the system at any layer — from simple CLI commands all the way up to a live visual dashboard with real-time events and multi-agent collaboration.

**Keywords:** universal architecture, system design, 10-layer, local-first, multi-tool, MCP, knowledge graph, hybrid search, dashboard, adaptive learning

---

> **[View Interactive Architecture Diagram](https://superlocalmemory.com/architecture)** — Explore all 10 layers, how they connect, and what each one does for you.

---

## Overview

The architecture has one core principle: **every layer enhances the one below it, and none replaces it.** Remove the visualization layer and the search still works. Remove the search layer and the storage still works. This makes the system resilient, upgradable, and easy to extend.

All layers share a single database on your machine — no sync, no duplication, no conflicts.

---

## The 10 Layers

| Layer | Name | What It Does For You |
|-------|------|---------------------|
| 9 | Visualization | Web dashboard — search, browse, explore, and manage everything visually |
| 8 | Hybrid Search | Three-signal retrieval that finds what you mean, not just what you typed |
| 7 | Universal Access | Works with 17+ tools simultaneously — all reading from the same database |
| 6 | MCP Integration | Seamless connection to Claude, Cursor, Windsurf, Perplexity, and more |
| 5 | Skills | Slash commands that work in any AI coding assistant |
| 4½ | Adaptive Learning | The memory system learns your preferences and improves results over time |
| 4 | Pattern Learning | Learns who you are as a developer — your tools, your style, your defaults |
| 3 | Knowledge Graph | Automatically connects related memories into a navigable map |
| 2 | Hierarchical Index | Fast navigation — finds context in milliseconds without scanning everything |
| 1 | Core Storage | A single SQLite database on your machine — private, durable, and portable |

---

## Layer-by-Layer Breakdown

### Layer 1 — Core Storage

Your memories live in a single file on your machine: `~/.claude-memory/memory.db`. Nothing goes to the cloud. Nothing is shared without your knowledge.

The database supports full-text search out of the box, automatic deduplication (the same memory is never stored twice), and importance scoring so you can prioritize what matters. Storage is compact — a well-used database with thousands of memories stays under a few megabytes.

**What this gives you:** A permanent, private, portable record of everything your AI tools have learned about your work.

### Layer 2 — Hierarchical Index

Raw storage is a flat list. The hierarchical index organizes that list into a navigable tree — grouped by project, topic, and context. When you search for something, the system doesn't scan every memory; it navigates the tree and finds the right branch.

**What this gives you:** Fast lookups even as your memory grows. Finding context takes milliseconds, not seconds.

### Layer 3 — Knowledge Graph

The knowledge graph discovers relationships between memories you never explicitly created. Save a memory about JWT tokens, another about session handling, and another about OAuth flows — the graph automatically clusters them together under "Authentication & Security."

When you search for "authentication," the graph finds not just memories that contain the word, but all memories in the authentication cluster, including ones that never used that exact term.

**What this gives you:** Hidden connections surfaced automatically. Search results that understand context, not just keywords.

```
Example graph output after saving 47 memories:

Cluster: "Authentication & Security" (12 memories)
  — JWT tokens, OAuth flows, session management, CSRF protection

Cluster: "React Components" (9 memories)
  — State management, lifecycle, props, component patterns

Cluster: "Database Operations" (7 memories)
  — Queries, indexing, migrations, connection pooling
```

[[Full guide: Knowledge Graph →|Knowledge-Graph-Guide]]

### Layer 4 — Pattern Learning

As you save memories over time, the system learns your preferences: which frameworks you reach for, how you approach testing, whether you default to REST or GraphQL, what languages you write in. Each preference comes with a confidence score that grows stronger as the pattern repeats.

This profile is available to inject into any AI session, so Claude or Cursor can make better suggestions from the first message — without you having to re-explain your stack every time.

**What this gives you:** An AI that already knows your defaults. No more "I prefer TypeScript, actually" corrections.

```
Example learned profile:

Frameworks:   React (73% confidence), Node.js (61%)
API style:    REST over GraphQL (81%)
Testing:      Jest preferred (65%)
Language:     TypeScript over JavaScript (69%)
```

[[Full guide: Pattern Learning →|Pattern-Learning-Explained]]

### Layer 4½ — Adaptive Learning (v2.7+)

This layer watches how you actually use your memories — which ones you click, which searches you run immediately after a recall, how long you spend reading a result — and uses those signals to improve future rankings. No thumbs-up buttons required.

The more you use the system, the better it gets at surfacing the right memory at the right moment.

**What this gives you:** Search results that get smarter over time, tuned to your actual behavior.

[[Full guide: Adaptive Learning →|Upgrading-to-v2.7]]

### Layer 5 — Skills

Skills are slash commands that work inside AI coding assistants — Claude Code, Continue.dev, Cody, and others. You don't need to leave your editor to save or recall memories.

| Skill | What It Does |
|-------|-------------|
| `/superlocalmemoryv2-remember` | Save a memory with optional tags and importance |
| `/superlocalmemoryv2-recall` | Search your memories |
| `/superlocalmemoryv2-list` | View recent memories |
| `/superlocalmemoryv2-status` | Check system health |
| `/superlocalmemoryv2-profile` | Switch active profile |
| `/superlocalmemoryv2-reset` | Reset the current session |
| `/superlocalmemoryv2-learning` | View learned patterns |

[[Full guide: Universal Skills →|Universal-Skills]]

### Layer 6 — MCP Integration

MCP (Model Context Protocol) is how modern AI tools communicate with external services. SuperLocalMemory runs a local MCP server that Cursor, Windsurf, Claude Desktop, Perplexity, and others connect to automatically.

Once connected, your AI tools can save and recall memories without any manual steps. When you tell Claude "remember this for later," it calls the MCP server directly. When Cursor needs context about a past decision, it searches through the same server.

The installer automatically detects and configures supported tools — you don't need to touch a config file manually.

**What this gives you:** Memory that works inside your AI tools, not alongside them.

[[Full guide: MCP Integration →|MCP-Integration]]

### Layer 7 — Universal Access

Any tool that speaks MCP, any terminal running the CLI, any script hitting the REST API — all of them talk to the same database. There is no "primary" access method. They are all equal.

**Four ways in:**
1. **MCP Protocol** — IDE integration (Claude Desktop, Cursor, Windsurf, Perplexity, and 13+ more)
2. **Universal Skills** — Slash commands inside AI assistants
3. **CLI** — Terminal commands (`slm remember`, `slm recall`, `slm list-recent`)
4. **REST API + Dashboard** — Web interface on port 8765

Adding a memory from the CLI shows up immediately in the dashboard. Recalling from Cursor uses the same index as searching from the dashboard. No sync delay, no duplication.

**What this gives you:** One memory system, reachable from everywhere.

### Layer 8 — Hybrid Search

Most search systems use a single strategy. SuperLocalMemory uses three simultaneously and combines the results.

| Strategy | Best At |
|----------|---------|
| Semantic search | Finding conceptually related content ("authentication patterns" finds JWT, OAuth, sessions) |
| Full-text search | Finding exact phrases ("expires after 24 hours" returns exact matches) |
| Graph-enhanced search | Finding everything in a cluster ("security" returns all members of the Security cluster) |

In hybrid mode (the default), all three run in parallel, results are normalized and merged, and you get the best of all three in a single ranked list.

**What this gives you:** Search that actually works — even when you can't remember the exact words you used when saving.

### Layer 9 — Visualization

The web dashboard gives you a visual interface for everything the system knows. No more text-only searches or reading through long lists.

**Five views:**
- **Memories** — Browse, filter, and manage all your memories
- **Knowledge Graph** — Interactive map of how your memories connect
- **Clusters** — Explore auto-discovered topic groups
- **Timeline** — Chronological view with importance color-coding
- **Live Events** — Real-time stream of every operation across all your tools
- **Agents** — See which AI tools are connected and what they've been doing
- **Patterns** — Your learned preferences and confidence scores
- **Settings** — Profile management, backups, configuration

**Launch:**
```bash
python ~/.claude-memory/ui_server.py
# Opens at http://localhost:8765
```

[[Full guide: Visualization Dashboard →|Visualization-Dashboard]]
[[Interactive graph guide: Using-Interactive-Graph →|Using-Interactive-Graph]]

---

## How the Layers Work Together

Here is what happens when you save a single memory:

```
You type a decision into Cursor
        |
        v
Layer 6 — MCP server receives it
        |
        v
Layer 1 — Stored in SQLite, hashed to prevent duplicates
        |
        v
Layer 2 — Indexed in the tree under the right project/context
        |
        v
Layer 3 — (On next graph build) Connected to related memories
        |
        v
Layer 4 — (Over time) Pattern frequency updated
        |
        v
Layer 7 — Available to every other tool immediately
        |
        v
Layer 8 — Searchable via semantic, full-text, and graph strategies
        |
        v
Layer 9 — Visible in the dashboard, included in Live Events stream
```

The whole chain completes in under 10 milliseconds for the immediate steps (storage, indexing, availability). Graph and pattern updates happen on rebuild, not on every write — so they don't add latency to your workflow.

---

## Two Databases

SuperLocalMemory uses two separate database files:

| Database | Location | What It Contains |
|----------|----------|-----------------|
| `memory.db` | `~/.claude-memory/memory.db` | Your memories — what you know |
| `learning.db` | `~/.claude-memory/learning.db` | Your preferences — how you work |

They are fully independent. You can delete `learning.db` to reset all learned preferences without losing a single memory. You can back up `memory.db` and restore it on another machine while leaving the learning data behind.

Both are backed up automatically. Both are local only.

---

## Performance at a Glance

| Operation | Time |
|-----------|------|
| Save a memory | < 10ms |
| Search (any strategy) | 10–80ms depending on corpus size |
| Dashboard load | < 500ms for 1,000 memories |
| Graph build | ~10 seconds for 1,000 memories (one-time, not per-save) |

[[Full benchmark data: Performance Benchmarks →|Performance-Benchmarks]]

---

## Related Pages

- [[Installation →|Installation]] — Set up in 5 minutes
- [[Quick Start Tutorial →|Quick-Start-Tutorial]] — First memory in 2 minutes
- [[MCP Integration →|MCP-Integration]] — Connect your AI tools
- [[Knowledge Graph Guide →|Knowledge-Graph-Guide]] — How clustering works
- [[Visualization Dashboard →|Visualization-Dashboard]] — Full dashboard guide
- [[Real-Time Event System →|Real-Time-Event-System]] — Live events and agent tracking
- [[Pattern Learning Explained →|Pattern-Learning-Explained]] — How the system learns your preferences
- [[Universal Skills →|Universal-Skills]] — Slash commands for AI assistants
- [[Why Local Matters →|Why-Local-Matters]] — Privacy, performance, and portability

---

[[← Back to Home|Home]]

---

**Created by Varun Pratap Bhardwaj**
*Solution Architect • SuperLocalMemory*

[GitHub](https://github.com/varun369/SuperLocalMemoryV2) • [Report Issue](https://github.com/varun369/SuperLocalMemoryV2/issues) • [Wiki](https://github.com/varun369/SuperLocalMemoryV2/wiki)
