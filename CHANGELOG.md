# Changelog

All notable changes to SuperLocalMemory V2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## Author

**Varun Pratap Bhardwaj**
*Solution Architect*

SuperLocalMemory V2 - Intelligent local memory system for AI coding assistants.

---

## [2.8.2] - 2026-03-04

### Fixed
- Windows compatibility for repository cloning (#7)
- Updated test assertions for v2.8 behavioral feature dimensions

---

## [2.8.0] - 2026-02-26

**Release Type:** Major Feature Release — "Memory That Manages Itself"

SuperLocalMemory now manages its own memory lifecycle, learns from action outcomes, and provides enterprise-grade compliance — all 100% locally on your machine.

### Added
- **Memory Lifecycle Management** — Memories automatically organize themselves over time based on usage patterns, keeping your memory system fast and relevant
- **Behavioral Learning** — The system learns what works by tracking action outcomes, extracting success patterns, and transferring knowledge across projects
- **Enterprise Compliance** — Full access control, immutable audit trails, and retention policy management for GDPR, HIPAA, and EU AI Act
- **6 New MCP Tools** — `report_outcome`, `get_lifecycle_status`, `set_retention_policy`, `compact_memories`, `get_behavioral_patterns`, `audit_trail`
- **Improved Search** — Lifecycle-aware recall that automatically promotes relevant memories and filters stale ones
- **Performance Optimized** — Real-time lifecycle management and access control

### Changed
- Enhanced ranking algorithm with additional signals for improved relevance
- Improved search ranking using multiple relevance factors
- Search results include lifecycle state information

### Fixed
- Configurable storage limits prevent unbounded memory growth

---

## [2.7.6] - 2026-02-22

### Improved
- Documentation organization and navigation

---

## [2.7.4] - 2026-02-16

### Added
- Per-profile learning — each profile learns its own preferences independently
- Thumbs up/down and pin feedback on memory cards
- Learning data management in Settings (backup + reset)
- "What We Learned" summary card in Learning tab

### Improved
- Smarter learning from your natural usage patterns
- Recall results improve automatically over time
- Privacy notice for all learning features
- All dashboard tabs refresh on profile switch

---

## [2.7.3] - 2026-02-16

### Improved
- Enhanced trust scoring accuracy
- Improved search result relevance across all access methods
- Better error handling for optional components

---

## [2.7.1] - 2026-02-16

### Added
- **Learning Dashboard Tab** — View your ranking phase, preferences, workflow patterns, and privacy controls
- **Learning API** — Endpoints for dashboard learning features
- **One-click Reset** — Reset all learning data directly from the dashboard

---

## [2.7.0] - 2026-02-16

**Release Type:** Major Feature Release — "Your AI Learns You"

SuperLocalMemory now learns your patterns, adapts to your workflow, and personalizes recall. All processing happens 100% locally — your behavioral data never leaves your machine.

### Added
- **Adaptive Learning System** — Detects your tech preferences, project context, and workflow patterns across all your projects
- **Personalized Recall** — Search results automatically re-ranked based on your learned preferences. Gets smarter over time.
- **Zero Cold-Start** — Personalization works from day 1 using your existing memory patterns
- **Multi-Channel Feedback** — Tell the system which memories were useful via MCP, CLI, or dashboard
- **Source Quality Scoring** — Learns which tools produce the most useful memories
- **Workflow Detection** — Recognizes your coding workflow sequences and adapts retrieval accordingly
- **Engagement Metrics** — Track memory system health locally with zero telemetry
- **Isolated Learning Data** — Behavioral data stored separately from memories. One-command erasure for full GDPR compliance.
- **3 New MCP Tools** — Feedback signal, pattern transparency, and user correction
- **2 New MCP Resources** — Learning status and engagement metrics
- **New CLI Commands** — Learning management, engagement tracking, pattern correction
- **New Skill** — View learned preferences in Claude Code and compatible tools
- **Auto Python Installation** — Installer now auto-detects and installs Python for new users

---

## [2.6.5] - 2026-02-16

### Added
- **Interactive Knowledge Graph** — Fully interactive visualization with zoom, pan, and click-to-explore
- **Mobile & Accessibility Support** — Touch gestures, keyboard navigation, and screen reader compatibility

---

## [2.6.0] - 2026-02-15

**Release Type:** Security Hardening & Scalability — "Battle-Tested"

### Added
- **Rate Limiting** — Protection against abuse with configurable thresholds
- **API Key Authentication** — Optional authentication for API access
- **CI Workflow** — Automated testing across multiple Python versions
- **Trust Enforcement** — Untrusted agents blocked from write and delete operations
- **Advanced Search Index** — Faster search at scale with graceful fallback
- **Hybrid Search** — Combined search across multiple retrieval methods
- **SSRF Protection** — Webhook URLs validated against malicious targets

### Improved
- Higher memory graph capacity with intelligent sampling
- Hardened profile isolation across all queries
- Bounded resource usage under high load
- Optimized index rebuilds for large databases
- Sanitized error messages — no internal details leaked
- Capped resource pools for stability

---

## [2.5.1] - 2026-02-13

**Release Type:** Framework Integration — "Plugged Into the Ecosystem"

### Added
- **LangChain Integration** — Persistent chat history for LangChain applications
- **LlamaIndex Integration** — Chat memory storage for LlamaIndex
- **Session Isolation** — Framework memories tagged separately from normal recall

---

## [2.5.0] - 2026-02-12

**Release Type:** Major Feature Release — "Your AI Memory Has a Heartbeat"

SuperLocalMemory transforms from passive storage to active coordination layer. Every memory operation now triggers real-time events.

### Added
- **Reliable Concurrent Access** — No more "database is locked" errors under multi-agent workloads
- **Real-Time Events** — Live event broadcasting across all connected tools
- **Subscriptions** — Durable and ephemeral event subscriptions with filters
- **Webhook Delivery** — HTTP notifications with automatic retry on failure
- **Agent Registry** — Track connected AI agents with protocol and activity monitoring
- **Memory Provenance** — Track who created or modified each memory, and from which tool
- **Trust Scoring** — Behavioral trust signals collected per agent
- **Dashboard: Live Events** — Real-time event stream with filters and stats
- **Dashboard: Agents** — Connected agents table with trust scores and protocol badges

### Improved
- Refactored core modules for reliability and performance
- Dashboard modernized with modular architecture

---

## [2.4.2] - 2026-02-11

### Fixed
- Profile isolation bug in dashboard — graph stats now filter by active profile

---

## [2.4.1] - 2026-02-11

### Added
- **Hierarchical Clustering** — Large knowledge clusters auto-subdivided for finer-grained topic discovery
- **Cluster Summaries** — Structured topic reports for every cluster in the knowledge graph

---

## [2.4.0] - 2026-02-11

**Release Type:** Profile System & Intelligence

### Added
- **Memory Profiles** — Single database, multiple profiles. Switch instantly from any IDE or CLI.
- **Auto-Backup** — Configurable automatic backups with retention policy
- **Confidence Scoring** — Statistical confidence tracking for learned patterns
- **Profile Management UI** — Create, switch, and delete profiles from the dashboard
- **Settings Tab** — Backup configuration, history, and profile management
- **Column Sorting** — Click headers to sort in Memories table

---

## [2.3.7] - 2026-02-09

### Added
- `--full` flag to show complete memory content without truncation
- Smart truncation for large memories

### Fixed
- CLI `get` command now retrieves memories correctly

---

## [2.3.5] - 2026-02-09

### Added
- **ChatGPT Connector** — Search and fetch memories from ChatGPT via MCP
- **Streamable HTTP Transport** — Additional transport option for MCP connections
- **Dashboard Enhancements** — Memory detail modal, dark mode, export, search score visualization

### Fixed
- Security improvement in dashboard event handling

---

## [2.3.0] - 2026-02-08

**Release Type:** Universal Integration

SuperLocalMemory now works across 16+ IDEs and CLI tools.

### Added
- **Auto-Configuration** — Automatic setup for Cursor, Windsurf, Claude Desktop, Continue.dev, Codex, Copilot, Gemini, JetBrains
- **Universal CLI** — `slm` command works in any terminal
- **Skills Installer** — One-command setup for supported editors
- **Tool Annotations** — Read-only, destructive, and open-world hints for all MCP tools

---

## [2.2.0] - 2026-02-07

**Release Type:** Feature Release — Advanced Search

### Added
- **Advanced Search** — Faster, more accurate search with multiple retrieval strategies
- **Query Optimization** — Spell correction, query expansion, and technical term preservation
- **Search Caching** — Frequently-used queries return near-instantly
- **Combined Search** — Results fused from multiple search methods for better relevance
- **Fast Vector Search** — Sub-10ms search at scale (optional)
- **Local Embeddings** — Semantic search with GPU acceleration (optional)
- **Modular Installation** — Install only what you need: core, UI, search, or everything

---

## [2.1.0-universal] - 2026-02-07

**Release Type:** Major Feature Release — Universal Integration

### Added
- **6 Universal Skills** — remember, recall, list-recent, status, build-graph, switch-profile
- **MCP Server** — Native IDE integration with tools, resources, and prompts
- **Attribution Protection** — Multi-layer protection ensuring proper credit
- **11+ IDE Support** — Cursor, Windsurf, Claude Desktop, Continue.dev, Cody, Aider, ChatGPT, Perplexity, Zed, OpenCode, Antigravity

---

## [2.0.0] - 2026-02-05

### Initial Release — Complete Rewrite

SuperLocalMemory V2 represents a complete architectural rewrite with intelligent knowledge graphs, pattern learning, and enhanced organization.

### Added
- **4-Layer Architecture** — Storage, Hierarchical Index, Knowledge Graph, Pattern Learning
- **Automatic Entity Extraction** — Discovers key topics and concepts from your memories
- **Intelligent Clustering** — Automatic thematic grouping of related memories
- **Pattern Learning** — Tracks your preferences across frameworks, languages, architecture, security, and coding style
- **Storage Optimization** — Progressive compression reduces storage by up to 96%
- **Profile Management** — Multi-profile support with isolated data

---

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **MAJOR:** Breaking changes (e.g., 2.0.0 → 3.0.0)
- **MINOR:** New features (backward compatible, e.g., 2.0.0 → 2.1.0)
- **PATCH:** Bug fixes (backward compatible, e.g., 2.1.0 → 2.1.1)

**Current Version:** v2.8.0
**Website:** [superlocalmemory.com](https://superlocalmemory.com)
**npm:** `npm install -g superlocalmemory`

---

## License

SuperLocalMemory V2 is released under the [MIT License](LICENSE).

---

**100% local. 100% private. 100% yours.**
