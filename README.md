<p align="center">
  <img src="https://superlocalmemory.com/assets/logo-mark.png" alt="SuperLocalMemory" width="200"/>
</p>

<h1 align="center">SuperLocalMemory V3.6.18</h1>
<p align="center"><strong>Cache. Compress. Remember. Three surfaces — proxy, MCP tools, or skill. Every setup covered.</strong><br/>
<em>To the best of our knowledge, the only zero-cloud agent memory that beats Mem0's zero-LLM score on LoCoMo. Mode A: 74.8% vs Mem0 64.2% — no GPU, no API key, on CPU.</em></p>
<p align="center"><code>v3.6.18</code> — <strong>Plugin-native. Profile-aware. Distributed-ready.</strong><br/>
Proxy: <code>slm wrap claude</code> &nbsp;·&nbsp; MCP: add <code>slm_compress</code> to your config &nbsp;·&nbsp; Skill: zero-config</p>
<p align="center"><strong>3 published research papers</strong> (arXiv preprints + Zenodo-archived) · <a href="https://arxiv.org/abs/2603.02240">arXiv:2603.02240</a> · <a href="https://arxiv.org/abs/2603.14588">arXiv:2603.14588</a> · <a href="https://arxiv.org/abs/2604.04514">arXiv:2604.04514</a></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.14588"><img src="https://img.shields.io/badge/arXiv-2603.14588-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper"/></a>
  <a href="#three-surfaces-proxy--mcp-tools--skill"><img src="https://img.shields.io/badge/Proxy_|_MCP_|_Skill-22c55e?style=for-the-badge" alt="Three Surfaces: Proxy, MCP Tools, Skill"/></a>
  <a href="https://pypi.org/project/superlocalmemory/"><img src="https://img.shields.io/pypi/v/superlocalmemory?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.npmjs.com/package/superlocalmemory"><img src="https://img.shields.io/npm/v/superlocalmemory?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/></a>
  <a href="https://www.gnu.org/licenses/agpl-3.0"><img src="https://img.shields.io/badge/License-AGPL_v3-blue.svg?style=for-the-badge" alt="AGPL v3"/></a>
  <a href="#eu-ai-act-compliance"><img src="https://img.shields.io/badge/EU_AI_Act-Design_Compliant-brightgreen?style=for-the-badge" alt="EU AI Act Design Compliant"/></a>
  <a href="https://superlocalmemory.com"><img src="https://img.shields.io/badge/Web-superlocalmemory.com-ff6b35?style=for-the-badge" alt="Website"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/MCP-Native-blue?style=for-the-badge" alt="MCP Native"/></a>
  <a href="#dual-interface-mcp--cli"><img src="https://img.shields.io/badge/CLI-Agent--Native-green?style=for-the-badge" alt="CLI Agent-Native"/></a>
  <a href="#multilingual-embedding-support"><img src="https://img.shields.io/badge/Multilingual-30%2B_Languages-ff69b4?style=for-the-badge" alt="Multilingual 30+ Languages"/></a>
</p>

---

## Why SuperLocalMemory?

Every hosted AI memory platform — Mem0 Cloud, Zep Cloud, Letta Cloud, EverMemOS Cloud — sends your data to cloud LLMs by default. Self-hosted variants exist but require Docker, a separate graph DB, or Ollama config, and most default to OpenAI until you flip env vars. After **August 2, 2026**, any of those cloud paths becomes a compliance question under the EU AI Act.

SuperLocalMemory V3 uses **mathematics instead of cloud compute** — differential geometry, algebraic topology, and stochastic analysis replace the work other systems need LLMs to do. Local-first out of the box. No Docker. No graph DB. No API keys. CPU-only.

**Benchmark results** (evaluated on [LoCoMo](https://arxiv.org/abs/2402.09714), the standard long-conversation memory benchmark, published April 2026):

| System | Score | Config | Cloud LLM required? | Open Source | Source |
|:-------|:-----:|:-------|:-------------------:|:-----------:|:-------|
| EverMemOS | 93.05% | Cloud (proprietary) | Yes | Core only | [evermind.ai](https://evermind.ai/) (Feb 2026) |
| Hindsight (LoComo10) | 92.0% | Cloud | Yes | No | [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io) (Apr 2026) |
| Mem0 (token-efficient) | 91.6% | Hybrid (Cohere/OpenAI) | Yes | Partial | [mem0.ai blog](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm) (Apr 16 2026) |
| **SLM V3 Mode C** | **87.7%** | Local + optional LLM | Optional (Ollama OK) | **Yes (AGPL-3.0)** | In-house, [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) |
| Zep v3 Cloud | 85.2% | Cloud | Yes | Community deprecated | [getzep.com](https://www.getzep.com/) |
| **SLM V3 Mode A** | **74.8%** | **Local, CPU-only, zero-LLM** | **No** | **Yes (AGPL-3.0)** | In-house, [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) |
| Mem0 (zero-retrieval-LLM) | 64.2% | Local baseline | No | Partial | Mem0 paper, zero-LLM row |

> **How to read this table.** Scores from different papers use different LoCoMo splits, judge models, and prompt variants. We do NOT claim these numbers are apples-to-apples across rows. Rows marked "In-house" were run by us; cited rows link to the vendor's public source and date. The only apples-to-apples comparison is **Mode A 74.8% vs Mem0 zero-retrieval-LLM 64.2%** (+10.6pp) — both are zero-LLM configurations. Mem0's 91.6% and EverMemOS's 93.05% use cloud LLMs; Mode C uses a local LLM (Ollama).

**What Mode A is:** CPU-only, SQLite-only, zero-LLM retrieval on published LoCoMo questions. To the best of our knowledge it is the only publicly-released local-first memory that clears Mem0's zero-LLM baseline on this benchmark. If another fully-local system hits similar numbers, please open an issue so we can update this table.

Mathematical layers contribute **+12.7 percentage points** average across 6 conversations (n=832 questions), with up to **+19.9pp on the most challenging dialogues**.

---

## Quick Start

```bash
# npm (recommended)
npm install -g superlocalmemory
slm setup       # Choose mode (A/B/C)
slm doctor      # Verify everything is working
```

```bash
# pip
pip install superlocalmemory
slm setup
slm doctor
```

```bash
# First use
slm remember "Alice works at Google as a Staff Engineer"
slm recall "What does Alice do?"
slm status
```

```bash
# Wrap your agent — starts proxy + sets environment + launches agent
slm wrap claude
# Your first repeat prompt → CACHE HIT → $0.00
# See savings: slm optimize savings --since 1
```

**Upgrading:** `pip install -U superlocalmemory && slm restart && slm doctor` — migration is automatic, no data loss.

---

## Three Pillars

### Memory

<a id="dual-interface-mcp--cli"></a>

Five-channel hybrid retrieval: Semantic (Fisher-Rao geodesic distance) + BM25 + Entity Graph + Temporal + Hopfield (associative/partial-query completion). RRF fusion, cross-encoder reranking, adaptive LightGBM ranking. All data stays local — SQLite + optional LanceDB/CozoDB.

Three mathematical contributions replace cloud LLM dependency:

1. **Fisher-Rao Retrieval Metric** — similarity scoring from the Fisher information structure of diagonal Gaussian families. To the best of our knowledge, the first public application of information geometry to agent memory retrieval.
2. **Sheaf Cohomology for Consistency** — algebraic topology detects contradictions via coboundary norms on the knowledge graph.
3. **Riemannian Langevin Lifecycle** — memory positions evolve on the Poincare ball; neglected memories self-archive, no hardcoded thresholds.

Auto-capture hooks (`slm hooks install`) fire only on real signals — topic pivot, web call, file edit — never on a timer. Fail-open, <10ms p99 hot path.

**Multi-scope memory (v3.6.15, opt-in):** keep memories `personal` (default), `shared` with named profiles, or `global` across the machine. Off by default — recall only ever returns your own facts until you turn sharing on, per call or in config. See **[docs/shared-memory.md](docs/shared-memory.md)**.

<a id="multilingual-embedding-support"></a>

**Multilingual:** plug in any OpenAI-compatible embedding endpoint — Ollama, vLLM, LiteLLM, `bge-m3`, `multilingual-e5`, `Qwen3-Embedding`. The math layer is language-agnostic; 30+ languages work at full retrieval quality. No cloud dependency, no code changes.

### Cache + Compress

<a id="three-surfaces-proxy--mcp-tools--skill"></a>

One engine, three ways in — choose the surface that fits your setup:

| Surface | How you use it | Requires proxy? | Window effect | Cache scope |
|---------|---------------|:---------------:|:-------------:|-------------|
| **A — Proxy** | `slm wrap claude` or `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` | **Yes** | Shrinks | Full-turn cache — every call |
| **B — MCP tools** | Add 5 tools to MCP config; call `slm_compress`, `slm_cache_set/get` | **No** | **Preserved (1M)** | Results you explicitly route through SLM |
| **C — Skill** | Copy `skills/slm-optimize/SKILL.md` → `~/.claude/skills/` | **No** | **Preserved (1M)** | Auto-applied by the agent per skill rules |

**The hard constraint:** The primary Claude conversation turn cannot be cached without a proxy. The MCP/skill path caches results you explicitly route through SLM (tool outputs, file reads, sub-model calls) — without a proxy the main conversation turn is not intercepted.

**How to choose:**
- Metered API (pay-per-token), want every call cached → **Proxy (A)**
- Pro/Max/Team subscription or any plan where you won't run a proxy → **MCP tools (B)** or **Skill (C)**
- Zero configuration → **Skill (C)**: install once, auto-compresses CLAUDE.md and large outputs
- Agent-controlled caching of repeated file reads → **MCP tools (B)**

**Cache:** exact-match SQLite lookup (SHA-256, zero false hits) + vCache-gated semantic (opt-in). **100% cost saved on a hit** (input + output tokens).

**Compress:** safe mode = lossless normalization (JSON/code/tool outputs, 60-95% fewer tokens); aggressive mode = LLMLingua-2 prose only (opt-in). CCR stores originals for byte-exact reversal. Anthropic 90% / OpenAI 50% prefix-cache discount alignment included. [CITATION-NEEDED-ONLINE: live provider prefix-cache discount rates]

**Savings dashboard:** `slm optimize savings --since 7` — live USD/INR/tokens saved. Hot-reload config, fail-open.

### Mesh

<a id="multi-machine-mesh-coordination"></a>

Run SLM on multiple machines and have agents coordinate as one team — no external broker, no Docker. HTTP-based sync every 30s, mDNS discovery (`SLM_MESH_DISCOVERY=on`), graceful offline queue.

```bash
# Machine A (broker)
export SLM_MESH_HOST=192.168.1.100
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init

# Machine B (client)
export SLM_MESH_PEER_URL=http://192.168.1.100:8765
export SLM_MESH_SHARED_SECRET=my-secret-key
slm init
```

8 mesh MCP tools: `mesh_peers`, `mesh_send`, `mesh_broadcast`, `mesh_project`, `mesh_inbox`, `mesh_pending`, `mesh_state`, `mesh_lock`.

Full docs: [docs/multi-machine.md](docs/multi-machine.md) · [docs/distributed-deployment.md](docs/distributed-deployment.md)

---

## Install Paths

| Path | Command | When |
|:-----|:--------|:-----|
| **npm** (recommended) | `npm install -g superlocalmemory` | Node 14+, installs Python deps automatically |
| **pip** | `pip install superlocalmemory` | Python 3.11+, direct install |
| **Claude Code Plugin** (WP-06) | `/plugin marketplace add qualixar/superlocalmemory` then `/plugin install superlocalmemory@qualixar` | Self-bootstraps venv, isolated SLM_DATA_DIR, additive — 14-tool core. Ships the skills/agents/hooks/commands |
| **Portable / IDE connect** (WP-08) | `slm connect <ide> [--here]` | Wire any IDE without reinstalling; `slm connect claude-code` → plugin pointer |

After any install path: `slm setup` → `slm doctor` → `slm warmup` (optional, pre-downloads ~500MB embedding model).

| Component | Size | When |
|:----------|:-----|:-----|
| Core libraries (numpy, scipy, networkx) | ~50MB | During install |
| Dashboard & MCP server (fastapi, uvicorn) | ~20MB | During install |
| Learning engine (lightgbm) | ~10MB | During install |
| Search engine (sentence-transformers, torch) | ~200MB | During install |
| Embedding model (nomic-embed-text-v1.5, 768d) | ~500MB | First use or `slm warmup` |
| **Mode B** requires [Ollama](https://ollama.com) + a model (`ollama pull llama3.2`) | ~2GB | Manual |

---

## MCP + Profiles

SLM supports two MCP transports:

**HTTP (recommended, v3.6.7+):**
```json
{ "mcpServers": { "superlocalmemory": { "type": "http", "url": "http://127.0.0.1:8765/mcp/" } } }
```
Or: `claude mcp add --transport http superlocalmemory http://127.0.0.1:8765/mcp/`

**stdio (universal fallback):**
```json
{ "mcpServers": { "superlocalmemory": { "command": "slm", "args": ["mcp"] } } }
```

### MCP Profiles (WP-01)

Control tool surface via `SLM_MCP_PROFILE`:

| Profile | Tools | Use case |
|:--------|:-----:|:---------|
| `core14` (default) | 14 | Memory core — `remember`, `recall`, `forget`, `session_init`, + mesh |
| `mesh8` | 8 | Mesh-only — multi-machine coordination |
| `full38` | 38 | Core + optimize + evolution + trust |
| `power50` | 50 | Full38 + admin + ingestion + compliance |
| `whole81` | 81 | Every tool (`SLM_MCP_ALL_TOOLS=1`) |

**Precedence:** `ALL` > `TOOLS` > `PROFILE` > `default`

```bash
export SLM_MCP_PROFILE=full38   # or core14 / mesh8 / power50 / whole81
slm mcp
```

Per-IDE configs available for Claude Code, Cursor, Windsurf, VS Code Copilot, Continue, Gemini CLI, JetBrains, Zed, and more (15 configs in `ide/configs/`). See [docs/ide-setup.md](docs/ide-setup.md).

---

## Claude Code Plugin

Install directly in Claude Code — no system-level npm/pip needed. This is how you
get the **skills, agents, hooks, commands, and rules** (the MCP server is
bootstrapped automatically). It is a two-step flow — add the marketplace once,
then install:

```bash
# 1. Add the Qualixar marketplace (one-time — the repo IS the marketplace)
/plugin marketplace add qualixar/superlocalmemory

# 2. Install the plugin
/plugin install superlocalmemory@qualixar
```

- Self-bootstraps a Python venv, installs all deps in an isolated `SLM_DATA_DIR`
- Registers the 14-tool core MCP surface (`core14` profile by default)
- Ships the SLM skills / agents / hooks / commands / rules
- Additive — does not replace an existing SLM install
- `slm connect claude-code` detects an existing plugin install and links them

> **Plugin vs `pip`/`npm`:** `pip install superlocalmemory` / `npm i -g superlocalmemory`
> give you the `slm` CLI + the MCP server (the *tools*). The **skills/agents/hooks/
> commands** come only through the plugin above. Use the plugin for Claude Code; use
> pip/npm for the CLI or other IDEs.

To update later: `/plugin marketplace update qualixar` then `/plugin install superlocalmemory@qualixar`.

---

## Modes + EU AI Act

<a id="eu-ai-act-compliance"></a>

| Mode | What | Cloud? | EU AI Act | Best For |
|:----:|:-----|:------:|:---------:|:---------|
| **A** | Local Guardian | **None** | **Compliant** | Privacy-first, air-gapped, enterprise |
| **B** | Smart Local | Local only (Ollama) | Compliant | Better answers, data stays local |
| **C** | Full Power | Cloud LLM | Partial | Maximum accuracy, research |

```bash
slm mode a   # Zero-cloud (default)
slm mode b   # Local Ollama
slm mode c   # Cloud LLM
```

**Mode A** is, to the best of our knowledge, the only publicly-released agent memory that runs with zero cloud calls while clearing Mem0's published LoCoMo score. All data stays on your device. No API keys. No GPU. Runs on 2 vCPUs + 4GB RAM.

The EU AI Act (Regulation 2024/1689) takes full effect **August 2, 2026**.

| Requirement | Mode A | Mode B | Mode C |
|:------------|:------:|:------:|:------:|
| Data sovereignty (Art. 10) | **Pass** | **Pass** | Requires DPA |
| Right to erasure (GDPR Art. 17) | **Pass** | **Pass** | **Pass** |
| Transparency (Art. 13) | **Pass** | **Pass** | **Pass** |
| No network calls during memory ops | **Yes** | **Yes** | No |

To the best of our knowledge, no existing agent memory system addresses EU AI Act compliance by architectural design. Modes A and B pass all checks — no personal data leaves the device during any memory operation.

Built-in compliance tools: GDPR Article 15/17 export + complete erasure, tamper-proof SHA-256 audit chain, data provenance tracking, ABAC policy enforcement. See [docs/compliance.md](docs/compliance.md).

---

## Advanced

| Topic | Link |
|:------|:-----|
| Full optimize docs | [docs/optimize-overview.md](docs/optimize-overview.md) · [docs/optimize-cli.md](docs/optimize-cli.md) · [docs/optimize-config.md](docs/optimize-config.md) |
| Distributed deployment | [docs/distributed-deployment.md](docs/distributed-deployment.md) |
| Multi-machine mesh | [docs/multi-machine.md](docs/multi-machine.md) |
| Auto-memory hooks | [docs/auto-memory.md](docs/auto-memory.md) |
| Architecture + math | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| CLI reference | [docs/cli-reference.md](docs/cli-reference.md) |
| MCP tools reference | [docs/mcp-tools.md](docs/mcp-tools.md) |
| Getting started | [docs/getting-started.md](docs/getting-started.md) |
| IDE setup (15 configs) | [docs/ide-setup.md](docs/ide-setup.md) |
| Skill evolution | [docs/skill-evolution.md](docs/skill-evolution.md) |
| V2 migration | [docs/migration-from-v2.md](docs/migration-from-v2.md) |
| Configuration | [docs/configuration.md](docs/configuration.md) |
| Wiki | [github.com/qualixar/superlocalmemory/wiki](https://github.com/qualixar/superlocalmemory/wiki) |

**Web dashboard:**
```bash
slm dashboard    # Opens at http://localhost:8765
```
17-tab sidebar with Knowledge Graph (Sigma.js WebGL, community detection), Health Monitor, Entity Explorer, Mesh Peers, Ingestion Status, Privacy blur mode. Cross-platform: macOS + Windows + Linux.

**Release history:**

| Version | Codename | Key Features |
|---|---|---|
| **v3.6.17** | Community | 8 contributor PRs (observability events, marker-bounded adapter writes, daemon port discovery, anthropic `api_base`, OpenMP workers, atomic-write rehash, `_jl` sentinel, LFS pointer); dashboard-feedback fix (#53/#59); env-tunable SQLite knobs + idle backoff; remote LLM test-probe (#40) |
| **v3.6.16** | Docs | Corrected Claude Code plugin install — adds the required `/plugin marketplace add` step; clarifies plugin vs pip/npm delivery |
| **v3.6.15** | Multi-scope | **Opt-in [shared memory](docs/shared-memory.md)** (personal/shared/global, off by default), default-deny scope at every read path, recall scope-race fix, contributor PRs #42/#43/#44, fixes #46–#49 |
| **v3.6.14** | Plugin-native | Claude Code Plugin (WP-06), MCP profiles (WP-01), IDE connect (WP-08), asset consolidation, UI polish (WP-12) |
| **v3.6.x** | Optimize Everywhere / Distributed-ready | Three surfaces (proxy/MCP/skill), `SLM_REMOTE=1` LAN mode, remote dashboard, custom LLM endpoints |
| **v3.5.0** | Scale-Ready | CozoDB/LanceDB, 6-channel recall <1s, Core Memory Block, context injection v2, score normalization |
| **v3.4.x** | Scale-Ready (foundation) | Tiered storage, graph pruning, Hopfield channel, LightGBM ranking, mDNS mesh discovery |
| **v3.3.x** | Foundation | BM25Plus, Fisher-Rao, sqlite-vec, RRF fusion, cross-encoder rerank. 3 published papers |

---

## Research Papers

SuperLocalMemory is backed by three published research papers (arXiv preprints + Zenodo DOIs). These are preprints — not conference-accepted or journal-published yet.

### Paper 3: The Living Brain (V3.3)
> **SuperLocalMemory V3.3: The Living Brain — Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2604.04514](https://arxiv.org/abs/2604.04514) · [Zenodo DOI: 10.5281/zenodo.19435120](https://zenodo.org/records/19435120)

### Paper 2: Information-Geometric Foundations (V3)
> **SuperLocalMemory V3: Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.14588](https://arxiv.org/abs/2603.14588) · [Zenodo DOI: 10.5281/zenodo.19038659](https://zenodo.org/records/19038659)

### Paper 1: Trust & Behavioral Foundations (V2)
> **SuperLocalMemory: A Structured Local Memory Architecture for Persistent AI Agent Context**
> Varun Pratap Bhardwaj (2026)
> [arXiv:2603.02240](https://arxiv.org/abs/2603.02240) · [Zenodo DOI: 10.5281/zenodo.18709670](https://zenodo.org/records/18709670)

### Cite This Work

```bibtex
@article{bhardwaj2026slmv33,
  title={SuperLocalMemory V3.3: The Living Brain — Biologically-Inspired
         Forgetting, Cognitive Quantization, and Multi-Channel Retrieval
         for Zero-LLM Agent Memory Systems},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2604.04514},
  year={2026},
  url={https://arxiv.org/abs/2604.04514}
}

@article{bhardwaj2026slmv3,
  title={Information-Geometric Foundations for Zero-LLM Enterprise Agent Memory},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.14588},
  year={2026}
}

@article{bhardwaj2026slm,
  title={A Structured Local Memory Architecture for Persistent AI Agent Context},
  author={Bhardwaj, Varun Pratap},
  journal={arXiv preprint arXiv:2603.02240},
  year={2026}
}
```

---

## Support / License / Qualixar

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. [Wiki](https://github.com/qualixar/superlocalmemory/wiki) for detailed documentation.

GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE).

For commercial licensing (closed-source, proprietary, or hosted use), see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or contact varun.pratap.bhardwaj@gmail.com.

Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar.

Part of [Qualixar](https://qualixar.com) · Author: [Varun Pratap Bhardwaj](https://varunpratap.com)

### Acknowledgments

- **[Everything Claude Code (ECC)](https://github.com/affaan-m/everything-claude-code)** — SLM's skill observation patterns were inspired by ECC's continuous learning architecture. SLM supports direct ingestion of ECC observations via `slm ingest --source ecc`. We recommend ECC for Claude Code users who want the deepest learning experience alongside SLM.
- **[HKUDS/OpenSpace](https://github.com/HKUDS/OpenSpace)** — The skill evolution research in SLM draws from the EvoSkills co-evolutionary verification concepts (arXiv:2604.01687). We adopted their 3-trigger evolution system and anti-loop guard patterns.

### Qualixar AI Agent Reliability Platform

Qualixar is building the open-source infrastructure for AI agent reliability engineering. Seven products, one coherent platform:

| Product | Purpose | Install |
|---------|---------|---------|
| **[SuperLocalMemory](https://github.com/qualixar/superlocalmemory)** | Persistent memory + learning | `npm install -g superlocalmemory` |
| **[Qualixar OS](https://github.com/qualixar/qualixar-os)** | Universal agent runtime | `npx qualixar-os` |
| **[SLM Mesh](https://github.com/qualixar/slm-mesh)** | P2P coordination across sessions | `npm i slm-mesh` |
| **[SLM MCP Hub](https://github.com/qualixar/slm-mcp-hub)** | Federate 430+ MCP tools | `pip install slm-mcp-hub` |
| **[AgentAssay](https://github.com/qualixar/agentassay)** | Token-efficient agent testing | `pip install agentassay` |
| **[AgentAssert](https://github.com/qualixar/agentassert-abc)** | Behavioral contracts + drift detection | `pip install agentassert-abc` |
| **[SkillFortify](https://github.com/qualixar/skillfortify)** | Formal verification for agent skills | `pip install skillfortify` |

**Zero cloud dependency. Local-first. EU AI Act compliant.**

Start here → **[qualixar.com](https://qualixar.com)** · [All papers on Qualixar HuggingFace](https://huggingface.co/Qualixar)

---

<p align="center">
  <sub>Built with mathematical rigor. Not in the race — here to help everyone build better AI memory systems.</sub>
</p>

---

## Star This Project

If this project solves a real problem for you, **please star the repo** — it helps other developers discover Qualixar and signals that the AI agent reliability community is growing.

[![Star History Chart](https://api.star-history.com/svg?repos=qualixar/superlocalmemory&type=Date)](https://star-history.com/#qualixar/superlocalmemory&Date)
