# Architecture Documentation

## SuperLocalMemory V2 - Technical Architecture

This document provides a comprehensive technical overview of SuperLocalMemory V2's architecture, design decisions, and implementation details.

---

## Table of Contents

- [System Overview](#system-overview)
- [Universal Integration](#universal-integration)
- [7-Layer Architecture](#7-layer-architecture)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Performance Characteristics](#performance-characteristics)
- [Security & Privacy](#security--privacy)
- [Scalability](#scalability)

---

## System Overview

SuperLocalMemory V2 is a **standalone, local-first, intelligent memory system** that transforms flat memory storage into a living knowledge graph with pattern learning capabilities.

Works independently or integrates with Claude CLI, GPT, local LLMs, or any AI assistant.

### Core Principles

1. **Local-First:** All data stored locally, zero external API calls
2. **Privacy-First:** No telemetry, no tracking, complete user control
3. **Intelligence:** Auto-discovery of relationships and patterns
4. **Modularity:** Independent layers with clear interfaces
5. **Performance:** Sub-second operations for typical workloads (< 500 memories)

### Architecture Philosophy

```
Simple Storage → Intelligent Organization → Adaptive Learning
     (SQLite)     (Graphs + Indexes)         (Pattern Recognition)
```

---

## Universal Integration

**Version 2.4.1** transforms SuperLocalMemory from Claude-Code-only to a universal memory system that works across 16+ IDEs and CLI tools with zero configuration.

### Three-Tier Access Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ACCESS METHODS                           │
├─────────────────────────────────────────────────────────────┤
│ TIER 1: MCP (Model Context Protocol)                       │
│  • Cursor IDE - Native AI tool access                      │
│  • Windsurf IDE - Built-in integration                     │
│  • Claude Desktop - Direct MCP support                     │
│  • Continue.dev - MCP context providers                    │
├─────────────────────────────────────────────────────────────┤
│ TIER 2: Skills & Commands                                  │
│  • Claude Code - /superlocalmemoryv2-*                     │
│  • Continue.dev - /slm-* slash commands                    │
│  • Cody - Custom commands via settings                     │
├─────────────────────────────────────────────────────────────┤
│ TIER 3: Universal CLI                                      │
│  • slm wrapper - Simple command syntax                     │
│  • aider-smart - Auto-context injection                    │
│  • Any terminal - Direct Python execution                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────────┐
         │ SuperLocalMemory   │
         │ Core (Unchanged)   │
         │ memory.db (SQLite) │
         └───────────────────┘
```

### MCP Server Implementation

**File:** `mcp_server.py` (450+ lines)

The MCP server provides native integration with modern AI tools:

**12 Tools:**
- `remember(content, tags, project)` - Save memories
- `recall(query, limit)` - Search memories
- `list_recent(limit)` - Recent memories
- `get_status()` - System status
- `build_graph()` - Build knowledge graph
- `switch_profile(name)` - Change profile
- `search(query)` - Search memories (OpenAI MCP spec for ChatGPT Connectors)
- `fetch(id)` - Fetch memory by ID (OpenAI MCP spec for ChatGPT Connectors)
- `backup_status()` - Auto-backup status
- `memory_used(memory_id, query, usefulness)` - Feedback for learning (v2.7)
- `get_learned_patterns(min_confidence, category)` - Retrieve learned patterns (v2.7)
- `correct_pattern(pattern_id, correct_value)` - Correct a learned pattern (v2.7)

**6 Resources:**
- `memory://recent/{limit}` - Recent memories feed
- `memory://stats` - System statistics
- `memory://graph/clusters` - Graph clusters
- `memory://patterns/identity` - Learned patterns
- `memory://learning/status` - Learning system status (v2.7)
- `memory://engagement` - Engagement metrics (v2.7)

**2 Prompts:**
- `coding_identity_prompt` - User's coding style
- `project_context_prompt` - Current project context

**Transport Support:**
- stdio (default) - For local IDE integration
- HTTP - For remote/network access
- streamable-http - For ChatGPT 2026+ Connectors

**Key Design:** Zero duplication - calls existing `memory_store_v2.py` functions

### Auto-Detection System

**File:** `install.sh` (enhanced)

Installation automatically detects and configures:

```bash
# Detect installed tools
[ -d "$HOME/Library/Application Support/Claude" ] && configure_claude_desktop
[ -d "$HOME/.cursor" ] && configure_cursor
[ -d "$HOME/.windsurf" ] && configure_windsurf
[ -d "$HOME/.continue" ] && configure_continue
```

**For each detected tool:**
1. Creates configuration from template
2. Substitutes {{INSTALL_DIR}} with actual path
3. Backs up existing configs
4. Installs new config
5. Reports success

**Zero manual configuration required.**

### Universal CLI Wrapper

**File:** `bin/slm`

Provides simple syntax for all operations:

```bash
slm remember "content"    # Calls superlocalmemoryv2-remember
slm recall "query"        # Calls superlocalmemoryv2-recall
slm list                  # Recent memories
slm status                # System health
slm profile list          # Profile management
slm graph build           # Knowledge graph
```

**Implementation:** Bash wrapper routing to existing CLI commands - no duplicate logic

### Backward Compatibility

**100% backward compatible:**
- All original commands work unchanged
- Database schema unchanged
- Profile system intact
- Skills system unchanged
- Zero breaking changes

**Upgrade path:**
1. Run `./install.sh`
2. Auto-detection configures new tools
3. Existing workflows continue working
4. New capabilities available immediately

### Integration Matrix

| Tool | Integration Method | Configuration |
|------|-------------------|---------------|
| Claude Code | Skills (unchanged) | Existing |
| Cursor | MCP stdio | Auto-detected |
| Windsurf | MCP stdio | Auto-detected |
| Claude Desktop | MCP stdio | Auto-detected |
| Continue.dev | MCP + Skills | Auto-detected |
| Cody | Custom commands | Auto-detected |
| Aider | Smart wrapper | Always available |
| Any terminal | slm wrapper | Always available |

### Design Decision: Additive Architecture

**Reasoning:**
- New files only - zero modifications to core src/
- Original commands unchanged
- Optional MCP layer on top
- Fallback to CLI if MCP unavailable
- Single source of truth (memory.db)

**Trade-offs:**
- Slight code duplication in wrappers (bash + MCP)
- More installation complexity (auto-detection)
- Multiple access paths to same data

**Benefits:**
- Zero risk to existing users
- Instant rollback (remove new files)
- Easy testing (each tier independent)
- Universal adoption without migration

---

## 7-Layer Architecture

SuperLocalMemory V2 uses a hierarchical, additive architecture where each layer builds on the previous without replacing it. Version 2.3.0 introduced universal access across 16+ IDEs, and subsequent releases (through 2.4.1) added profiles, hierarchical clustering, and community summaries.

```
┌─────────────────────────────────────────────────────────────────┐
│                   SuperLocalMemory V2                           │
│                   7-Layer Universal Architecture                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Layer 7: Universal Access                                      │
│  Three access methods: MCP Protocol, Skills, CLI               │
│  All share one database (~/.claude-memory/memory.db)           │
│  Output: Unified access across 16+ IDEs and tools              │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 6: MCP Integration (mcp_server.py)                       │
│  12 tools, 6 resources, 2 prompts                              │
│  Auto-configured for Claude Desktop, Cursor, Windsurf, etc.   │
│  Output: Native AI tool access via Model Context Protocol      │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Skills Layer (skills/*)                               │
│  7 slash-command skills for Claude Code, Continue.dev, Cody    │
│  slm-remember, slm-recall, slm-status, slm-list-recent,       │
│  slm-build-graph, slm-switch-profile, slm-show-patterns       │
│  Output: Familiar /command interface across AI tools           │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Pattern Learning (Identity Profiles)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Frequency        │  │  Context         │                   │
│  │  Analysis         │  │  Analysis        │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Terminology      │  │  Confidence      │                   │
│  │  Learning         │  │  Scoring         │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  Output: Identity profiles with confidence scores              │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Knowledge Graph (GraphRAG)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  TF-IDF Entity   │  │  Leiden          │                   │
│  │  Extraction      │  │  Clustering      │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Auto-naming     │  │  Relationship    │                   │
│  │  Clusters        │  │  Discovery       │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  Output: Clusters of related memories, entity relationships    │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Hierarchical Index (PageIndex-inspired)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Tree Structure  │  │  Parent-Child    │                   │
│  │  Management      │  │  Relationships   │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Fast Navigation │  │  Contextual      │                   │
│  │  Paths           │  │  Grouping        │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  Output: Hierarchical organization, breadcrumb navigation      │
└─────────────────────────────────────────────────────────────────┘
                            ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Raw Storage (SQLite Foundation)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Full-Text       │  │  Vector          │                   │
│  │  Search (FTS5)   │  │  Embeddings      │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │  Tags &          │  │  Compression     │                   │
│  │  Metadata        │  │  Archives        │                   │
│  └──────────────────┘  └──────────────────┘                   │
│                                                                 │
│  Output: Persistent storage, fast search, metadata             │
└─────────────────────────────────────────────────────────────────┘
```

**Optional Add-on Layers (v2.2.0):**
- **Advanced Search:** BM25 engine, HNSW vector index, hybrid search
- **Visualization:** Web dashboard with timeline, graph view, statistics

---

## Component Details

### Layer 1: Raw Storage (SQLite)

**Responsibility:** Persistent storage, basic search, metadata management

**Components:**

1. **memory_store_v2.py** - Core CRUD operations
   - Add, update, delete memories
   - Tag management
   - Search interface

2. **SQLite Schema:**
   ```sql
   -- Main memories table
   CREATE TABLE memories (
       id INTEGER PRIMARY KEY,
       content TEXT NOT NULL,
       timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
       tags TEXT,  -- JSON array
       metadata TEXT,  -- JSON object
       tier INTEGER DEFAULT 1,  -- Compression tier
       parent_id INTEGER,  -- For tree structure
       FOREIGN KEY (parent_id) REFERENCES memories(id)
   );

   -- Full-text search index
   CREATE VIRTUAL TABLE memories_fts USING fts5(
       content, tags,
       content=memories,
       content_rowid=id
   );

   -- Graph tables
   CREATE TABLE graph_clusters (
       id INTEGER PRIMARY KEY,
       name TEXT,
       description TEXT
   );

   CREATE TABLE graph_cluster_members (
       cluster_id INTEGER,
       memory_id INTEGER,
       FOREIGN KEY (cluster_id) REFERENCES graph_clusters(id),
       FOREIGN KEY (memory_id) REFERENCES memories(id)
   );

   -- Pattern learning tables
   CREATE TABLE learned_patterns (
       id INTEGER PRIMARY KEY,
       category TEXT,
       subcategory TEXT,
       pattern_text TEXT,
       confidence REAL,
       frequency INTEGER,
       last_seen DATETIME
   );
   ```

3. **Compression System** (memory_compression.py)
   - Progressive summarization
   - Tier-based compression (Tier 1→2→3)
   - Cold storage archival

**Performance:**
- Full-text search: Sub-11ms median for typical databases (see wiki Performance Benchmarks for measured data)
- Insert: <10ms
- Tag search: ~30ms

---

### Layer 2: Hierarchical Index (PageIndex)

**Responsibility:** Tree-based organization, contextual grouping

**Components:**

1. **tree_manager.py** - Tree structure operations
   - Parent-child relationship management
   - Breadcrumb generation
   - Subtree queries

**Key Features:**
- Fast navigation through related memories
- Contextual grouping (e.g., "Project Alpha" → subtasks)
- Efficient ancestor/descendant queries

**Schema Integration:**
```sql
-- Parent-child relationships
parent_id INTEGER FOREIGN KEY → memories(id)

-- Query: Get all children
SELECT * FROM memories WHERE parent_id = ?

-- Query: Get breadcrumb path
WITH RECURSIVE ancestors AS (...)
```

**Use Cases:**
- Project hierarchies
- Conversation threading
- Multi-step task tracking

---

### Layer 3: Knowledge Graph (GraphRAG)

**Responsibility:** Relationship discovery, auto-clustering, entity extraction

**Components:**

1. **graph_engine.py** - Graph construction and queries
   - TF-IDF entity extraction
   - Leiden clustering algorithm
   - Cluster naming via LLM patterns
   - Relationship discovery

**Algorithm Pipeline:**

```
Memories → TF-IDF → Entity Extraction → Similarity Matrix → Leiden Clustering → Named Clusters
```

**Step-by-Step:**

1. **Entity Extraction (TF-IDF):**
   ```python
   # Extract top terms from each memory
   tfidf_vectorizer = TfidfVectorizer(
       max_features=20,
       stop_words='english',
       ngram_range=(1, 2)
   )
   entities = tfidf_vectorizer.fit_transform(memories)
   ```

2. **Similarity Calculation:**
   ```python
   # Cosine similarity between memories
   similarity_matrix = cosine_similarity(entities)
   ```

3. **Leiden Clustering:**
   ```python
   # Community detection in similarity graph
   import leidenalg
   communities = leidenalg.find_partition(graph, resolution=1.0)
   ```

4. **Auto-Naming:**
   ```python
   # Extract common terms from cluster members
   cluster_name = most_frequent_terms(cluster_memories)
   # Example: "Authentication & Security"
   ```

**Performance:**
- Build time: <30ms (20 memories)
- Build time: ~2s (100 memories)
- Query time: <10ms (related memories)

**Output:**
- Clusters with auto-generated names
- Entity relationships
- Related memory suggestions

---

### Layer 4: Pattern Learning (xMemory-inspired)

**Responsibility:** Learn user preferences, coding style, terminology

**Components:**

1. **pattern_learner.py** - Pattern extraction and scoring
   - Frequency analysis
   - Context analysis
   - Confidence scoring
   - Identity profile generation

**Analysis Categories:**

```python
CATEGORIES = {
    "preferences": {
        "framework": ["React", "Vue", "Angular", ...],
        "language": ["Python", "JavaScript", "Go", ...],
        "architecture": ["microservices", "monolith", ...],
        "security": ["JWT", "OAuth", "API keys", ...]
    },
    "style": {
        "code_preferences": ["functional", "OOP", ...],
        "priorities": ["performance", "readability", ...]
    },
    "terminology": {
        "common_terms": [...],
        "technical_vocabulary": [...]
    }
}
```

**Confidence Scoring:**

```python
confidence = frequency / total_memories

# Example:
# "React" appears in 6/10 memories → 60% confidence
# "Vue" appears in 1/10 memories → 10% confidence
```

**Output Format:**

```json
{
  "preferences": {
    "framework": {
      "React": 0.60,
      "Angular": 0.20
    },
    "language": {
      "Python": 0.50,
      "JavaScript": 0.40
    }
  },
  "style": {
    "priorities": {
      "performance": 0.53,
      "readability": 0.30
    }
  }
}
```

**Use Cases:**
- Personalized AI assistant context
- Style guide generation
- Skill tracking
- Learning trajectory analysis

---

## Data Flow

### Write Path (Adding Memory)

```
User Input
    ↓
memory_store_v2.py: add()
    ↓
SQLite: INSERT into memories table
    ↓
FTS5: Update full-text index
    ↓
[Optional] tree_manager: Set parent_id
    ↓
[Async] graph_engine: Rebuild trigger
    ↓
[Async] pattern_learner: Update trigger
```

### Read Path (Search)

```
User Query
    ↓
memory_store_v2.py: search()
    ↓
SQLite FTS5: Full-text search
    ↓
Results + Metadata
    ↓
[Optional] graph_engine: Enrich with related memories
    ↓
[Optional] pattern_learner: Add context suggestions
    ↓
User Output
```

### Graph Build Path

```
Trigger: Manual or scheduled
    ↓
graph_engine.py: build()
    ↓
Fetch all memories from SQLite
    ↓
TF-IDF: Extract entities
    ↓
Cosine Similarity: Calculate relationships
    ↓
Leiden Algorithm: Detect communities
    ↓
Auto-naming: Generate cluster names
    ↓
SQLite: INSERT into graph_clusters, graph_cluster_members
    ↓
Statistics output
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Database | SQLite | 3.35+ | Persistent storage |
| Search | FTS5 | Built-in | Full-text search |
| Language | Python | 3.8+ | Core implementation |
| Clustering | Leiden Algorithm | Custom impl | Community detection |
| Entity Extraction | TF-IDF | scikit-learn-style | Term importance |
| CLI | Bash scripts | - | User commands |

### Python Standard Library Only

**No external dependencies required for core features:**
- `sqlite3` - Database
- `json` - Configuration, metadata
- `re` - Text processing
- `datetime` - Timestamps
- `hashlib` - ID generation
- `collections` - Data structures

**Optional (for advanced features):**
- `scikit-learn` - Full TF-IDF implementation
- `leidenalg` - Advanced clustering

**Fallback implementations provided** for systems without optional dependencies.

---

## Design Decisions

### Decision 1: SQLite Over NoSQL

**Reasoning:**
- Local-first: No server setup required
- ACID transactions: Data integrity guaranteed
- FTS5: Built-in full-text search
- Widely available: Pre-installed on most systems
- Zero configuration: Works out of the box

**Trade-offs:**
- Single-writer: Not suitable for high-concurrency scenarios
- Limited to local disk: No distributed access (feature, not bug)

---

### Decision 2: 7-Layer Additive Architecture

**Reasoning:**
- Modularity: Each layer can be disabled independently
- Progressive enhancement: System works even if upper layers fail
- Clear separation: Easy to understand and maintain
- Testability: Each layer can be tested in isolation
- Universal access: Layers 5-7 enable multi-IDE support without changing core

**Trade-offs:**
- Slight redundancy: Some data duplicated across layers
- Build time: Graph/pattern layers require periodic rebuilds
- More installation complexity: Auto-detection for 16+ IDEs

---

### Decision 3: Local-Only, No External APIs

**Reasoning:**
- Privacy: User data never leaves machine
- Reliability: No internet dependency
- Speed: No network latency
- Cost: No API bills

**Trade-offs:**
- No cloud sync: Multi-device requires manual export/import
- Limited ML: No access to large models (by design)

---

### Decision 4: TF-IDF + Leiden Clustering

**Reasoning:**
- TF-IDF: Fast, lightweight, no training required
- Leiden: Better than Louvain, finds quality communities
- Deterministic: Same input = same output
- Scalable: O(n log n) complexity

**Alternatives considered:**
- Vector embeddings: Too heavy, requires models
- LDA: Slower, less interpretable
- Simple keyword matching: Too simplistic

---

### Decision 5: Confidence Scoring Over Binary Classification

**Reasoning:**
- Nuance: Captures uncertainty in patterns
- User control: Filter by confidence threshold
- Adaptability: Patterns evolve as memories grow
- Transparency: Users see why patterns are suggested

**Example:**
```
React: 60% confidence → High priority suggestion
Vue: 10% confidence → Low priority, exploratory
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Add memory | O(1) | <10ms |
| Search (FTS5) | O(log n) | ~11ms median (100 memories) |
| Graph build | O(n²) worst, O(n log n) avg | ~2s (100 memories) |
| Pattern update | O(n) | <2s (100 memories) |
| Find related | O(1) | <10ms |

### Space Complexity

| Component | Storage | Notes |
|-----------|---------|-------|
| Raw memories | ~1KB/memory | Typical text memory |
| FTS5 index | ~30% overhead | Full-text search index |
| Graph data | ~100 bytes/memory | Cluster memberships |
| Patterns | ~1KB total | Aggregate statistics |
| Compression (Tier 2) | 60% reduction | Summarized old memories |
| Compression (Tier 3) | 96% reduction | Archived to JSON |

**Example:** 500 memories ≈ 500KB raw + 150KB index + 50KB graph = ~700KB total

---

## Security & Privacy

### Threat Model

**Assumptions:**
- User has physical access to machine
- Local filesystem is trusted
- No network threats (local-only)

**Protections:**

1. **No External Data Leaks:**
   - Zero API calls
   - No telemetry
   - No auto-updates

2. **Data Integrity:**
   - SQLite ACID transactions
   - Automatic backups before resets

3. **Access Control:**
   - Relies on filesystem permissions
   - Standard Unix/macOS security model

**Not Protected Against:**
- Physical access to machine
- Root/admin access
- Filesystem malware

**Recommendation:** Use full-disk encryption for sensitive data.

---

## Scalability

### Current Limits (Tested)

| Memories | Build Time | Search Time (median) | Database Size |
|----------|-----------|----------------------|---------------|
| 100 | 0.28s | 10.6ms | ~150KB |
| 500 | ~5s | 65.2ms | ~700KB |
| 1,000 | 10.6s | 124.3ms | 1.50 MB |
| 5,000 | 277s | 1,172ms | ~6.8 MB |

### Scaling Strategies

**For 1000+ memories:**
1. **Profile Splitting:** Separate contexts (work, personal, learning)
2. **Compression:** Auto-archive old memories (Tier 3)
3. **Incremental Builds:** Graph updates instead of full rebuilds
4. **Selective Loading:** Load only active profile

**Future Optimizations:**
- Graph delta updates (vs. full rebuild)
- Lazy pattern loading
- Memory pagination for large searches

---

## Extension Points

### Adding New Layers

```python
# Layer 5 example: Predictive recommendations
class RecommendationEngine:
    def suggest_next_memory(self, context):
        # Use patterns + graph to predict what user might add next
        pass
```

### Adding New Pattern Categories

```python
# In pattern_learner.py
CATEGORIES["preferences"]["testing"] = [
    "pytest", "jest", "unittest", "TDD", "BDD"
]
```

### Custom Clustering Algorithms

```python
# In graph_engine.py
def custom_clustering(similarity_matrix):
    # Implement alternative clustering
    pass
```

---

## Conclusion

SuperLocalMemory V2's architecture provides:
- **Simplicity:** Standard tools (SQLite, Python stdlib)
- **Intelligence:** Auto-discovery of patterns and relationships
- **Privacy:** 100% local, zero external dependencies
- **Modularity:** Independent layers, clear interfaces
- **Performance:** Sub-second operations for typical workloads

**Design Philosophy:** Start simple (Layer 1), add intelligence progressively (Layers 2-4), enable universal access (Layers 5-7), maintain local-first principles throughout.

---

## Further Reading

- [Compression System](COMPRESSION-README.md)
- [CLI Commands Reference](CLI-COMMANDS-REFERENCE.md)
- [MCP Troubleshooting](MCP-TROUBLESHOOTING.md)
- [Universal Integration Guide](UNIVERSAL-INTEGRATION.md)
- [Wiki: Knowledge Graph Guide](https://github.com/varun369/SuperLocalMemoryV2/wiki/Knowledge-Graph-Guide)
- [Wiki: Pattern Learning](https://github.com/varun369/SuperLocalMemoryV2/wiki/Pattern-Learning-Explained)

---

## Author

**Varun Pratap Bhardwaj**
*Solution Architect*

SuperLocalMemory V2 - Intelligent local memory system for AI coding assistants.

---

**Questions about architecture?**

See [CONTRIBUTING.md](CONTRIBUTING.md) for development discussions or open an issue on GitHub.
