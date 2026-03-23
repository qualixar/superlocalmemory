# Active Memory (V3.1)

SuperLocalMemory V3.1 transforms the memory system from a passive database into an **active intelligence layer** that learns from your usage patterns and improves over time — at zero token cost.

## How It Works

Every time you recall a memory, the system collects learning signals:

```
Recall → Feedback Signal → Co-Retrieval Graph → Confidence Boost → Adaptive Ranking
```

No LLM tokens are spent. All learning happens through mathematical signals computed locally.

## Three Learning Phases

| Phase | Signals Needed | What Changes |
|-------|---------------|-------------|
| **1. Baseline** | 0-19 | Standard cross-encoder ranking |
| **2. Rule-Based** | 20+ | Heuristic boosts: recency, access frequency, trust score |
| **3. ML Model** | 200+ | LightGBM model trained on YOUR specific usage patterns |

Phase transitions are automatic. Each recall generates ~5 signals (one per returned fact). Typical users reach Phase 2 in a day of normal work and Phase 3 within a week.

## Four Learning Signals

### 1. Co-Retrieval
When memories are retrieved together repeatedly, they form implicit connections. The system learns that these memories are related — even if they don't share keywords.

### 2. Confidence Lifecycle
- **Accessed facts get boosted** (+0.02 per recall, capped at 1.0)
- **Unused facts decay** (-0.001/day after 7 days of no access, floor at 0.1)
- This creates a natural "memory importance" ranking without manual curation.

### 3. Channel Performance
SLM uses 4 retrieval channels (semantic, BM25, entity graph, temporal). The system tracks which channel produces the best results for different query types and adjusts channel weights accordingly.

### 4. Entropy Gap
When new content arrives, the system measures how "surprising" it is relative to existing memories. High-entropy content (genuinely new information) gets prioritized for deeper indexing.

## Auto-Capture

SLM can automatically detect and store decisions, bug fixes, and preferences from your conversations:

```bash
slm observe "We decided to use PostgreSQL because of JSONB support"
# Auto-captured: decision (confidence: 0.75)
```

Detection patterns:
- **Decisions**: "decided", "chose", "switched to", "using X because"
- **Bug fixes**: "fixed", "root cause was", "resolved by"
- **Preferences**: "always use", "prefer", "convention is"

## Auto-Recall (Session Context)

At the start of every session, the system can automatically inject relevant context:

```bash
slm session-context  # Returns top-10 relevant memories for current project
```

### Claude Code Hooks (Invisible Integration)

```bash
slm hooks install  # One-time setup
```

This installs a Claude Code hook that auto-injects memory context at the start of every session. The developer never types a command — context appears automatically.

## MCP Tools

Three new MCP tools for AI assistants:

| Tool | Purpose | When to Call |
|------|---------|-------------|
| `session_init` | Get project context | Once at session start |
| `observe` | Auto-capture content | After decisions, bug fixes, preferences |
| `report_feedback` | Explicit learning signal | When a recalled memory was useful/not useful |

## Sleep-Time Consolidation

A background maintenance process runs periodically:
- Decays confidence on unused memories
- Deduplicates near-identical facts
- Generates behavioral patterns from accumulated data
- Auto-retrains the ML ranker when enough signals accumulate

Trigger manually: `slm consolidate` or via the dashboard's Learning tab.

## Dashboard

The Learning tab shows:
- **Signal count** and phase progression (0 → 20 → 200)
- **Tech preferences** learned from your memories
- **Temporal patterns** (when you work on what)
- **Channel performance** (which retrieval channel works best)

The Behavioral tab shows:
- **Learned patterns** with confidence scores
- **Outcome tracking** (success/failure/partial)
- **Cross-project pattern transfer**

## Competitive Advantage

| System | Learning Cost | Learning Method |
|--------|-------------|----------------|
| Mem0 | LLM call per operation | Cloud extraction |
| Zep | LLM call per operation | Temporal KG |
| Letta | LLM call per operation | Agent self-writing |
| **SLM V3.1** | **$0 (zero tokens)** | **Mathematical signals** |

SLM is the only memory system that learns without spending tokens.

---

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)
