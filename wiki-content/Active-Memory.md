# Active Memory (V3.1)

This page describes the active-memory components introduced in V3.1. The
current release includes local feedback, outcome, co-retrieval, and adaptive
ranking machinery. It does not guarantee that ranking improves for every
corpus, and it does not treat result exposure as proof of relevance.

## How It Works

Recall can emit local telemetry that later feedback or outcomes can qualify:

```
Recall → Exposure telemetry → Explicit feedback/outcome → Adaptive ranking
```

The feedback/ranking path itself can operate locally. Provider-backed
extraction or other optional integrations have separate network and token
behavior.

## Three Learning Phases

| Phase | Signals Needed | What Changes |
|-------|---------------|-------------|
| **1. Baseline** | Insufficient labeled outcomes | Configured retrieval and reranking |
| **2. Rule-Based** | Local signals available | Declared heuristic ranking adjustments |
| **3. ML Model** | Training gate satisfied | Optional local learned ranker |

Signal counts and training readiness depend on actual use and explicit outcome
coverage. V3.7 publishes no time-to-training guarantee.

## Four Learning Signals

### 1. Co-Retrieval
When memories are retrieved together repeatedly, they form implicit connections. The system learns that these memories are related — even if they don't share keywords.

### 2. Lifecycle signals
Recency, access, trust, and lifecycle state can affect internal ranking utility.
They do not change `relevance_score` or `memory_confidence` into answer
probabilities. See [Score Contract v2](Retrieval-Score-Contract).

### 3. Channel Performance
Current candidate producers are dense semantic, BM25 lexical, temporal,
Hopfield associative, and spreading activation. Entity-graph information is a
post-fusion score enhancement in the current implementation.

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

This explicitly installs supported Claude Code hooks. The current package
installers do not install hooks or edit IDE configuration implicitly.

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

## Limitations

Adaptive ranking needs representative feedback and held-out evaluation. A
retrieved fact is only an exposure event until explicit feedback or a qualified
outcome exists. V3.7 does not publish a uniqueness claim, guaranteed learning
curve, or calibrated answer-confidence result for this subsystem.

---

Part of [Qualixar](https://qualixar.com) | Author: [Varun Pratap Bhardwaj](https://varunpratap.com)
