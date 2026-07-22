# Auto-Memory
> SuperLocalMemory V3 Documentation
> https://superlocalmemory.com | Part of Qualixar

SuperLocalMemory captures and recalls context automatically. Install it once, then forget about it — your AI assistant gets smarter over time without any manual effort.

---

## How Auto-Capture Works

When you work with an AI assistant that has SLM connected, certain types of information are automatically stored as memories:

| What gets captured | Example |
|-------------------|---------|
| **Decisions** | "Let's use WebSocket instead of SSE" |
| **Bug fixes** | "The crash was caused by a null pointer in the auth middleware" |
| **Preferences** | "I prefer functional components over class components" |

### What does NOT get captured

- Raw code blocks (too noisy, changes too fast)
- Casual conversation ("thanks", "sounds good")
- Repeated content within the configured observation debounce window
- Content that does not match the enabled decision, bug-fix, or preference admission rules

### How the system decides what to capture

Auto-capture admission is a lightweight rules step. It matches configured
decision, bug-fix, and preference patterns and returns a category, reason, and
confidence. Rejected observations are not acknowledged as stored.

Accepted observations are submitted durably to M018 before the caller receives
`captured: true`. This is separate from materialization: the operation first
becomes `queryable`, then enrichment advances it to `complete` or records a
retryable `failed` state. Entropy, consolidation, graph, temporal, provenance,
and projector work belong to materialization, not admission.

## How Auto-Recall Works

Before your AI assistant responds to a question, SLM automatically searches for relevant memories and injects them as context.

### The flow

```
You ask a question
    |
    v
SLM runs a recall query using your question as the search input
    |
    v
Relevant memories are injected into the AI's context window
    |
    v
The AI responds with awareness of your past decisions, preferences, and project context
```

### What this looks like in practice

**Without SLM:**
> You: "What database should I use for the new service?"
> AI: Generic advice about PostgreSQL vs MySQL vs MongoDB...

**With SLM:**
> You: "What database should I use for the new service?"
> AI: "Based on your previous decision to standardize on PostgreSQL 16 (stored March 5), and your preference for managed services on AWS (stored February 20), I'd recommend Amazon RDS for PostgreSQL. Your auth service and payment service already use PostgreSQL, so this keeps the stack consistent."

The AI did not "remember" this on its own. SLM injected the relevant memories before the AI generated its response.

### Memory is evidence, not instruction

Recalled content can include old prompts, imported text, or hostile
instructions. SLM therefore renders it inside one reference-only untrusted
evidence boundary, with provenance, size budgets, recognized-secret redaction,
and forged-boundary neutralization. If the mandatory renderer fails, SLM omits
the memory context instead of falling back to raw text.

IDE instruction files contain only static product protocol. SLM retrieves
dynamic memory through MCP at runtime rather than copying recalled text into a
trusted Cursor, Copilot, or Antigravity rules file.

## Configuration

### Toggle auto-capture and auto-recall

In `~/.superlocalmemory/config.json`:

```json
{
  "auto_capture": true,
  "auto_recall": true
}
```

Set either to `false` to disable. When disabled, you can still use `slm remember` and `slm recall` manually.

### Adjust recall sensitivity

```json
{
  "recall_threshold": 0.3
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `recall_threshold` | `0.3` | Minimum relevance score (0.0 to 1.0). Lower = more memories, possibly less relevant. Higher = fewer but more precise. |

> **Recall result count:** The default is 20 results per query (`CANONICAL_RECALL_LIMIT`). Override per-call with the `--limit N` flag (CLI) or the `limit` parameter (MCP `recall` tool). There is no config file key for this default.

### Adjust capture sensitivity

```json
{
  "capture_threshold": 0.5
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `capture_threshold` | `0.5` | Minimum information value to auto-capture. Lower = capture more. Higher = capture only high-value statements. |

## Manual Override

You always have full control:

```bash
# Explicitly store something
slm remember "The API rate limit on production is 500 req/min, staging is 100 req/min"

# Explicitly recall
slm recall "rate limits"

# Delete a memory
slm forget --id 42
```

Manual operations work regardless of auto-capture/auto-recall settings.

## Learning Over Time

SLM's adaptive learning system observes which memories are recalled frequently, which are marked helpful or outdated, and adjusts its behavior:

- **Frequently helpful memories** get higher ranking in future recalls
- **Memories marked "outdated"** are deprioritized or flagged for review
- **Usage patterns** inform what types of information to prioritize for capture

You can see what the system has learned:

```bash
slm patterns            # Show learned patterns
slm patterns correct 5  # Correct pattern #5 if it's wrong
```

## Privacy

Core auto-capture and recall storage use the configured local data root. Mode A
does not require a cloud model provider for core memory operations, but optional
connectors, cloud backup, proxy providers, dependency/model downloads, and
other explicitly enabled integrations can use the network. Mode C sends the
constructed model request—including selected memory evidence—to the configured
cloud provider. Review that provider's retention and privacy terms before use.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
