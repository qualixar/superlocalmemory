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
| **Architecture choices** | "We're going with a microservices approach for the payment system" |
| **Preferences** | "I prefer functional components over class components" |
| **Project context** | "The staging server is at 10.0.1.50, port 8080" |
| **People and roles** | "Sarah is the lead on the mobile team" |
| **Corrections** | "Actually, the deadline is March 20, not March 15" |

### What does NOT get captured

- Raw code blocks (too noisy, changes too fast)
- Casual conversation ("thanks", "sounds good")
- Repeated information already in memory
- Content filtered by the entropy gate (redundant or low-value)

### How the system decides what to capture

The ingestion pipeline scores each candidate memory on:

1. **Information value** — Does this add new knowledge not already stored?
2. **Specificity** — Is this a concrete fact or a vague statement?
3. **Reusability** — Is this likely to be useful in a future session?

Memories that score below the threshold are discarded. This keeps your database focused and retrieval quality high.

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
  "max_recall_results": 10,
  "recall_threshold": 0.3
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `max_recall_results` | `10` | Maximum memories injected per query |
| `recall_threshold` | `0.3` | Minimum relevance score (0.0 to 1.0). Lower = more memories, possibly less relevant. Higher = fewer but more precise. |

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

Auto-capture and auto-recall happen entirely within SLM on your machine. In Mode A, no data leaves your device at any point. In Mode C, recall queries are sent to your configured LLM provider, but the memories themselves remain local.

---

*SuperLocalMemory V3 — Copyright 2026 Varun Pratap Bhardwaj. AGPL-3.0-or-later. Part of Qualixar.*
