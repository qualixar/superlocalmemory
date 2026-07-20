# Auto-Memory

SuperLocalMemory can automatically capture and recall memories without explicit commands. This page explains how auto-capture and auto-recall work.

## Auto-Recall

**What it does:** When you start a conversation in your IDE, SuperLocalMemory automatically retrieves relevant memories and injects them into the AI's context.

**How it works:**
1. Your IDE sends the conversation context to the MCP server (`slm mcp`)
2. SuperLocalMemory extracts key terms and entities from your message
3. It runs a full recall (five candidate producers: semantic, keyword/BM25, temporal, spreading-activation, Hopfield; plus entity-graph enhancement)
4. The top results are returned to your IDE
5. Your IDE includes these memories in the AI's system prompt

**What this means for you:** Your AI knows about your past work without you having to say "recall" or "remember." It just knows.

**Example:**
```
You: "Can you help me debug the auth service?"
AI: "Based on your previous work, the auth service uses JWT tokens with 24-hour expiry
     and refresh tokens lasting 30 days. Last time you debugged it, the issue was
     related to clock skew between services. Let me help..."
```

The AI referenced memories you stored days or weeks ago — automatically.

## Auto-Capture

**What it does:** SuperLocalMemory can automatically store important information from your conversations without you running `slm remember`.

**What gets captured:**
- Decisions ("We chose PostgreSQL for the user service")
- Bug fixes ("Fixed the race condition in the queue processor")
- Configuration details ("Deploy to us-east-1, use t3.large instances")
- Preferences ("Always use TypeScript strict mode")
- Project context ("The frontend uses React 19 with Server Components")

**What does NOT get captured:**
- Casual conversation
- Questions without answers
- Temporary debugging output
- Sensitive data marked as excluded

**How it works:**
1. Your IDE conversation flows through the MCP server
2. An entropy gate evaluates each message for information density
3. High-information messages are extracted into structured facts
4. Facts are stored with entities, timestamps, and graph connections
5. Low-information messages are ignored

## Memory is evidence, not instruction

Recalled content can include old prompts, imported text, or hostile
instructions. SLM renders it inside one reference-only untrusted evidence
boundary, with provenance, size budgets, recognized-secret redaction, and
forged-boundary neutralization. If the mandatory renderer fails, SLM omits the
memory context instead of falling back to raw text.

IDE instruction files contain only static product protocol. Dynamic memory is
retrieved through MCP at runtime rather than copied into a trusted Cursor,
Copilot, or Antigravity rules file.

## Configuration

Auto-capture and auto-recall behavior is configured through your IDE's MCP integration. The MCP server (`slm mcp`) handles both automatically when connected.

**To check what's stored:**

```bash
slm recall "recent"          # See recent memories
slm trace "recent"           # See with channel breakdown
slm health                   # Check overall system state
```

**To manually store something the auto-capture missed:**

```bash
slm remember "The critical detail that was missed"
```

**To delete something auto-captured incorrectly:**

```bash
slm forget "the incorrect memory"
```

## How Auto-Capture Decides What to Store

The entropy gate uses several signals:

1. **Information density** — Messages with specific facts, names, numbers, or decisions score higher
2. **Novelty** — Information that is not already stored scores higher
3. **Entity presence** — Messages mentioning people, projects, tools, or services score higher
4. **Temporal markers** — Messages with dates, deadlines, or time references are captured
5. **Decision language** — Phrases like "we decided," "the fix was," "going with" trigger capture

## Privacy Note

Auto-capture processes data submitted through configured MCP and hook surfaces;
enabled importers and connectors have their own data scope. Core storage uses
the configured local data root. Mode A does not require a cloud model provider
for core memory operations, but optional connectors, backup, proxy providers,
and dependency/model downloads can use the network. Mode C sends the
constructed model request—including selected memory evidence—to the configured
cloud provider. Review that provider's retention and privacy terms before use.

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
