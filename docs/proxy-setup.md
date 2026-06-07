# Proxy Setup — All CLIs & Editors
> SuperLocalMemory V3.6 Documentation
> https://superlocalmemory.com | Part of Qualixar — AI Reliability Engineering

SLM Optimize Proxy intercepts every LLM API call and applies three cost-reduction levers — cache, compress, and align — before the call reaches the provider. This page shows exactly how to wire it into every supported CLI and editor.

---

## Prerequisites

```bash
pip install -U superlocalmemory  # Python
# or
npm i -g superlocalmemory         # Node.js / npm global

slm optimize on        # Enable the Optimize module
slm proxy              # Start the proxy on port 8765
slm optimize status    # Verify: Proxy: running on :8765
```

The proxy runs at `http://127.0.0.1:8765`. It supports three surfaces:

| Surface | Protocol | Path |
|---------|----------|------|
| **Anthropic Messages** | Anthropic Messages API | `/v1/messages` |
| **OpenAI Compatible** | OpenAI Chat Completions API | `/v1/chat/completions` |
| **Gemini Native** | Google Gemini generateContent API | `/v1beta/models/*` |
| **Gemini OpenAI-compat** | Gemini via OpenAI-format JSON | `/v1beta/openai/chat/completions` |

> **v3.6.3 — Cache and compression now fully available for Claude Code, Claude Desktop, and Codex CLI.**
> The tool-bypass guard that permanently blocked caching for tool-bearing requests has been removed.
> All three clients are now covered end-to-end: cache fires on every request, streaming responses are
> accumulated and stored post-stream, and compression runs on every new message batch.
> The Gemini native surface also gains full cache + compression support in this release.

---

## What Gets Cached

Cache fires for **all requests** — including those with tools. As of v3.6.3, the tool-bypass guard that previously blocked caching for Claude Code, Claude Desktop, and Codex CLI has been removed. The proxy caches the full response (including `tool_use` content blocks) and replays it as a proper SSE stream on the next identical call.

| Client | Tools present? | Cache fires? | Compress fires? |
|--------|---------------|:------------:|:---------------:|
| Python `anthropic` SDK (no tools) | No | ✓ Yes | ✓ Yes |
| Node.js `@anthropic-ai/sdk` (no tools) | No | ✓ Yes | ✓ Yes |
| OpenAI Python/Node SDK (no tools) | No | ✓ Yes | ✓ Yes |
| Cursor / Windsurf (depends on config) | Sometimes | ✓ Yes | ✓ Yes |
| **Claude Code CLI** | **Always** | **✓ Yes (v3.6.3)** | ✓ Yes |
| **Claude Desktop** | **Always** | **✓ Yes (v3.6.3)** | ✓ Yes |
| **Codex CLI** | **Always** | **✓ Yes (v3.6.3)** | ✓ Yes |
| AGY / Antigravity (Claude/OpenAI models) | Sometimes | ✓ Yes (if proxiable) | ✓ Yes |
| AGY / Antigravity (Gemini models) | Sometimes | ⚠ See AGY section | ⚠ See AGY section |
| Gemini CLI (native) | Rarely | ✓ Yes (v3.6.3) | ✓ Yes |
| Raw `curl` / scripts without tools | No | ✓ Yes | ✓ Yes |

> **How tool-use caching works:** The proxy accumulates the full SSE stream (including all `tool_use` blocks), assembles the complete JSON message, and stores it under the request hash. On a cache hit, it replays the stored JSON as a properly formed SSE stream — emitting `content_block_start` (type `tool_use`), `input_json_delta` chunks, and `message_stop` — exactly as the real API would. The client sees no difference.

---

## Claude Code CLI

**Recommended: persistent settings (one-time setup)**

```bash
# Option A — write ANTHROPIC_BASE_URL into ~/.claude/settings.json (permanent)
slm wrap claude --persistent

# Option B — launch Claude Code with proxy active for this session only
slm wrap claude
```

`slm wrap claude` sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` and launches Claude Code.
`--persistent` writes it permanently to `~/.claude/settings.json` so every Claude Code session goes through the proxy automatically, even when launched from a fresh terminal.

**Manual setup (if `slm wrap` is unavailable):**

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765"
  }
}
```

Restart Claude Code. Verify: `echo $ANTHROPIC_BASE_URL` inside a Claude Code bash tool — should print `http://127.0.0.1:8765`.

> **Note:** Claude Code uses streaming and always includes tools. As of v3.6.3, caching and compression are fully active for Claude Code — both tool-bearing and tool-free requests are cached. On a cache hit, the proxy replays the complete SSE stream (including tool_use blocks) without hitting Anthropic's API.

---

## Claude Desktop

Claude Desktop is a GUI macOS app. Environment variables must be set at the OS level so the app inherits them at launch.

**Method 1 — launchctl (recommended, takes effect at next login):**

```bash
launchctl setenv ANTHROPIC_BASE_URL "http://127.0.0.1:8765"
# Quit and relaunch Claude Desktop for the change to take effect.
```

**Method 2 — persistent via ~/.zshenv (applies to all apps launched from shell):**

```bash
echo 'export ANTHROPIC_BASE_URL="http://127.0.0.1:8765"' >> ~/.zshenv
# Requires a full logout/login or relaunch from Terminal.
```

> **Note:** Claude Desktop sets its own `ANTHROPIC_BASE_URL` at startup. If the app overrides your setting, method 1 (launchctl) takes precedence for new app launches. Verify by checking proxy stats after a Claude Desktop conversation — you should see new misses/hits appear.

---

## Cursor

Cursor makes OpenAI-compatible calls. Set the base URL to point to SLM proxy.

**Via Cursor settings UI:**

1. Open Cursor → Settings → Models
2. Set **OpenAI Base URL** (or **Custom Base URL**) to: `http://127.0.0.1:8765`
3. Keep your existing API key — the proxy passes it through untouched

**Via environment variable (for Cursor launched from Terminal):**

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765 cursor .
```

**Or `slm wrap cursor`:**

```bash
slm wrap cursor
```

---

## Windsurf

Same OpenAI-compatible surface as Cursor.

```bash
slm wrap windsurf
# or set OPENAI_BASE_URL=http://127.0.0.1:8765 in Windsurf's model settings
```

---

## AGY / Antigravity IDE

**Status as of v3.6.3: partial.** AGY is a compiled Google Cloud Code binary. Binary analysis
shows it routes Claude model calls through **Google Vertex AI** and Gemini model calls through
the Generative Language API — using its own OAuth token store, not standard env var overrides.

**What does NOT work:**
- `ANTHROPIC_BASE_URL=http://127.0.0.1:8765 agy ...` — ignored. AGY routes Claude via Vertex AI.
- `GOOGLE_GENAI_BASE_URL=http://127.0.0.1:8765 agy ...` — ignored. AGY uses its own auth layer.

**What may work (if you use AGY with a third-party OpenAI-compat endpoint):**

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 agy -p "your prompt"
```

Only works for AGY integrations that explicitly read `OPENAI_BASE_URL` — not the default Gemini
or Claude paths.

> **Roadmap:** Full AGY proxy support is the primary v3.7 target. Google is deprecating Gemini CLI
> on June 19, 2026 — AGY is the successor and we are prioritising native proxy integration for it.
> Watch [GitHub releases](https://github.com/qualixar/superlocalmemory/releases) for updates.

---

## Gemini CLI

> **⚠ Deprecation notice:** Google is retiring the Gemini CLI on June 19, 2026. Use AGY (Antigravity)
> as the successor. See the AGY section above for proxy status.

The Gemini CLI reads `GOOGLE_GENAI_BASE_URL`. As of **v3.6.3**, the Gemini native surface (`/v1beta/models/*`)
has full cache + compression support — no longer pass-through only.

```bash
GOOGLE_GENAI_BASE_URL=http://127.0.0.1:8765 gemini -p "your prompt"
# or
slm wrap gemini
```

Cache and compression are active for streaming and non-streaming Gemini calls. The proxy assembles
chunked SSE into a single `generateContent` JSON, stores it under the request hash, and on a cache
hit replays it as a properly chunked SSE stream.

---

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://127.0.0.1:8765/v1",   # ← SLM proxy
)

# All completions, embeddings go through the proxy — caching + compression active
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

---

## Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-anthropic-key",
    base_url="http://127.0.0.1:8765",   # ← SLM proxy
)

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Streaming also works — the proxy accumulates the stream, stores it in cache on completion, and serves future identical calls from cache as a re-emitted SSE stream:

```python
with client.messages.stream(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

---

## Node.js / TypeScript — Anthropic SDK

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseURL: "http://127.0.0.1:8765",   // ← SLM proxy
});

const message = await client.messages.create({
  model: "claude-haiku-4-5-20251001",
  max_tokens: 256,
  messages: [{ role: "user", content: "Hello" }],
});
```

---

## Node.js / TypeScript — OpenAI SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:8765/v1",   // ← SLM proxy
});

const completion = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "Hello" }],
});
```

---

## SDK Adapter (Zero-Config Wrapper)

If you prefer not to change your base URL, use the in-process adapter:

```python
from superlocalmemory.optimize.adapters.openai_adapter import withSLM
from openai import OpenAI

client = withSLM(OpenAI())           # wraps in-process — same interface
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

```typescript
import { withSLM } from "superlocalmemory/optimize/adapters/openai";
import OpenAI from "openai";

const client = withSLM(new OpenAI());
```

---

## LangChain / LlamaIndex

Set `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL` as an environment variable before importing the library — both LangChain and LlamaIndex read these from the environment.

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8765/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8765
python your_script.py
```

---

## Raw curl

```bash
# Anthropic
curl http://127.0.0.1:8765/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":32,"messages":[{"role":"user","content":"Hello"}]}'

# OpenAI
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "content-type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Codex CLI

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 codex "your task"
# or
slm wrap codex
```

---

## Verifying Savings

After wiring any client, verify the proxy is seeing calls:

```bash
slm optimize savings --since 1          # savings in the last 1 hour
curl -s http://127.0.0.1:8765/api/optimize/stats | python3 -m json.tool
```

Or open the dashboard: `slm serve` → http://localhost:8700 → **Optimize** tab.

Expected after one cache hit:
```json
{
  "hits": 1,
  "misses": 1,
  "cache_entry_count": 1,
  "tokens_saved_output": 42,
  ...
}
```

---

## Troubleshooting

### "Connection refused" on port 8765

The proxy daemon is not running. Start it:

```bash
slm proxy           # start proxy
slm optimize status # verify
```

### Proxy running but no savings after many calls

Check the following:
1. **Cache enabled?** Run `slm optimize status` — verify `cache: on`.
2. **Requests are identical?** The cache key is a hash of `(model, messages, tools)`. If messages differ on every call (e.g. a different system prompt or timestamp), each call is a miss.
3. **Too few messages to compress?** Compression protects the last 4 messages. Conversations with ≤4 messages will show no compression savings (expected). Real savings appear in longer sessions with 10+ messages.
4. **Check stats:** `curl -s http://127.0.0.1:8765/api/optimize/stats | python3 -m json.tool` — look at `misses` vs `hits`.

### `ZlibError: Decompression error` in Claude Code

You have both `ANTHROPIC_BASE_URL` set by the system (e.g. inherited from a parent Claude Code session) AND in `settings.json`. The inherited value may conflict. Unset the system-level variable:

```bash
unset ANTHROPIC_BASE_URL  # in your terminal before launching
# Then launch via settings.json: ANTHROPIC_BASE_URL will be applied by Claude Code at startup
```

### Dashboard shows 0 entries after clear

Cache was cleared or the daemon was restarted — normal. Make a fresh call to populate it.

---

## All Supported CLIs at a Glance

| CLI / SDK | Surface | Setup method |
|-----------|---------|-------------|
| Claude Code CLI | Anthropic | `slm wrap claude` or `settings.json` env block |
| Claude Desktop | Anthropic | `launchctl setenv ANTHROPIC_BASE_URL http://127.0.0.1:8765` |
| Cursor | OpenAI compat | Cursor settings → Custom Base URL |
| Windsurf | OpenAI compat | `slm wrap windsurf` |
| AGY / Antigravity | Anthropic / OpenAI | Partial — see AGY section. Full support in v3.7. |
| Gemini CLI (deprecated Jun 19) | Gemini native | `GOOGLE_GENAI_BASE_URL=http://127.0.0.1:8765 gemini ...` |
| Codex CLI | OpenAI compat | `slm wrap codex` |
| Python anthropic SDK | Anthropic | `base_url="http://127.0.0.1:8765"` |
| Python openai SDK | OpenAI compat | `base_url="http://127.0.0.1:8765/v1"` |
| Node.js @anthropic-ai/sdk | Anthropic | `baseURL: "http://127.0.0.1:8765"` |
| Node.js openai SDK | OpenAI compat | `baseURL: "http://127.0.0.1:8765/v1"` |
| LangChain | OpenAI compat | `OPENAI_BASE_URL` env var |
| LlamaIndex | OpenAI compat | `OPENAI_BASE_URL` env var |
| SLM SDK adapter | In-process | `withSLM(OpenAI())` |
| Raw curl | Both | Direct HTTP to port 8765 |
| Any OpenAI-compatible client | OpenAI compat | Set base URL to `http://127.0.0.1:8765/v1` |

---

*Part of [Qualixar](https://qualixar.com) — AI Reliability Engineering | Created by [@varunPbhardwaj](https://x.com/varunPbhardwaj) | Subscribe [@myhonestdiary](https://youtube.com/@myhonestdiary)*
