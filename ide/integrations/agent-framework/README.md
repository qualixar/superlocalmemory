# agent-framework-superlocalmemory

[Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
memory + history providers backed by the local data root of
[SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory).

Give an Agent Framework agent durable conversation history and memory
enrichment in your local SLM data root, visible through every other SLM
surface. Optional SLM providers, connectors, backup, and downloads have
separate network behavior.

## Prerequisites

- Python 3.11+
- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory) installed in the same environment
- `agent-framework-core >= 1.5.0` (the release that introduced `before_run` / `after_run`)

## Installation

```bash
pip install agent-framework-superlocalmemory
```

Or from source:

```bash
cd ide/integrations/agent-framework
pip install -e .
```

## Quick start

```python
from agent_framework_superlocalmemory import (
    SuperLocalMemoryContextProvider,
    SuperLocalMemoryHistoryProvider,
)

# Persist conversation history to SLM
history = SuperLocalMemoryHistoryProvider()

# Inject recent session memory before each run, persist the turn after
memory = SuperLocalMemoryContextProvider(max_recall=10)

# Attach to an agent per the Agent Framework provider API, e.g.:
#   agent = ChatAgent(..., context_providers=[memory], history_provider=history)
```

## What each provider does

- **`SuperLocalMemoryHistoryProvider`** implements `get_messages` /
  `save_messages`, storing each message as a durable SLM memory keyed by session.
- **`SuperLocalMemoryContextProvider`** overrides `before_run` (inject recent
  session memory via `context.extend_instructions`) and `after_run` (persist the
  turn's input messages and the agent response).

Both delegate persistence to a framework-free message store that is
independently tested against the SLM engine.

## Status and scope

Written against **Microsoft Agent Framework GA** (`agent-framework-core`). The
Python 1.5.0 release replaced the beta `invoking`/`invoked` hooks with
`before_run`/`after_run`; this package targets that current API and pins
`agent-framework-core>=1.5.0`.

- Persistence (append / list / clear per session) is verified against the SLM
  engine via the framework-free core.
- `Message` / `AgentSession` / `AgentResponse` objects are converted
  defensively (role/text extraction) because their exact field layout is not
  fully pinned in docs; validate in CI against the installed framework version.
- `before_run` injects recent session **history**; semantic recall across the
  wider SLM store is a documented follow-up.

## License

AGPL-3.0 — see [LICENSE](../../../LICENSE).

## Links

- [SuperLocalMemory V3](https://github.com/qualixar/superlocalmemory)
- [Documentation](https://superlocalmemory.com/)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/)
