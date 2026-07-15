# Python API Reference

> SuperLocalMemory V3 documentation · AGPL-3.0-or-later

The Python API is for applications running inside an isolated Python
environment. For an end-user CLI install, prefer `uv tool` or `pipx`; for a
project dependency, create and activate that project's virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install superlocalmemory
```

Do not add paths from an npm installation to `sys.path`. The npm wrapper owns a
private runtime for its CLI and is not a Python SDK installation contract.

## MemoryEngine

```python
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.core.engine import MemoryEngine
from superlocalmemory.core.modes import Mode

config = SLMConfig.for_mode(Mode.A)
engine = MemoryEngine(config)

try:
    fact_ids = engine.store(
        "The mobile client uses OAuth 2.0 with PKCE",
        session_id="session-42",
        metadata={"source": "architecture-review"},
    )

    response = engine.recall("mobile authentication", limit=5)
    for result in response.results:
        print(result.rank_position, result.relevance_score, result.fact.content)
finally:
    engine.close()
```

`MemoryEngine` initializes lazily on first use. Call `close()` for every owned
engine so database, worker, and optional-backend resources have an explicit
lifecycle.

### `store`

```python
fact_ids = engine.store(
    content,
    session_id="",
    session_date=None,
    speaker="",
    role="user",
    metadata=None,
    scope="personal",
    shared_with=None,
)
```

The canonical Python write returns `list[str]` fact IDs. `scope` accepts
`personal`, `shared`, or `global`; `shared_with` contains profile IDs allowed to
read a shared fact. Shared/global recall is opt-in and remains subject to the
configured scope policy.

The daemon, MCP, CLI, hooks, and Python API attach a trusted actor before
persistence. A caller-supplied agent label is attribution metadata, not an
authentication credential.

### `recall`

```python
response = engine.recall(
    query,
    profile_id=None,
    mode=None,
    limit=10,
    agent_id="unknown",
    session_id=None,
    include_global=None,
    include_shared=None,
)
```

`recall` returns a `RecallResponse`, not a list. Iterate `response.results`; the
stored fact is `result.fact`.

## Retrieval Score Contract v2

| Result field | Meaning |
|---|---|
| `relevance_score` | Query-relative relevance, bounded to `0.0..1.0` |
| `ranking_score` | Internal ranking utility; diagnostic, not a probability |
| `memory_confidence` | Confidence attached to the stored assertion |
| `trust_score` | Evidence-policy trust signal |
| `rank_position` | One-based result position |
| `channel_scores` | Recorded per-channel contributions |

For one compatibility release, `score` aliases `relevance_score` and
`confidence` aliases `memory_confidence`. New code should use the explicit
names.

V3.7 does not publish calibrated answer confidence:

```python
assert response.score_contract_version == "2"
assert response.calibration_status == "uncalibrated"
assert response.calibration_id is None
assert response.answer_confidence is None
```

Do not transform retrieval scores into answer probability. See
[Retrieval Score Contract v2](retrieval-score-contract.md) for abstention and
calibration semantics.

## Retrieval composition

The current engine can run five candidate producers when their dependencies
are healthy: dense semantic, BM25 lexical, temporal, Hopfield associative, and
spreading activation. Entity-graph information can enhance scores after fusion
but does not create an independent candidate. Optional reranking and adaptive
learning can change ranking utility.

Use CLI or MCP trace output to inspect the channels that actually contributed
in a deployed configuration.

## Framework adapters

LangChain and LlamaIndex adapters are package surfaces with their own installed
artifact contract. Use them only with the exact released adapter documentation
and compatibility matrix. This page deliberately does not publish generic
framework snippets: earlier snippets referenced a removed V2 storage API, and
adapter compatibility must be proven against the frozen V3.7 artifacts before
it is presented as supported.

## HTTP and MCP identity

Loopback mutations receive a code-derived local actor. Python-owned daemon
clients use the private capability and exact target-instance descriptor. The
same-origin dashboard uses an install token, while configured API keys or mesh
credentials authorize their documented remote surfaces. Caller-provided
`agent_id` values never replace these credentials.

The private daemon capability is process/filesystem state and must not be
embedded in browser JavaScript, checked into source, or logged.
