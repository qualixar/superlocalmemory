# Python API Reference

The Python API and Python-installed CLI are for applications running inside an
activated virtual environment. The other primary CLI path is the npm global
package, which owns its private Python runtime.

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
    profile_id=None,
    slot=None,
    slot_value=None,
    source_file=None,
)
```

The canonical Python write returns `list[str]` fact IDs. `profile_id` routes this one
write to a named profile without moving the active one. `scope` accepts
`personal`, `shared`, or `global`; `shared_with` contains profile IDs allowed to
read a shared fact. Shared/global recall is opt-in and remains subject to the
configured scope policy.

`slot` and `slot_value` name the one thing the memory states the current value
of, as `subject/relation` or `subject/relation/qualifier` with the value in a
few words, exactly as `remember(slot=…, slot_value=…)` on MCP, the CLI and HTTP.
The same value later changes nothing; a different one opens a dispute. A
`source_file` (an absolute path to a readable local file) is fingerprinted now,
so a later recall can say the file changed or is gone. A malformed slot or a
path that is not a readable file raises `ValueError` before anything is stored;
neither can be set through `metadata`.

The daemon, MCP, CLI, hooks, and Python API attach a trusted actor before
persistence. A caller-supplied agent label is attribution metadata, not an
authentication credential.

### `recall`

```python
response = engine.recall(
    query,
    profile_id=None,
    mode=None,
    limit=20,
    agent_id="unknown",
    session_id=None,
    fast=None,
    *,
    include_global=None,
    include_shared=None,
    window=None,
    as_of=None,
    known_as_of=None,
    valid_at=None,
    include_unknown=False,
    answer_check=None,
    facets=None,
    show_weak=False,
    include_related=False,
)
```

`recall` returns a `RecallResponse`, not a list. Iterate `response.results`; the
stored fact is `result.fact`. `profile_id` recalls from a named profile without
moving the active one. The time arguments behave as in
[Recall](recall.md#time-travel), and `answer_check` is `"full"` (the default) or
`"no_reorder"`. The narrowing filters of the CLI and MCP surfaces (`project`, `saved_by`, `about`,
`kind`, `tags`) reach the engine as one `facets` object
(`superlocalmemory.retrieval.facets.Facets`).

The Python API returns the engine's answer as it is: when no memory supports the question,
`response.results` is empty and `response.abstention_note` is "That isn't in SuperLocalMemory.".
`show_weak=True` keeps the weak matches (each `result.weak`), and `include_related=True` returns up
to three close memories with `result.tier == "related"` when the question is real but not answered
exactly. `response.support_tier` is `supported`, `related` or `none`, and `response.warming_up` is true
while models are still loading. Depth and `max_tokens` are applied by the daemon, MCP and CLI
surfaces on top of this response ([Recall](recall.md#layered-recall)).

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

Without a configured [Answer check](answer-check.md), SLM does not publish
calibrated answer confidence:

```python
assert response.score_contract_version == "2"
assert response.calibration_status == "uncalibrated"
assert response.calibration_id is None
assert response.answer_confidence is None
```

Every result also carries `support` (the answer-support probability), `weak` and `tier`. Do not
transform retrieval scores into answer probability. See
[Retrieval Score Contract v2](retrieval-score-contract.md) for abstention and
calibration semantics.

## Retrieval composition

The current engine can run five candidate producers when their dependencies
are healthy: dense semantic, BM25 lexical, temporal, Hopfield associative, and
spreading activation (plus a picture channel when pictures and documents are on). Entity-graph information can enhance scores after fusion
but does not create an independent candidate. Optional reranking and adaptive
learning can change ranking utility.

Use CLI or MCP trace output to inspect the channels that actually contributed
in a deployed configuration.

## Framework adapters

Nine adapter packages back LangGraph, Semantic Kernel, Microsoft Agent Framework,
LangChain, LlamaIndex, CrewAI, AutoGen, Google ADK and OpenAI Agents memory
interfaces with SLM. See [Framework Adapters](framework-adapters.md) for the
classes, requirements and how to install each.

## HTTP and MCP identity

Loopback mutations receive a code-derived local actor. Python-owned daemon
clients use the private capability and exact target-instance descriptor. The
same-origin dashboard uses an install token, while configured API keys or mesh
credentials authorize their documented remote surfaces. Caller-provided
`agent_id` values never replace these credentials.

Configuring an SLM API key does not revoke the local OS-user trust boundary:
uncredentialed loopback mutations remain accepted and are protected from
browser CSRF by the origin guard. Non-loopback callers must authenticate.

The private daemon capability is process/filesystem state and must not be
embedded in browser JavaScript, checked into source, or logged.
