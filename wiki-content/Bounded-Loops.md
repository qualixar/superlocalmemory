# Bounded Loops

Bounded loops give an agent a durable, gate-verified iteration primitive
backed by SLM memory. The core invariant: an **independent gate** decides
when the loop is done, never the agent's own claim of completion.

Every lap is persisted to the SLM data root with tag `loop:<name>`, visible
via `slm recall`, `slm loop history`, and the dashboard. Loop history survives
process restarts in the same database as all other SLM memories.

## Three surfaces

Bounded loops ship on three surfaces:

| Surface | Entry point | Gate mechanism |
|---|---|---|
| CLI | `slm loop demo \| history \| show` | Deterministic (demo only) |
| Plugin command | `/slm-loop` (Claude plugin + Codex skill + agent) | Configurable via command |
| MCP tools | `slm_loop_run` / `slm_loop_history` / `slm_loop_show` | SLM recall query (`gate_query`) |

## MCP tools

The MCP tools are available in the `code`, `full`, `power`, and `whole`
profiles. See [MCP Tools](MCP-Tools) for the full profile reference.

### `slm_loop_run`

Run one bounded, gate-verified loop to a terminal outcome. Blocks (polling
the gate) until the gate passes or a bound is exhausted. Every lap is
persisted to SLM memory.

The gate checks whether a memory matching `gate_query` is retrievable with
confidence `>= gate_min_score` — making this a safe multi-agent coordination
primitive: one agent waits, under strict bounds, for a memory another agent
will write into shared SLM.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Loop identifier; used as the ledger key and `loop:<name>` tag |
| `gate_query` | str | required | SLM recall query; loop converges when a matching memory is found |
| `gate_min_score` | float | `0.0` | Minimum relevance score to treat a recall result as passing |
| `max_iterations` | int | `20` | Maximum laps before forcing termination (hard cap: 200) |
| `max_wallclock_s` | float | `15.0` | Wall-clock budget in seconds (hard cap: 120.0) |
| `poll_interval_s` | float | `1.0` | Seconds between gate polls (minimum: 0.25) |
| `max_tokens` | int | `0` | Optional token budget (0 = unlimited) |
| `no_progress_window` | int | `0` | Laps without change before early-stopping (0 = disabled) |

Returns a dict with `ok`, `status`, `reason`, `laps`, `run_id`, and
`termination`. On internal failure the tool returns `ok: False` with a
message rather than raising an exception.

### `slm_loop_history`

List recorded runs for a loop name (read-only).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | str | required | Loop name to query |
| `limit` | int | `20` | Maximum runs to return |

Returns `name`, `runs` (list of `{run_id, laps, final_status, ts}`).

### `slm_loop_show`

Show every lap of one run (read-only).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `run_id` | str | required | Run identifier returned by `slm_loop_run` |
| `limit` | int | `200` | Maximum laps to return |

Returns `run_id`, `name`, `laps` (list of `{lap, ts, decision, passed, detail, budget}`).

## CLI

The `slm loop` command provides read-only inspection and a keyless convergence
demo. It is not a substitute for the MCP `slm_loop_run` tool; the demo uses a
hardcoded stub proposer.

```bash
# Run the convergence demo: gate fails on laps 1-2, passes on lap 3
slm loop demo
slm loop demo --iterations 10 --json

# List recorded runs for a named loop
slm loop history
slm loop history --name my-loop-name --json

# Show every lap of a specific run
slm loop show <run_id>
slm loop show <run_id> --json
```

All subcommands accept `--json` for agent-native structured output.

## Durable ledger

Every lap is written to SLM memory with:

- **Tag** `loop:<name>` — enables `slm recall` and dashboard filtering
- **Run ID** — a stable identifier for `slm loop show` and `slm_loop_show`
- **Lap fields** — lap number, gate decision (`pass`/`fail`), detail text, token budget

Ledger writes are fail-open: a memory write error is counted and surfaced in
the run result rather than aborting a running loop. The gate is the
correctness guarantee; the ledger is observability.

## Example: multi-agent coordination via MCP

One agent writes a memory when its work is verified. A second agent waits
for that memory using `slm_loop_run`:

```python
# Agent A writes the signal
slm.remember("review complete: PR #42 is approved", tags="review,pr-42")

# Agent B waits for that signal (via MCP)
result = await slm_loop_run(
    name="wait-for-pr-review",
    gate_query="PR #42 approved",
    gate_min_score=0.5,
    max_iterations=30,
    max_wallclock_s=60.0,
    poll_interval_s=2.0,
)
```

Agent B's loop converges as soon as Agent A's memory becomes retrievable with
score >= 0.5. Neither agent polls a shared queue or a file; coordination flows
through the SLM memory layer that both agents already use.

## Terminal statuses

| Status | Meaning |
|---|---|
| `converged` | Gate passed; loop reached its goal |
| `max_iterations` | Hit the iteration limit without convergence |
| `max_wallclock` | Hit the wall-clock budget without convergence |
| `max_tokens` | Hit the token budget without convergence |
| `no_progress` | `no_progress_window` laps elapsed with no change |

---
*Part of [Qualixar](https://qualixar.com) | Created by [Varun Pratap Bhardwaj](https://varunpratap.com)*
