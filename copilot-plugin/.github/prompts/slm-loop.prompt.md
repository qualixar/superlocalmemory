---
name: slm-loop
description: Gate-verified bounded loops with SuperLocalMemory as the durable ledger. Use when a task has a checkable acceptance condition and you must iterate until an INDEPENDENT gate passes — never stopping because the agent believes it is done. `slm_loop_run` (MCP) waits, under hard bounds, for a recall gate to pass; `slm loop demo` shows the control flow keyless; `slm loop history` and `slm loop show <run_id>` (or `slm_loop_history` / `slm_loop_show`) inspect past runs, whose every lap is stored as queryable SLM memory (tag `loop:<name>`). Terminal statuses are DONE / HALT / PAUSE / KILLED / ERROR — report them exactly, never converting HALT/PAUSE/ERROR into success.
version: "4.1.25"
agent: agent
tools:
  - slm_loop_run
  - slm_loop_history
  - slm_loop_show
  - recall
  - Bash
---

# slm-loop — Bounded, gate-verified agent loops

## The one rule

A bounded loop is complete **only when an independent gate passes** — not when
the agent claims it is finished. The agent's own "I'm done" signal is recorded
for audit and is *never* used to terminate the loop. If you take one thing from
this skill: **the gate is the authority.**

Use a bounded loop whenever the goal has a mechanical, checkable contract: a
test suite, a JSON schema, a linter, a reconciliation rule, a citation checker,
a security scan. When the goal is subjective, keep a human approval gate.

## What SLM runs, and what you run

SLM's loop engine runs the laps, enforces the bounds, calls the gate and writes
the ledger. It does **not** start your tests or linters: it carries no
subprocess or sandbox machinery. What each surface gives you:

| Surface | What it does |
|---|---|
| `slm_loop_run` (MCP) | A *watcher*: each lap is a recall of `gate_query`; the loop ends when a confident match appears, or a bound trips. Use it to wait for a verification or coordination memory another agent will write (for example "build passed"). |
| `slm loop demo` (CLI) | A keyless convergence demo: a stub proposer, a deterministic gate that passes on lap 3, every lap recorded. Proves the engine and ledger work end to end. |
| `slm loop history`, `slm loop show` / `slm_loop_history`, `slm_loop_show` | Read the ledger of past runs. |
| `superlocalmemory.loops.run_bounded_loop` (Python) | The engine itself, for code that supplies its own runner and gate callables. |

For a task whose gate is a command (`pytest -q`, a schema validation), you run
that command yourself each lap with Bash and read its exit code; the discipline
below still applies, but SLM is not recording those laps. For graph runs with
receipts, use the separate Bounded Loops product; SLM can take a read-only
snapshot of its finished runs with `observe_bounded_loop_evidence(workspace)`
(an absolute path), which records observations only and never changes recall or
ranking.

## What SLM adds

Every lap SLM runs is written as a durable, queryable memory (tagged `loop:<name>`,
session `loop:<run_id>`). That makes a run:

- **auditable** — inspect the decision, gate verdict and budget for each lap;
- **historical** — a run's ledger survives across sessions;
- **discoverable** — visible via `slm recall`, the dashboard, and any
  SLM-integrated tool, alongside everything else the agent remembers.

## MCP: `slm_loop_run`

```
slm_loop_run(
  name: str,                    # 1–128 chars; also the memory tag
  gate_query: str,              # the recall the gate checks each lap (up to 2000 chars)
  gate_min_score: float = 0.0,  # minimum top-result score to pass
  max_iterations: int = 20,     # hard lap cap, 1–200
  max_wallclock_s: float = 15.0,# hard time cap; 0 disables; never more than 120
  poll_interval_s: float = 1.0, # wait between laps; at least 0.25
  max_tokens: int = 0,          # optional token budget; 0 disables
  no_progress_window: int = 0,  # halt after N no-change laps; leave 0 for a watcher
  require_support: bool = False,# pass only if the answer check ran and judged it sufficient
)
```

The call **blocks** until the gate passes or a bound trips, then returns
`{ok, status, reason, passed, laps, run_id, ledger: [{lap, decision, passed, detail}]}`.
The gate recalls at most three memories per lap and ignores the loop's own
ledger entries, so only a memory written by someone else can satisfy it. A
recall the answer check judged insufficient never passes, even with a high
score; with `require_support=True` an unchecked recall (check off, busy,
loading, out of time) does not pass either. When the answer check is on, each
lap asks it once about those memories; with the online check that is a request
to the provider and may be billed. `slm_loop_run` is not available to remote
callers.

```
slm_loop_history(name, limit=20)   # runs recorded under a loop name
slm_loop_show(run_id, limit=200)   # every lap of one run, in order
```

## CLI

```bash
slm loop demo [--iterations N] [--json]   # keyless convergence demo
slm loop history [--name NAME] [--json]   # list recorded runs
slm loop show <run_id> [--json]           # every lap of one run
```

There is no `slm loop run`.

## The bounds

A loop runs inside a safety envelope. Any bound tripping ends the run with
`HALT` (never a success):

- **max_iterations** — a hard lap cap.
- **no_progress_window** — consecutive no-change laps before halting a spinning
  agent.
- **token budget / wall-clock** — cumulative ceilings, checked again after each
  lap, so an overshooting lap halts even if its gate would pass.
- **kill switch** — a non-empty `SLM_LOOP_KILL` in the loop's process
  environment stops it before the next lap, with status `KILLED`.
- **approval rung** — L1 (report), L2 (assisted, pauses for approval), L3
  (unattended). L2 and L3 require approval before a passing gate is accepted as
  DONE unless approval is explicitly configured off. `slm_loop_run` and the demo
  run at L1, so `PAUSE` arises only from code that sets a higher rung.

## Terminal statuses — report exactly

| Status   | Meaning |
|----------|---------|
| `DONE`   | The independent gate passed **and** approval was granted or not required. |
| `HALT`   | A bound tripped (iterations, no-progress, token, or wall-clock). |
| `PAUSE`  | The gate passed but required approval is not yet granted. |
| `KILLED` | The external kill switch tripped between laps. |
| `ERROR`  | The runner or gate failed to execute; inspect the lap detail. |

Say `DONE` only when the status is exactly `DONE`. Never describe `HALT`,
`PAUSE`, or `ERROR` as success. When halted, name which bound tripped; when
paused, name the approval needed; when errored, quote the short detail.

## Reporting workflow

1. Run or resume the loop.
2. Read back the ledger with `slm_loop_show` or `slm loop show <run_id>` (or
   `slm recall` on tag `loop:<name>`).
3. Report the exact terminal status, the lap count, and the gate's final
   verdict. Include the `run_id` so the run can be re-inspected later.

## Gate discipline

- The gate verifies; the runner only proposes. They are separate.
- Prefer a typed, parseable gate (a test exit code, a schema validation, a
  scanner report) over a vague check. A missing tool, an empty report, or a
  crashed scanner is **not** a clean pass — fail closed.
- Never use "an LLM decides it looks good" as the gate. That reintroduces the
  exact failure mode bounded loops exist to remove.

---

## Related skills

- `slm-status` — confirm SLM is healthy before relying on the ledger.
- `slm-recall` — query a loop's laps directly (`loop:<name>` tag), and the
  answer-check fields the recall gate depends on.
- `slm-session` — session lifecycle around a longer loop run.

---

SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later
