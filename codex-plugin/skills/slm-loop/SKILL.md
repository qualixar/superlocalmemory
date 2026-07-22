---
name: slm-loop
description: Run gate-verified bounded loops with SuperLocalMemory as the durable ledger. Use when a task has a checkable acceptance condition (tests, schema, lint, reconciliation) and you must iterate until an INDEPENDENT gate passes — never stopping just because the agent believes it is done. `slm loop demo` runs a keyless convergence demo; `slm loop history` and `slm loop show <run_id>` inspect past runs whose every lap is persisted as queryable SLM memory (tag `loop:<name>`). Terminal statuses are DONE / HALT / PAUSE / KILLED / ERROR — report them exactly, never converting HALT/PAUSE/ERROR into success.
when_to_use: "run a bounded loop, gate-verified task, iterate until tests pass, verify against an independent gate, don't trust the agent's own done claim, slm loop, convergence loop, loop until green, loop ledger, resume a loop"
allowed-tools: Bash, slm_recall
---

# slm-loop — Bounded, gate-verified agent loops

## The one rule

A bounded loop is complete **only when an independent gate passes** — not when
the agent claims it is finished. The agent's own "I'm done" signal is recorded
for audit and is *never* used to terminate the loop. If you take one thing from
this skill: **the gate is the authority.**

Use a bounded loop whenever the goal has a mechanical, checkable contract: a
test suite, a JSON schema, a linter, a reconciliation rule, a citation checker,
a security scan. When the goal is subjective, keep a human approval gate (see
rungs below).

## What SLM adds

Every lap is written to SuperLocalMemory as a durable, queryable memory (tagged
`loop:<name>`, session `loop:<run_id>`). That makes a run:

- **auditable** — inspect the decision + gate verdict + budget for each lap;
- **resumable / historical** — a run's ledger survives across sessions;
- **discoverable** — visible via `slm recall`, the dashboard, and any
  SLM-integrated tool, alongside everything else the agent remembers.

## CLI

```bash
slm loop demo [--iterations N] [--json]   # keyless convergence demo
slm loop history [--name NAME] [--json]   # list recorded runs
slm loop show <run_id> [--json]           # every lap of one run
```

`slm loop demo` proposes a fix, checks it against a deterministic gate that
passes on lap 3, and records the run — a zero-setup way to see the control flow
and confirm the SLM-backed ledger works end to end.

## The bounds

A loop runs inside a safety envelope. Any bound tripping ends the run with
`HALT` (never a success):

- **max_iterations** — a hard lap cap.
- **no_progress_window** — consecutive no-change laps before halting a spinning
  agent.
- **token budget / wall-clock** — cumulative ceilings.
- **kill switch** — set `SLM_LOOP_KILL` to stop before the next lap.
- **approval rung** — L1 (report), L2 (assisted, pauses for approval), L3
  (unattended). L2/L3 require approval before a passing gate is accepted as
  DONE unless approval is explicitly configured off.

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
2. Read back the ledger with `slm loop show <run_id>` (or `slm recall` on tag
   `loop:<name>`).
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
- `slm-recall` — query a loop's laps directly (`loop:<name>` tag).
- `slm-session` — session lifecycle around a longer loop run.

---

SuperLocalMemory v3.8.0 · Qualixar · AGPL-3.0-or-later
