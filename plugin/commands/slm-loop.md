---
description: Run a task as a bounded, gate-verified loop backed by SuperLocalMemory — iterate until an independent gate passes, never on the agent's own claim.
argument-hint: <task that has a checkable acceptance condition (tests, schema, lint, reconciliation)>
---

Run this task as a **bounded loop** using SuperLocalMemory as the durable ledger:

> $ARGUMENTS

Delegate the execution to the **slm-loop-runner** agent and follow the
**slm-loop** skill. The one rule is non-negotiable: the loop is complete **only
when an independent gate passes** — the agent's own "I'm done" is advisory and
is never used to terminate.

Procedure:

1. **Name the gate.** State the mechanical check that proves this task is done
   (e.g. `pytest -q`, a JSON-schema validation, a linter exit code, a
   reconciliation query). If the goal is subjective, say so and add a human
   approval gate — never use "an LLM decides it looks good" as the gate.
2. **State the bounds.** Max iterations, a no-progress window, and any token or
   wall-clock budget, before starting.
3. **Iterate.** Each lap: propose a change, then run the gate independently.
   Every lap persists as queryable SLM memory (tag `loop:<name>`); inspect with
   `slm loop history` and `slm loop show <run_id>`, so the run is auditable and
   resumable. `slm loop demo` shows the control flow end to end.
4. **Report the exact terminal status** — `DONE` (gate passed + approved),
   `HALT` (a bound tripped), `PAUSE` (approval pending), `KILLED` (kill switch),
   or `ERROR` (runner/gate failed). Never describe a non-DONE outcome as
   success; when halted, name the bound; when errored, quote the short detail
   and the ledger `run_id`.
