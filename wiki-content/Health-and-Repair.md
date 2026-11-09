# Health and Repair

SLM checks itself in six separate ways, and repairs most problems without losing a memory. This page names each mechanism, because "watchdog" is not one feature: it is several small ones that cover different failures.

## Pre-flight and component health

`slm doctor` runs a checklist and `slm doctor --fix` repairs what it can by re-downloading missing models or installing the vector extension. `--quick` is fast, `--deep` runs a full SQLite integrity check. A component registry runs read-only probes, and a healer acts on what they find, with firm limits: it never uses sudo, never pulls an Ollama model on its own and never installs a multi-gigabyte dependency unasked. In the dashboard, **Health** shows components with a **Retry now** button. The daemon also self-heals at start, removing only lock and process files that are provably dead.

## Store integrity and repair

`slm db integrity` reports five separate answers (pages, relations, fidelity, projections and repair) and changes nothing. `slm db repair` previews by default. With `--apply` it fixes orphan rows, stale or orphaned vectors, erased-text leftovers and settled obligations, in small receipted batches (100 rows per batch with a 50 ms pause by default). `--undo RUN_ID` puts back what a run changed. `slm db fidelity` lists facts whose text drifted from what their memory said. In the dashboard, Health has a **Memory store** card that runs the once-per-upgrade check, and **Repair now** takes a backup copy first. Why batches and receipts: a repair that holds the database for minutes would stall every other writer, and a repair you cannot undo is a risk.

## Memory health: describing older memories and offering repairs

Older memories are often thinly described: no tags, no project, no entities, no dates, no kind. SLM improves this by itself, in the background, and the Memory health card on the dashboard's Health pane (and `slm enrich status` in a terminal) shows how far it has got. A worker does the job only while your computer is idle, in small steps, and it steps aside the moment you use SLM. It is on by default.

**What it does on its own.** The automatic work only adds. It fills in what is missing and never rewrites what a person or an agent chose. It derives tags, a project name, entity names, observation dates and an event date from the memory's own words, and suggests a kind (rule, decision and so on) using plain rules, or a small bounded model on this machine if you have one running. A kind that the saving agent or you set is always kept. SLM does not invent dates or labels to make a percentage look better: an event date is filled only when the text carries one, and the date a memory was saved is never used as the date it became true. The text of the memory is never changed. Before the first automatic change SLM takes one safety copy of your stores, and every run after that can be undone from its own record.

**What waits for you.** Repairs that move or delete something never run on their own. The worker only counts them and shows the count: entity links that point at aliases or junk entities, hidden memories that are still in use and can be moved back, and saves that did not finish. Each has a **Fix** button on the card, or `slm enrich fix entity-links`, `slm enrich fix refile-hidden` or `slm enrich fix repair-saves`. SLM asks before it fixes anything and, for saves that did not finish, says plainly that the fix can't be undone and that a safety copy is taken first. A repair of search sketches, which only rebuilds something missing from a vector that is already stored, runs automatically.

**Undo.** **Undo the last change** on the card, or `slm enrich undo` (the latest run that changed something, after a confirmation; pass a run id to choose one), puts back what that run changed. If you changed a memory since, that change is kept and the result says how many were kept: "Undone: 12 changes restored. 2 kept because they were changed since." Once you undo a repair, SLM does not run it again on its own and leaves out the items it touched, until you ask with `slm enrich fix <repair> --include-undone`. `slm enrich pause RUN_ID` and `resume RUN_ID` stop and restart a run, and memories keep working while paused.

**What the card says.** It reads "Improving your memories: 1,200 of 4,300" with one of a few plain sentences: waiting for the computer to be idle, improving in the background, paused, or "Up to date. Checked again while your computer is idle". A failure shows as "Some improvements are waiting and will be tried again", retried with a growing delay; error names stay in the log. `slm doctor` has two matching rows. **Memory description** shows tag, kind, date and entity coverage with the current state, and **Memory repairs** appears for each repair waiting for you, with the command to run.

**Stubs.** Some memories are only bookkeeping: an empty structure, "session ended at" followed by a timestamp, or a label such as "receipt id" followed by a machine-shaped id (a UUID, a long hex string or a 12-digit number). SLM marks these as stubs. A stub is kept, never deleted, and still listed, counted and exported; the Memories list shows a **stub** badge ("Kept, but never used as an answer"). What changes is that recall does not offer a stub as an answer. The rule is deliberately narrow: a receipt number or a date alone is not a stub, and neither is anything a person could mean, such as "use port 8765". If you decide a memory is an answer, that decision is stored and always wins. To make that decision from the terminal, `slm hide <fact_id>` keeps a memory out of answers and session starts (for a vague note the rule cannot recognise), and `slm hide --undo <fact_id>` makes it an answer again; neither deletes or edits anything ([CLI Reference](CLI-Reference)).

Memory health is for the owner or administrator: reading it and acting on it need management permission, because it counts and changes the whole store.

## Operations recovery

`slm ops list`, `resolve` and `status`, and the MCP tools `list_failed_operations` and `resolve_operation` (in the `whole` profile) surface the dead-letter queue, stuck ingestion, exhausted projection obligations and writer stalls. The dashboard shows the same under Governance, Admin: Ops Health.

## Process watchdogs

A health monitor kills the heaviest worker when total memory passes a budget (40 percent of RAM, with a 2,500 MB floor, unless `SLM_RSS_BUDGET_MB` says otherwise) and kills workers that have been silent for 60 seconds. Every worker also watches its parent process and exits if the parent dies. The picture worker has its own memory cap (`SLM_MEDIA_WORKER_RSS_LIMIT_MB`, 1,600 MB by default). `slm reap` kills orphaned SLM processes whose IDE parent has died, and `slm restart` is the heavy option: it kills orphans, cleans state, starts again and verifies health.

## Math-layer health

`slm health` and the MCP `health` and `consistency_check` tools check the mathematical layers, including a sheaf-based contradiction check. In the dashboard, the Health pane's status strip shows the daemon, memory database, math layers and mesh broker, with trust signals and connected agents below.

## Escape hatches

`slm disable` writes a `.disabled` marker in the data folder and stops the daemon; `slm enable` removes it. Both are for the case where you need SLM to stand down without uninstalling it.

Code: `core/enrichment.py`, `core/metadata_health.py`, `core/health_repairs.py`, `core/description_metadata.py`, `encoding/non_answer.py`, `cli/enrich_cmd.py`, `server/routes/memory_health.py`, `ui/js/od-memory-health.js`, `core/component_registry.py`, `core/component_healer.py`, `storage/integrity_health.py`, `storage/integrity_repair.py`, `storage/integrity_receipts.py`, `cli/integrity_cmd.py`, `core/health_monitor.py`, `infra/process_reaper.py`, `infra/self_heal.py`, `server/routes/integrity.py`.
