# Importing from other apps

Part of 4.2.30 (WS-9). SuperLocalMemory can read the export files of ChatGPT, Claude, Mem0 and Letta,
a folder of OKF notes, and plain Markdown (`AGENTS.md`, `CLAUDE.md`, `MEMORY.md` with its topic files),
and bring what is in them into your memory. It is on by default and runs only when you start an import
(`slm flags set portability.import off` turns it off), and the kill switch
(`slm flags kill on`) stops it.

An import is one unit. You see what a file holds before anything is saved, it runs in the background
without slowing the agents that use SuperLocalMemory at the same time, and **Undo** takes the whole run
back exactly.

## The flow

1. **Preview.** `slm import preview FILE` (or choose a file on the dashboard). It says what it found in
   plain words, for example "We found 214 conversations from ChatGPT, March 2024 to October 2026. 12 empty
   ones will be skipped.", counts what will be left out and why, and how much is already in your memory.
   Nothing is saved. The file waits in a private folder until you press Import, then it is deleted.
2. **Import.** `slm import run FILE` (or **Import**). It runs as a background job in the daemon: batches of
   saves through the normal writer, progress you can read (`slm import status`), a **Stop** that keeps what
   was saved (`slm import cancel`). Between batches it lets recalls finish and waits for the enrichment queue
   to shrink, so an agent recalling or saving at the same time never fails and never sees
   "database is locked".
3. **Undo.** `slm import undo RUN_ID` (or **Undo this import**). It removes what that run saved and nothing
   else, in the background.

## What it accepts

| Source | File | What becomes memory |
|---|---|---|
| ChatGPT | `conversations.json`, the numbered `conversations-000.json`, `-001`... set, the zip, one conversation, or `{"conversations": [...]}` | One memory per turn, one session per conversation. Only the branch you were shown (from `current_node` up through `parent`) is read; replies you edited or regenerated away are counted and skipped. |
| Claude (claude.ai) | `conversations.json` (`chat_messages`, `sender`, ISO times), the zip, one conversation; `memories.json` when it is in the export | One memory per turn (`human` is stored as `user`); each entry of `memories.json` is one memory. `thinking` and tool blocks are never imported. |
| Claude memory text | A text file you copied out of Claude's memory settings | Each bullet and each paragraph is one memory; a leading date like `[2026-01-10]` is read as when it was said. |
| Mem0 | `get_all` output as a bare list, `{"results": [...]}`, the platform's paginated envelope (several saved pages are merged), or `{"memories": [...]}` | One memory per item. Items that had expired or were replaced are skipped. |
| Letta | An agent file (`.af`) or a bare array of memory blocks | One memory per non-empty block; one session per agent for `user` and `assistant` messages. **Archival memory is not in an `.af` file**, so it cannot be imported; the preview says so every time. |
| OKF | A folder or zip of Markdown concept files with front matter (only `type` is required) | One memory per section of each concept. `index.md` and `log.md` are skipped. OKF has no speaker, role or created time, so none is invented. |
| Markdown | Any `.md` file or folder: `AGENTS.md`, `CLAUDE.md`, `MEMORY.md` and topic files | One memory per section, with its heading path as the first line. A file that is only a list of links (an index) is skipped. |

Unknown fields are ignored, never an error. A file that matches none of these is refused with a message
that names every format this version knows; name the format yourself with `--format` (chatgpt, claude,
claude-memory, mem0, letta, okf, markdown) when you are sure of it. Empty, system-only, hidden and tool-only
turns are skipped and counted. Credentials found in the text are stored as typed placeholders, as for any save.

Sources worth knowing about: the shape of ChatGPT's export comes from documentation by third parties (OpenAI's
help pages could not be read when this was built), and `memories.json` in Claude's export is described the same
way. Both importers accept every variant seen and ignore what they do not know. ChatGPT's saved-memory list
is not in its export; copy it into a text file and import that as Claude memory text or Markdown.

## What an import may and may not do

Imported memories have the origin `imported_file`, the bottom of the origin lattice (M-15):

- They **never replace** what you remember. Anything in the file that is already in your memory (same text,
  ignoring case and spacing) is left out and counted as "already in your memory"; the memory that was there is
  not touched, not even its date. Importing the same file twice saves nothing the second time.
- They **never move a slot head**. A value that disagrees with what a slot already holds opens a dispute
  (`origin_below_head`) that no judge answers (a run causes **0** judge calls, `JUDGE_CALLS_PER_RUN`), and only
  a recorded decision settles it (`slm disputes`, `settle_dispute`, the dashboard). One run lists at most
  **50** such disputes (`DISPUTES_PER_RUN`); past that the text is still kept and the run says how many it left
  out. These are routing constants: they protect the review queue and decide nothing about what is true.
- They never become a rule, a pin or a core block by themselves; that needs one explicit, recorded
  promotion, as for any text below `agent_inferred` (see the instruction firewall). They are still
  recalled and shown at session start, as evidence.
- The import is a decision under D-49: any caller may start it, it is recorded as kind `import` with who and the
  channel, listed under **Recent decisions by your agents**, and its Undo there rolls the run back.

## Undo, exactly

A run records the memory each save created (`import_runs`, `import_run_items`, M085). Undo removes those
memories with everything derived from them: facts, keyword, vector, graph and scene copies, entity records and
profile summaries the run created, consolidation records, derivation lineage, summaries that cite them, the
ingestion records that held the original text and the trust score of the run's own actor (every save is written
as `import:<run id>`). The tests compare every row of every table before and after, with the ledgers that only
ever record that something happened (write receipts, the ids-only change feed, queued vector removals, the
decision log) and the search index's own internal segments set aside, and the difference is empty, for an
import that was only saved and for one that was fully enriched, for each importer, and for a run that was
stopped half way. An erased memory leaves its run, and erasing or folding a profile takes its runs with it
(I-11). If something is still being enriched when you undo, the run says so and undoing again finishes it.

## Limits

An upload through the dashboard or HTTP is at most **64 MB** after decoding (`MAX_UPLOAD_BYTES`; a request
that says it is bigger is refused before its body is read). From the command line a path on this computer can
be up to 512 MB in total (a ChatGPT export is split by its maker long before that). A zip may unpack to at most
256 MB. One import runs at a time per profile. All of these are routing constants.

## Where it lives

- CLI: `slm import preview|run|status|list|cancel|undo` ([reference](cli-reference.md#slm-import-4230)).
- HTTP: `GET /api/import/runs`, `GET /api/import/runs/{id}`, `POST /api/import/preview` (review-read role),
  `POST /api/import/run`, `POST /api/import/runs/{id}/cancel`, `POST /api/import/runs/{id}/undo` (memory-update
  role), `POST /api/import/switch` (this computer, installation-wide authority). A caller from another machine is
  governed by its role, and only a caller on this computer can name a path.
- Dashboard: the **Bring your memories from another app** card on the home pane.
- MCP: **no new tool** (tool caps); an agent runs `slm import` in its shell.
