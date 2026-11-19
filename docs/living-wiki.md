# The living wiki: compiled pages and Markdown both ways

Part of 4.2.30 (WS-6). SuperLocalMemory can compile pages from your memories and write them to a
folder of Markdown files, and read your edits to those files back. Everything is on by default; nothing is written until you
turn it on, and nothing in it changes a memory without you.

## What a page is

A page is derived text about one subject. There are three kinds:

- **Current state**: one page per subject that has slots (the value a thing holds now, and since
  when). A value two memories disagree about is shown as a dispute, never as a value.
- **Story**: one page per family of decisions (memories of kind `decision` that share a slot
  subject or an entity), one dated sentence per decision, in time order.
- **Topic**: one page per entity or tag that has enough live memories, with the first sentences
  of those memories.

**Every sentence is extractive.** It is either a *span*, a sentence copied word for word from a
memory it cites, or a *template*: a fixed frame filled with slot values and the verbatim text of
cited memories ("slm cadence has had this value since 2026-03-01."). No model writes any
sentence. A model-written sentence would need its own CITATION_SUPPORT certificate, and none
exists. Each sentence carries the ids of the memories it cites, and is checked again every time a
page is read: a span must still be found word for word in a live memory, and a template must
still render to exactly the same words from the current slots. A sentence that fails is hidden,
the page says it is `stale`, and the next rebuild proposes the page again.

**Each page says how many sentences were withheld**, and why: not found word for word in the
memory it cites, over the size of a page, too long to be one sentence, or its memory's source
file changed. A page never fills the gap.

**A page is never a memory.** It has origin `derived`, it is stored in its own table
(`compiled_pages`, M083) and it is never admitted to recall. Erasing a memory removes the
sentences that cite it from every page and file (see below).

## How pages are made

Pages enter the store only as a **ProposalSet** of kind `create_page`, so they follow the rules of
[proposed changes](cli-reference.md#slm-proposals-4230): applying needs your approval (or a
certificate that covers it, and none exists), an earlier version of a rebuilt page is kept, and
`slm proposals rollback` puts the page back exactly as it was (or removes a page that was new).

There are two ways to make them:

1. **You or an agent rebuild.** `slm wiki rebuild` or **Rebuild pages** on the dashboard
   compiles the pages and applies them in one step, recorded (who, channel, time) and undone
   with `slm proposals rollback`.
2. **The dream cycle.** With `compiler.dream` on, the background maintenance cycle looks over
   each profile at most every six hours and queues one ProposalSet for review: a page for every
   subject whose page is new or changed, and a merge for each group of exact duplicates (the same
   text after case, spacing and end punctuation are folded; never a pinned memory, never one whose
   source file changed). It uses rules only and asks no model; it checks every sentence
   before it queues the page; it writes only the proposal tables; and it never applies anything.
   A profile with a dream set still waiting is not given another, and what you rejected is not
   proposed again unchanged. Review it with `slm proposals` or the dashboard's **Proposed
   changes** card.

## The Markdown folder (`compiler.wiki`)

```bash
slm flags set compiler.wiki on      # or `shadow`: show what would be written, write nothing
slm wiki build                      # write the approved pages
slm wiki dir                        # where the folder is
```

The folder is `SLM_WIKI_DIR`, or the `dir` in `wiki.json` in your data folder (`slm wiki dir PATH`
sets it), or `wiki/` in your data folder, with one folder per profile. No request to the daemon
can name a path.

Each sentence is a block in the file:

```markdown
<!-- slm:block fact=ab12cd34ef567890 hash=0123456789abcdef slot=sl_9f3a:7 type=span -->
SLM ships every week.
<!-- /slm:block -->
```

The marker carries the memory the block cites, the hash of its text as written, and the version of
the slot that memory fills (`-` if none). `slm wiki build` never overwrites a file you edited since
it wrote it; it says so and keeps your file.

### Editing a file

Change the words of a block and run `slm wiki sync` or press **Read my edits** on
the dashboard. SLM finds edited blocks **by hash**, and for each one:

- the new words are screened like any save (a credential becomes a `[REDACTED:TYPE:LAST4]`
  placeholder when `trust.secret_screen` is on) and stored as a correction waiting for review,
  hidden from recall, with origin `external_edit`: a person editing a file on their own disk is
  close to `user_direct`, but nothing tells that person from a tool or an agent that wrote the same
  file, so the edit starts as `agent_inferred` and becomes `user_confirmed` only when you approve
  it;
- a ProposalSet of kind `wiki_edit` is queued, one per edit, so you decide each on its own;
- when you apply it, it is a **compare-and-swap** on the slot version the block was written
  against. If the slot is unchanged, the correction is applied. If the slot moved on since, the
  edit is *not* allowed to replace anything: its words are kept as a memory next to the slot's
  current value and a **dispute** asks you which is true now (the same dispute you answer with
  `slm disputes settle`). Rolling the set back undoes exactly that.

You can edit a span line, a story line (its first sentence) and a current-value line (the whole
value). Other template lines are computed from slot values; edit the memory they cite.

**A deleted line is never a forget.** Deleting a block, or emptying it, changes nothing and is only
counted. SLM ignores its own writes by hash: a file whose hash is the one it recorded is not read.
To forget a memory, forget it in SLM.

### Erasing a memory

Erasing a memory scrubs the sentences that cite it from the compiled pages and rewrites the files
without those blocks (a file left with no block is deleted); anything you edited elsewhere in the
file is kept. Versions and proposals that quote the memory go with it.

## Reading pages

- MCP: `get_memory_summary(kind="page")` lists pages; with `target=<page id>` reads one, with
  `sentences[].facts` (the memory ids), `withheld` and `stale`.
- CLI: `slm wiki status|diff --json`.
- HTTP: `GET /api/wiki/pages`, `GET /api/wiki/pages/{id}`, `GET /api/wiki/diff` (local callers with
  the review-read role). `POST /api/wiki/build`, `/rebuild` and `/sync` need the memory-update role;
  who decided and the channel come from the connection and are recorded (D-49).
- Dashboard: the **Living wiki** card lists pages, shows each sentence with the memory it comes
  from (**Open memory** searches for it), and has **Rebuild pages** and **Read my edits**.

Read a page for an overview; use `recall` for a specific question or the freshest answer: a page
can lag a memory by up to one rebuild.

## Limits

- Pages are about what the store knows, not about anything else. They are as good as the slot
  keys and entities behind them.
- A page holds at most 60 sentences; a topic page takes at most two sentences from each memory.
- The kill switch reads `compiler.wiki` and `compiler.dream` as off. It does not delete pages or
  files.
- Moving a memory between profiles (a fold) deletes the source profile's pages; the dream cycle
  compiles them again for the target.
