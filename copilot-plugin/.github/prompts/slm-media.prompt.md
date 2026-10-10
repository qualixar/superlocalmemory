---
name: slm-media
description: Save pictures and PDFs in SuperLocalMemory and find them again by describing them. Covers when to save a picture or document, remember_media, remember_document, get_media, media_status and media_upload_link, how recall returns pictures and PDF pages, the limits, what stays private, turning the feature on with `slm media` (16 GB of memory needed), connecting folders with `slm sources`, and how web apps add files through one-time upload links. Off by default.
version: "4.1.25"
agent: agent
tools:
  - remember_media
  - remember_document
  - get_media
  - media_status
  - media_upload_link
  - recall
  - Bash
---

# slm-media — Pictures, PDFs and Folders

SuperLocalMemory can keep a picture or a PDF as a memory and find it later by
describing what it shows or what it says. A picture model runs on this
computer (EmbeddingGemma 2). OCR reads the text inside pictures. Each PDF
page is indexed by its picture and by its text. No generative model looks at
the files, so SLM cannot describe a picture to you: it finds it, and you look
at the thumbnail it returns.

This feature is **off by default**. Pictures live in their own database
(`media.db`); turning it on does not change any existing memory or embedding.

---

## When to save a picture or a PDF

Save one when the user asks, or when the file is itself the thing worth
finding again: a screenshot of an error they will meet again, a diagram of a
design decision, a receipt, a scanned page, a PDF spec. Do not save a file
just because it was shown in the conversation.

Add words with `content`. The picture is found by what it shows, by the text
read inside it, and by the words you add, so a short sentence about why it
matters ("Staging dashboard after the Postgres 17 move") helps recall.

Never save a picture or PDF that shows a password, key, token, card number,
government id or private data about a third party. The text read from a file
is scanned like any saved text, but a screenshot of a secret should not be
saved at all. Ask the user first if you are unsure.

---

## Are the tools there?

The five tools below are listed only while pictures and documents are on, and
only in the `full`, `power` and `whole` MCP tool sets. They are not in `core`
or `code`. The tool set is fixed when the MCP server starts, so after the
feature is turned on, restart the MCP session (`slm restart` starts the
picture worker too). Do not guess: run

```bash
slm media status
```

It says whether the feature is off, set up, still setting up, or failed
(`slm doctor` explains a failure). The same status is in the dashboard under
**Documents & Images**.

### Turning it on (ask the user first)

```bash
slm media enable    # shows the download size and a disk check, then asks; --yes skips the question
```

Turning it on downloads about 1.5 GB of models and tools, and needs a
computer with 16 GB of memory (the check passes from 15 GiB). On a smaller
computer `enable` refuses with the reason and exits with code 4; text memory
keeps working. Never turn it on, or pass `--yes`, without the user saying
yes in this conversation. Never set `SLM_MEDIA_ALLOW_LOW_RAM=1` yourself: it
removes the memory check and the owner must decide that.

Supported: macOS on Apple Silicon, Windows x64, Linux x86_64 and Linux ARM64
(aarch64), Python 3.12 to 3.14. Not supported: Intel Macs and Windows on ARM.

Other commands:

```bash
slm media status              # what is on and how set-up is going
slm media disable             # off again; memories are kept (--remove-files also deletes the downloaded models)
slm media gc                  # report picture records without a memory and files without a record; removes nothing
slm media gc --apply          # remove them (owner or admin); never touches SLM's databases
slm media repair --dry-run    # count pictures that cannot be found by what they show yet
slm media repair              # re-embed them (owner or admin); run again if it says pictures are left
```

All take `--json`. Run `slm media repair` when a save says a picture "can't be
found by what it shows yet", or after the picture model changed.

---

## Tool reference

### `remember_media` — save a picture

```
remember_media(
  path: str = "", download_url: str = "", base64: str = "", file = None,   # give exactly one
  content: str = "", tags: str = "", profile_id: str = "",
  idempotency_key: str = "", scope: str = "", shared_with: str = "",
) -> dict
```

Give exactly one of `path` (a file on this computer), `download_url` (an
https link), `base64`, or `file` (a picture attached in a ChatGPT chat;
ChatGPT fills it in, never write it by hand). Two or none is refused
(`invalid_request`). `scope` (personal, project, shared, global) and
`shared_with` (comma-separated profile ids) work as in `remember`: leave them
unset unless the user asked to share. Formats: PNG, JPEG, GIF (first frame)
and WebP.

Returns `success`, `status`, and on success a `media_id` and a `resource`,
`slm://media/<media_id>`. Report what the tool returned; do not say "saved"
unless `success` is true.

### `remember_document` — save a PDF

```
remember_document(
  path: str = "", base64: str = "", download_url: str = "", file = None,   # give exactly one
  file_name: str = "", content: str = "", tags: str = "", profile_id: str = "",
  idempotency_key: str = "", scope: str = "", shared_with: str = "",
) -> dict
```

Same sources and rules as `remember_media`, for a PDF. The work continues in
the background; the answer has a `job_id` (and `document_id`). Follow it with
`media_status`. Each page is indexed by its picture and its text layer.

### `media_status` — follow a document

```
media_status(job_id: str, profile_id: str = "") -> dict
```

`job_id` is the 32-character hex id from `remember_document`. Poll sparingly
(every few seconds at most) and stop when the job is finished or failed. A PDF that cannot
be finished (password-protected, over 500 pages, a page that takes too long)
is marked not finished with the reason in plain words. Pages read before it
stopped stay in memory. The dashboard has a **Try again** button, or drop the
same file again.

### `get_media` — look at a saved picture

```
get_media(media_id: str, variant: str = "thumb", profile_id: str = "") -> image
```

Returns the thumbnail (WebP) as an image, never as text. Only `thumb`
exists. A thumbnail is at most 32 KB.

### `media_upload_link` — for web apps only

```
media_upload_link(kind: str = "image", note: str = "", profile_id: str = "") -> dict
```

`kind` is `image` or `document`; `note` is the words to save with the file.
Only an app on another computer can call it (an app on this computer gets
`not_for_local` and passes a `path` instead). It returns `url`, `expires_at`
and `max_mb`. Tell the person to open the link in any browser, pick the file
and press Save. The link works once, expires in 10 minutes, belongs to the app
that asked for it, and must not be shared. You cannot see the upload: do not
say the file was saved until the person tells you it was.

### Errors you will see

A refused call returns `success: false`, `status: "refused"`, a `code` and a
plain `error`; `retryable: true` means try again later.

| `code` | Meaning | What to do |
|---|---|---|
| `invalid_request` | Not exactly one source, a bad job id, or a `file` object without `download_url` and `file_id` | Fix the call |
| `not_for_remote` | This app has not been given pictures and documents | Ask the owner to allow it (see below) |
| `path_not_for_remote` | A web app tried to name a file on the owner's computer | Send a link or data, or use `media_upload_link` |
| `too_large_for_remote` | A web app pasted more than 512 KB | Use `media_upload_link` |
| `not_allowed`, `refused` | The daemon refused (role, size, type, a link host not on the allowed list) | Read `error`; do not retry the same call |
| `DAEMON_UNAVAILABLE` | SLM is not running | `slm status`, then `slm restart`; retry once |
| `error` | Unexpected failure, `retryable: true` | Retry once |

---

## How recall returns pictures and pages

A normal `recall` searches pictures and pages beside text memories. A result
that came from a picture or a PDF page has a `media` block:

```json
{
  "media": {
    "media_id": "<32 hex>",
    "kind": "image",
    "thumbnail_uri": "slm://media/<media_id>/thumb",
    "page": null,
    "document_id": null,
    "citation": ""
  }
}
```

For a PDF page, `kind` is `page`, `page` is the page number, `document_id`
names the document and `citation` reads `page 12`. When you call `recall` from
an app on this computer, up to three thumbnails follow the JSON as image
blocks; the thumbnails are never inside `structuredContent`. Use `get_media`
for any other result.

Rules for reading them:

- A score ranks results; it is not a probability that the picture shows what
  was asked. Look at the thumbnail before you say it does.
- If `abstained` is `true`, say you do not have a picture or page that answers
  the question. See `slm-recall`.
- When you quote a PDF, give the page (`citation`) so the person can check it.
- A picture with no readable text is found by what it shows only. It no
  longer comes back for a question about pictures that were never saved.

---

## Limits

| What | Limit |
|---|---|
| One picture | 25 MB from a file, the dashboard or an upload link; 8 MB as pasted `base64` from an app on this computer; 512 KB when a web app pastes it |
| One PDF | 100 MB (change with `SLM_DOC_MAX_MB`); 25 MB pasted from an app on this computer; 512 KB when a web app pastes it |
| PDF pages | 500. A longer PDF is refused as a whole, not cut short |
| Text kept from a picture | The first 8,000 characters |
| Pictures and PDFs per profile | 2 GB |
| Upload links | 3 open at a time per connected app, 20 uploads per app per day |

---

## Privacy: what stays and what leaves

- The models run on this computer. Pictures and PDFs are stored on this
  computer in `media.db` and its picture folder. Location data (GPS) in a
  photo is never read.
- Erasing a memory erases its picture. Removing a PDF hides all its pages.
- Reading a file by `path` needs the owner or an admin in a company
  workspace. SLM's own data folder can never be named as a path or a source.
- A web app sees a picture or page only when it holds the pictures and
  documents permission and the text read from that file held no secret or
  personal data. Folders connected with `slm sources` are never shown to web
  apps. A web app cannot change, delete, pin or replace a memory it is not
  allowed to see: it is answered as if that memory did not exist.
- Pictures saved before 4.1.25 are re-checked in the background and become
  visible to web apps once their text is clean. They stay fully searchable on
  this computer.

---

## Web apps: two yeses, then an upload link

A web app (ChatGPT, Claude on the web, Grok Bot, Muse, Composio) can use
these tools only when all of these are true. This is why a call may be
refused with `not_for_remote` or `not_allowed`.

1. The approval page ticked **Allow images and documents** (and, for saving,
   **Allow saving memories**).
2. The **Web access** row in the dashboard's **Connected apps** has the switch
   **Let these apps save and read pictures and documents** on, or
   `slm remote keys allow web-<connection id> media` was run on the SLM
   computer (find the key with `slm remote keys list`).
3. The key is a write key.

The owner does these steps; an agent cannot. Turning pictures off for the
connection also ends any upload link not yet used.

How a web app adds a file:

- **ChatGPT with an attachment.** Pass the attachment as `file` to
  `remember_media` or `remember_document`. The computer downloads it only from
  ChatGPT's own file hosts plus hosts the owner lists in `SLM_MEDIA_URL_HOSTS`.
  If a save is refused with "That host is not on the allowed list", the owner
  must add the host. Never type a link into `download_url` yourself and expect
  it to work: links a model writes need the owner's list.
- **Anything else, or ChatGPT on a phone.** Call `media_upload_link`, show the
  link, and wait for the person to say they saved the file.

A web app saves to its key's own profile only, cannot name a path, and cannot
pass a file the owner's list does not allow. More: `docs/remote-access/hosts.md`
and `docs/pictures-and-documents.md`.

---

## Folders and Obsidian vaults

```bash
slm sources add ~/Notes --kind obsidian   # or --kind folder; shows what would be read, then asks
slm sources list
slm sources report <id>                   # what was skipped, held back or failed
slm sources rescan <id>
slm sources remove <id> [--purge]         # --purge also erases the memories the folder gave
slm sources forget-empty <id>             # for a folder you emptied on purpose
```

A source is read-only: SLM mirrors Markdown, text, canvas files, PDFs and
PNG, JPEG and WebP images into memory and never writes to the folder. Files
that look like they hold a secret are held back and listed in `report`. PDFs
and pictures inside a folder are read only while pictures and documents are
on; files skipped earlier are read on the next scan once they are. Connecting
a folder needs the owner or an admin. `add` asks before it connects and
`remove --purge` needs the typed word `erase`; without a terminal both need
`--yes`, so do not pass `--yes` unless the user said yes to that folder. In
the dashboard, **Choose folder** opens the computer's own folder picker.

---

## Upgrade memory engine (preview)

With pictures and documents on, `slm embedder upgrade` (or the card in
Settings) re-reads existing memories with the same on-device model that reads
pictures, so text and pictures share one search space. It runs in the
background, recall keeps working, the previous engine's data stays until the
owner frees it (`slm embedder forget-previous`), and `slm embedder rollback`
goes back. It is opt-in and never starts on its own. If it cannot run, the
command prints the reason. Offer it; do not run it unasked.

---

## Related skills

- `slm-recall` — reading results and the answer check
- `slm-remember` — text memories; use it for facts, this skill for files
- `slm-web-access` — connecting web apps and the permissions above
- `slm-mesh` — bots can pass `media:<id>` references in messages
- `slm-governance` — roles that gate path reads and folders
- `slm-status` — `slm media status` and `slm doctor`

---

*SuperLocalMemory v4.1.25 · Qualixar · AGPL-3.0-or-later*
