# Signed memories, the keyed audit chain and `slm why`

Part of 4.2.30 (WS-5). Off by default: `slm flags set trust.signing shadow|on` (`shadow` and `on`
do the same thing, because signing only writes its own table), and the next daemon start picks
it up.

## What it covers, and what it does not

Each memory gets a signature that a copy of your files cannot forge. The key lives in your
operating system's keychain and nowhere else.

**It detects** a memory that was changed, added or moved by something that has the *file* but
not your *login session*: a synced or restored copy, another user account on the computer, a
disk image, a backup someone edited. That party cannot read your keychain, so it cannot make a
signature that checks.

**It does not stop** a process running as you. That process can ask the keychain for the key the
same way SuperLocalMemory does. Signing is tamper evidence, not access control.

## The key

* macOS Keychain, the Windows Credential Manager (which Windows protects with DPAPI) or the
  Secret Service (libsecret on Linux): the same secure store SLM already uses for remote
  connections. Where none is available, signing is **unavailable** and says so. SLM never writes
  a key file as a fallback.
* The daemon reads the key **once, when it starts**, with a five-second limit, and holds it in
  memory. A hook never asks for it and neither does recall. `slm doctor --truth`, `slm signing`
  and `slm why --verify` are commands you run on purpose, so they read it themselves, once.
* **A lost key loses no data.** Memories signed under it are reported as "signed under another
  key", never as modified, and nothing is deleted. Sign them again under the new key with
  `slm signing resign` (any caller: an agent passes `--yes`; recorded as a decision; daemon stopped). It is never
  done automatically: a pass that re-signed everything with a new key would make a copied,
  edited file look valid.

## What is signed

HMAC-SHA256 over one canonical JSON object: the profile, the memory's id, the SHA-256 of its text
and its `created_at`. Stored in `fact_signatures` (migration M082), with the id of the key that
made it (a digest of the key, never the key). No memory text is stored there.

Signed when the daemon saves the memory (key held, flag on) and, for memories saved by a hook, a
command-line save or before the flag was turned on, by a small background batch in the daemon's
maintenance cycle (200 per batch). A signature is written once; the batch never rewrites the
signature of a memory that has changed, so a changed memory stays "modified" until you decide.

### The signing mark: a removed signature is not a new memory

Someone with the file can change a memory's text and also delete its signature row. If the batch
then signed "whatever is unsigned", the changed memory would read "signed, and unchanged". So the
daemon keeps a second record, `signing_progress` (one row, no memory text): the highest memory row
number it has signed up to, the time signing first ran, and a MAC under the same keychain key. A
memory saved through the daemon moves the mark up to itself the moment it is signed; the batch
moves it up over the memories it signs.

* The batch signs only unsigned memories **above** the mark. An unsigned memory **at or below** it
  was signed once, so it reads **"signature missing"**: `slm doctor --truth` fails, `slm why` says
  "Its signature was removed", the dashboard says "no longer have their signature". The batch never
  signs it.
* Someone with the file cannot make a mark that checks, so cannot lower it. A mark that is gone
  while signed memories remain, or that does not check, is a failure too, and the batch signs
  nothing until you look. A mark made under another key (you lost the key) is reported, not a
  failure, and waits for you.
* **`slm signing resign`** (any caller, recorded as a decision, daemon stopped) is the one door back:
  it shows how many memories lost their signature and asks; on yes it signs them, and the
  memories signed under an old key, as they stand now, and writes the mark again. Look at them
  first with `slm why FACT_ID`.
* The first time signing runs it seals the memories already in the store, in the same bounded
  batches (the mark moves up as each batch is signed), so a signature removed between two batches
  of that first sealing is already refused.

What it still does not cover, plainly:

* A memory that existed when signing first ran and was already changed then is sealed as it was.
* The mark is a row number. Someone who replaces a memory by a **new row above the mark** (new text,
  no signature) makes it look like a memory saved later, and the batch signs it. A copy of an older
  valid mark put back lowers the protection to that point. Signatures can only vouch for what the
  daemon has seen; they are tamper evidence for a copy of the file, not for a person who rebuilds a
  row and the table around it.
* Erasing the newest memories and then saving with a process that holds no key (a hook, or a
  command-line save with the daemon down) before the next batch can reuse an erased row number at or
  below the mark: that memory reads "signature missing" until you run `slm signing resign`. A save
  through the daemon is signed at admission and never shows this. A repair that rewrites a
  memory's text (restoring text a redaction blanked) also removes its signature and, for the same
  reason, waits for `slm signing resign`.
* **Not signed: `pinned`, the memory kind (rule, decision, ...) and the taint class** (the origin
  classes of `fact_trust`). Someone with the file can flip a memory to "pinned" and "rule", or
  give it a trusted taint class, and its text signature still checks. They are left out on
  purpose: these fields change legitimately through processes that do not hold the key (a hook,
  a pin from the command line or an agent, the kind classifier, an explicit promotion), so signing them would either make
  every legitimate change read "modified" or put the key into those processes, which D-16 rules
  out ("never from a hook"). What that leaves open: a rewritten file can promote a memory to a
  standing instruction with its signature intact. The instruction firewall (`trust.taint`) does not
  close it either, because the taint class lives in the same file. `slm doctor --truth` and
  `slm why` therefore report the text signature, not the standing-instruction fields. Treat a
  copied or restored file you did not make yourself as untrusted for rules and pins, and review
  your pins and standing rules after restoring it (`slm trust pending` lists those below the
  firewall floor).

## Profile folds, export and import

* **A profile fold** (deleting a profile and folding its memories into another) moves each
  memory and its signature row; only `profile_id` changes. The row keeps `signed_profile_id`,
  the profile the signature covers. The daemon's batch verifies the old signature over that
  profile and, only if it holds, signs again for the new one. The fold itself never holds the
  key, so it does not sign. Until the batch runs the memory is counted "moved by a fold", not
  modified. A bad signature stays bad through a fold.
* **A repair that rewrites a memory's text** (restoring text a redaction blanked) drops that
  memory's signature; it reads "signature missing" until you run `slm signing resign` (see the
  signing mark above).
* **Export** (`slm gdpr export` and the dashboard's export) writes memories, not signatures: a signature means
  nothing off the machine whose keychain made it.
* **Import** saves memories through the normal save path, so the importing computer signs them
  with its own key when its daemon holds one. An imported memory is signed as "unchanged since it
  arrived here", which says nothing about the exporter.
* **A backup or restore point** is a copy of the store, signatures included. Restored on the same
  computer and login it verifies; restored on another computer it reports "signed under another
  key" for every memory, which is correct: that computer's keychain does not hold the key.
* **Erasing a memory** erases its signature row in the same transaction.

## The keyed audit chain

The audit chain (`audit_chain.db`) is a hash chain. Until now each link was a plain SHA-256, so
someone who could rewrite the file could recompute every hash. With `trust.signing` on, each new
link is an **HMAC under the same keychain key**, so a copy made without your login session cannot
be rewritten and still verify.

* An **anchor entry** (`key_anchor`) is written before the first keyed link, and again whenever
  the key changes (a lost or replaced key) or the chain restarts empty. It records the new key id,
  the previous key id and why.
* Links written before keying began, and by a process that does not hold the key (a command run
  while the daemon is down), are plain links and stay verifiable by their plain hash. The report
  counts plain links that follow the first keyed one. They are shown, not trusted as keyed.
* Links made under an earlier key cannot be checked by HMAC once that key is gone; their order and
  previous-links are still checked and the report counts them as "keyed under a key this computer
  does not hold". **When this computer does hold a key, those links pass only if a later anchor
  keyed under your current key stands behind them** (that anchor's HMAC covers the hash of the
  entry before it, so it vouches for everything written earlier). Someone with the file and no key
  could give every link, and the anchor, a made-up key id and rewrite an entry; the order, the
  previous-links and the external anchor would still agree. With your key present that now fails:
  "no anchor under your current key vouches for them". After a lost key, the daemon writes that
  anchor at its next audit entry, so the check reads as failed until then. Without a key nothing
  can be said about those links beyond their order. Still open: the content of links made under a
  key that is gone (only the head they ended on is vouched for), and a keyed stretch rewritten as
  plain links.
* A keyed link must carry the key the latest anchor declared, so a plain or foreign link cannot be
  relabelled as keyed to escape its check.

Verify with `slm doctor --truth` (it reads the chain beside the store, read-only) or
`slm signing status`.

## `slm doctor --truth`: the signatures check

Reports how many memories are signed and unsigned, then verifies a **random sample of 500**. The
sample size is routing-only: it bounds the work of a doctor run and is not a detection
guarantee. A sample finds one modified memory among N with probability about 500/N. Use
`--signatures-all` (or `slm signing status --all`) to check every signed memory. It names modified
memories by id and never prints their text. A modified memory or a broken audit link makes the
command exit 1; unsigned memories, signatures under another key and plain links after keying began
are reported and are not failures.

The dashboard's **Truth checks** card (Health page, **Run checks**) and **Trust overview** card (**Check now**) say the same two results in words: how many memories are signed, whether any no longer matches its signature, and whether the audit log is intact (`GET /api/truth/checks`, `GET /api/trust/overview`, both read-only). The dashboard checks with the key the daemon already holds in memory and never asks the keychain; if the daemon does not hold it (signing is off or unavailable), it gives the counts and says nothing was checked, never "intact".

## `slm why`

```bash
slm why FACT_ID [--json] [--verify] [--deliveries N]
```

Everything that explains one memory, read-only (no daemon, no write): where it came from (class,
taint, the raw signals, the mapping row that decided it and why that row exists, the parents that
lowered the taint); whether the instruction firewall holds it back and who approved it; its
signature; its slots, open disputes and source-file state; derived work that mentions it
(versions and proposals, by id and kind); and the last times it was shown to an agent, each with
the reason in words (session start, a prompt, a cue, an explicit recall, its rank, and whether
randomised withholding left it out). `--verify` reads the key once to check the signature.

The same report is `GET /api/trust/why/{fact_id}` (the review-read role on the profile; a memory
in another profile is a 404), `fetch(fact_ids, why=true)` on the MCP tools (at most 10 memories per
call; no new tool, so the tool caps do not change), and the **Why was this shown?** section of a
memory's detail in the dashboard (read when you ask).
