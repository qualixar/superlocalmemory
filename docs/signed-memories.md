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
  `slm signing resign` (you, at an interactive terminal, with the daemon stopped). It is never
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

Limits, plainly: signing attests "unchanged since it was signed", not "never altered before". The
first time signing runs it seals what is in the store. A signature row deleted by someone with
the file makes that memory read "unsigned" (reported, never as intact), but the next batch signs
it as it then stands. Moving a memory to another profile by editing both tables by hand looks
like a profile fold and is signed again.

## Profile folds, export and import

* **A profile fold** (deleting a profile and folding its memories into another) moves each
  memory and its signature row; only `profile_id` changes. The row keeps `signed_profile_id`,
  the profile the signature covers. The daemon's batch verifies the old signature over that
  profile and, only if it holds, signs again for the new one. The fold itself never holds the
  key, so it does not sign. Until the batch runs the memory is counted "moved by a fold", not
  modified. A bad signature stays bad through a fold.
* **A repair that rewrites a memory's text** (restoring text a redaction blanked) drops that
  memory's signature, so it is signed again with the restored text.
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
  does not hold".
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
