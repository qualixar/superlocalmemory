# Answer Check

Answer check lets your memory say "I don't have that" instead of guessing. Recall already ranks stored memories by relevance and returns the best matches. Answer check adds a second, separate judgment: it looks at the top few results and decides whether they actually answer the question, which is a different thing from whether they are related to it. A memory can be the closest match SLM has and still not answer what you asked. The check's verdict is reported on every recall it ran for, and it is one of the inputs to the support check that decides whether recall returns the results, returns only the closest labelled ones, or returns nothing with "That isn't in SuperLocalMemory." ([How Recall Works](How-Recall-Works)). The check alone hides nothing: with it off, recall still decides support from its other signals.

That distinction is the core of AI Reliability Engineering: an agent that knows the difference between "found something related" and "found the answer" fails safely instead of confidently making things up. It runs after the evidence floor, which only asks whether a result earned any real evidence at all; see [How Recall Works](How-Recall-Works).

## The three choices

Open **Settings, Answer check**. Exactly one of these runs at a time.

| Choice | Where it runs | What leaves your computer |
|---|---|---|
| **On this Mac** (private, recommended) | A small model called Laya, about 1.1 GB downloaded once and about 1.1 GB of memory while it is on. Apple Silicon only | Nothing |
| **Online with Jev** (needs a key) | A hosted provider you choose and pay directly, TypeSafe or OpenRouter, with your own key | Your question and the top 3 memories for it. Off until you add a key and tick a consent box |
| **Off** | Nowhere | Nothing |

Until you choose, the check is in its starting state, called `auto`. Auto uses the on-device check whenever a working install exists that passed its own check; with no install it behaves exactly like Off. On Apple Silicon SLM installs the on-device model for you (see Setting it up), so there auto is normally on from the first day. **Auto never turns the online option on by itself, and it never resolves to Jev**, so data leaves your computer only when you chose that. Once you pick one of the three, your choice sticks: picking Off means a later setup does not turn the check back on until you pick On this Mac. Switching SLM's mode (A, B or C) never turns Jev back on.

## What recall tells you

A recall response carries these fields, and the CLI and your assistant's session get one extra line when there is something to say. The first two come from the support check, the last three from the answer check.

| Field | Meaning |
|---|---|
| `tier` | `supported`, `related` or `none`: whether a memory supports the question, only close ones exist, or nothing is in memory |
| `abstained` | `true` when no memory supports the question, or when nothing confident was found at all |
| `abstention_note` | The plain sentence for a person to read, for example "That isn't in SuperLocalMemory." |
| `abstention_reason` | `unsupported` or `judged_insufficient` when memories were found but none answer it; `evidence_floor` or `no_candidates` when nothing confident was found in the first place |
| `answer_confidence` | How confident the check is that the top results answer the question, 0.0 to 1.0 |
| `answer_check_status` | What happened on this recall: `judged`, `off`, `skipped`, `busy`, `warming` or `unavailable` |

If no memory supports the question, recall returns no results (or only the labelled closest ones, for a person), and the assistant sees a line telling it to say it does not have the answer, or to ask, and not to present these memories as the answer. The instructions SLM ships for Claude Code, Codex, Cursor, Copilot, Antigravity and Hermes all say the same thing. A [bounded loop](Bounded-Loops) waiting on a memory honours it too: a recall the check calls insufficient can never pass that loop's gate.

A check can be skipped, and skipping costs only the verdict. It is skipped for a context load or a system recall, when there are no results, when too little of the recall's time budget is left, or when the online option would have had to read another profile's memory. A question's check shares the question's own time, so on a large store it still runs: in the measurement of 24 questions over a 27,000-memory store it ran on all 24, and the check itself takes 30 to 140 ms. A hook's recall keeps its short three-second limit, so it can still skip the check. The check only ever sees the handful of results recall was already about to return. It never runs its own search and never sees more than you do. A picture with no readable text is reported as not judged, because a text model would only be scoring its label.

## Reordering results with Jev (optional)

Inside **Online with Jev** there is one more switch, **Also use Jev to reorder results**. It is off by default, upgrading never turns it on, and it has its own notice and its own consent because it sends more than the answer check alone. The same single request also asks Jev which of your top memories answers the question most directly, and recall puts them in that order. It sends your question and up to 20 memories instead of 3, each shortened and stripped of credential-shaped text first. Nothing is removed or hidden; only the order of the top results changes. If the provider is slow or answers something unexpected, recall returns the usual order as if the switch were off. Laya never reorders.

On the project's benchmark of long conversations (LoCoMo, 1,531 questions), the memory that answers the question came first 62 percent of the time with reordering on against 49 percent without it, and reordering adds about 0.4 seconds to a recall. That figure is from the project documentation; the repository's `docs/benchmarks.md` has the protocol.

**At session start (4.2.17).** With reordering on, a session start uses it too: the SessionStart hook (`slm session-context`) and `session_init` ask Jev once for the order of the memories they load, against your first message or, without one, the project ("project context <folder>"). Only the order is used. The answer check in that request is discarded, because loading context is not a question, so a session start never says "That isn't in SuperLocalMemory" and nothing is hidden. It waits at most 1.5 seconds and keeps the usual order on any failure. With reordering off nothing is sent.

The reorder switches itself off, and stays off until you tick it again, when you switch to On this Mac or Off, remove the key, untick the Jev notice or change provider.

## Privacy, by choice

**On this Mac:** nothing about your questions or memories leaves the machine for this feature. The model runs in a local worker that is forbidden from using the network, and the only network use is the one-time setup download.

**Online with Jev:** each checked recall sends the question and the top 3 memories to the provider you chose. Known vendor key formats, a password in a connection URL and a value after a credential label such as `password=` or `token=` are stripped first. A hex string or UUID counts as a credential only when it follows a key-like label, so a commit id or checksum is sent as it is. Ordinary personal details are not stripped, so do not turn this on for stores that hold client, confidential or personal material. If what the check would read includes a memory owned by another profile, such as a shared or global result, the check is skipped for that recall: your consent to send covers your own memories, not a teammate's. The consent flag must be literally true; a hand-edited "true" string does not count. Withdrawing consent turns the online check off immediately, and SLM confirms it has actually stopped before saying so. See [Privacy and Data Flow](Privacy-and-Data-Flow).

## Setting it up

Everything is configured in the dashboard. Open **Settings, Answer check**. For **On this Mac**, press **Set up** and SLM downloads and verifies the model in the background with a progress bar, until the status reads Ready. **Remove** stops the model and frees the disk, and **Use an install you already have** lets you point at a Python install and model you already have. For **Online with Jev**, pick a provider, paste a key and save it (only the last four characters are ever shown back), press **Test** to send one real request, and tick the consent box, which spells out exactly what is sent. **On Apple Silicon the on-device check is installed by default.** Interactive setup asks "Set it up now? [Y/n]" with yes as the default. An unattended setup (`slm setup --auto`, CI, or any run without a terminal) installs it without asking, and says how to opt out. After an upgrade, the first `slm` command offers it the same way: it asks in a terminal, and without a terminal it starts the install in the background and logs it to `runtimes/laya/install-after-upgrade.log` in your data folder. The offer goes to the error stream, so it never corrupts a command's output or a hook's JSON. The check does not depend on Mode A, B or C; it runs in whichever mode you use.

To opt out, run `slm setup --no-laya` or set `SLM_SKIP_LAYA=1` (`true`, `yes` and `on` also count; the old `SLM_INSTALL_LAYA=0` still works). Answering no once is remembered, so you are not asked again. SLM never installs it on hardware other than Apple Silicon, over an install you already have, or when you have set the check to Off or to the online option. While the model is still downloading or warming up, recall behaves as if the check were off, and the support check works from its other signals.

## The Answer Check tab

The dashboard's first tab shows the check at work: which check runs and why, a **Try it** box that shows what retrieval found next to what the check said and how long each took against the 3-second limit, how often it said "I don't have that" over 24 hours, 7 days or 30 days (shown only after at least 20 recalls were judged, with its 95 percent range), typical and slow timings, and a live list of recent checks.

The history keeps outcomes and timings only, never your question, any memory text, memory ids or agent names. It stays on this machine in `learning.db`, separately for each profile, for 30 days and at most 10,000 checks per profile. **Clear history** removes it; a data export includes it, and erasing a profile erases it too.

## Troubleshooting

| Problem | What to do |
|---|---|
| The Mac is not Apple Silicon | On this Mac needs Apple Silicon. Use Online with Jev, or leave the check off |
| A company network blocks the download | Open **Use an install you already have** under the On this Mac panel and point it at a Python interpreter and model you already have, for example one copied from another machine. For the cache folder give either your `HF_HOME` or the `hub` folder inside it. SLM accepts only an interpreter it can trust: the `python` inside a Python environment, owned by you or the system, changeable by no other account, and not in a temporary folder or under `/Users/Shared` |
| "The saved key file was changed outside SLM" | The key file no longer looks like the one SLM wrote, so SLM will not read it and the online option stays off. Choose **Remove key**, then paste your key again |
| A key test fails | The result is in plain language: key not accepted (check or regenerate it), no credit left (top up with your provider), too many requests (wait a minute), or the service did not answer or could not be reached (try again shortly) |
| "Answer check runs in another SLM process" | Expected, not an error. When several SLM processes share one data folder, only one runs the on-device model and the rest report their recalls as if the check were off. Recalls from the CLI, MCP tools and dashboard normally go through the daemon, which usually holds the model |
| Reclaiming disk | The on-device model and its environment live in `runtimes/laya` in your data folder. **Remove** in the dashboard deletes it. Switching to Online with Jev or Off stops the model but keeps the files so you can switch back without a download. An install you pointed SLM at is never deleted by SLM |

Code: `core/answer_check_stage.py`, `core/judge_selection.py`, `retrieval/sufficiency.py`, `retrieval/answer_check_status.py`, `retrieval/laya_transport.py`, `retrieval/jev_judge.py`, `retrieval/jev_rerank.py`, `core/laya_worker.py`, `server/routes/answer_check.py`.
