# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Routing constants of the portability pack (WS-9, D-19).

Every number here bounds work, memory or risk. None of them decides what is true about a
memory, and none is a tuned ranking weight (D-05): they say how big a file may be, how many
saves go out at once, and how many contradictions one import may put in front of a reader.
"""

from __future__ import annotations

#: The flag that switches the pack on (``core/feature_flags``); off by default (D-19).
FLAG = "portability.import"

#: An upload through HTTP (dashboard or remote app): at most this many bytes after decoding.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
#: A file or folder named by a path on this computer (command line): at most this many bytes
#: in total. Bigger exports are split by their makers (ChatGPT writes ``conversations-000.json``).
MAX_LOCAL_BYTES = 512 * 1024 * 1024
#: Everything a zip may unpack to, and how many files it may hold (a zip bomb stops here).
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_FILES = 5_000
#: A folder of Markdown files: how many files are read.
MAX_FOLDER_FILES = 2_000
#: How long a staged upload waits for the person to press Import, then it is deleted.
STAGING_TTL_S = 24 * 3600

#: One memory per section, and a section longer than this is cut at paragraph breaks.
SECTION_LIMIT = 4_000
#: One conversation turn longer than this is cut at paragraph breaks (the writer's own limit
#: is 24,000 characters; a turn this long is a pasted document).
TURN_LIMIT = 12_000

#: Saves sent to the writer before the job lets recalls through and records its progress.
BATCH = 25
#: Longest the job waits for recalls in flight between two batches (``recall_gate``).
YIELD_MAX_S = 5.0
#: A short rest between batches so agents using SLM at the same time are never starved.
REST_BETWEEN_BATCHES_S = 0.15

#: How many contradictions with existing memory one run may open for a reader. Beyond this the
#: imported text is still kept (it is simply not listed as a question), and the run says how
#: many it left out (``disputes_unlisted``). Routing only: it protects the review queue.
DISPUTES_PER_RUN = 50
#: How many judge calls one run may cause. Zero: an imported fact is at the bottom of the
#: origin lattice, so no model could retire what it contradicts, and asking one would only
#: spend its calls. A contradiction waits for a recorded decision instead.
JUDGE_CALLS_PER_RUN = 0

#: Rollback: how many memories are removed in one short transaction, and how long it waits
#: for a save that is still being enriched to settle before it looks again.
ROLLBACK_BATCH = 50
ROLLBACK_SETTLE_S = 6.0
ROLLBACK_PASSES = 4

#: Preview samples (counts only; no memory text is ever shown on a preview).
SKIP_REASONS_SHOWN = 6
