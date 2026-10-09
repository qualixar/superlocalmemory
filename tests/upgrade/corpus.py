# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Invented memories and fixed recall queries for the upgrade/downgrade check.

Everything here is synthetic: people, projects and numbers are made up.  Each
memory ends with a reference ``QX-nnnn`` so a recalled result can be mapped back
to its corpus id by the reference alone, whatever the stored wording becomes.
"""

from __future__ import annotations

import re
from typing import NamedTuple

REF_PATTERN = re.compile(r"\bQX-(\d{4})\b")


class Memory(NamedTuple):
    id: str
    text: str
    tag: str


class Query(NamedTuple):
    text: str
    expected: tuple[str, ...]


_ROWS: tuple[tuple[str, str], ...] = (
    ('Priya Raman approved a budget of 48200 euros for Project Alder on 2026-03-14.', 'budget'),
    ('Tomas Lindqvist moved the Birchwood release from 2026-05-02 to 2026-06-19.', 'schedule'),
    ('Aiko Tanabe chose PostgreSQL 16 over MariaDB for the Cedar ledger service.', 'decision'),
    ('Ibrahim Okafor measured 212 ms p95 latency on the Dunmore gateway after the cache change.', 'perf'),
    ('Marta Kowalczyk runs the Thursday retro for the Elmstead team at 15:30 Warsaw time.', 'ritual'),
    ('Declan Whitfield rotated the Fenwick signing keys on 2026-02-09 and logged ticket 7731.', 'security'),
    ('Sofia Marchetti prefers dark roast coffee and refuses meetings before 09:00.', 'preference'),
    ('Kwame Mensah owns the Garnet mobile app and reports to Priya Raman.', 'org'),
    ('The Hollin cluster has 14 nodes, each with 64 GB memory, in the Frankfurt region.', 'infra'),
    ('Project Alder deadline is 2026-09-30 and the team of 6 engineers is on track.', 'schedule'),
    ('Birchwood uses feature flags stored in a YAML file called flags-prod.yaml.', 'config'),
    ('Cedar ledger retention is 7 years because the auditors in Lisbon require it.', 'compliance'),
    ('Dunmore gateway returns HTTP 429 after 600 requests per minute per API key.', 'limits'),
    ('Elmstead team agreed to freeze merges every Friday after 14:00.', 'process'),
    ('Fenwick certificate expires on 2027-01-21 and renewal costs 310 euros.', 'security'),
    ('Garnet app crashed 38 times last week, mostly on Android 13 devices.', 'quality'),
    ('Hollin nightly backup starts at 02:15 and takes about 47 minutes.', 'ops'),
    ('Ibrahim Okafor wrote the incident review for the Dunmore outage of 2026-04-03.', 'incident'),
    ('Marta Kowalczyk proposed moving Elmstead to trunk-based development in Q3.', 'proposal'),
    ('Aiko Tanabe is vacationing in Kyoto from 2026-08-10 to 2026-08-24.', 'calendar'),
    ('The Juniper dataset contains 1.2 million rows of anonymised sensor readings.', 'data'),
    ('Kestrel build pipeline takes 18 minutes and caches dependencies in object storage.', 'ci'),
    ('Lantern dashboard shows revenue in Swiss francs rounded to the nearest 100.', 'reporting'),
    ('Sofia Marchetti decided that Lantern alerts go to the on-call rota, not email.', 'decision'),
    ('Declan Whitfield documented the Mulberry onboarding checklist with 12 steps.', 'docs'),
    ('Nettle service uses Redis 7 with a 4 GB limit and allkeys-lru eviction.', 'infra'),
    ('Orchard research spike concluded vector search was not needed for under 5000 documents.', 'research'),
    ('Kwame Mensah asked for a Garnet accessibility audit before the November launch.', 'request'),
    ('Pennant invoices are issued on the 25th and paid within 30 days by Brightwater Ltd.', 'finance'),
    ('Quince migration moved 9 databases to the new Oslo datacentre over a weekend.', 'ops'),
    ("Priya Raman's team wrote a postmortem template with sections for impact and timeline.", 'docs'),
    ('Rowan intern programme hosts 5 students from June to August each year.', 'hr'),
    ('Sorrel mobile SDK version 3.8.2 dropped support for iOS 14.', 'release'),
    ('Tamarind search latency target is 150 ms for the 95th percentile.', 'perf'),
    ('Umber legacy cron jobs were replaced by a scheduler with retry and alerting.', 'ops'),
    ('Tomas Lindqvist reviews Birchwood pull requests every morning before standup.', 'ritual'),
    ('Vetch pricing tiers are Basic at 9 dollars, Plus at 29 dollars and Pro at 79 dollars.', 'pricing'),
    ('Willow style guide bans abbreviations in user-facing error messages.', 'docs'),
    ('Yarrow load test peaked at 3400 concurrent sessions before errors appeared.', 'perf'),
    ('Zephyr team moved its design reviews to Wednesdays at 11:00 Lisbon time.', 'ritual'),
)

MEMORIES: tuple[Memory, ...] = tuple(
    Memory(id=f"QX-{1000 + i}", text=f"{text} Ref QX-{1000 + i}", tag=tag)
    for i, (text, tag) in enumerate(_ROWS, start=1)
)

_QUERIES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ('who approved the Alder budget in euros', (1,)),
    ('when was the Birchwood release moved', (2,)),
    ('which database did Cedar ledger choose', (3,)),
    ('Dunmore gateway p95 latency after cache', (4,)),
    ('Fenwick signing keys rotation ticket', (6,)),
    ('Hollin cluster nodes Frankfurt memory', (9,)),
    ('Garnet app crashes Android', (16,)),
    ('Hollin nightly backup start time', (17,)),
    ('Juniper dataset rows sensor readings', (21,)),
    ('Nettle Redis eviction limit', (26,)),
    ('Vetch pricing tiers dollars', (37,)),
    ('Yarrow load test concurrent sessions', (39,)),
)

QUERIES: tuple[Query, ...] = tuple(
    Query(text=text, expected=tuple(f"QX-{1000 + n}" for n in nums))
    for text, nums in _QUERIES
)


def ref_of(text: str) -> str | None:
    """Return the corpus id (``QX-nnnn``) found in ``text``, or None."""
    match = REF_PATTERN.search(text or "")
    return f"QX-{match.group(1)}" if match else None
