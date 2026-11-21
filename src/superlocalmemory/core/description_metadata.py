# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Conservative descriptions of a memory, without inventing event dates."""

from __future__ import annotations

import re

from superlocalmemory.core.enriched_fact import temporal_fields
from superlocalmemory.encoding.entity_mentions import extract_mentions
from superlocalmemory.encoding.entity_validity import valid_names
from superlocalmemory.encoding.non_answer import is_stub
from superlocalmemory.encoding.temporal_parser import TemporalParser
from superlocalmemory.storage.models import MemoryRecord

VERSION = 'description-v1'
MAX_CONTENT = 24_000


def parent_labels(db, memory: dict) -> dict:
    text = memory['content'][:MAX_CONTENT]
    tags = sorted(set(re.findall(r'(?<![\w])#([A-Za-z][\w-]{1,40})\b', text)))[:12]
    project = None
    patterns = (r'\b([A-Z][\w.-]{1,60})\s+(?:is|was)\s+(?:our|my|the|a)\s+project\b',
                r'\bproject\s*:\s*([A-Z][\w.-]{1,60})\b',
                r'\b([A-Z][\w.-]{1,60})\s+project\b')
    for pattern in patterns:
        names = valid_names(re.findall(pattern, text), text)
        if len(set(names)) == 1:
            project = names[0]
            break
    if not tags:
        tags = valid_names(extract_mentions(text), text)[:8]
    # Same profile, session AND sharing boundary; only unanimous sibling labels.
    if memory.get('session_id'):
        siblings = db.execute(
            'SELECT metadata_json FROM memories WHERE profile_id=? AND session_id=? '
            'AND memory_id!=? AND scope=? AND shared_with IS ? LIMIT 20',
            (memory['profile_id'], memory['session_id'], memory['memory_id'],
             memory.get('scope', 'personal'), memory.get('shared_with')))
        from superlocalmemory.storage.json_columns import load_json_column

        # A sibling whose metadata is bytes, not text, offers no label.
        values = [load_json_column(r['metadata_json'], {}, column='memories.metadata_json')
                  for r in siblings]
        for key, found in (('tags', tags), ('project', project)):
            candidates = [v[key] for v in values if v.get(key)]
            if not found and len(candidates) >= 2 and all(v == candidates[0] for v in candidates):
                if key == 'tags':
                    tags = candidates[0]
                else:
                    project = candidates[0]
    return {k: v for k, v in {'tags': tags, 'project': project}.items() if v}


def fact_description(fact, memory: dict) -> dict:
    record = MemoryRecord(memory_id=fact.memory_id, profile_id=fact.profile_id,
                          content=memory['content'],
                          session_date=memory.get('session_date') or fact.created_at,
                          created_at=memory['created_at'])
    dates = temporal_fields(fact, record, TemporalParser(fact.created_at))
    return {
        # When the memory was recorded: a stand-in for a missing observation
        # date, labelled inferred by the runner and never used as an event start.
        'observation_date': fact.observation_date or fact.created_at,
        'referenced_date': fact.referenced_date or dates.get('referenced_date'),
        'interval_start': fact.interval_start or dates.get('interval_start'),
        'interval_end': fact.interval_end or dates.get('interval_end'),
        'entities_json': valid_names(
            list(dict.fromkeys([*fact.entities, *extract_mentions(fact.content[:MAX_CONTENT])])),
            fact.content[:MAX_CONTENT]),
        'non_answer': int(is_stub(fact.content)),
    }
