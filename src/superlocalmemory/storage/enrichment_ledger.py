# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Compare-and-set metadata with attribution and conflict-safe undo."""

from __future__ import annotations

import base64
import json

from superlocalmemory.core.description_metadata import VERSION
from superlocalmemory.storage.json_columns import is_damaged, load_json_column

FIELDS = {
    'atomic_facts': frozenset({'observation_date', 'referenced_date', 'interval_start',
                              'interval_end', 'valid_from', 'entities_json',
                              'canonical_entities_json', 'non_answer', 'memory_kind',
                              'memory_kind_source', 'memory_kind_confidence',
                              'memory_kind_recipe', 'memory_kind_at', 'lifecycle'}),
    'memories': frozenset({'tags', 'project'}),
    'fact_temporal_validity': frozenset({'valid_from'}),
    'fact_retention': frozenset({'lifecycle_zone', 'last_computed_at'}),
    'polar_embeddings': frozenset({'qjl_bits'}),
    'embedding_quantization_metadata': frozenset({'compressed_size_bytes'}),
}
IDS = {'atomic_facts': 'fact_id', 'memories': 'memory_id',
       'fact_temporal_validity': 'fact_id', 'canonical_entities': 'entity_id',
       'fact_retention': 'fact_id', 'polar_embeddings': 'fact_id',
       'embedding_quantization_metadata': 'fact_id'}


def read(db, table, target_id, profile_id, field):
    if field not in FIELDS[table]:
        raise ValueError('not an enrichable field')
    rows = db.execute(f'SELECT * FROM {table} WHERE {IDS[table]}=? AND profile_id=?',
                      (target_id, profile_id))
    if not rows:
        return None, False
    row = dict(rows[0])
    if table == 'memories':
        if is_damaged(row['metadata_json']):
            return None, False  # bytes that are not text: never merged into or written over
        return load_json_column(row['metadata_json'], {}, column='memories.metadata_json',
                                row=target_id).get(field), True
    if field not in row:
        return None, False  # an older schema without this column: nothing to add
    return row[field], True


def write(db, table, target_id, profile_id, field, value):
    if table == 'memories':
        rows = db.execute('SELECT metadata_json FROM memories WHERE memory_id=? AND profile_id=?',
                          (target_id, profile_id))
        if is_damaged(rows[0]['metadata_json']):
            return  # bytes that are not text: left as they are
        metadata = load_json_column(rows[0]['metadata_json'], {}, column='memories.metadata_json',
                                    row=target_id)
        if value is None:
            metadata.pop(field, None)
        else:
            metadata[field] = value
        db.execute('UPDATE memories SET metadata_json=? WHERE memory_id=? AND profile_id=?',
                   (json.dumps(metadata), target_id, profile_id))
    else:
        db.execute(f'UPDATE {table} SET {field}=? WHERE {IDS[table]}=? AND profile_id=?',
                   (value, target_id, profile_id))


INFERRED = 'inferred'  # a value enrichment filled in that the fact never carried itself


def record(db, run_id, profile_id, table, target_id, field, before, after,
           *, fact_id=None, memory_id=None, source='enrichment'):
    db.execute(
        'INSERT INTO enrichment_changes(run_id,profile_id,fact_id,memory_id,target_table,'
        'target_id,field,before_json,after_json,source,enrichment_version) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (run_id, profile_id, fact_id, memory_id, table, target_id, field,
         dumps(before), dumps(after), source, VERSION))


def add(db, run_id, profile_id, table, target_id, field, value, *, replace_guess=False,
        source='enrichment'):
    if value is None:
        return False
    before, exists = read(db, table, target_id, profile_id, field)
    if not exists or before == value or (before not in (None, '', '[]', 0) and not replace_guess):
        return False
    record(db, run_id, profile_id, table, target_id, field, before, value,
           fact_id=target_id if table != 'memories' else None,
           memory_id=target_id if table == 'memories' else None, source=source)
    write(db, table, target_id, profile_id, field, value)
    return True


def undo_changes(db, run_id):
    restored = conflicts = 0
    changed_facts = set()
    rows = db.execute('SELECT * FROM enrichment_changes WHERE run_id=? ORDER BY change_id DESC',
                      (run_id,))
    for raw in rows:
        row = dict(raw)
        table, target_id, field = row['target_table'], row['target_id'], row['field']
        before, after = loads(row['before_json']), loads(row['after_json'])
        if field == '__row__' and table == 'fact_temporal_validity':
            current = db.execute('SELECT * FROM fact_temporal_validity WHERE fact_id=? '
                                 'AND profile_id=?', (target_id, row['profile_id']))
            if current and dict(current[0]) == after:
                db.execute('DELETE FROM fact_temporal_validity WHERE fact_id=? AND profile_id=?',
                           (target_id, row['profile_id']))
                restored += 1
            continue
        if table == 'atomic_facts' and field.startswith('memory_kind'):
            source = db.execute('SELECT memory_kind_source FROM atomic_facts WHERE fact_id=? '
                                'AND profile_id=?', (target_id, row['profile_id']))
            if source and source[0]['memory_kind_source'] in ('caller', 'user'):
                conflicts += 1
                continue
        if table == 'canonical_entities':
            used = db.execute('SELECT 1 FROM fact_entity_associations WHERE entity_id=? LIMIT 1',
                              (target_id,))
            if not used:
                db.execute('DELETE FROM canonical_entities WHERE entity_id=? AND profile_id=?',
                           (target_id, row['profile_id']))
                restored += 1
            continue
        current, exists = read(db, table, target_id, row['profile_id'], field)
        if not exists:
            continue  # an erased fact must stay erased
        if current != after:
            conflicts += 1
            continue  # a later user/caller edit always wins
        write(db, table, target_id, row['profile_id'], field, before)
        restored += 1
        if table == 'atomic_facts' and field == 'canonical_entities_json':
            ids = load_json_column(before, [], column='atomic_facts.canonical_entities_json',
                                   row=target_id)
            # Only remove memberships added by this run; existing memberships
            # (including their count bookkeeping) stay byte-for-byte intact.
            added = set(load_json_column(after, [], column='atomic_facts.canonical_entities_json',
                                         row=target_id)) - set(ids)
            for entity_id in added:
                db.execute('DELETE FROM fact_entity_associations WHERE fact_id=? '
                           'AND profile_id=? AND entity_id=?',
                           (target_id, row['profile_id'], entity_id))
        if table == 'atomic_facts':
            changed_facts.add((target_id, row['profile_id']))
    from superlocalmemory.storage.entity_index import record_fact_entities
    for fact_id, profile_id in changed_facts:
        fact = db.get_fact(fact_id, profile_id)
        if fact:
            record_fact_entities(db, fact_id, profile_id, fact.canonical_entities)
    return {'restored': restored, 'conflicts': conflicts}


def remap_entity_ids(conn, profile_id, mapping):
    """Follow a profile fold's entity merge (old id -> surviving id) in the ledger.

    The fold rewrites every moved fact's ``canonical_entities_json`` the same
    way, so without this an undo reads the merge as a later user edit and
    keeps the run's entities (WP-19 review M3). Runs inside the fold's own
    transaction, on its raw connection.
    """
    if not mapping:
        return
    rows = conn.execute(
        'SELECT change_id,before_json,after_json FROM enrichment_changes '
        "WHERE profile_id=? AND target_table='atomic_facts' "
        "AND field='canonical_entities_json'", (profile_id,)).fetchall()
    for change_id, before, after in rows:
        conn.execute('UPDATE enrichment_changes SET before_json=?,after_json=? WHERE change_id=?',
                     (_remapped(before, mapping), _remapped(after, mapping), change_id))


def _remapped(stored, mapping):
    value = loads(stored)
    try:
        ids = json.loads(value) if isinstance(value, str) else None
    except ValueError:
        ids = None
    if not isinstance(ids, list):
        return stored
    # Exactly the fold's own rewrite (profile_fold._repoint_json): map, de-duplicate.
    return dumps(json.dumps(list(dict.fromkeys(mapping.get(i, i) for i in ids))))


def dumps(value):
    return json.dumps(value, default=lambda v: {'_binary': base64.b64encode(v).decode('ascii')})


def loads(value):
    return json.loads(value, object_hook=lambda v: base64.b64decode(v['_binary'])
                      if set(v) == {'_binary'} else v)
