# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The ordered catalogue of migrations ``migration_runner`` applies.

Split out of ``migration_runner`` (which re-exports ``MIGRATIONS``,
``DEFERRED_MIGRATIONS`` and every ``_Mnnn`` alias) to keep that module within
the size limit. Moving an entry here changes nothing about when it runs.
"""

from __future__ import annotations

from superlocalmemory.storage._migration_internals import Migration
from superlocalmemory.storage.migrations import (
    M001_add_signal_features_columns as _M001,
)
from superlocalmemory.storage.migrations import (
    M002_model_state_history as _M002,
)
from superlocalmemory.storage.migrations import (
    M003_migration_log as _M003,
)
from superlocalmemory.storage.migrations import (
    M004_cross_platform_sync_log as _M004,
)
from superlocalmemory.storage.migrations import (
    M005_bandit_tables as _M005,
)
from superlocalmemory.storage.migrations import (
    M006_action_outcomes_reward as _M006,
)
from superlocalmemory.storage.migrations import (
    M007_pending_outcomes as _M007,
)
from superlocalmemory.storage.migrations import (
    M009_model_lineage as _M009,
)
from superlocalmemory.storage.migrations import (
    M010_evolution_config as _M010,
)
from superlocalmemory.storage.migrations import (
    M011_archive_and_merge as _M011,
)
from superlocalmemory.storage.migrations import (
    M012_shadow_observations as _M012,
)
from superlocalmemory.storage.migrations import (
    M013_bi_temporal_columns as _M013,
)
from superlocalmemory.storage.migrations import (
    M014_v345_scale_ready as _M014,
)
from superlocalmemory.storage.migrations import (
    M015_add_pinned_column as _M015,
)
from superlocalmemory.storage.migrations import (
    M016_add_scope_support as _M016,
)
from superlocalmemory.storage.migrations import (
    M017_ccq_scope_column as _M017,
)
from superlocalmemory.storage.migrations import (
    M018_ingestion_operations as _M018,
)
from superlocalmemory.storage.migrations import (
    M019_derivation_lineage as _M019,
)
from superlocalmemory.storage.migrations import (
    M020_model_state_integrity as _M020,
)
from superlocalmemory.storage.migrations import (
    M021_ingestion_log_profile as _M021,
)
from superlocalmemory.storage.migrations import (
    M022_entity_aliases_profile as _M022,
)
from superlocalmemory.storage.migrations import (
    M023_mesh_profile_isolation as _M023,
)
from superlocalmemory.storage.migrations import (
    M024_rbac_users_roles as _M024,
)
from superlocalmemory.storage.migrations import (
    M025_perf_indexes as _M025,
)
from superlocalmemory.storage.migrations import (
    M026_rbac_memberships_fk as _M026,
)
from superlocalmemory.storage.migrations import (
    M027_transferable_patterns_profile as _M027,
)
from superlocalmemory.storage.migrations import (
    M028_fact_entity_associations as _M028,
)
from superlocalmemory.storage.migrations import (
    M029_behavioral_history_indexes as _M029,
)
from superlocalmemory.storage.migrations import (
    M030_entity_explorer_indexes as _M030,
)
from superlocalmemory.storage.migrations import (
    M031_dead_letter_operations as _M031,
)
from superlocalmemory.storage.migrations import (
    M032_write_coordinator_admission as _M032,
)
from superlocalmemory.storage.migrations import (
    M033_projection_transactions as _M033,
)
from superlocalmemory.storage.migrations import (
    M034_obligation_integrity as _M034,
)
from superlocalmemory.storage.migrations import (
    M035_erasure_receipts as _M035,
)
from superlocalmemory.storage.migrations import (
    M036_vector_row_map as _M036,
)
from superlocalmemory.storage.migrations import (
    M037_manifest_hmac_version as _M037,
)
from superlocalmemory.storage.migrations import (
    M038_learning_feedback_channel as _M038,
)
from superlocalmemory.storage.migrations import (
    M039_scene_fact_members as _M039,
)
from superlocalmemory.storage.migrations import (
    M040_agent_experience_receipts as _M040,
)
from superlocalmemory.storage.migrations import (
    M041_external_evidence_receipts as _M041,
)
from superlocalmemory.storage.migrations import (
    M042_correction_case_ledger as _M042,
)
from superlocalmemory.storage.migrations import (
    M043_quarantine_display_summaries as _M043,
)
from superlocalmemory.storage.migrations import (
    M044_play_carries_its_own_evidence as _M044,
)
from superlocalmemory.storage.migrations import (
    M045_fact_outcome_score as _M045,
)
from superlocalmemory.storage.migrations import (
    M046_prospective_memory_has_its_own_name as _M046,
)
from superlocalmemory.storage.migrations import (
    M047_fisher_vectors_are_stored_like_every_other_vector as _M047,
)
from superlocalmemory.storage.migrations import (
    M048_upcoming_holds_only_what_is_upcoming as _M048,
)
from superlocalmemory.storage.migrations import (
    M049_a_schema_version_marker_is_one_row as _M049,
)
from superlocalmemory.storage.migrations import (
    M050_execution_learning_v2 as _M050,
)
from superlocalmemory.storage.migrations import (
    M051_lifecycle_is_recomputed_not_resampled as _M051,
)
from superlocalmemory.storage.migrations import (
    M052_memory_kinds as _M052,
)
from superlocalmemory.storage.migrations import (
    M053_answer_check_history as _M053,
)
from superlocalmemory.storage.migrations import (
    M054_saved_views as _M054,
)
from superlocalmemory.storage.migrations import (
    M055_learning_attribution as _M055,
)
from superlocalmemory.storage.migrations import (
    M056_recall_channel_indexes as _M056,
)
from superlocalmemory.storage.migrations import (
    M057_temporal_retrieval_version as _M057,
)
from superlocalmemory.storage.migrations import (
    M058_recall_fact_cover as _M058,
)
from superlocalmemory.storage.migrations import (
    M059_erasure_visibility as _M059,
)
from superlocalmemory.storage.migrations import (
    M060_memory_description as _M060,
)
from superlocalmemory.storage.migrations import (
    M061_fact_hub_baseline as _M061,
)
from superlocalmemory.storage.migrations import (
    M062_hook_bookkeeping_is_flagged as _M062,
)
from superlocalmemory.storage.migrations import (
    M063_blank_temporal_dates_are_null as _M063,
)
from superlocalmemory.storage.migrations import (
    M064_orphan_index_rows_are_removed as _M064,
)
from superlocalmemory.storage.migrations import (
    M065_fabricated_dates_are_removed as _M065,
)
from superlocalmemory.storage.migrations import (
    M066_wrongly_archived_memories_are_restored as _M066,
    M067_projection_follows_every_fact_change as _M067,
)

# Order matters: M003 creates the log table. The runner handles M003's own
# bootstrap (it can't record itself before it exists).
MIGRATIONS: list[Migration] = [
    Migration(name=_M003.NAME, db_target="learning", ddl=_M003.DDL),
    Migration(name=_M001.NAME, db_target="learning", ddl=_M001.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M002.NAME, db_target="learning", ddl=_M002.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M005.NAME, db_target="learning", ddl=_M005.DDL,
              dependencies=(_M003.NAME,)),
    # M009 extends learning_model_state (created by M002).
    Migration(name=_M009.NAME, db_target="learning", ddl=_M009.DDL,
              dependencies=(_M002.NAME,)),
    # M020 owns post-release integrity repair. M002 remains byte-for-byte
    # compatible with databases that recorded its historical DDL hash.
    Migration(name=_M020.NAME, db_target="learning", ddl=_M020.DDL,
              dependencies=(_M002.NAME,)),
    # M010 creates evolution_config + evolution_llm_cost_log (learning.db).
    Migration(name=_M010.NAME, db_target="learning", ddl=_M010.DDL,
              dependencies=(_M003.NAME,)),
    # M012 creates shadow_observations (learning.db) — paired NDCG@10
    # observations for ShadowTest persistence across daemon restart.
    Migration(name=_M012.NAME, db_target="learning", ddl=_M012.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M004.NAME, db_target="memory", ddl=_M004.DDL),
    # M007 creates pending_outcomes (memory.db, LLD-00 §1.2).
    Migration(name=_M007.NAME, db_target="memory", ddl=_M007.DDL),
    # M018 is additive and independent of runtime-bootstrapped tables.
    Migration(name=_M018.NAME, db_target="memory", ddl=_M018.DDL),
    # M024 creates RBAC tables (users / memberships / sessions). Independent
    # brand-new tables, so it runs pre-engine-init.
    Migration(name=_M024.NAME, db_target="memory", ddl=_M024.DDL),
    Migration(name=_M019.NAME, db_target="memory", ddl=_M019.DDL,
              dependencies=(_M018.NAME,)),
    # M031 creates dead_letter_operations — standalone table, no FK to engine-
    # bootstrapped tables, so it can run during apply_all (before engine init).
    Migration(name=_M031.NAME, db_target="memory", ddl=_M031.DDL),
    # M032 is standalone and must precede daemon readiness: typed writes use
    # this append-only receipt ledger for durable idempotency.
    Migration(name=_M032.NAME, db_target="memory", ddl=_M032.DDL),
    Migration(name=_M033.NAME, db_target="memory", ddl=_M033.DDL),
    Migration(name=_M034.NAME, db_target="memory", ddl=_M034.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M035.NAME, db_target="memory", ddl=_M035.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M036.NAME, db_target="memory", ddl=_M036.DDL,
              dependencies=(_M033.NAME,)),
    Migration(name=_M037.NAME, db_target="memory", ddl=_M037.DDL,
              dependencies=(_M033.NAME, _M035.NAME)),
    # Main-line M033 is renumbered in V4 because V4 already owns M033-M037.
    # It repairs the legacy learning_feedback schema before any reader mines
    # channel patterns.
    Migration(name=_M038.NAME, db_target="learning", ddl=_M038.DDL,
              dependencies=(_M003.NAME,)),
    # Receipt writes are a learning-plane concern and must never share the
    # memory.db recall lock domain.  The tables are self-contained: profile
    # lifecycle performs explicit cross-store erasure rather than an FK.
    Migration(name=_M040.NAME, db_target="learning", ddl=_M040.DDL,
              dependencies=(_M003.NAME,)),
    Migration(name=_M041.NAME, db_target="learning", ddl=_M041.DDL,
              dependencies=(_M040.NAME,)),
    Migration(name=_M050.NAME, db_target="learning", ddl=_M050.DDL,
              dependencies=(_M041.NAME,)),
    # M053: the Answer Check history (4.1.20). A learning-plane table for the
    # same reason as M040: its writer must never hold memory.db's recall lock.
    Migration(name=_M053.NAME, db_target="learning", ddl=_M053.DDL,
              dependencies=(_M003.NAME,)),
    # M054: saved views (4.1.21, issue #113). A person's named recall queries;
    # learning.db so the upgrade copies only the smaller database.
    Migration(name=_M054.NAME, db_target="learning", ddl=_M054.DDL,
              dependencies=(_M003.NAME,)),
    # M055: durable learning attribution (4.1.25). One outcome settles one
    # play (bandit_plays.outcome_id, unique) and a shadow test records which
    # arm and model served each recall. learning.db only; additive.
    Migration(name=_M055.NAME, db_target="learning", ddl=_M055.DDL,
              dependencies=(_M005.NAME, _M044.NAME)),
    # Review-gated correction metadata is self-contained in memory.db. It
    # contains identifiers only and does not alter temporal fact state.
    Migration(name=_M042.NAME, db_target="memory", ddl=_M042.DDL,
              dependencies=(_M032.NAME,)),
    # M052 adds five nullable memory-kind columns to atomic_facts and two new
    # tables. Eager on purpose, unlike the other atomic_facts migrations: ADD
    # COLUMN with no constraint reads no row, so it is safe before engine init,
    # and running here means journal replay never writes a fact the columns do
    # not exist for. On a fresh install atomic_facts is absent and only the two
    # tables are created; schema.py creates the columns with the table.
    Migration(name=_M052.NAME, db_target="memory", ddl=_M052.DDL),
    # M044 lets a bandit play record which memories it showed, so the reward
    # proxy can settle it from evidence instead of always falling through to
    # the 120-second neutral default. Additive column on M005's bandit_plays,
    # and eager on purpose: nothing bootstraps that table at engine init, so
    # there is no reason to defer it.
    Migration(name=_M044.NAME, db_target="learning", ddl=_M044.DDL,
              dependencies=(_M005.NAME,)),
    # M006 + M011 are deliberately NOT here — see DEFERRED_MIGRATIONS below.
]


# Deferred migrations run AFTER ``MemoryEngine.initialize()`` has called
# ``storage.schema.create_all_tables`` to bootstrap runtime tables such as
# ``action_outcomes``. Running them during ``apply_all`` (which fires BEFORE
# engine init on daemon startup) would blow up with "no such table".
#
# ``learning.database.fetch_training_examples`` already checks
# ``_migration_applied("M006_action_outcomes_reward")`` and falls back to the
# position proxy when the column is absent, so a failed deferred apply never
# crashes the trainer — it just keeps the old label path.
DEFERRED_MIGRATIONS: list[Migration] = [
    # M028 captures an atomic_facts rowid high-water mark before readiness.
    # atomic_facts/canonical_entities are bootstrapped by MemoryEngine.
    Migration(name=_M028.NAME, db_target="memory", ddl=_M028.DDL,
              dependencies=(_M018.NAME,)),
    Migration(name=_M006.NAME, db_target="memory", ddl=_M006.DDL),
    # M011 extends atomic_facts + creates memory_archive / memory_merge_log.
    # atomic_facts is bootstrapped at engine init, so M011 defers alongside M006.
    Migration(name=_M011.NAME, db_target="memory", ddl=_M011.DDL),
    # M013 adds bi-temporal columns (valid_from / valid_until) to
    # atomic_facts. Deferred for the same engine-init-bootstrap reason
    # as M011.
    Migration(name=_M013.NAME, db_target="memory", ddl=_M013.DDL),
    Migration(name=_M014.NAME, db_target="memory", ddl=_M014.DDL),
    # M015 adds pinned column to atomic_facts (v3.4.65 core-memory pins).
    Migration(name=_M015.NAME, db_target="memory", ddl=_M015.DDL),
    # M016 adds scope and shared_with columns to 5 core tables for
    # multi-scope memory support (personal/global/shared).
    Migration(name=_M016.NAME, db_target="memory", ddl=_M016.DDL),
    # M017 adds scope to the engine-bootstrapped CCQ consolidation table.
    Migration(name=_M017.NAME, db_target="memory", ddl=_M017.DDL),
    # M021 rebuilds ingestion_log with a profile-scoped dedup constraint.
    # Deferred: ingestion_log is created at engine init (apply_v343_schema).
    Migration(name=_M021.NAME, db_target="memory", ddl=_M021.DDL),
    # M022 adds profile_id to entity_aliases, backfilled from the parent entity.
    # Deferred: entity_aliases is created at engine init (create_all_tables).
    Migration(name=_M022.NAME, db_target="memory", ddl=_M022.DDL),
    # M023 profile-scopes every mesh coordination table (peers/messages/state/
    # locks/events). Deferred: mesh tables are created at engine init
    # (apply_v343_schema), same as M021.
    Migration(name=_M023.NAME, db_target="memory", ddl=_M023.DDL),
    # M025 adds hot-path perf indexes (atomic_facts dedup, mesh cleanup/list).
    Migration(name=_M025.NAME, db_target="memory", ddl=_M025.DDL),
    # M026 rebuilds rbac_memberships with a profiles FK (ON DELETE CASCADE) so
    # deleting a profile cascade-purges its role grants (SEC-H-01 defense-in-
    # depth). Deferred: the FK target `profiles` is created at engine init.
    Migration(name=_M026.NAME, db_target="memory", ddl=_M026.DDL),
    # M027 rebuilds transferable_patterns with profile_id + UNIQUE(profile_id,
    # pattern_type, key) to prevent cross-profile preference contamination (H-01,
    # cycle-3 audit). Deferred: CrossProjectAggregator creates the table on first
    # consolidation run, not during engine init or apply_all. apply() is a no-op
    # when the table is absent (first install after the schema change).
    Migration(name=_M027.NAME, db_target="learning", ddl=_M027.DDL),
    # M029 indexes behavioral tables bootstrapped during engine initialization.
    Migration(name=_M029.NAME, db_target="memory", ddl=_M029.DDL),
    # M030 bounds Entity Explorer pagination and profile-summary ranking.
    Migration(name=_M030.NAME, db_target="memory", ddl=_M030.DDL),
    # Main-line M034 is renumbered in V4. It must remain deferred because its
    # backfill joins engine-bootstrapped memory_scenes and atomic_facts.
    Migration(name=_M039.NAME, db_target="memory", ddl=_M039.DDL),
    # M043 withholds model-written summaries from the retrieval corpus and
    # un-hides the memories they displaced. Deferred because it reads and
    # writes atomic_facts + fact_retention, both bootstrapped at engine init —
    # the same reason M011/M013/M015/M016 are deferred. apply_deferred takes a
    # verified snapshot before the first migration it actually applies, so the
    # store is recoverable.
    Migration(name=_M043.NAME, db_target="memory", ddl=_M043.DDL,
              dependencies=(_M011.NAME,)),
    # M045 holds the per-fact outcome score. Deferred because its backfill
    # reads action_outcomes, which engine init bootstraps — the same reason
    # M006 and M011 are deferred. Depends on M006 for the reward column it
    # averages.
    Migration(name=_M045.NAME, db_target="memory", ddl=_M045.DDL,
              dependencies=(_M006.NAME,)),
    # M046 renames the fact type used for planned future events, which means
    # rebuilding atomic_facts to widen a CHECK constraint SQLite cannot alter.
    # Deferred for the same reason as M043: atomic_facts is bootstrapped at
    # engine init, and apply_deferred takes a verified snapshot before the first
    # migration it applies, so a table rebuild has something to fall back to.
    # Depends on M043 so the two never contend for the same table in one pass.
    Migration(name=_M046.NAME, db_target="memory", ddl=_M046.DDL,
              dependencies=(_M043.NAME,)),
    # M047 rewrites the two Fisher vectors on each fact as float32 rather than
    # as decimal text. Deferred because it walks every fact in atomic_facts,
    # which engine init bootstraps. It changes no schema and both forms stay
    # readable, so it is resumable and an interrupted store still works.
    # Depends on M046 so a table rebuild and a full-table update never run in
    # the same pass over the same table.
    Migration(name=_M047.NAME, db_target="memory", ddl=_M047.DDL,
              dependencies=(_M046.NAME,)),
    # M048 finishes what M046 started: M046 renamed the type used for planned
    # events without re-reading a single one of them, so the same wrongly-filed
    # rows now carry a more confident name. Depends on M046 for the rename.
    Migration(name=_M048.NAME, db_target="memory", ddl=_M048.DDL,
              dependencies=(_M046.NAME,)),
    # M049 gives schema_version the unique constraint its six writers all
    # assumed it had. Every one uses INSERT OR IGNORE, which ignores nothing
    # without a constraint, so each appended a duplicate per run: seven distinct
    # versions held as 3,496 rows on one store and 234,348 on another. No
    # dependency -- it touches a bookkeeping table no other migration reads.
    Migration(name=_M049.NAME, db_target="memory", ddl=_M049.DDL),
    # M051 clears lifecycle positions that were thermal noise, so the
    # maintenance backfill recomputes them from the forgetting curve.
    # No dependency: it clears one column no other migration reads.
    Migration(name=_M051.NAME, db_target="memory", ddl=_M051.DDL),
    Migration(name=_M056.NAME, db_target="memory", ddl=_M056.DDL),
    Migration(name=_M057.NAME, db_target="memory", ddl=_M057.DDL),
    Migration(name=_M058.NAME, db_target="memory", ddl=_M058.DDL),
    Migration(name=_M059.NAME, db_target="memory", ddl=_M059.DDL,
              dependencies=(_M035.NAME,)),
    Migration(name=_M060.NAME, db_target="memory", ddl=_M060.DDL),
    # M061 adds the hub-baseline table recall reads (retrieval/hub_baseline.py).
    # Deferred: its delete trigger watches atomic_facts.
    Migration(name=_M061.NAME, db_target="memory", ddl=_M061.DDL),
    # M062 flags the hooks' bookkeeping already stored (atomic_facts.non_answer,
    # a column M060 adds). Deferred: it reads atomic_facts, which engine init
    # creates.
    Migration(name=_M062.NAME, db_target="memory", ddl=_M062.DDL,
              dependencies=(_M060.NAME,)),
    # M063 stores blank temporal dates as NULL so the temporal slice cache
    # (M057) can serve. Deferred: it reads temporal_events.
    Migration(name=_M063.NAME, db_target="memory", ddl=_M063.DDL,
              dependencies=(_M057.NAME,)),
    # M064 removes the index rows of memories that no longer exist (a delete
    # that failed partway). Deferred: it reads atomic_facts and vector_row_map,
    # which engine init creates.
    Migration(name=_M064.NAME, db_target="memory", ddl=_M064.DDL),
    # M065 takes the made-up dates of builds before 4.1.6 out of facts and their
    # date rows, with an undo record. Deferred: it reads temporal_events,
    # fact_context and bm25_tokens, which engine init creates.
    Migration(name=_M065.NAME, db_target="memory", ddl=_M065.DDL),
    # M066 restores memories the hours-clock ELC step archived (#266) to the zone
    # the store-scaled clock gives them, with an undo record. Deferred: it reads
    # fact_retention and the decay inputs, which engine init creates.
    Migration(name=_M066.NAME, db_target="memory", ddl=_M066.DDL),
    # M067 queues a fact for the graph and vector copies whenever its visibility
    # or entities change (#296). Deferred: it watches columns engine init and
    # earlier migrations add, and the queue table M033 creates.
    Migration(name=_M067.NAME, db_target="memory", ddl=_M067.DDL,
              dependencies=(_M033.NAME,)),
]
