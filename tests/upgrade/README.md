# Upgrade and downgrade fixtures

Synthetic data only: every person, project and number in `corpus.py` is invented.

## What each check proves

1. **Upgrade in one start.** A copy of an old store is opened by this checkout
   through the same entry point a user hits (`slm serve start`).  `learning.db`
   must report `slm_schema_version` 54 and all 40 memories must be unchanged.
2. **Snapshot.** The upgrade leaves a `*-pre-migration.db` under
   `pre-migration-snapshots/`, and restoring it with
   `restore_pre_migration_snapshot` gives back identical row counts and contents.
3. **Recall.** 12 fixed queries are run twice against 4.1.24 (measuring its own
   run-to-run noise) and once against the upgraded store.  The top-5 overlap of the
   upgraded store with 4.1.24 must be at least the overlap 4.1.24 has with itself.
   Without an embedding model both sides use keyword-only recall; the report says so.
4. **Downgrade.** `slm db prepare-downgrade` runs on the upgraded copy, then 4.1.24
   opens it, answers the 12 queries and loses no memory.

## Files

| file | purpose |
|------|---------|
| `corpus.py` | 40 invented memories, 12 queries, expected ids |
| `build_fixture.py` | builds a store with one released version (own venv, own CLI) |
| `upgrade_check.py` | runs the four checks, writes a JSON verdict |
| `test_corpus.py` | fast tests (default run) |
| `test_upgrade_fixtures.py` | end-to-end, marked `slow` |

## Rebuild and run

```
python tests/upgrade/build_fixture.py --version 4.1.20 --out /scratch/fixtures/4.1.20 --venv /scratch/venvs/4.1.20
python tests/upgrade/upgrade_check.py --fixture /scratch/fixtures/4.1.20 \
    --old-python /scratch/venvs/4.1.24/bin/python --new-python .venv/bin/python
SLM_UPG_FIXTURES=/scratch/fixtures SLM_UPG_OLD_PY=/scratch/venvs/4.1.24/bin/python pytest tests/upgrade -m slow
```

Stores are not committed: a venv with the full dependency set is several GB and
the stores are cheap to rebuild.  Every step works on copies in a scratch directory
with its own `HOME`, so a real `~/.superlocalmemory` is never touched.
