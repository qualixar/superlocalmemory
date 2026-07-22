# H7 (3.7.9): sync_tier_changes used an invalid ``:update`` op (CozoScript has
# no such statement) with string-interpolated entity IDs — so every call threw
# a parse error that ``except: pass`` swallowed. Cozo tier sync therefore never
# worked. These tests prove the :put-rebind rewrite actually updates tiers,
# preserves other columns, is a no-op for unknown IDs, and is injection-safe.
import pytest

from superlocalmemory.graph.cozo_backend import CozoDBGraphBackend, _COZO_AVAILABLE

pytestmark = pytest.mark.skipif(not _COZO_AVAILABLE, reason="cozo not installed")


@pytest.fixture
def backend(tmp_path):
    b = CozoDBGraphBackend(str(tmp_path / "cozo"))
    yield b
    b._db.close()


def _tier(b, eid):
    rows = b._db.run("?[tier] := *entity{id, tier}, id = $id", {"id": eid}).values.tolist()
    return rows[0][0] if rows else None


def test_sync_tier_changes_promotes_and_demotes(backend):
    backend.add_entity("e1", "Alice", "person")
    backend.add_entity("e2", "Bob", "person")
    assert _tier(backend, "e1") == "hot"
    backend.sync_tier_changes(added=["e1"], removed=["e2"])
    assert _tier(backend, "e1") == "active"
    assert _tier(backend, "e2") == "cold"


def test_sync_tier_changes_preserves_other_columns(backend):
    backend.add_entity("e1", "Alice", "person", properties={"k": "v"})
    backend.sync_tier_changes(added=["e1"], removed=[])
    rows = backend._db.run(
        "?[name, entity_type, properties] := "
        "*entity{id, name, entity_type, properties}, id = 'e1'"
    ).values.tolist()
    assert rows[0][0] == "Alice"
    assert rows[0][1] == "person"
    assert "v" in rows[0][2]


def test_sync_tier_changes_unknown_id_is_noop(backend):
    backend.add_entity("e1", "Alice", "person")
    backend.sync_tier_changes(added=["GHOST"], removed=[])
    ghost = backend._db.run("?[id] := *entity{id}, id = 'GHOST'").values.tolist()
    assert ghost == []  # no stub row created for a non-existent entity


def test_sync_tier_changes_is_injection_safe(backend):
    backend.add_entity("e1", "Alice", "person")
    backend.sync_tier_changes(added=["e1'; :rm entity; %"], removed=[])
    count = backend._db.run("?[count(id)] := *entity{id}").values.tolist()[0][0]
    assert count == 1                    # table not dropped
    assert _tier(backend, "e1") == "hot"  # real e1 untouched


def test_health_check_counts_fact_entity(backend):
    backend._db.put("fact_entity", [
        {"fact_id": "f1", "entity_id": "e1", "profile_id": "default"},
        {"fact_id": "f1", "entity_id": "e2", "profile_id": "default"},
    ])
    hc = backend.health_check()
    assert hc["status"] == "active"
    assert hc["fact_entity"] == 2


def test_entity_ids_returns_all_ids(backend):
    backend.add_entity("e1", "A", "person")
    backend.add_entity("e2", "B", "person")
    assert sorted(backend.entity_ids()) == ["e1", "e2"]
