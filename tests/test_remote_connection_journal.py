"""Remote enrollment metadata only; fixtures never open the live memory root."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os

import pytest

try:
    from superlocalmemory.remote_connections.journal import EnrollmentJournal, JournalConflict
except ImportError:
    EnrollmentJournal = None
    JournalConflict = RuntimeError


def intent(profile="default"):
    return {"host": "muse", "profile_id": profile, "remote_opt_in": True,
            "permissions": {"read": True, "write": False, "correction": False, "session": False}}


@pytest.fixture
def journal(tmp_path):
    assert EnrollmentJournal is not None, "enrollment journal implementation missing"
    return EnrollmentJournal(tmp_path / "remote-connections")


def test_implementation_exists():
    assert EnrollmentJournal is not None


def test_retry_and_restart_preserve_identity(journal, tmp_path):
    one = journal.begin("owner", "default", "a" * 32, intent())
    two = EnrollmentJournal(tmp_path / "remote-connections").begin("owner", "default", "a" * 32, intent())
    assert one.connection_id == two.connection_id
    assert one.installation_id == two.installation_id == journal.installation_id
    assert one.state == "pending" and one.version == 1


def test_changed_intent_conflicts(journal):
    journal.begin("owner", "default", "a" * 32, intent())
    changed = intent(); changed["permissions"]["write"] = True
    with pytest.raises(JournalConflict, match="intent_conflict"):
        journal.begin("owner", "default", "a" * 32, changed)
    assert len(journal.list("owner", "default")) == 1


def test_context_isolation(journal):
    a = journal.begin("owner-a", "default", "a" * 32, intent())
    b = journal.begin("owner-b", "default", "a" * 32, intent())
    c = journal.begin("owner-a", "other", "a" * 32, intent("other"))
    assert len({a.connection_id, b.connection_id, c.connection_id}) == 3
    assert [x.connection_id for x in journal.list("owner-a", "other")] == [c.connection_id]
    with pytest.raises(JournalConflict, match="not_found"):
        journal.get("owner-b", "default", a.connection_id)


def test_concurrent_retries_create_one_record(journal):
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: journal.begin("owner", "default", "a" * 32, intent()).connection_id, range(24)))
    assert len(set(ids)) == 1


def test_one_dispatch_lease_and_stale_callback_fence(journal):
    row = journal.begin("owner", "default", "a" * 32, intent())
    lease = journal.claim("owner", "default", row.connection_id)
    assert lease is not None
    assert journal.claim("owner", "default", row.connection_id) is None
    assert not journal.acknowledge("owner", "default", row.connection_id, "wrong", lease.version, "gateway-reference")
    assert journal.acknowledge("owner", "default", row.connection_id, lease.token, lease.version, "gateway-reference")
    current = journal.get("owner", "default", row.connection_id)
    assert current.state == "pending" and current.requested
    assert journal.claim("owner", "default", row.connection_id) is None


def test_expired_lease_can_be_reclaimed_without_duplicate_identity(tmp_path):
    assert EnrollmentJournal is not None
    clock = [100.0]
    journal = EnrollmentJournal(tmp_path / "remote", clock=lambda: clock[0])
    row = journal.begin("owner", "default", "a" * 32, intent())
    first = journal.claim("owner", "default", row.connection_id)
    clock[0] = 131.0
    second = journal.claim("owner", "default", row.connection_id)
    assert second.token != first.token and second.version > first.version
    assert not journal.acknowledge("owner", "default", row.connection_id, first.token, first.version, "old")


def test_cancellation_cannot_be_revived_by_late_receipt(journal):
    row = journal.begin("owner", "default", "a" * 32, intent())
    lease = journal.claim("owner", "default", row.connection_id)
    cancelled = journal.cancel("owner", "default", row.connection_id, lease.version)
    assert cancelled.state == "cancelled" and cancelled.cleanup_pending
    assert not journal.acknowledge("owner", "default", row.connection_id, lease.token, lease.version, "late")
    again = journal.begin("owner", "default", "a" * 32, intent())
    assert again.state == "cancelled" and again.connection_id == row.connection_id
    assert journal.claim("owner", "default", row.connection_id) is None


def test_cancel_requires_current_version(journal):
    row = journal.begin("owner", "default", "a" * 32, intent())
    with pytest.raises(JournalConflict, match="version_conflict"):
        journal.cancel("owner", "default", row.connection_id, 100)


@pytest.mark.parametrize("change", [
    lambda x: x.update(remote_opt_in=False),
    lambda x: x.update(host="evil"),
    lambda x: x.update(host=[]),
    lambda x: x.update(host={}),
    lambda x: x.update(profile_id="other"),
    lambda x: x.update(unknown="value"),
    lambda x: x["permissions"].update(write="yes"),
    lambda x: x["permissions"].update(correction=True),
])
def test_invalid_consent_never_creates_record(journal, change):
    payload = intent(); change(payload)
    with pytest.raises(ValueError):
        journal.begin("owner", "default", "a" * 32, payload)
    assert journal.list("owner", "default") == []


def test_input_mutation_does_not_enlarge_persisted_consent(journal):
    payload = intent(); original = deepcopy(payload)
    row = journal.begin("owner", "default", "a" * 32, payload)
    payload["permissions"]["write"] = True
    assert journal.get("owner", "default", row.connection_id).intent == original


def test_owner_only_files_and_symlink_rejection(journal, tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX modes; Windows ACL covered separately")
    assert (journal.path.stat().st_mode & 0o777) == 0o600
    assert (journal.path.parent.stat().st_mode & 0o777) == 0o700
    link = tmp_path / "link"; link.symlink_to(journal.path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe_journal_path"):
        EnrollmentJournal(link)


def test_expired_dispatch_receipt_is_not_authoritative(tmp_path):
    assert EnrollmentJournal is not None
    clock = [100.0]
    journal = EnrollmentJournal(tmp_path / "remote", clock=lambda: clock[0])
    row = journal.begin("owner", "default", "a" * 32, intent())
    lease = journal.claim("owner", "default", row.connection_id)
    clock[0] = 131.0
    assert not journal.acknowledge("owner", "default", row.connection_id, lease.token, lease.version, "late")


def test_invalid_idempotency_keys_and_owners(journal):
    for key in [None, "bad", "z" * 32]:
        with pytest.raises(ValueError):
            journal.begin("owner", "default", key, intent())
    with pytest.raises(ValueError):
        journal.begin("not an owner", "default", "a" * 32, intent())


def test_pending_capacity_does_not_prevent_safe_retry(journal):
    for index in range(32):
        journal.begin("owner", "default", f"{index:032x}", intent())
    assert journal.begin("owner", "default", "0" * 32, intent()).state == "pending"
    with pytest.raises(JournalConflict, match="capacity_exhausted"):
        journal.begin("owner", "default", "f" * 32, intent())


def test_cancel_before_dispatch_needs_no_remote_cleanup(journal):
    row = journal.begin("owner", "default", "a" * 32, intent())
    cancelled = journal.cancel("owner", "default", row.connection_id, 1)
    assert not cancelled.cleanup_pending
    assert journal.cancel("owner", "default", row.connection_id, 1) == cancelled

def test_generic_compatible_mcp_client_uses_same_explicit_consent_boundary(journal):
    payload=intent();payload['host']='other_mcp'
    row=journal.begin('owner','default','a'*32,payload)
    assert row.host=='other_mcp' and row.state=='pending'
