import pytest
from superlocalmemory.remote_connections.runtime import NativeConnectionRuntime
from superlocalmemory.remote_connections.native_enrollment import NativeEnrollmentStore
from superlocalmemory.remote_connections.journal import EnrollmentJournal
from tests.test_remote_native_enrollment_store import Backend, record

@pytest.mark.asyncio
async def test_callback_rejects_unknown_state_before_cloud_or_key_creation(tmp_path):
    runtime=NativeConnectionRuntime(None,EnrollmentJournal(tmp_path/'journal'),NativeEnrollmentStore(tmp_path/'secure',backend=Backend()),current_profile=lambda:'profile',can_manage=lambda owner,profile:True,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback')
    with pytest.raises(ValueError,match='invalid_callback_state'):await runtime.callback('x'*43,'synthetic-code')

@pytest.mark.asyncio
async def test_callback_refuses_revoked_creator_permission(tmp_path):
    store=NativeEnrollmentStore(tmp_path/'secure',backend=Backend());store.save(record())
    runtime=NativeConnectionRuntime(None,EnrollmentJournal(tmp_path/'journal'),store,current_profile=lambda:'profile',can_manage=lambda owner,profile:False,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback')
    with pytest.raises(ValueError,match='creator_permission_changed'):await runtime.callback('s'*43,'synthetic-code')

def test_confirmed_remote_cleanup_preserves_terminal_local_cancellation(tmp_path):
    journal=EnrollmentJournal(tmp_path/'journal')
    payload={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    row=journal.begin('owner','profile','a'*32,payload)
    lease=journal.claim('owner','profile',row.connection_id)
    assert journal.acknowledge('owner','profile',row.connection_id,lease.token,lease.version,'remote-reference')
    row=journal.get('owner','profile',row.connection_id)
    row=journal.cancel('owner','profile',row.connection_id,row.version)
    assert row.cleanup_pending
    journal.clear_cleanup('owner','profile',row.connection_id)
    row=journal.get('owner','profile',row.connection_id)
    assert row.state=='cancelled' and not row.cleanup_pending
