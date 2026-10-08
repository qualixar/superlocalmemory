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

def test_dashboard_runtime_installation_has_no_keyring_or_network_start(tmp_path,monkeypatch):
    from fastapi import FastAPI
    from types import SimpleNamespace
    from superlocalmemory.remote_connections.runtime import install_runtime
    import superlocalmemory.remote_connections.native_enrollment as native
    def forbidden():raise AssertionError('keyring must stay closed before opt-in')
    monkeypatch.setattr(native,'_native_backend',forbidden)
    app=FastAPI();app.state.daemon_descriptor=SimpleNamespace(port=18767)
    runtime=install_runtime(app,root=tmp_path)
    assert app.state.remote_connections is runtime.service
    assert not runtime._companions

@pytest.mark.asyncio
async def test_callback_completes_scope_bound_credential_without_secrets_in_public_status(tmp_path,monkeypatch):
    import json,time
    from dataclasses import replace
    from superlocalmemory.remote_connections.credentials import CredentialVault
    from superlocalmemory.server.remote_keys import RemoteKeyStore
    journal=EnrollmentJournal(tmp_path/'journal')
    payload={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    pending=journal.begin('owner','profile','a'*32,payload)
    lease=journal.claim('owner','profile',pending.connection_id)
    assert journal.acknowledge('owner','profile',pending.connection_id,lease.token,lease.version,pending.connection_id)
    backend=Backend();store=NativeEnrollmentStore(tmp_path/'secure',backend=backend)
    row=replace(record(),installation_id=journal.installation_id,connection_id=pending.connection_id,intent_json=json.dumps(payload))
    store.save(row)
    runtime=NativeConnectionRuntime(None,journal,store,current_profile=lambda:'profile',can_manage=lambda owner,profile:True,redirect_uri=row.redirect_uri)
    runtime.keys=RemoteKeyStore(tmp_path/'remote_keys.json')
    class Provider:
        async def exchange(self,row,code):return replace(row,access_token='synthetic-access')
        async def provision(self,row):return {'device_token':'d'*64,'generation':1,'expires_at_ms':int(time.time()*1000)+60000}
    runtime.provider=Provider()
    started=[]
    async def start(row):started.append(row.connection_id)
    monkeypatch.setattr(runtime,'start',start)
    assert await runtime.callback(row.state,'synthetic-code')==pending.connection_id
    credential=CredentialVault(store.root/'connector',backend=backend).load(row.installation_id,row.owner,row.profile,row.connection_id)
    assert credential and credential.profile=='profile'
    assert started==[pending.connection_id]
    assert store.by_state(row.state) is None
    status=json.dumps(runtime.service.status(row.owner,row.profile))
    assert 'synthetic-access' not in status and 'PRIVATE KEY' not in status


def enrolled_runtime(tmp_path, *, session=False, expired=False):
    import json
    from dataclasses import replace
    from superlocalmemory.server.remote_keys import RemoteKeyStore
    journal=EnrollmentJournal(tmp_path/'journal')
    payload={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':session}}
    pending=journal.begin('owner','profile','a'*32,payload)
    lease=journal.claim('owner','profile',pending.connection_id)
    journal.acknowledge('owner','profile',pending.connection_id,lease.token,lease.version,pending.connection_id)
    backend=Backend();store=NativeEnrollmentStore(tmp_path/'secure',backend=backend,clock=lambda:100)
    row=replace(record(),installation_id=journal.installation_id,connection_id=pending.connection_id,intent_json=json.dumps(payload),expires_at_ms=101000 if expired else 9999999999999)
    store.save(row)
    if expired:store=NativeEnrollmentStore(tmp_path/'secure',backend=backend,clock=lambda:102)
    runtime=NativeConnectionRuntime(None,journal,store,current_profile=lambda:'profile',can_manage=lambda owner,profile:True,redirect_uri=row.redirect_uri)
    runtime.keys=RemoteKeyStore(tmp_path/'remote_keys.json')
    return runtime,row

@pytest.mark.asyncio
async def test_expired_pending_cancel_uses_bootstrap_verifier_and_terminal_secure_tombstone(tmp_path):
    runtime,row=enrolled_runtime(tmp_path,expired=True)
    calls=[]
    class Provider:
        async def cancel_bootstrap(self,value):
            calls.append(value.verifier)
            return {'cancelled':True}
    runtime.provider=Provider()
    current=runtime.journal.get(row.owner,row.profile,row.connection_id)
    result=await runtime.service.cancel(row.owner,row.profile,row.connection_id,current.version)
    assert result['state']=='cancelled' and not result['cleanup_pending']
    assert calls==[row.verifier]
    assert runtime.store.by_connection(row.connection_id,for_cleanup=True) is None
    with pytest.raises(ValueError):runtime.store.save(row)

@pytest.mark.asyncio
async def test_cancel_during_delayed_local_key_add_never_starts_and_revokes_own_key(tmp_path,monkeypatch):
    import asyncio,threading,time
    from dataclasses import replace
    runtime,row=enrolled_runtime(tmp_path)
    entered=threading.Event();release=threading.Event();original=runtime.keys.add;started=[]
    def delayed(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args,**kwargs)
    monkeypatch.setattr(runtime.keys,'add',delayed)
    class Provider:
        async def exchange(self,value,code):return replace(value,access_token='synthetic')
        async def provision(self,value):return {'device_token':'d'*64,'generation':1,'expires_at_ms':int(time.time()*1000)+60000}
        async def revoke(self,value):return {'revoked':True}
    runtime.provider=Provider()
    async def start(value):started.append(value.connection_id)
    monkeypatch.setattr(runtime,'start',start)
    callback=asyncio.create_task(runtime.callback(row.state,'code'))
    assert await asyncio.to_thread(entered.wait,1)
    current=runtime.journal.get(row.owner,row.profile,row.connection_id)
    cancel=asyncio.create_task(runtime.service.cancel(row.owner,row.profile,row.connection_id,current.version))
    try:
        for _ in range(100):
            if runtime.journal.get(row.owner,row.profile,row.connection_id).state=='cancelled':break
            await asyncio.sleep(.005)
        assert runtime.journal.get(row.owner,row.profile,row.connection_id).state=='cancelled'
    finally:release.set()
    with pytest.raises(ValueError,match='enrollment_cancelled'):await callback
    assert (await cancel)['state']=='cancelled'
    assert not started and not any(key.active for key in runtime.keys.list())

@pytest.mark.asyncio
async def test_session_only_consent_provisions_session_capable_origin_key(tmp_path,monkeypatch):
    import time
    from dataclasses import replace
    runtime,row=enrolled_runtime(tmp_path,session=True)
    class Provider:
        async def exchange(self,value,code):return replace(value,access_token='synthetic')
        async def provision(self,value):return {'device_token':'d'*64,'generation':1,'expires_at_ms':int(time.time()*1000)+60000}
    runtime.provider=Provider()
    async def start(value):pass
    monkeypatch.setattr(runtime,'start',start)
    await runtime.callback(row.state,'code')
    key=runtime.keys.list()[0]
    assert key.scope=='write' and key.profile=='profile'

@pytest.mark.asyncio
async def test_completed_opt_in_restarts_stopped_companion(tmp_path):
    from dataclasses import replace
    runtime,row=enrolled_runtime(tmp_path)
    row=replace(row,completed=True);runtime.store.save(row)
    class Stopped:
        _running=False
        async def start(self):self._running=True
    stopped=Stopped();runtime._companions[row.connection_id]=stopped
    await runtime.resume(row.owner,row.profile)
    assert stopped._running

@pytest.mark.asyncio
async def test_startup_restores_only_completed_authorized_current_profile_opt_ins(tmp_path,monkeypatch):
    from dataclasses import replace
    runtime,row=enrolled_runtime(tmp_path)
    runtime.store.save(replace(row,completed=True));started=[]
    async def start(value):started.append(value.connection_id)
    monkeypatch.setattr(runtime,'start',start)
    await runtime.restore()
    assert started==[row.connection_id]
    started.clear();runtime.current_profile=lambda:'another-profile'
    await runtime.restore()
    assert not started

@pytest.mark.asyncio
async def test_local_only_restore_never_opens_keyring(tmp_path,monkeypatch):
    store=NativeEnrollmentStore(tmp_path/'secure',backend=Backend())
    runtime=NativeConnectionRuntime(None,EnrollmentJournal(tmp_path/'journal'),store,current_profile=lambda:'profile',can_manage=lambda owner,profile:True,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback')
    def forbidden(*args,**kwargs):raise AssertionError('no opt-in, no keyring')
    monkeypatch.setattr(store,'by_connection',forbidden)
    await runtime.restore()
    assert not runtime._companions

@pytest.mark.asyncio
async def test_interrupted_callback_waits_for_mutating_thread_before_disconnect_ack(tmp_path,monkeypatch):
    import asyncio,threading,time
    from dataclasses import replace
    runtime,row=enrolled_runtime(tmp_path)
    entered=threading.Event();release=threading.Event();finished=threading.Event();original=runtime.keys.add
    def delayed(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        try:return original(*args,**kwargs)
        finally:finished.set()
    monkeypatch.setattr(runtime.keys,'add',delayed)
    class Provider:
        async def exchange(self,value,code):return replace(value,access_token='synthetic')
        async def provision(self,value):return {'device_token':'d'*64,'generation':1,'expires_at_ms':int(time.time()*1000)+60000}
        async def revoke(self,value):return {'revoked':True}
    runtime.provider=Provider()
    callback=asyncio.create_task(runtime.callback(row.state,'code'))
    assert await asyncio.to_thread(entered.wait,1)
    callback.cancel()
    current=runtime.journal.get(row.owner,row.profile,row.connection_id)
    cancel=asyncio.create_task(runtime.service.cancel(row.owner,row.profile,row.connection_id,current.version))
    try:
        await asyncio.sleep(.05)
        assert not cancel.done(), 'disconnect must wait for the mutation already in progress'
    finally:release.set()
    with pytest.raises(asyncio.CancelledError):await callback
    await cancel
    assert finished.is_set()
    assert not any(key.active for key in runtime.keys.list())


def test_expired_enrollment_is_identified_without_exposing_credentials(tmp_path):
    runtime, row = enrolled_runtime(tmp_path, expired=True)
    status = runtime.service.status('owner', 'profile')
    current = status['connections'][0]
    assert current['sign_in_state'] == 'expired'
    assert current['authorization_expires_at_ms'] == row.expires_at_ms
    assert current['verified'] is False


@pytest.mark.asyncio
async def test_restart_revokes_old_request_and_concurrent_retries_share_new_intent(tmp_path):
    import asyncio
    from superlocalmemory.remote_connections.service import GatewayReceipt
    runtime, row = enrolled_runtime(tmp_path, expired=True)
    version = runtime.journal.get('owner', 'profile', row.connection_id).version
    calls = []
    async def cancel(owner, profile, identifier):
        calls.append(('cancel', identifier))
        return True
    runtime.cancel = cancel
    class Provider:
        async def enroll(self, **kwargs):
            calls.append(('enroll', kwargs['connection_id']))
            return GatewayReceipt(kwargs['connection_id'])
    runtime.service.provider = Provider()
    a, b = await asyncio.gather(*[runtime.service.restart('owner', 'profile', row.connection_id, version) for _ in range(2)])
    assert a['connection_id'] == b['connection_id'] != row.connection_id
    assert len([x for x in calls if x[0] == 'enroll']) == 1
    old = runtime.journal.get('owner', 'profile', row.connection_id)
    assert old.state == 'cancelled' and not old.cleanup_pending
    new = runtime.journal.get('owner', 'profile', a['connection_id'])
    assert new.intent == old.intent


@pytest.mark.asyncio
async def test_restart_fails_closed_until_old_remote_cleanup_is_confirmed(tmp_path):
    from superlocalmemory.remote_connections.journal import JournalConflict
    runtime, row = enrolled_runtime(tmp_path, expired=True)
    async def cancel(*args): return False
    runtime.cancel = cancel
    version = runtime.journal.get('owner', 'profile', row.connection_id).version
    with pytest.raises(JournalConflict, match='cleanup_pending'):
        await runtime.service.restart('owner', 'profile', row.connection_id, version)
    assert len(runtime.journal.list('owner', 'profile')) == 1


@pytest.mark.asyncio
async def test_restart_identity_is_canonical_across_accepted_versions_and_service_recreation(tmp_path):
    from superlocalmemory.remote_connections.runtime import ManagedConnectionService
    from superlocalmemory.remote_connections.service import GatewayReceipt
    runtime, row = enrolled_runtime(tmp_path, expired=True)
    version = runtime.journal.get('owner', 'profile', row.connection_id).version
    async def cancel(*args): return True
    runtime.cancel = cancel
    class Provider:
        async def enroll(self, **kwargs): return GatewayReceipt(kwargs['connection_id'])
    provider = Provider();runtime.service.provider = provider
    first = await runtime.service.restart('owner', 'profile', row.connection_id, version)
    replacement = ManagedConnectionService(runtime.journal, provider, hosts=runtime.service.hosts, runtime=runtime)
    second = await replacement.restart('owner', 'profile', row.connection_id, version + 1)
    assert first['connection_id'] == second['connection_id']
    assert len([x for x in runtime.journal.list('owner', 'profile') if x.state == 'pending']) == 1


@pytest.mark.asyncio
async def test_failed_cleanup_can_resume_with_current_cancelled_version(tmp_path):
    from superlocalmemory.remote_connections.journal import JournalConflict
    from superlocalmemory.remote_connections.service import GatewayReceipt
    runtime, row = enrolled_runtime(tmp_path, expired=True)
    async def cancel_no(*args): return False
    runtime.cancel = cancel_no
    version = runtime.journal.get('owner', 'profile', row.connection_id).version
    with pytest.raises(JournalConflict, match='cleanup_pending'):
        await runtime.service.restart('owner', 'profile', row.connection_id, version)
    cancelled = runtime.journal.get('owner', 'profile', row.connection_id)
    async def cancel_yes(*args): return True
    runtime.cancel = cancel_yes
    class Provider:
        async def enroll(self, **kwargs): return GatewayReceipt(kwargs['connection_id'])
    runtime.service.provider = Provider()
    result = await runtime.service.restart('owner', 'profile', row.connection_id, cancelled.version)
    assert result['state'] == 'pending'
    assert len([x for x in runtime.journal.list('owner', 'profile') if x.state == 'pending']) == 1


def _verify_runtime(tmp_path, outcomes):
    """Runtime whose provider verify() yields the given outcomes in order (Exception = failure)."""
    import asyncio
    from types import SimpleNamespace
    runtime=NativeConnectionRuntime(None,EnrollmentJournal(tmp_path/'journal'),NativeEnrollmentStore(tmp_path/'secure',backend=Backend()),current_profile=lambda:'profile',can_manage=lambda owner,profile:True,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback')
    row=SimpleNamespace(connection_id='c'*32)
    runtime.store=SimpleNamespace(by_connection=lambda identifier:row)
    async def current(_row):return None
    runtime._current=current
    calls=[]
    class Provider:
        async def exchange(self,latest,code):return latest
        async def verify(self,latest):
            calls.append(1);outcome=outcomes[min(len(calls),len(outcomes))-1]
            if isinstance(outcome,Exception):raise outcome
            return outcome
    runtime.provider=Provider()
    runtime._verify_retry_base_s=0.01
    runtime._epochs[row.connection_id]=1;runtime._states[row.connection_id]='transport_ready'
    return runtime,row,calls,asyncio

@pytest.mark.asyncio
async def test_transient_verification_failure_recovers_without_a_reconnect(tmp_path):
    ok={'verified':True,'connection_id':'c'*32}
    runtime,row,calls,asyncio=_verify_runtime(tmp_path,[TimeoutError(),ok])
    await runtime.verify(row,1)
    assert runtime._states[row.connection_id]=='verification_unavailable'
    for _ in range(200):
        if runtime._states[row.connection_id]=='ready_for_client':break
        await asyncio.sleep(0.01)
    assert runtime._states[row.connection_id]=='ready_for_client' and len(calls)==2
    await runtime.shutdown() if hasattr(runtime,'shutdown') else None

@pytest.mark.asyncio
async def test_verification_retry_never_applies_to_a_changed_connection(tmp_path):
    ok={'verified':True,'connection_id':'c'*32}
    runtime,row,calls,asyncio=_verify_runtime(tmp_path,[TimeoutError(),ok])
    await runtime.verify(row,1)
    runtime._epochs[row.connection_id]=2;runtime._states[row.connection_id]='reconnecting'
    await asyncio.sleep(0.2)
    assert runtime._states[row.connection_id]=='reconnecting' and row.connection_id not in runtime._verified and len(calls)==1

@pytest.mark.asyncio
async def test_verification_retries_are_bounded(tmp_path):
    runtime,row,calls,asyncio=_verify_runtime(tmp_path,[TimeoutError()])
    await runtime.verify(row,1)
    await asyncio.sleep(1.0)
    assert runtime._states[row.connection_id]=='verification_unavailable'
    assert len(calls)==1+5
