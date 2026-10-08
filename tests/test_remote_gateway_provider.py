import pytest
from superlocalmemory.remote_connections.gateway_provider import CloudGatewayProvider
from superlocalmemory.remote_connections.native_enrollment import NativeEnrollmentStore
from tests.test_remote_native_enrollment_store import Backend

@pytest.mark.asyncio
async def test_opt_in_enrollment_registers_native_pkce_and_returns_sign_in(tmp_path):
    requests=[]
    async def http(path, **kwargs):
        requests.append((path,kwargs))
        if path=='/oauth/register':return {'client_id':'synthetic-client'}
        if path=='/bootstrap':return {'connection_id':'a'*32,'authorize_url':'https://auth.superlocalmemory.com/owner-login?connection_id='+'a'*32}
        raise AssertionError(path)
    store=NativeEnrollmentStore(tmp_path,backend=Backend())
    provider=CloudGatewayProvider(store,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    receipt=await provider.enroll(installation_id='installation',connection_id='a'*32,owner='owner',profile='profile',intent={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}})
    assert receipt.authorization_url.endswith('a'*32)
    row=store.by_connection('a'*32)
    assert row and row.client_id=='synthetic-client'
    assert 'PRIVATE KEY' not in str(requests)
    assert 'code_challenge_method=S256' in requests[-1][1]['json']['authorizationUrl']

@pytest.mark.asyncio
async def test_enrollment_retry_preserves_client_and_pkce(tmp_path):
    count=0;payloads=[]
    async def http(path,**kwargs):
        nonlocal count
        if path=='/oauth/register':count+=1;return {'client_id':'synthetic-client'}
        payloads.append(kwargs['json'])
        return {'connection_id':'a'*32,'authorize_url':'https://auth.superlocalmemory.com/owner-login?connection_id='+'a'*32}
    provider=CloudGatewayProvider(NativeEnrollmentStore(tmp_path,backend=Backend()),redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    intent={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    for _ in range(2):await provider.enroll(installation_id='installation',connection_id='a'*32,owner='owner',profile='profile',intent=intent)
    assert count==1
    assert payloads[0]==payloads[1]

@pytest.mark.asyncio
async def test_two_connections_share_desktop_registration_and_key(tmp_path):
    count=0;payloads=[]
    async def http(path,**kwargs):
        nonlocal count
        if path=='/oauth/register':count+=1;return {'client_id':'synthetic-client'}
        value=kwargs['json'];payloads.append(value)
        return {'connection_id':value['connectionId'],'authorize_url':'https://auth.superlocalmemory.com/owner-login?connection_id='+value['connectionId']}
    provider=CloudGatewayProvider(NativeEnrollmentStore(tmp_path,backend=Backend()),redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    intent={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    for connection in ('a'*32,'b'*32):await provider.enroll(installation_id='installation',connection_id=connection,owner='owner',profile='profile',intent=intent)
    assert count==1
    assert payloads[0]['deviceJwk']==payloads[1]['deviceJwk']

@pytest.mark.asyncio
async def test_interrupted_connection_save_recovers_existing_desktop_client(tmp_path):
    store=NativeEnrollmentStore(tmp_path,backend=Backend());count=0
    original_save=store.save;failed=False
    def interrupted(row):
        nonlocal failed
        if row.client_id and not failed:
            failed=True
            raise ValueError('synthetic-store-interruption')
        return original_save(row)
    store.save=interrupted
    async def http(path,**kwargs):
        nonlocal count
        if path=='/oauth/register':count+=1;return {'client_id':'synthetic-client'}
        return {'connection_id':'a'*32,'authorize_url':'https://auth.superlocalmemory.com/owner-login?connection_id='+'a'*32}
    provider=CloudGatewayProvider(store,redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    intent={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    with pytest.raises(ValueError):await provider.enroll(installation_id='installation',connection_id='a'*32,owner='owner',profile='profile',intent=intent)
    await provider.enroll(installation_id='installation',connection_id='a'*32,owner='owner',profile='profile',intent=intent)
    assert count==1

@pytest.mark.asyncio
async def test_slow_secure_store_does_not_block_unrelated_coroutine(tmp_path):
    import asyncio,threading
    entered=threading.Event();release=threading.Event()
    class Slow(Backend):
        def get_password(self,service,name):
            entered.set();release.wait(2)
            return super().get_password(service,name)
    async def http(path,**kwargs):
        if path=='/oauth/register':return {'client_id':'synthetic-client'}
        return {'connection_id':'a'*32,'authorize_url':'https://auth.superlocalmemory.com/owner-login?connection_id='+'a'*32}
    provider=CloudGatewayProvider(NativeEnrollmentStore(tmp_path,backend=Slow()),redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    intent={'host':'muse','profile_id':'profile','remote_opt_in':True,'permissions':{'read':True,'write':False,'correction':False,'session':False}}
    task=asyncio.create_task(provider.enroll(installation_id='installation',connection_id='a'*32,owner='owner',profile='profile',intent=intent))
    try:
        assert await asyncio.to_thread(entered.wait,1)
        await asyncio.wait_for(asyncio.sleep(0),0.1)
    finally:release.set()
    await task

@pytest.mark.asyncio
async def test_expired_completed_record_refresh_persists_rotated_tokens(tmp_path):
    from dataclasses import replace
    from tests.test_remote_native_enrollment_store import record
    backend=Backend();store=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:100)
    row=replace(record(),expires_at_ms=101000,completed=True,client_id='client',access_token='old-access',refresh_token='old-refresh',access_expires_ms=1)
    store.save(row)
    later=NativeEnrollmentStore(tmp_path,backend=backend,clock=lambda:102);requests=[]
    async def http(path,**kwargs):
        requests.append((path,kwargs))
        return {'access_token':'new-access','refresh_token':'new-refresh','expires_in':3600,'scope':'slm:connect'}
    provider=CloudGatewayProvider(later,redirect_uri=row.redirect_uri,http=http)
    updated=await provider.exchange(later.by_connection(row.connection_id),'')
    assert requests[0][1]['data']['grant_type']=='refresh_token'
    assert later.by_connection(row.connection_id)==updated
    assert updated.refresh_token=='new-refresh' and later.by_state(row.state) is None


def _proof_target(proof):
    import base64,json
    payload=proof.split('.')[1];payload+='='*(-len(payload)%4)
    return json.loads(base64.urlsafe_b64decode(payload))['htu']

@pytest.mark.asyncio
async def test_connected_apps_calls_use_owner_token_and_endpoint_bound_proof(tmp_path):
    from dataclasses import replace
    from tests.test_remote_native_enrollment_store import record
    requests=[]
    async def http(path,**kwargs):
        requests.append((path,kwargs))
        return {'apps':[]} if path=='/owner/apps' else {'revoked':True,'version':2}
    provider=CloudGatewayProvider(NativeEnrollmentStore(tmp_path,backend=Backend()),redirect_uri='http://127.0.0.1:18767/api/v3/connections/callback',http=http)
    row=replace(record(),access_token='owner-token',completed=True)
    assert await provider.list_apps(row)=={'apps':[]}
    assert await provider.revoke_app(row,'app-1',1)=={'revoked':True,'version':2}
    (list_path,list_kwargs),(revoke_path,revoke_kwargs)=requests
    assert list_path=='/owner/apps' and revoke_path=='/owner/apps/revoke'
    assert list_kwargs['headers']['Authorization']=='Bearer owner-token'
    assert _proof_target(list_kwargs['headers']['DPoP']).endswith('/owner/apps')
    assert _proof_target(revoke_kwargs['headers']['DPoP']).endswith('/owner/apps/revoke')
    assert revoke_kwargs['json']=={'authorization_id':'app-1','expected_version':1}

def test_only_app_removal_reports_conflict_and_missing_distinctly():
    assert CloudGatewayProvider._removal_error('/owner/apps/revoke',409)=='version_conflict'
    assert CloudGatewayProvider._removal_error('/owner/apps/revoke',404)=='not_found'
    assert CloudGatewayProvider._removal_error('/owner/apps/revoke',503) is None
    assert CloudGatewayProvider._removal_error('/owner/revoke',409) is None
