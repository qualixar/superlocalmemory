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
