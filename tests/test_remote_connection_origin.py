"""Private in-process adapter; runs the existing remote listener wrapper."""
import base64
import json
import time
import pytest
from superlocalmemory.remote_connections.credentials import ConnectorCredential
from superlocalmemory.remote_connections.codec import MAX_RESPONSE_BYTES
try:
    from superlocalmemory.remote_connections.origin import CanonicalMcpOrigin
except ImportError:
    CanonicalMcpOrigin=None

def credential():return ConnectorCredential("install-a","owner-a","default","a"*32,1,int(time.time()*1000)+60000,"a"*64,"slmr_"+"b"*43)
def frame():return {"v":1,"kind":"request","id":"x","generation":1,"deadlineAt":int(time.time()*1000)+1000,"headers":[["content-type","application/json"]],"bodyBase64":base64.b64encode(b'{"jsonrpc":"2.0","id":1,"method":"tools/list"}').decode()}

@pytest.mark.asyncio
async def test_origin_uses_one_app_with_private_remote_identity_and_opaque_body():
    assert CanonicalMcpOrigin is not None
    observed=[]
    async def app(scope,receive,send):
        observed.append((scope,await receive()))
        await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"application/json"),(b"set-cookie",b"private=secret")]})
        await send({"type":"http.response.body","body":b'{"result":{}}'})
    response=await CanonicalMcpOrigin(app)(frame(),credential())
    scope,body=observed[0]
    assert scope["slm_remote_listener"] is True and scope["slm_remote_host_verified"] is True
    assert scope["path"]=="/mcp/" and scope["method"]=="POST"
    assert scope["client"][0]=="remote-listener-peer"
    assert dict(scope["headers"])[b"authorization"]==b"Bearer slmr_"+b"b"*43
    assert body["body"]==base64.b64decode(frame()["bodyBase64"])
    assert response.headers==(("content-type","application/json"),)
    assert response.body==b'{"result":{}}'

@pytest.mark.asyncio
async def test_origin_bounds_response_before_transport_buffers_it():
    assert CanonicalMcpOrigin is not None
    async def app(scope,receive,send):
        await send({"type":"http.response.start","status":200,"headers":[]})
        await send({"type":"http.response.body","body":b'a'*(MAX_RESPONSE_BYTES+1)})
    with pytest.raises(ValueError,match="origin_unavailable"):await CanonicalMcpOrigin(app)(frame(),credential())

@pytest.mark.asyncio
async def test_origin_cannot_follow_redirect_or_expose_provider_exception():
    assert CanonicalMcpOrigin is not None
    async def app(scope,receive,send):
        await send({"type":"http.response.start","status":307,"headers":[(b"location",b"https://evil.example")]})
        await send({"type":"http.response.body","body":b''})
    with pytest.raises(ValueError,match="origin_redirect_denied"):await CanonicalMcpOrigin(app)(frame(),credential())
    async def fail(*args):raise RuntimeError("SECRET engine exception")
    with pytest.raises(ValueError,match="origin_unavailable") as error:await CanonicalMcpOrigin(fail)(frame(),credential())
    assert "SECRET" not in str(error.value)

@pytest.mark.asyncio
async def test_origin_never_uses_install_token_or_global_api_key():
    assert CanonicalMcpOrigin is not None
    from dataclasses import replace
    async def app(*args):raise AssertionError("must not call app")
    with pytest.raises(ValueError):await CanonicalMcpOrigin(app)(frame(),replace(credential(),origin_key="global-api-key"))

@pytest.mark.asyncio
async def test_origin_reaches_real_pinned_sdk_transport_without_host_bypass():
    from fastapi import FastAPI
    from mcp.server.mcpserver import MCPServer
    server=MCPServer('SLM transport verification')
    @server.tool()
    async def recall(query: str) -> dict:
        return {'synthetic':query}
    mcp_app=server.streamable_http_app(stateless_http=True,json_response=True,streamable_http_path='/',host='127.0.0.1')
    app=FastAPI();app.mount('/mcp',mcp_app)
    payload={'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-06-18','capabilities':{},'clientInfo':{'name':'synthetic','version':'1'}}}
    packet=frame();packet['headers'].append(['accept','application/json']);packet['bodyBase64']=base64.b64encode(json.dumps(payload).encode()).decode()
    async with mcp_app.router.lifespan_context(mcp_app):
        response=await CanonicalMcpOrigin(app)(packet,credential())
    assert response.status==200
    assert json.loads(response.body)['result']['protocolVersion']=='2025-06-18'
