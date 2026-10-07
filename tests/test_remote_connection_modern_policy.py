"""Modern discovery through the existing remote policy, preserving legacy use."""
import json
from tests.test_security.test_remote_tool_policy import READ_KEY, WRITE_KEY, _run
from superlocalmemory.server import remote_tool_policy as policy
import pytest

class Discovery:
    def __init__(self):self.reached=[]
    async def __call__(self,scope,receive,send):
        request=json.loads((await receive())["body"]);self.reached.append(request)
        result={"cacheScope":"public","ttlMs":86400000,"resultType":"complete","supportedVersions":["2026-07-28","2025-11-25"],"capabilities":{"tools":{"listChanged":True},"resources":{},"prompts":{}},"instructions":"SECRET host instructions","_meta":{"private":"SECRET"}}
        body=json.dumps({"jsonrpc":"2.0","id":request["id"],"result":result}).encode()
        await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"application/json"),(b"cache-control",b"public, max-age=86400")]})
        await send({"type":"http.response.body","body":body})

@pytest.mark.parametrize("principal",[READ_KEY,WRITE_KEY])
def test_modern_discovery_is_allowed_scoped_and_never_publicly_cached(principal):
    stub=Discovery();messages=[]
    body=json.dumps({"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}}).encode()
    status,result,_=_run(body,principal,stub=stub,sent_out=messages)
    assert status==200 and stub.reached
    assert result["result"]["cacheScope"]=="private" and result["result"]["ttlMs"]==0
    assert set(result["result"]["capabilities"])=={"tools"}
    assert "SECRET" not in json.dumps(result)
    headers=dict(messages[0]["headers"])
    assert headers[b"cache-control"]==b"no-store"

def test_discovery_does_not_enable_admin_or_subscription_methods():
    assert "server/discover" in policy.ALLOWED_METHODS
    for method in ["subscriptions/listen","resources/read","prompts/get","mesh/send"]:
        body=json.dumps({"jsonrpc":"2.0","id":1,"method":method}).encode()
        assert _run(body,WRITE_KEY)[1]["error"]["code"]==-32601

def test_remote_json_filter_limits_response_buffer_before_forwarding():
    class Oversized(Discovery):
        async def __call__(self,scope,receive,send):
            await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"application/json")]})
            await send({"type":"http.response.body","body":b'a'*(4*1024*1024+1)})
    status,result,_=_run(json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/list"}).encode(),READ_KEY,stub=Oversized())
    assert status==502 and result["error"]=="remote_answer_too_large"
