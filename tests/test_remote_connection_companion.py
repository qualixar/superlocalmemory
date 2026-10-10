"""Opt-in lifecycle uses fake sockets; canonical origin tests are separate."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import time
import pytest
from superlocalmemory.remote_connections.codec import compact
from superlocalmemory.remote_connections.credentials import ConnectorCredential
try:
    from superlocalmemory.remote_connections.companion import Companion
except ImportError:
    Companion=None

def credential():return ConnectorCredential("install-a","owner-a","default","a"*32,1,int(time.time()*1000)+60000,"a"*64,"slmr_"+"b"*43)
async def wait_until(condition):
    for _ in range(100):
        if condition():return
        await asyncio.sleep(.005)
    raise AssertionError("bounded lifecycle wait expired")

class Socket:
    def __init__(self):self.queue=asyncio.Queue();self.sent=[];self.closed=False;self.queue.put_nowait(compact({"v":1,"kind":"ready","generation":1}))
    async def recv(self):return await self.queue.get()
    async def send(self,text):
        self.sent.append(text)
        if text=="ping":self.queue.put_nowait("pong")

@pytest.fixture
def harness():
    assert Companion is not None
    states=[]; calls=[]; socket=Socket()
    async def load():return credential()
    async def exchange(*args):raise AssertionError("no requests expected")
    @asynccontextmanager
    async def dial(endpoint,token):
        calls.append((endpoint,token))
        try:yield socket
        finally:socket.closed=True
    return states,calls,socket,load,exchange,dial

@pytest.mark.asyncio
async def test_disabled_companion_loads_nothing_and_opens_no_socket(harness):
    states,calls,socket,load,exchange,dial=harness
    async def fail():raise AssertionError("must not load")
    companion=Companion(enabled=False,load_credential=fail,exchange=exchange,dial=dial,on_state=states.append)
    await companion.start();assert states==["disabled"] and not calls
    await companion.stop()

@pytest.mark.asyncio
async def test_ready_heartbeat_and_stop_are_owned_by_one_companion(harness):
    states,calls,socket,load,exchange,dial=harness
    companion=Companion(enabled=True,load_credential=load,exchange=exchange,dial=dial,on_state=states.append,heartbeat_ms=10)
    await companion.start();await companion.start();await wait_until(lambda:"transport_ready" in states)
    await wait_until(lambda:"ping" in socket.sent)
    await companion.stop();assert socket.closed and states[-1]=="stopped" and len(calls)==1
    assert calls[0][0]=="wss://connect.superlocalmemory.com/connector"

@pytest.mark.asyncio
async def test_expired_or_missing_credential_stops_remote_only(harness):
    states,calls,socket,load,exchange,dial=harness
    for value in [None,replace(credential(),expires_at_ms=1)]:
        async def expired():return value
        companion=Companion(enabled=True,load_credential=expired,exchange=exchange,dial=dial,on_state=states.append)
        await companion.start();await wait_until(lambda:states and states[-1]=="authorization_required")
        assert not calls;await companion.stop()

@pytest.mark.asyncio
async def test_stop_fences_delayed_secure_store_load(harness):
    states,calls,socket,load,exchange,dial=harness
    started=asyncio.Event();release=asyncio.Event()
    async def delayed():started.set();await release.wait();return credential()
    companion=Companion(enabled=True,load_credential=delayed,exchange=exchange,dial=dial,on_state=states.append)
    await companion.start();await started.wait();await companion.stop();release.set();await asyncio.sleep(0)
    assert calls==[] and states[-1]=="stopped"

@pytest.mark.asyncio
async def test_network_loss_reconnects_and_never_replays_a_request(harness):
    states,calls,socket,load,exchange,dial=harness
    companion=Companion(enabled=True,load_credential=load,exchange=exchange,dial=dial,on_state=states.append,retry_ms=5)
    await companion.start();await wait_until(lambda:"transport_ready" in states)
    socket.queue.put_nowait(b"binary is forbidden")
    await wait_until(lambda:len(calls)>=2)
    await companion.stop();assert "reconnecting" in states

@pytest.mark.asyncio
async def test_handshake_auth_denial_does_not_retry_forever(harness):
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response
    from websockets.datastructures import Headers
    states,calls,socket,load,exchange,dial=harness
    @asynccontextmanager
    async def denied(endpoint,token):
        calls.append(endpoint)
        raise InvalidStatus(Response(401,"Unauthorized",Headers()))
        yield socket
    companion=Companion(enabled=True,load_credential=load,exchange=exchange,dial=denied,on_state=states.append,retry_ms=5)
    await companion.start();await wait_until(lambda:states and states[-1]=="authorization_required")
    await asyncio.sleep(.02);assert len(calls)==1;await companion.stop()

def test_invalid_lifecycle_configuration_rejected(harness):
    states,calls,socket,load,exchange,dial=harness
    with pytest.raises(ValueError):Companion(enabled=True,load_credential=load,exchange=exchange,dial=dial,on_state=states.append,retry_ms=0)

@pytest.mark.asyncio
async def test_connection_end_reason_is_logged_without_secrets(harness,caplog):
    # A relay 503 is only diagnosable if the laptop records why its socket ended.
    states,calls,socket,load,exchange,dial=harness
    caplog.set_level("INFO",logger="superlocalmemory.remote_connections.companion")
    companion=Companion(enabled=True,load_credential=load,exchange=exchange,dial=dial,on_state=states.append,retry_ms=5)
    await companion.start();await wait_until(lambda:"transport_ready" in states)
    socket.queue.put_nowait(b"binary is forbidden")
    await wait_until(lambda:len(calls)>=2)
    await companion.stop()
    text="\n".join(record.getMessage() for record in caplog.records)
    assert "remote_companion_state state=transport_ready" in text
    assert "remote_companion_connection_ended reason=connected_then_lost" in text
    assert "slmr_" not in text and "a"*32 not in text

@pytest.mark.asyncio
async def test_connection_failure_logs_error_class_only(harness,caplog):
    states,calls,socket,load,exchange,_=harness
    caplog.set_level("INFO",logger="superlocalmemory.remote_connections.companion")
    attempts=[]
    @asynccontextmanager
    async def failing(endpoint,token):
        attempts.append(1)
        raise RuntimeError("SECRET dial detail")
        yield
    companion=Companion(enabled=True,load_credential=load,exchange=exchange,dial=failing,on_state=states.append,retry_ms=5)
    await companion.start();await wait_until(lambda:len(attempts)>=2)
    await companion.stop()
    text="\n".join(record.getMessage() for record in caplog.records)
    assert "remote_companion_connection_failed error=RuntimeError" in text
    assert "SECRET" not in text

@pytest.mark.asyncio
async def test_a_busy_credential_store_is_retried_not_treated_as_misconfiguration(harness):
    from superlocalmemory.remote_connections.credentials import CredentialError
    states,calls,socket,load,exchange,dial=harness
    attempts=[]
    async def busy_then_ready():
        attempts.append(1)
        if len(attempts)<3:
            raise CredentialError("credential_store_busy")
        return credential()
    companion=Companion(enabled=True,load_credential=busy_then_ready,exchange=exchange,dial=dial,on_state=states.append,retry_ms=5)
    await companion.start()
    await wait_until(lambda:"transport_ready" in states)
    await companion.stop()
    assert "configuration_error" not in states and len(attempts)==3

@pytest.mark.asyncio
async def test_a_broken_credential_is_still_a_configuration_error(harness):
    from superlocalmemory.remote_connections.credentials import CredentialError
    states,calls,socket,load,exchange,dial=harness
    async def broken():
        raise CredentialError("invalid_connector_credential")
    companion=Companion(enabled=True,load_credential=broken,exchange=exchange,dial=dial,on_state=states.append,retry_ms=5)
    await companion.start()
    await wait_until(lambda:"configuration_error" in states)
    await companion.stop()
    assert calls==[]

def test_the_dial_says_it_reads_grants(monkeypatch):
    import websockets.asyncio.client as client
    from superlocalmemory.remote_connections import companion as module
    from superlocalmemory.remote_connections.proof import DeviceSigner
    seen={}
    class Fake:
        def __init__(self,endpoint,**kwargs):seen.update(kwargs)
    monkeypatch.setattr(client,"connect",Fake)
    module._dial("wss://connect.superlocalmemory.com/connector","tok",DeviceSigner.generate().private_pem)
    headers=seen["additional_headers"]
    assert headers["x-slm-connector-features"]=="grant-v1"
    assert headers["Authorization"]=="Bearer tok" and headers["DPoP"]
