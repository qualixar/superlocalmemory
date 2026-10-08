"""Bounded Python companion execution; no public sockets or live database."""
import asyncio
import base64
import time
import pytest
from superlocalmemory.remote_connections.codec import compact, decode_frame
from superlocalmemory.remote_connections.credentials import ConnectorCredential
try:
    from superlocalmemory.remote_connections.session import RelaySession, OriginResponse
except ImportError:
    RelaySession=OriginResponse=None

def credential():
    return ConnectorCredential("install-a","owner-a","default","a"*32,1,int(time.time()*1000)+60000,"a"*64,"slmr_"+"b"*43)

def frame(identifier="request-a",generation=1,**changes):
    return {"v":1,"kind":"request","id":identifier,"generation":generation,"deadlineAt":int(time.time()*1000)+1000,"headers":[["content-type","application/json"]],"bodyBase64":base64.b64encode(b'{"synthetic":true}').decode(),**changes}

@pytest.mark.asyncio
async def test_session_requires_ready_and_forwards_opaque_bytes():
    assert RelaySession is not None
    sent=[]; closed=[]; calls=[]
    async def exchange(request,private):
        calls.append((base64.b64decode(request["bodyBase64"]),private))
        return OriginResponse(200,(("content-type","application/json"),),b'{"ok":true}')
    async def send(text):sent.append(text)
    session=RelaySession(credential(),exchange=exchange,send=send,close=closed.append)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    assert session.ready
    await session.receive(compact(frame()));await session.wait_idle()
    assert calls[0][0]==b'{"synthetic":true}'
    assert decode_frame(sent[0])["status"]==200
    await session.stop()

@pytest.mark.asyncio
@pytest.mark.parametrize("text",[compact(frame()),'{"v":1,"kind":"ready","generation":0}','{"v":1,"kind":"ready","generation":1,"extra":true}'])
async def test_bad_handshake_closes_without_origin_calls(text):
    assert RelaySession is not None
    closed=[]
    async def fail(*args):raise AssertionError("must not reach origin")
    session=RelaySession(credential(),exchange=fail,send=fail,close=closed.append)
    await session.receive(text)
    assert closed==["connector_protocol_error"] and not session.ready
    await session.stop()

@pytest.mark.asyncio
async def test_cancellation_and_stop_fence_late_origin_response():
    assert RelaySession is not None
    sent=[]; started=asyncio.Event();cancelled=asyncio.Event()
    async def exchange(*args):
        started.set()
        try:await asyncio.Event().wait()
        except asyncio.CancelledError:cancelled.set();return OriginResponse(200,(),b'late')
    async def send(text):sent.append(text)
    session=RelaySession(credential(),exchange=exchange,send=send,close=lambda _:None)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    await session.receive(compact(frame()));await started.wait()
    await session.receive(compact({"v":1,"kind":"cancel","id":"request-a","generation":1}))
    await session.wait_idle();await asyncio.wait_for(cancelled.wait(),1)
    assert sent==[]
    await session.stop()

@pytest.mark.asyncio
async def test_deadline_and_backend_failure_are_bounded_and_sanitized():
    assert RelaySession is not None
    sent=[]
    async def exchange(*args):raise RuntimeError("SECRET private failure")
    async def send(text):sent.append(decode_frame(text))
    session=RelaySession(credential(),exchange=exchange,send=send,close=lambda _:None)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    await session.receive(compact(frame(deadlineAt=0)))
    await session.receive(compact(frame("second")));await session.wait_idle()
    assert [x["status"] for x in sent]==[504,502]
    assert "SECRET" not in str(sent)
    await session.stop()

@pytest.mark.asyncio
async def test_wrong_generation_and_duplicate_active_id_close_session():
    assert RelaySession is not None
    closed=[]
    async def exchange(*args):await asyncio.Event().wait()
    async def send(text):pass
    session=RelaySession(credential(),exchange=exchange,send=send,close=closed.append)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    await session.receive(compact(frame()))
    await session.receive(compact(frame()))
    assert closed==["connector_protocol_error"]
    await session.stop()
    session=RelaySession(credential(),exchange=exchange,send=send,close=closed.append)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    await session.receive(compact(frame(generation=2)))
    assert len(closed)==2
    await session.stop()

@pytest.mark.asyncio
async def test_concurrency_cap_rejects_ninth_request_without_origin():
    assert RelaySession is not None
    sent=[]
    async def exchange(*args):await asyncio.Event().wait()
    async def send(text):sent.append(decode_frame(text))
    session=RelaySession(credential(),exchange=exchange,send=send,close=lambda _:None)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    for n in range(9):await session.receive(compact(frame(f"request-{n}")))
    assert sent[0]["status"]==429
    await session.stop()


async def _ready_session(sent, exchange):
    async def send(text):sent.append(decode_frame(text))
    session=RelaySession(credential(),exchange=exchange,send=send,close=lambda _:None)
    await session.receive(compact({"v":1,"kind":"ready","generation":1}))
    return session

async def _ok(*args):
    return OriginResponse(200,(("content-type","application/json"),),b'{"ok":true}')

@pytest.mark.asyncio
async def test_correct_but_slow_origin_gets_the_relay_budget_not_five_seconds():
    # A real database during maintenance can take more than 5 s; that is not a failure.
    sent=[];session=await _ready_session(sent,_ok)
    await session.receive(compact(frame(deadlineAt=int(time.time()*1000)+20000)));await session.wait_idle()
    assert [x["status"] for x in sent]==[200]
    await session.stop()

@pytest.mark.asyncio
async def test_laptop_clock_behind_the_relay_is_tolerated_and_wait_is_capped(monkeypatch):
    # deadlineAt comes from Cloudflare's clock. A laptop 3 s behind sees 28 s remaining.
    import superlocalmemory.remote_connections.session as module
    waits=[];original=module.asyncio.wait_for
    async def recording(awaitable,timeout):
        waits.append(timeout);return await original(awaitable,timeout)
    monkeypatch.setattr(module.asyncio,"wait_for",recording)
    sent=[];session=await _ready_session(sent,_ok)
    await session.receive(compact(frame(deadlineAt=int(time.time()*1000)+28000)));await session.wait_idle()
    assert [x["status"] for x in sent]==[200]
    assert waits and max(waits)<=25.0
    await session.stop()

@pytest.mark.asyncio
async def test_deadline_beyond_budget_and_clock_tolerance_is_rejected():
    sent=[];calls=[]
    async def exchange(*args):calls.append(args);return await _ok()
    session=await _ready_session(sent,exchange)
    await session.receive(compact(frame(deadlineAt=int(time.time()*1000)+40000)));await session.wait_idle()
    assert [x["status"] for x in sent]==[400] and calls==[]
    await session.stop()
