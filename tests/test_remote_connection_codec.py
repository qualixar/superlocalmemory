"""Python companion wire compatibility with the existing TypeScript relay."""
import base64
import json
import pytest
try:
    from superlocalmemory.remote_connections.codec import decode_frame, encode_frame, FrameError
except ImportError:
    decode_frame=encode_frame=None
    FrameError=RuntimeError

def request(**changes):
    return {"v":1,"kind":"request","id":"wire_1","generation":1,"deadlineAt":2000,
            "headers":[["Content-Type","application/json"]],"bodyBase64":base64.b64encode(b'{"synthetic":true}').decode(),**changes}
def wire(frame):return json.dumps(frame,separators=(",",":"),ensure_ascii=False)

def test_private_wire_roundtrip_preserves_original_mcp_bytes():
    assert decode_frame is not None
    message=request(); result=decode_frame(wire(message))
    assert result==message and encode_frame(result)==wire(message)
    assert base64.b64decode(result["bodyBase64"])==b'{"synthetic":true}'

@pytest.mark.parametrize("frame",[
    request(v=True),request(generation=True),request(generation=0),request(generation=2**53),
    request(id="../bad"),request(extra="unknown"),request(bodyBase64="AB=="),request(bodyBase64="AA"),
    request(headers=[["Authorization","secret"]]),request(headers=[["X-Install-Token","secret"]]),
    request(headers=[["content-type","application/json"],["Content-Type","application/json"]]),
    request(headers=[["content-type","bad\r\nheader"]]),request(headers=[["Mcp-Param-profile","default"]]),
])
def test_malformed_or_credential_bearing_wire_is_refused(frame):
    assert decode_frame is not None
    with pytest.raises(FrameError):decode_frame(wire(frame))

@pytest.mark.parametrize("text",[' {}','{"v":1,"v":1,"kind":"cancel","id":"x","generation":1}', '\ufeff{"v":1}', '[1]', '{'])
def test_noncanonical_duplicate_or_bom_frame_is_refused(text):
    assert decode_frame is not None
    with pytest.raises(FrameError):decode_frame(text)

def test_schema_curated_parameter_headers_are_request_only():
    assert decode_frame is not None
    frame=request(headers=[["Mcp-Param-profile","default"]])
    assert decode_frame(wire(frame),param_headers=("Mcp-Param-profile",))==frame
    response={"v":1,"kind":"response","id":"wire_1","generation":1,"status":200,"headers":frame["headers"],"bodyBase64":""}
    with pytest.raises(FrameError):decode_frame(wire(response),param_headers=("Mcp-Param-profile",))

def test_frame_and_body_limits_apply_before_processing():
    assert decode_frame is not None
    with pytest.raises(FrameError):decode_frame(" "*(8*1024*1024+1))
    with pytest.raises(FrameError):decode_frame(wire(request(bodyBase64=base64.b64encode(b'a'*(1024*1024+1)).decode())))

def test_response_and_cancel_contract():
    assert decode_frame is not None
    for frame in [{"v":1,"kind":"cancel","id":"x","generation":1},{"v":1,"kind":"response","id":"x","generation":1,"status":204,"headers":[],"bodyBase64":""}]:
        assert decode_frame(encode_frame(frame))==frame

@pytest.mark.parametrize("options",[("Authorization",), ("Mcp-Param-bad header",), "Mcp-Param-name"])
def test_invalid_curated_header_configuration_fails_closed(options):
    assert decode_frame is not None
    with pytest.raises(FrameError):decode_frame(wire(request()),param_headers=options)
