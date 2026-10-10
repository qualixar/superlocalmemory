"""The dashboard's streaming upload route: the sizes the pane advertises really get through (audit F4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.documents.submit import DocumentReceipt
from superlocalmemory.media.ingest import MediaReceipt
from superlocalmemory.server.routes import media_dashboard_upload as up
from tests.test_media._real_roles import Roles

LOCAL, REMOTE = ("127.0.0.1", 50000), ("10.1.2.3", 50000)
MB = 1024 * 1024
PNG = b"\x89PNG\r\n\x1a\n"
PDF = b"%PDF-1.7\n"
URL = "/api/v3/media/upload"


class Db:
    def execute(self, sql, params=()):
        return [{"one": 1}] if params and params[0] in ("default", "p2") else []


class Spy:
    def __init__(self):
        self.images, self.docs, self.governed = [], [], []


@pytest.fixture()
def spy(monkeypatch):
    s = Spy()

    def image(inp, **kw):
        s.images.append((inp, kw))
        return MediaReceipt("stored", media_id="m" * 32, memory_id="mem1")

    def doc(inp, **kw):
        assert Path(inp.path).is_file(), "the PDF is on disk while it is ingested"
        s.docs.append((inp, kw, Path(inp.path).stat().st_size))
        return DocumentReceipt("processing", document_id="d" * 32, job_id="j" * 32)

    monkeypatch.setattr(up, "remember_media", image)
    monkeypatch.setattr(up, "submit_document", doc)
    monkeypatch.setattr(up, "enforce_remember_governance", lambda *a, **k: s.governed.append(k))
    return s


def make(*, actor="authenticated:test", client=LOCAL, runtime=True):
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=SimpleNamespace(pii_redaction=False),
                                       _db=Db(), _hooks=None)
    if runtime:
        app.state.canonical_remember_runtime = object()

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(up.router)
    return TestClient(app, client=client)


def test_the_limits_are_the_ones_the_pane_advertises() -> None:
    assert up.IMAGE_LIMIT == 25 * MB and up.PDF_LIMIT == 100 * MB


def test_an_image_the_json_route_would_refuse_is_accepted(spy) -> None:
    body = PNG + b"x" * (15 * MB)  # base64 of this is 20M chars, past the JSON route's 12M cap
    r = make().post(f"{URL}?kind=image", content=body)
    assert r.status_code == 200 and r.json()["status"] == "stored"
    (inp, kw), = spy.images
    assert inp.data == body and kw["profile_id"] == "default" and kw["actor_id"] == "authenticated:test"
    assert kw["runtime"] is not None


def test_a_pdf_the_json_route_would_refuse_is_streamed_to_a_file_and_cleaned_up(spy) -> None:
    body = PDF + b"x" * (30 * MB)  # past the JSON route's ~25 MB cap
    r = make().post(f"{URL}?kind=pdf&file_name=big.pdf", content=body)
    assert r.status_code == 202 and r.json()["status"] == "processing"
    (inp, kw, size), = spy.docs
    assert size == len(body) and inp.file_name == "big.pdf" and kw["profile_id"] == "default"
    assert not Path(inp.path).exists() and not Path(inp.path).parent.exists()


def test_the_scratch_file_is_not_inside_the_data_folder(spy) -> None:
    from superlocalmemory.infra.data_root import overlaps_data_root

    make().post(f"{URL}?kind=pdf", content=PDF + b"x")
    assert not overlaps_data_root(spy.docs[0][0].path)


@pytest.mark.parametrize("kind,limit_name", [("image", "IMAGE_LIMIT"), ("pdf", "PDF_LIMIT")])
def test_a_body_over_the_limit_is_refused_while_streaming(spy, monkeypatch, kind, limit_name) -> None:
    monkeypatch.setattr(up, limit_name, 1000)
    r = make().post(f"{URL}?kind={kind}", content=b"x" * 1001)
    assert r.status_code == 413
    assert spy.images == [] and spy.docs == []


def test_a_declared_size_over_the_limit_is_refused_before_reading(spy, monkeypatch) -> None:
    monkeypatch.setattr(up, "IMAGE_LIMIT", 1000)
    r = make().post(f"{URL}?kind=image", content=b"x" * 1001, headers={"content-length": "1001"})
    assert r.status_code == 413 and spy.images == []


def test_a_stream_without_a_length_is_still_capped(spy, monkeypatch) -> None:
    monkeypatch.setattr(up, "PDF_LIMIT", 1000)

    def chunks():
        for _ in range(5):
            yield b"x" * 400

    r = make().post(f"{URL}?kind=pdf", content=chunks())
    assert r.status_code == 413 and spy.docs == []


def test_an_unknown_kind_is_refused(spy) -> None:
    assert make().post(f"{URL}?kind=video", content=b"x").status_code == 422
    assert make().post(URL, content=b"x").status_code == 422


def test_only_local_callers_with_the_write_credential(spy) -> None:
    assert make(client=REMOTE).post(f"{URL}?kind=image", content=PNG).status_code == 403
    assert make(actor="").post(f"{URL}?kind=image", content=PNG).status_code == 403
    assert spy.images == []


def test_an_image_needs_the_memory_writer(spy) -> None:
    assert make(runtime=False).post(f"{URL}?kind=image", content=PNG).status_code == 503


def test_the_save_goes_through_the_same_governance_as_the_json_routes(spy) -> None:
    make().post(f"{URL}?kind=image", content=PNG)
    assert len(spy.governed) == 1 and spy.governed[0]["profile"] == "default"


def test_a_refusal_comes_back_as_422_with_the_reason(monkeypatch, spy) -> None:
    monkeypatch.setattr(up, "remember_media", lambda inp, **kw: MediaReceipt("refused", reason="not a picture"))
    r = make().post(f"{URL}?kind=image", content=b"zzz")
    assert r.status_code == 422 and r.json()["detail"] == "not a picture"


def test_an_unknown_profile_is_404(spy) -> None:
    assert make().post(f"{URL}?kind=image&profile_id=nope", content=PNG).status_code == 404


@pytest.mark.parametrize("role,code", [("owner", 200), ("admin", 200), ("member", 200), ("viewer", 403)])
def test_roles_are_the_ones_pasted_data_needs(tmp_path, spy, role, code) -> None:
    roles = Roles(tmp_path)
    roles.client.app.include_router(up.router)
    r = roles.client.post(f"{URL}?kind=image", content=PNG, headers=roles.headers(role))
    assert r.status_code == code
    assert len(spy.images) == (1 if code == 200 else 0)


def test_an_outsider_cannot_probe_profile_names(tmp_path, spy) -> None:
    roles = Roles(tmp_path)
    roles.client.app.include_router(up.router)
    codes = {roles.client.post(f"{URL}?kind=image&profile_id={p}", content=PNG,
                               headers=roles.headers("outsider")).status_code for p in ("p2", "nope")}
    assert codes == {403}


def test_the_real_ingest_takes_what_this_route_hands_it(tmp_path, monkeypatch) -> None:
    """The size checks inside the ingest agree with the route: 20 MB of image bytes, a 30 MB PDF file."""
    from superlocalmemory.documents import submit
    from superlocalmemory.media import files, ingest
    from superlocalmemory.media.ingest import MediaInput

    root = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    assert len(ingest._read_input(MediaInput(data=PNG + b"x" * (20 * MB)))) > 20 * MB
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pdf = scratch / "upload.bin"
    pdf.write_bytes(PDF + b"x" * (30 * MB))
    staged, _sha, size = submit._stage(MediaInput(path=pdf, file_name="big.pdf"), root)
    assert size == pdf.stat().st_size and staged.parent == files.tmp_dir(root)
