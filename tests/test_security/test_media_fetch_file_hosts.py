# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratapbhardwaj.com

"""ChatGPT's own file hosts: trusted only for a call that carried a ``file`` object, and every guard stays."""

from __future__ import annotations

import httpx
import pytest

from superlocalmemory.core import media_fetch
from superlocalmemory.core.media_fetch import MediaFetchRefused, fetch_media, fetch_media_to_file

PUBLIC = "93.184.216.34"
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 20
CHAT_HOST = "files.chat.example"
CHAT_LINK = f"https://{CHAT_HOST}/file-1?sig=SECRET"
OWN_HOST = "mine.example"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.setattr(media_fetch, "DEFAULT_FILE_HOSTS", (CHAT_HOST,))


def ok(_request):
    return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})


def get(url, handler=ok, *, hosts="", remote=True, file_param=False, addrs=(PUBLIC,), **kw):
    return fetch_media(url, remote=remote, allowed_hosts=hosts, file_param=file_param,
                       resolver=lambda host, port: list(addrs),
                       transport=httpx.MockTransport(handler), **kw)


def test_the_shipped_default_is_a_tuple_of_plain_lowercase_host_names(monkeypatch):
    monkeypatch.undo()  # drop the test default: look at the value that ships
    shipped = media_fetch.DEFAULT_FILE_HOSTS
    assert isinstance(shipped, tuple)
    assert all(h and h == h.lower() and not any(c in h for c in "*/:@ ") for h in shipped)


def test_a_file_object_makes_the_default_host_trusted_with_no_user_list():
    assert get(CHAT_LINK, file_param=True).data == PNG


def test_a_model_typed_link_to_the_same_host_still_needs_the_users_list():
    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, file_param=False)
    assert "until a host list is set" in e.value.reason and "SECRET" not in e.value.reason
    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, hosts=OWN_HOST, file_param=False)
    assert "not on the allowed list" in e.value.reason
    assert get(CHAT_LINK, hosts=CHAT_HOST, file_param=False).data == PNG  # the user's own list works


def test_the_defaults_are_merged_with_the_users_list_not_replacing_it():
    assert get(f"https://{OWN_HOST}/a.png", hosts=OWN_HOST, file_param=True).data == PNG
    assert get(CHAT_LINK, hosts=OWN_HOST, file_param=True).data == PNG
    with pytest.raises(MediaFetchRefused) as e:
        get("https://other.example/a.png", hosts=OWN_HOST, file_param=True)
    assert "not on the allowed list" in e.value.reason


def test_a_file_object_with_an_unlisted_host_is_refused_for_a_remote_caller():
    with pytest.raises(MediaFetchRefused):
        get("https://other.example/a.png", file_param=True)


def test_with_no_default_hosts_a_remote_file_object_still_needs_the_users_list(monkeypatch):
    monkeypatch.setattr(media_fetch, "DEFAULT_FILE_HOSTS", ())
    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, file_param=True)
    assert "until a host list is set" in e.value.reason
    assert get(CHAT_LINK, hosts=CHAT_HOST, file_param=True).data == PNG


def test_a_local_caller_with_no_user_list_is_not_narrowed_by_the_defaults():
    assert get("https://anywhere.example/a.png", remote=False, file_param=True).data == PNG


@pytest.mark.parametrize("url", [
    f"http://{CHAT_HOST}/f", f"https://u:p@{CHAT_HOST}/f", "file:///etc/passwd",
])
def test_https_only_and_no_userinfo_still_apply_to_a_file_object(url):
    with pytest.raises(MediaFetchRefused):
        get(url, file_param=True)


def test_a_default_host_that_resolves_to_a_private_address_is_refused():
    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, file_param=True, addrs=(PUBLIC, "10.0.0.5"))
    assert "private" in e.value.reason and "SECRET" not in e.value.reason


def test_a_redirect_off_the_trusted_hosts_is_refused():
    def handler(request):
        if request.headers["host"] == CHAT_HOST:
            return httpx.Response(302, headers={"location": "https://evil.example/x.png"})
        return httpx.Response(200, content=PNG)

    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, handler, file_param=True)
    assert "not on the allowed list" in e.value.reason


def test_a_redirect_to_a_private_address_from_a_trusted_host_is_refused():
    def handler(request):
        return httpx.Response(302, headers={"location": f"https://{CHAT_HOST}/again"})

    with pytest.raises(MediaFetchRefused):
        get(CHAT_LINK, handler, file_param=True, addrs=("127.0.0.1",))


def test_the_size_cap_still_applies_to_a_file_object():
    with pytest.raises(MediaFetchRefused) as e:
        get(CHAT_LINK, lambda r: httpx.Response(200, content=b"x" * 100), file_param=True, max_bytes=50)
    assert "too large" in e.value.reason


# -- streaming into a file (documents) -------------------------------------------------

def stream(url, handler, *, sink=None, **kw):
    chunks = sink if sink is not None else []
    kw.setdefault("hosts", "")
    info = fetch_media_to_file(url, chunks.append, remote=kw.pop("remote", True), allowed_hosts=kw.pop("hosts"),
                               file_param=kw.pop("file_param", False),
                               resolver=lambda host, port: [PUBLIC],
                               transport=httpx.MockTransport(handler), **kw)
    return info, chunks


def pdf_chunks(_request):
    return httpx.Response(200, content=iter([b"%PDF-1.4 ", b"a" * 10, b"b" * 10]),
                          headers={"content-type": "application/pdf"})


def test_a_document_streams_chunk_by_chunk_and_reports_where_it_came_from():
    info, chunks = stream(CHAT_LINK, pdf_chunks, file_param=True, max_bytes=100)
    assert b"".join(chunks) == b"%PDF-1.4 " + b"a" * 10 + b"b" * 10 and len(chunks) >= 2
    assert info.size == 29 and info.content_type_header == "application/pdf"
    assert info.final_url == f"https://{CHAT_HOST}/file-1"


def test_a_document_link_asks_for_a_pdf_and_names_a_document_in_refusals():
    seen = []

    def handler(request):
        seen.append(request.headers["accept"])
        return httpx.Response(404)

    with pytest.raises(MediaFetchRefused) as e:
        stream(CHAT_LINK, handler, file_param=True, noun="document")
    assert "pdf" in seen[0] and "document" in e.value.reason and "image" not in e.value.reason


def test_a_document_over_the_cap_is_refused_while_streaming():
    with pytest.raises(MediaFetchRefused) as e:
        stream(CHAT_LINK, pdf_chunks, file_param=True, max_bytes=20, noun="document")
    assert "That document is too large" in e.value.reason


def test_a_document_link_needs_the_same_hosts_as_an_image_link():
    with pytest.raises(MediaFetchRefused):
        stream(CHAT_LINK, pdf_chunks, file_param=False)
    with pytest.raises(MediaFetchRefused):
        stream("https://other.example/a.pdf", pdf_chunks, file_param=True)


def test_a_slow_document_stops_at_the_deadline():
    ticks = iter(range(0, 1000, 5))
    with pytest.raises(MediaFetchRefused) as e:
        stream(CHAT_LINK, pdf_chunks, file_param=True, timeout_s=8, clock=lambda: float(next(ticks)))
    assert "too long" in e.value.reason
