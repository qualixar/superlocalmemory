# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The image-link fetcher refuses anything that could reach a private network."""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from superlocalmemory.core import media_fetch
from superlocalmemory.core.media_fetch import MediaFetchRefused, fetch_media, unwrap, vet_address

PUBLIC = "93.184.216.34"
PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 20


def resolver_for(*addrs):
    return lambda host, port: list(addrs)


def run(url, handler, *, addrs=(PUBLIC,), remote=False, hosts="", **kw):
    seen: list[httpx.Request] = []

    def wrapped(request):
        seen.append(request)
        return handler(request)

    kw.setdefault("resolver", resolver_for(*addrs))
    out = fetch_media(url, remote=remote, transport=httpx.MockTransport(wrapped), allowed_hosts=hosts, **kw)
    return out, seen


def ok(_request):
    return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})


@pytest.mark.parametrize("addr", [
    "127.0.0.1", "10.1.2.3", "172.16.0.5", "192.168.1.1", "169.254.169.254", "100.64.0.1", "0.0.0.0",
    "224.0.0.1", "::1", "::", "fe80::1", "fc00::1", "ff02::1",
    "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::7f00:1",
    "64:ff9b::a00:1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "64:ff9b:1::1",
    "2002:7f00:1::", "2002:a00:1::", "2001:0:4136:e378:8000:63bf:3fff:fdd2",
])
def test_non_global_addresses_are_refused(addr):
    assert vet_address(addr) is False


@pytest.mark.parametrize("addr", ["93.184.216.34", "8.8.8.8", "2606:4700::1", "::ffff:8.8.8.8", "64:ff9b::808:808"])
def test_public_addresses_pass(addr):
    assert vet_address(addr) is True


def test_unwrap_extracts_the_embedded_ipv4():
    assert unwrap(ipaddress.ip_address("::ffff:127.0.0.1")) == (ipaddress.ip_address("127.0.0.1"),)
    assert ipaddress.ip_address("10.0.0.1") in unwrap(ipaddress.ip_address("64:ff9b::a00:1"))
    assert ipaddress.ip_address("127.0.0.1") in unwrap(ipaddress.ip_address("2002:7f00:1::"))
    assert unwrap(ipaddress.ip_address("93.184.216.34")) == ()


def test_a_good_link_is_fetched():
    out, seen = run("https://img.example.com/a.png?token=SECRET", ok)
    assert out.data == PNG and out.content_type_header == "image/png"
    assert out.final_url == "https://img.example.com/a.png"  # no query or userinfo kept
    assert len(seen) == 1 and seen[0].method == "GET"


@pytest.mark.parametrize("url", [
    "http://img.example.com/a.png", "ftp://img.example.com/a.png", "file:///etc/passwd",
    "https://zed:hunter2@img.example.com/a.png", "https:///a.png", "not a url", "",
])
def test_only_plain_https_links_are_allowed(url):
    with pytest.raises(MediaFetchRefused) as e:
        run(url, ok)
    assert "hunter2" not in e.value.reason and "zed" not in e.value.reason


def test_an_ip_literal_to_a_private_address_is_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://127.0.0.1/a.png", ok)
    with pytest.raises(MediaFetchRefused):
        run("https://[::ffff:7f00:1]/a.png", ok)


def test_a_dns_answer_with_one_private_address_is_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", ok, addrs=(PUBLIC, "10.0.0.5"))


def test_an_empty_dns_answer_is_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", ok, addrs=())


def test_a_dns_failure_is_refused_without_the_url():
    def boom(host, port):
        raise OSError("nope")

    with pytest.raises(MediaFetchRefused) as e:
        run("https://img.example.com/a.png?k=SECRET", ok, resolver=boom)
    assert "SECRET" not in e.value.reason


def test_the_connection_goes_to_the_vetted_ip_with_the_original_host():
    _out, seen = run("https://img.example.com/p/a.png?x=1", ok)
    req = seen[0]
    assert req.url.host == PUBLIC and req.url.path == "/p/a.png" and req.url.query == b"x=1"
    assert req.headers["host"] == "img.example.com"
    assert req.extensions["sni_hostname"] == "img.example.com"


def test_an_ipv6_answer_is_connected_to_in_brackets():
    _out, seen = run("https://img.example.com/a.png", ok, addrs=("2606:4700::1",))
    assert seen[0].url.host == "2606:4700::1" and seen[0].headers["host"] == "img.example.com"
    assert str(seen[0].url).startswith("https://[2606:4700::1]")


def test_the_name_is_resolved_once_so_a_rebind_cannot_slip_in():
    calls = []

    def resolver(host, port):
        calls.append(host)
        return [PUBLIC] if len(calls) == 1 else ["127.0.0.1"]

    _out, seen = run("https://img.example.com/a.png", ok, resolver=resolver)
    assert calls == ["img.example.com"] and seen[0].url.host == PUBLIC


def test_each_redirect_is_resolved_and_vetted_again():
    def handler(request):
        if request.headers["host"] == "img.example.com":
            return httpx.Response(302, headers={"location": "https://evil.example.net/x.png"})
        return httpx.Response(200, content=PNG)

    def resolver(host, port):
        return [PUBLIC] if host == "img.example.com" else ["169.254.169.254"]

    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", handler, resolver=resolver)


def test_a_redirect_to_a_private_literal_or_http_is_refused():
    for loc in ("https://10.0.0.1/x.png", "http://img.example.com/x.png", "https://[::1]/x"):
        h = lambda r, loc=loc: httpx.Response(301, headers={"location": loc})  # noqa: E731
        with pytest.raises(MediaFetchRefused):
            run("https://img.example.com/a.png", h)


def test_up_to_three_redirects_are_followed_and_a_fourth_is_refused():
    def chain(limit):
        def handler(request):
            n = int(request.url.path.strip("/") or 0)
            if n < limit:
                return httpx.Response(302, headers={"location": f"/{n + 1}"})
            return httpx.Response(200, content=PNG)
        return handler

    out, seen = run("https://img.example.com/", chain(3))
    assert out.data == PNG and len(seen) == 4
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/", chain(4))


def test_a_redirect_without_a_location_is_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", lambda r: httpx.Response(302))


def test_a_stream_past_the_cap_is_aborted():
    pulled = []

    def stream():
        for _ in range(40):
            pulled.append(1)
            yield b"y" * (1024 * 1024)

    with pytest.raises(MediaFetchRefused) as e:
        run("https://img.example.com/big.png", lambda r: httpx.Response(200, content=stream()))
    assert "25 MB" in e.value.reason and len(pulled) <= 27


def test_a_declared_length_past_the_cap_is_refused_early():
    h = lambda r: httpx.Response(200, headers={"content-length": str(26 * 1024 * 1024)}, content=b"z")  # noqa: E731
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/big.png", h)


def test_the_total_time_is_bounded():
    now = [0.0]

    def stream():
        for _ in range(5):
            now[0] += 4.0
            yield b"slow"

    with pytest.raises(MediaFetchRefused) as e:
        run("https://img.example.com/a.png", lambda r: httpx.Response(200, content=stream()), clock=lambda: now[0])
    assert "too long" in e.value.reason


def test_a_connect_timeout_is_a_refusal():
    def handler(request):
        raise httpx.ConnectTimeout("t")

    with pytest.raises(MediaFetchRefused) as e:
        run("https://img.example.com/a.png?k=SECRET", handler)
    assert "SECRET" not in e.value.reason


def test_an_error_status_is_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", lambda r: httpx.Response(404))


def test_remote_callers_with_no_allow_list_are_refused():
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", ok, remote=True, hosts="")


def test_local_callers_with_no_allow_list_may_fetch_any_public_host():
    out, _ = run("https://img.example.com/a.png", ok, remote=False, hosts="")
    assert out.data == PNG


def test_the_allow_list_limits_hosts_and_redirect_targets(monkeypatch):
    out, _ = run("https://img.example.com/a.png", ok, remote=True, hosts="other.test, IMG.example.com")
    assert out.data == PNG
    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", ok, remote=False, hosts="other.test")

    def hop(request):
        return httpx.Response(302, headers={"location": "https://cdn.other.net/x.png"})

    with pytest.raises(MediaFetchRefused):
        run("https://img.example.com/a.png", hop, remote=False, hosts="img.example.com")


def test_the_allow_list_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SLM_MEDIA_URL_HOSTS", "only.example.org")
    with pytest.raises(MediaFetchRefused):
        fetch_media("https://img.example.com/a.png", remote=False, resolver=resolver_for(PUBLIC),
                    transport=httpx.MockTransport(ok))


def test_the_client_ignores_the_environment_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    seen = {}
    real = httpx.Client

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(media_fetch.httpx, "Client", spy)
    run("https://img.example.com/a.png", ok)
    assert seen["trust_env"] is False and seen["follow_redirects"] is False


def test_refusal_reasons_never_hold_the_query_or_userinfo():
    for url in ("https://zed:hunter2@img.example.com/a.png?k=SECRET", "http://img.example.com/a.png?k=SECRET",
                "https://10.0.0.1/a.png?k=SECRET"):
        with pytest.raises(MediaFetchRefused) as e:
            run(url, ok)
        assert "SECRET" not in e.value.reason and "hunter2" not in e.value.reason


def test_the_gate_guard_lists_the_fetcher():
    from tests.test_security.test_outbound_gate_guard import REVIEWED

    assert any(site[0].endswith("core/media_fetch.py") for site in REVIEWED)
