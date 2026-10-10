# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Deterministic facets: links, paths, dates, versions, code names, tags."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.tagging import EXTRACTED_KEY, Extracted, extract_deterministic
from superlocalmemory.tagging.extractors import MAX_ITEMS, MAX_SCAN_CHARS


def test_key_name() -> None:
    assert EXTRACTED_KEY == "_slm_extracted"


def test_url_strips_query_fragment_userinfo() -> None:
    got = extract_deterministic(
        "see https://user:pw@Example.COM:8080/a/b?token=abc#frag now"
    )
    assert got.urls == ("https://example.com:8080/a/b",)
    assert "token" not in str(got.as_metadata())
    assert "pw" not in str(got.as_metadata())


def test_url_trailing_punctuation_and_parens() -> None:
    got = extract_deterministic(
        "(visit https://a.io/x). Also https://b.io/y, and https://c.io/z(1) ok"
    )
    assert got.urls == ("https://a.io/x", "https://b.io/y", "https://c.io/z(1)")


def test_url_path_is_not_also_a_path() -> None:
    got = extract_deterministic("https://example.com/docs/guide")
    assert got.paths == ()


def test_non_http_urls_ignored() -> None:
    assert extract_deterministic("ftp://host/a/b").urls == ()


def test_paths_posix_home_windows() -> None:
    got = extract_deterministic(
        r"edit /etc/nginx/nginx.conf and ~/notes/todo.md then C:\Users\me\x.txt"
    )
    assert got.paths == ("/etc/nginx/nginx.conf", "~/notes/todo.md", r"C:\Users\me\x.txt")


def test_not_paths() -> None:
    got = extract_deterministic("half is 1/2, use and/or, or /single")
    assert got.paths == ()


def test_dates_validated() -> None:
    got = extract_deterministic("on 2026-02-28 and 2026-02-30 and 2026-13-01")
    assert got.dates == ("2026-02-28",)


def test_versions_vs_ip_vs_date() -> None:
    got = extract_deterministic(
        "upgrade to v4.1.25 or 2.0-rc1, host 192.168.1.10, built 2026-03-04, v1.2.3.4"
    )
    assert got.versions == ("v4.1.25", "2.0-rc1")
    assert got.dates == ("2026-03-04",)


def test_code_ids_backticks_only() -> None:
    got = extract_deterministic(
        "call `write_queryable` and `engine.MemoryEngine` and `Foo::bar` plus `camelCase`, "
        "not plain_snake or `two words`"
    )
    assert got.code_ids == ("write_queryable", "engine.MemoryEngine", "Foo::bar", "camelCase")


def test_hashtags() -> None:
    got = extract_deterministic(
        "# Heading\n#Project/Alpha and #idea, issue #123, colour #fff #a1b2c3, a&#38; x/#no"
    )
    assert got.hashtags == ("project/alpha", "idea")


def test_dedup_and_order() -> None:
    got = extract_deterministic("#b #a #b https://x.io/1 https://x.io/1?q=2 https://w.io/2")
    assert got.hashtags == ("b", "a")
    assert got.urls == ("https://x.io/1", "https://w.io/2")


def test_caps() -> None:
    text = " ".join(f"https://h{i}.io/p" for i in range(17))
    assert len(extract_deterministic(text).urls) == MAX_ITEMS == 16
    long_url = "https://example.com/" + "a" * 300
    assert extract_deterministic(long_url).urls == ()


def test_text_beyond_scan_limit_ignored() -> None:
    text = "x" * MAX_SCAN_CHARS + " https://late.io/a #late 2026-01-01"
    assert extract_deterministic(text).is_empty()


def test_empty_and_prose() -> None:
    assert extract_deterministic("").is_empty()
    got = extract_deterministic("The team agreed to ship on Friday after the review.")
    assert got.is_empty() and got.as_metadata() == {}


def test_type_error() -> None:
    with pytest.raises(TypeError):
        extract_deterministic(b"bytes")  # type: ignore[arg-type]


def test_pii_markers_produce_nothing() -> None:
    assert extract_deterministic("mail [PII:EMAIL] or [PII:PHONE] today").is_empty()


def test_as_metadata_order_and_type() -> None:
    got = extract_deterministic("#t 2026-01-02 https://a.io/x v1.2")
    assert isinstance(got, Extracted)
    assert list(got.as_metadata()) == ["urls", "dates", "versions", "hashtags"]


def test_timing_24k_mixed_text() -> None:
    chunk = (
        "See https://example.com/a/b?x=1 at /var/log/app.log on 2026-03-04 for v1.2.3 "
        "of `do_thing` #tag 192.168.0.1 1/2 and/or plain words here. "
    )
    text = (chunk * 300)[:MAX_SCAN_CHARS]
    extract_deterministic(text)
    best = min(_time_once(text) for _ in range(5))
    print(f"extraction_ms_24k={best * 1000:.2f}")
    assert best < 0.020


def _time_once(text: str) -> float:
    start = time.perf_counter()
    extract_deterministic(text)
    return time.perf_counter() - start
