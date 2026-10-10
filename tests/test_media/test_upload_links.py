"""The one-time upload link store: minting, ordered chunks, single use, limits and cleanup."""

from __future__ import annotations

import os
import stat
import sqlite3

import pytest

from superlocalmemory.media import upload_links as ul

CONN = "a" * 32
OTHER = "b" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100
PDF = b"%PDF-1.7\n" + b"0" * 100


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def links(tmp_path, clock):
    return ul.UploadLinks(tmp_path, clock=clock)


def mint(links, kind="image", conn=CONN, note=""):
    return links.mint(conn, "key1", "personal", kind, note)


def refused(code, fn, *args, **kwargs):
    with pytest.raises(ul.UploadError) as caught:
        fn(*args, **kwargs)
    assert caught.value.code == code
    assert caught.value.message and "\n" not in caught.value.message
    return caught.value


def test_mint_returns_an_unguessable_token_and_stores_only_its_hash(links, tmp_path):
    first, second = mint(links), mint(links)
    assert len(first.token) == 43 and first.token != second.token
    assert first.max_bytes == 25 * 1024 * 1024 and first.expires_at == 1_000_600
    assert mint(links, "document").max_bytes == 100 * 1024 * 1024
    raw = (tmp_path / "media" / "uploads.db").read_bytes()
    assert first.token.encode() not in raw
    assert stat.S_IMODE((tmp_path / "media" / "uploads.db").stat().st_mode) == 0o600


def test_mint_refuses_a_bad_kind_and_a_long_note(links):
    refused("invalid_kind", links.mint, CONN, "k", "personal", "video", "")
    refused("note_too_long", links.mint, CONN, "k", "personal", "image", "x" * 2001)


def test_at_most_three_open_links_per_connection(links, clock):
    for _ in range(3):
        mint(links)
    refused("too_many_open", mint, links)
    mint(links, conn=OTHER)  # another connection has its own allowance
    clock.now += 601  # the old ones expired
    mint(links)


def test_info_hides_why_an_unknown_or_foreign_token_failed(links):
    link = mint(links)
    info = links.info(link.token, CONN)
    assert (info.kind, info.max_bytes, info.expires_at) == ("image", 25 * 1024 * 1024, link.expires_at)
    first = refused("invalid_link", links.info, "z" * 43, CONN)
    second = refused("invalid_link", links.info, link.token, OTHER)
    assert first.message == second.message
    refused("invalid_link", links.info, "short", CONN)


def test_an_expired_link_is_refused(links, clock):
    link = mint(links)
    clock.now += 601
    refused("expired", links.info, link.token, CONN)
    refused("expired", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG)


def test_chunks_are_appended_in_order_into_a_private_file(links, tmp_path):
    link = mint(links)
    body = PNG + b"1" * 50
    assert links.accept_chunk(link.token, CONN, 0, len(body), body[:60]) == 60
    assert links.accept_chunk(link.token, CONN, 1, len(body), body[60:]) == len(body)
    row = links.find(link.token, CONN)
    path = links.temp_path(row.upload_id)
    assert path.read_bytes() == body and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.parent == tmp_path / "media" / "tmp"
    assert row.state == "receiving" and row.received == len(body)


def test_out_of_order_duplicate_and_overlong_chunks_are_refused(links):
    link = mint(links)
    body = PNG + b"1" * 50
    links.accept_chunk(link.token, CONN, 0, len(body), body[:60])
    refused("bad_order", links.accept_chunk, link.token, CONN, 2, len(body), body[60:])
    refused("size_changed", links.accept_chunk, link.token, CONN, 1, len(body) + 1, body[60:])
    refused("too_much_data", links.accept_chunk, link.token, CONN, 1, len(body), body[60:] + b"x")


def test_chunk_size_and_total_are_bounded_by_the_laptop_not_the_gateway(links):
    link = mint(links)
    refused("chunk_too_large", links.accept_chunk, link.token, CONN, 0, 10 ** 8, PNG + b"x" * ul.MAX_CHUNK_BYTES)
    refused("too_large", links.accept_chunk, link.token, CONN, 0, 25 * 1024 * 1024 + 1, PNG)
    refused("empty", links.accept_chunk, link.token, CONN, 0, 100, b"")
    refused("empty", links.accept_chunk, link.token, CONN, 0, 0, PNG)


def test_the_first_chunk_must_look_like_the_kind(links):
    image, document = mint(links), mint(links, "document")
    refused("wrong_type", links.accept_chunk, image.token, CONN, 0, len(PDF), PDF)
    refused("wrong_type", links.accept_chunk, document.token, CONN, 0, len(PNG), PNG)
    refused("wrong_type", links.accept_chunk, image.token, CONN, 0, 20, b"hello world, not an image")
    assert links.accept_chunk(document.token, CONN, 0, len(PDF), PDF) == len(PDF)
    for head in (b"\xff\xd8\xff\xe0" + b"0" * 20, b"RIFF\x00\x00\x00\x00WEBP" + b"0" * 20):
        other = mint(links, conn=OTHER)
        assert links.accept_chunk(other.token, OTHER, 0, len(head), head) == len(head)
        links.begin_finish(other.token, OTHER)  # leaves the open-link allowance free for the next loop


def test_a_wrong_connection_cannot_send_chunks(links):
    link = mint(links)
    refused("invalid_link", links.accept_chunk, link.token, OTHER, 0, len(PNG), PNG)


def test_restarting_at_chunk_zero_replaces_the_partial_file_up_to_three_times(links):
    link = mint(links)
    for _ in range(3):
        links.accept_chunk(link.token, CONN, 0, 500, PNG)
    refused("too_many_attempts", links.accept_chunk, link.token, CONN, 0, 500, PNG)
    path = links.temp_path(links.find(link.token, CONN).upload_id)
    assert path.read_bytes() == PNG


def test_daily_limit_counts_started_uploads_per_connection(links, clock):
    for _ in range(ul.MAX_UPLOADS_PER_DAY):
        link = mint(links)
        links.accept_chunk(link.token, CONN, 0, len(PNG), PNG)
        links.begin_finish(link.token, CONN)
        links.finish_failed(links.find(link.token, CONN).upload_id, {"ok": False, "code": "refused", "message": "no"})
    link = mint(links)
    refused("daily_limit", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG)
    other = mint(links, conn=OTHER)
    assert links.accept_chunk(other.token, OTHER, 0, len(PNG), PNG) == len(PNG)
    clock.now += 86_401
    fresh = mint(links)
    assert links.accept_chunk(fresh.token, CONN, 0, len(PNG), PNG) == len(PNG)


def test_finish_needs_every_byte_and_runs_once(links):
    link = mint(links)
    refused("not_started", links.begin_finish, link.token, CONN)
    links.accept_chunk(link.token, CONN, 0, len(PNG) + 5, PNG)
    refused("incomplete", links.begin_finish, link.token, CONN)
    links.accept_chunk(link.token, CONN, 1, len(PNG) + 5, b"12345")
    plan = links.begin_finish(link.token, CONN)
    assert plan.action == "run" and plan.row.kind == "image" and plan.row.profile_id == "personal"
    again = links.begin_finish(link.token, CONN)
    assert again.action == "working"
    links.finish_done(plan.row.upload_id, {"ok": True, "done": True, "message": "Saved to your memory."})
    replay = links.begin_finish(link.token, CONN)
    assert replay.action == "result" and replay.result["message"] == "Saved to your memory."
    refused("used", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG)
    assert not links.temp_path(plan.row.upload_id).exists()


def test_a_failed_save_is_final_and_a_retryable_one_keeps_the_bytes(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG)
    plan = links.begin_finish(link.token, CONN)
    links.finish_retry(plan.row.upload_id)
    assert links.find(link.token, CONN).state == "receiving"
    assert links.temp_path(plan.row.upload_id).exists()
    plan = links.begin_finish(link.token, CONN)
    links.finish_failed(plan.row.upload_id, {"ok": False, "code": "refused", "message": "Not an image."})
    assert links.begin_finish(link.token, CONN).result["message"] == "Not an image."
    refused("used", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG)
    assert not links.temp_path(plan.row.upload_id).exists()


def test_a_started_upload_gets_twenty_minutes_from_its_start(links, clock):
    link = mint(links)
    clock.now += 500
    links.accept_chunk(link.token, CONN, 0, len(PNG) + 5, PNG)
    clock.now += 700  # past the ten minutes the link was made for
    assert links.accept_chunk(link.token, CONN, 1, len(PNG) + 5, b"12345") == len(PNG) + 5
    clock.now += 1300
    refused("expired", links.begin_finish, link.token, CONN)


def test_cleanup_removes_expired_links_their_files_and_strays(links, clock, tmp_path):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, 500, PNG)
    upload_id = links.find(link.token, CONN).upload_id
    stray = links.temp_dir / "upload-stray.part"
    stray.write_bytes(b"x")
    os.utime(stray, (clock.now - 7200, clock.now - 7200))
    young = links.temp_dir / "upload-young.part"
    young.write_bytes(b"x")
    os.utime(young, (clock.now, clock.now))
    assert links.cleanup() == 1 and links.temp_path(upload_id).exists() and not stray.exists()
    clock.now += 1300
    assert links.cleanup() == 1
    assert not links.temp_path(upload_id).exists() and young.exists()
    clock.now += 90_000
    links.cleanup()
    with sqlite3.connect(tmp_path / "media" / "uploads.db") as db:
        assert db.execute("SELECT COUNT(*) FROM upload_links").fetchone()[0] == 0


def test_the_note_travels_with_the_link(links):
    link = mint(links, note="the whiteboard from Monday")
    assert links.find(link.token, CONN).note == "the whiteboard from Monday"
