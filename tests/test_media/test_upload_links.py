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
NONCE = "n" * 22
OTHER_NONCE = "o" * 22


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
    refused("expired", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG, NONCE)


def test_chunks_are_appended_in_order_into_a_private_file(links, tmp_path):
    link = mint(links)
    body = PNG + b"1" * 50
    assert links.accept_chunk(link.token, CONN, 0, len(body), body[:60], NONCE) == 60
    assert links.accept_chunk(link.token, CONN, 1, len(body), body[60:], NONCE) == len(body)
    row = links.find(link.token, CONN)
    path = links.temp_path(row.upload_id)
    assert path.read_bytes() == body and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.parent == tmp_path / "media" / "tmp"
    assert row.state == "receiving" and row.received == len(body)


def test_out_of_order_duplicate_and_overlong_chunks_are_refused(links):
    link = mint(links)
    body = PNG + b"1" * 50
    links.accept_chunk(link.token, CONN, 0, len(body), body[:60], NONCE)
    refused("bad_order", links.accept_chunk, link.token, CONN, 2, len(body), body[60:], NONCE)
    refused("size_changed", links.accept_chunk, link.token, CONN, 1, len(body) + 1, body[60:], NONCE)
    refused("too_much_data", links.accept_chunk, link.token, CONN, 1, len(body), body[60:] + b"x", NONCE)


def test_chunk_size_and_total_are_bounded_by_the_laptop_not_the_gateway(links):
    link = mint(links)
    refused("chunk_too_large", links.accept_chunk, link.token, CONN, 0, 10 ** 8, PNG + b"x" * ul.MAX_CHUNK_BYTES, NONCE)
    refused("too_large", links.accept_chunk, link.token, CONN, 0, 25 * 1024 * 1024 + 1, PNG, NONCE)
    refused("empty", links.accept_chunk, link.token, CONN, 0, 100, b"", NONCE)
    refused("empty", links.accept_chunk, link.token, CONN, 0, 0, PNG, NONCE)


def test_the_first_chunk_must_look_like_the_kind(links):
    image, document = mint(links), mint(links, "document")
    refused("wrong_type", links.accept_chunk, image.token, CONN, 0, len(PDF), PDF, NONCE)
    refused("wrong_type", links.accept_chunk, document.token, CONN, 0, len(PNG), PNG, NONCE)
    refused("wrong_type", links.accept_chunk, image.token, CONN, 0, 20, b"hello world, not an image", NONCE)
    assert links.accept_chunk(document.token, CONN, 0, len(PDF), PDF, NONCE) == len(PDF)
    for head in (b"\xff\xd8\xff\xe0" + b"0" * 20, b"RIFF\x00\x00\x00\x00WEBP" + b"0" * 20):
        other = mint(links, conn=OTHER)
        assert links.accept_chunk(other.token, OTHER, 0, len(head), head, NONCE) == len(head)
        links.begin_finish(other.token, OTHER, NONCE)  # leaves the open-link allowance free for the next loop


def test_a_wrong_connection_cannot_send_chunks(links):
    link = mint(links)
    refused("invalid_link", links.accept_chunk, link.token, OTHER, 0, len(PNG), PNG, NONCE)


def test_restarting_at_chunk_zero_replaces_the_partial_file_up_to_three_times(links, clock):
    link = mint(links)
    for attempt in range(3):
        links.accept_chunk(link.token, CONN, 0, 500, PNG, f"{attempt}" * 22)
        clock.now += 61  # the earlier try went quiet
    refused("too_many_attempts", links.accept_chunk, link.token, CONN, 0, 500, PNG, "9" * 22)
    path = links.temp_path(links.find(link.token, CONN).upload_id)
    assert path.read_bytes() == PNG


def test_daily_limit_counts_started_uploads_per_connection(links, clock):
    for _ in range(ul.MAX_UPLOADS_PER_DAY):
        link = mint(links)
        links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
        links.begin_finish(link.token, CONN, NONCE)
        links.finish_failed(links.find(link.token, CONN).upload_id, {"ok": False, "code": "refused", "message": "no"})
    link = mint(links)
    refused("daily_limit", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG, NONCE)
    other = mint(links, conn=OTHER)
    assert links.accept_chunk(other.token, OTHER, 0, len(PNG), PNG, NONCE) == len(PNG)
    clock.now += 86_401
    fresh = mint(links)
    assert links.accept_chunk(fresh.token, CONN, 0, len(PNG), PNG, NONCE) == len(PNG)


def test_finish_needs_every_byte_and_runs_once(links):
    link = mint(links)
    refused("not_started", links.begin_finish, link.token, CONN, NONCE)
    links.accept_chunk(link.token, CONN, 0, len(PNG) + 5, PNG, NONCE)
    refused("incomplete", links.begin_finish, link.token, CONN, NONCE)
    links.accept_chunk(link.token, CONN, 1, len(PNG) + 5, b"12345", NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    assert plan.action == "run" and plan.row.kind == "image" and plan.row.profile_id == "personal"
    again = links.begin_finish(link.token, CONN, NONCE)
    assert again.action == "working"
    links.finish_done(plan.row.upload_id, {"ok": True, "done": True, "message": "Saved to your memory."})
    replay = links.begin_finish(link.token, CONN, NONCE)
    assert replay.action == "result" and replay.result["message"] == "Saved to your memory."
    refused("used", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG, NONCE)
    assert not links.temp_path(plan.row.upload_id).exists()


def test_a_failed_save_is_final_and_a_retryable_one_keeps_the_bytes(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_retry(plan.row.upload_id)
    assert links.find(link.token, CONN).state == "receiving"
    assert links.temp_path(plan.row.upload_id).exists()
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)  # the page sends the file again
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_failed(plan.row.upload_id, {"ok": False, "code": "refused", "message": "Not an image."})
    assert links.begin_finish(link.token, CONN, NONCE).result["message"] == "Not an image."
    refused("used", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG, NONCE)
    assert not links.temp_path(plan.row.upload_id).exists()


def test_a_started_upload_gets_twenty_minutes_from_its_start(links, clock):
    link = mint(links)
    clock.now += 500
    links.accept_chunk(link.token, CONN, 0, len(PNG) + 5, PNG, NONCE)
    clock.now += 700  # past the ten minutes the link was made for
    assert links.accept_chunk(link.token, CONN, 1, len(PNG) + 5, b"12345", NONCE) == len(PNG) + 5
    clock.now += 1300
    refused("expired", links.begin_finish, link.token, CONN, NONCE)


def test_cleanup_removes_expired_links_their_files_and_strays(links, clock, tmp_path):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, 500, PNG, NONCE)
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


def test_housekeeping_runs_at_first_use_then_at_most_hourly(links, clock):
    stray = links.temp_dir / "upload-old.part"
    stray.write_bytes(b"x")
    os.utime(stray, (clock.now - 7200, clock.now - 7200))
    mint(links)  # first use after start cleans up
    assert not stray.exists()
    again = links.temp_dir / "upload-old2.part"
    again.write_bytes(b"x")
    os.utime(again, (clock.now - 7200, clock.now - 7200))
    mint(links, conn=OTHER)
    assert again.exists()  # not within the hour
    clock.now += 3601
    mint(links, conn=OTHER)
    assert not again.exists()


def test_cleanup_creates_nothing_on_a_computer_that_never_made_a_link(tmp_path):
    assert ul.UploadLinks(tmp_path).cleanup() == 0
    assert not (tmp_path / "media").exists()


# -- the file cannot be swapped mid-upload -------------------------------------

def test_a_second_upload_cannot_restart_a_link_that_is_moving(links, clock):
    link = mint(links)
    body = PNG + b"12345"
    links.accept_chunk(link.token, CONN, 0, len(body), PNG, NONCE)
    refused("in_progress", links.accept_chunk, link.token, CONN, 0, len(body), PNG, OTHER_NONCE)
    clock.now += 30
    refused("in_progress", links.accept_chunk, link.token, CONN, 0, len(body), PNG, OTHER_NONCE)
    assert links.find(link.token, CONN).received == len(PNG)  # the first upload is untouched
    assert links.accept_chunk(link.token, CONN, 1, len(body), b"12345", NONCE) == len(body)


def test_a_chunk_or_finish_with_another_nonce_is_refused(links):
    link = mint(links)
    body = PNG + b"12345"
    links.accept_chunk(link.token, CONN, 0, len(body), PNG, NONCE)
    refused("in_progress", links.accept_chunk, link.token, CONN, 1, len(body), b"12345", OTHER_NONCE)
    refused("in_progress", links.begin_finish, link.token, CONN, OTHER_NONCE)
    assert links.find(link.token, CONN).received == len(PNG)
    links.accept_chunk(link.token, CONN, 1, len(body), b"12345", NONCE)
    refused("in_progress", links.begin_finish, link.token, CONN, OTHER_NONCE)
    assert links.begin_finish(link.token, CONN, NONCE).action == "run"


def test_a_quiet_upload_can_be_taken_over_after_a_minute(links, clock):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, 500, PNG, NONCE)
    clock.now += 61
    assert links.accept_chunk(link.token, CONN, 0, 500, PNG, OTHER_NONCE) == len(PNG)
    refused("in_progress", links.accept_chunk, link.token, CONN, 1, 500, b"x", NONCE)  # the old one lost it


def test_the_same_nonce_cannot_restart_its_own_upload(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, 500, PNG, NONCE)
    refused("bad_order", links.accept_chunk, link.token, CONN, 0, 500, PNG, NONCE)


@pytest.mark.parametrize("bad", ["", "short", "n" * 21, "n" * 23, "n" * 21 + "!", "n" * 21 + " "])
def test_a_malformed_nonce_is_refused_everywhere(links, bad):
    link = mint(links)
    refused("invalid_request", links.accept_chunk, link.token, CONN, 0, 500, PNG, bad)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    refused("invalid_request", links.begin_finish, link.token, CONN, bad)


def test_a_retry_after_a_warming_picture_model_can_start_a_new_upload(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_retry(plan.row.upload_id)
    assert links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, OTHER_NONCE) == len(PNG)


# -- a save that never ended, and links that must die with their consent -------

def test_a_save_that_never_ended_is_failed_after_the_finisher_timeout(links, clock):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    clock.now += 200
    assert links.begin_finish(link.token, CONN, NONCE).action == "working"
    clock.now += 101  # past 300 s since the save began
    ended = links.begin_finish(link.token, CONN, NONCE)
    assert ended.action == "result" and ended.result["ok"] is False
    assert "interrupted" in ended.result["message"] and "\n" not in ended.result["message"]
    assert links.find(link.token, CONN).state == "failed"
    assert not links.temp_path(plan.row.upload_id).exists()


def test_cleanup_fails_finishing_rows_older_than_the_finisher_timeout(links, clock):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    links.begin_finish(link.token, CONN, NONCE)
    clock.now += 299
    links.cleanup()
    assert links.find(link.token, CONN).state == "finishing"
    clock.now += 2
    links.cleanup()
    assert links.find(link.token, CONN).state == "failed"


def test_failing_every_open_link_of_a_connection_leaves_other_connections_alone(links):
    a, b, c = mint(links), mint(links), mint(links, conn=OTHER)
    links.accept_chunk(b.token, CONN, 0, len(PNG), PNG, NONCE)
    assert links.fail_open_links(CONN) == 2
    for gone in (a, b):
        refused("used", links.accept_chunk, gone.token, CONN, 0, len(PNG), PNG, NONCE)
        result = links.begin_finish(gone.token, CONN, NONCE).result
        assert result["ok"] is False and "no longer works" in result["message"]
    assert not links.temp_path(links.find(b.token, CONN).upload_id).exists()
    assert links.accept_chunk(c.token, OTHER, 0, len(PNG), PNG, NONCE) == len(PNG)
    assert links.fail_open_links(CONN) == 0


# -- a link belongs to the app that asked for it (audit F6) -------------------------

def test_a_link_remembers_the_authorization_that_issued_it(links):
    minted = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-a")
    assert links.find(minted.token, CONN).authorization_id == "app-a"
    assert mint(links).token and links.find(mint(links).token, CONN).authorization_id == ""


def test_failing_one_apps_links_leaves_the_other_apps_links_open(links):
    a = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-a")
    a2 = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-a")
    b = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-b")
    links.accept_chunk(a2.token, CONN, 0, len(PNG), PNG, NONCE)
    assert links.fail_open_links(CONN, authorization_id="app-a") == 2
    for gone in (a, a2):
        refused("used", links.accept_chunk, gone.token, CONN, 0, len(PNG), PNG, NONCE)
    assert not links.temp_path(links.find(a2.token, CONN).upload_id).exists()
    assert links.accept_chunk(b.token, CONN, 0, len(PNG), PNG, NONCE) == len(PNG)


def test_failing_one_apps_links_also_ends_links_made_before_apps_were_recorded(links):
    legacy = mint(links)
    assert links.fail_open_links(CONN, authorization_id="app-a") == 1
    refused("used", links.accept_chunk, legacy.token, CONN, 0, len(PNG), PNG, NONCE)


def test_links_of_apps_that_are_no_longer_listed_are_failed(links):
    a = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-a")
    b = links.mint(CONN, "key1", "personal", "image", "", authorization_id="app-b")
    other = links.mint(OTHER, "key1", "personal", "image", "", authorization_id="app-c")
    assert links.fail_unlisted_authorizations(CONN, {"app-b"}) == 1
    refused("used", links.accept_chunk, a.token, CONN, 0, len(PNG), PNG, NONCE)
    assert links.accept_chunk(b.token, CONN, 0, len(PNG), PNG, NONCE) == len(PNG)
    assert links.info(other.token, OTHER).kind == "image"


def test_an_uploads_database_from_before_this_column_is_upgraded(tmp_path, clock):
    path = tmp_path / "media"
    path.mkdir()
    old = sqlite3.connect(path / "uploads.db")
    old.execute(OLD_DDL)
    old.commit()
    old.close()
    store = ul.UploadLinks(tmp_path, clock=clock)
    minted = store.mint(CONN, "key1", "personal", "image", "", authorization_id="app-a")
    assert store.find(minted.token, CONN).authorization_id == "app-a"


OLD_DDL = """CREATE TABLE upload_links (
  upload_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE,
  connection_id TEXT NOT NULL, key_id TEXT NOT NULL, profile_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('image','document')), note TEXT NOT NULL,
  max_bytes INTEGER NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('open','receiving','finishing','done','failed')),
  total INTEGER NOT NULL DEFAULT 0, received INTEGER NOT NULL DEFAULT 0,
  next_index INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL, started_at INTEGER,
  result_json TEXT NOT NULL DEFAULT '', nonce TEXT, touched_at INTEGER)"""


# -- a warming save answers honestly instead of "another upload is using this link" -

WARMING = {"ok": False, "code": "warming", "message": "The picture tools are starting."}


def test_finish_after_a_warming_save_returns_the_warming_result_not_in_progress(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    assert plan.action == "run"
    links.finish_retry(plan.row.upload_id, WARMING)
    again = links.begin_finish(link.token, CONN, NONCE)  # the gateway asks again with the same nonce
    assert again.action == "result" and again.result["code"] == "warming"
    assert again.result["message"] == WARMING["message"]
    assert links.temp_path(plan.row.upload_id).exists()


def test_a_default_warming_message_is_plain_when_none_is_given(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_retry(plan.row.upload_id)
    result = links.begin_finish(link.token, CONN, NONCE).result
    assert result["code"] == "warming" and "again in a minute" in result["message"]


def test_a_new_upload_after_a_warming_save_restarts_and_finishes(links):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_retry(plan.row.upload_id, WARMING)
    assert links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, OTHER_NONCE) == len(PNG)
    assert links.get(plan.row.upload_id).result_json == ""
    assert links.begin_finish(link.token, CONN, OTHER_NONCE).action == "run"
    links.finish_done(plan.row.upload_id, {"ok": True, "done": True, "message": "Saved to your memory."})
    assert links.begin_finish(link.token, CONN, OTHER_NONCE).result["done"] is True


def test_an_expired_link_still_expires_after_a_warming_save(links, clock):
    link = mint(links)
    links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
    plan = links.begin_finish(link.token, CONN, NONCE)
    links.finish_retry(plan.row.upload_id, WARMING)
    clock.now += ul.STARTED_TTL_S + 5
    refused("expired", links.begin_finish, link.token, CONN, NONCE)


# -- the same pictures the saver takes (audit: GIF) -------------------------------

@pytest.mark.parametrize("head", [b"GIF87a" + b"0" * 10, b"GIF89a" + b"0" * 10])
def test_gif_is_a_picture_the_link_accepts(links, head):
    assert ul.looks_like("image", head)
    assert not ul.looks_like("document", head)
    link = mint(links)
    gif = head + b"0" * 50
    assert links.accept_chunk(link.token, CONN, 0, len(gif), gif, NONCE) == len(gif)


def test_a_gif_header_that_is_not_one_is_still_refused():
    assert not ul.looks_like("image", b"GIF99a" + b"0" * 10)
    assert not ul.looks_like("image", b"GIF")


# -- a full disk is said plainly (audit MU-M2) ---------------------------------

def _disk_full(monkeypatch, *, after=0):
    import errno

    real = ul.UploadLinks._write
    calls = {"n": 0}

    def write(path, data, flags):
        calls["n"] += 1
        if calls["n"] > after:
            raise OSError(errno.ENOSPC, "No space left on device")
        real(path, data, flags)

    monkeypatch.setattr(ul.UploadLinks, "_write", staticmethod(write))


def test_a_full_disk_on_the_first_chunk_is_a_plain_refusal_and_costs_no_attempt(links, monkeypatch):
    from superlocalmemory.media import files

    link = mint(links)
    _disk_full(monkeypatch)

    err = refused("disk_full", links.accept_chunk, link.token, CONN, 0, len(PNG), PNG, NONCE)

    assert err.message == files.DISK_FULL
    row = links.find(link.token, CONN)
    assert row.state == "open" and row.attempts == 0


def test_a_full_disk_on_a_later_chunk_is_a_plain_refusal_and_keeps_what_arrived(links, monkeypatch):
    body = PNG + b"x" * 50
    link = mint(links)
    _disk_full(monkeypatch, after=1)
    assert links.accept_chunk(link.token, CONN, 0, len(body), body[:60], NONCE) == 60

    refused("disk_full", links.accept_chunk, link.token, CONN, 1, len(body), body[60:], NONCE)

    row = links.find(link.token, CONN)
    assert row.received == 60
    assert links.temp_path(row.upload_id).stat().st_size == 60   # no half chunk left behind


def test_other_write_errors_are_not_called_disk_full(links, monkeypatch):
    import errno

    def write(path, data, flags):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(ul.UploadLinks, "_write", staticmethod(write))
    link = mint(links)

    with pytest.raises(OSError):
        links.accept_chunk(link.token, CONN, 0, len(PNG), PNG, NONCE)
