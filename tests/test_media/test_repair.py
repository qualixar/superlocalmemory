"""A picture saved without its vector says so, and `slm media repair` fixes it (audit F9)."""

from __future__ import annotations

import pytest

from superlocalmemory.media import ingest
from superlocalmemory.media.repair import repair
from tests.helpers.env_capabilities import NO_VECTOR_SEARCH_REASON, vector_search_available
from tests.test_media.test_ingest import env, png, root, save, store  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON)


@pytest.fixture()
def vectorless(env, monkeypatch):
    """A picture saved while the vector write fails."""
    real = env.store.put_vector

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(env.store, "put_vector", boom)
    receipt = save(env, content="the cafe")
    monkeypatch.setattr(env.store, "put_vector", real)
    return receipt


def test_a_failed_vector_write_is_still_stored_but_the_receipt_says_what_is_missing(vectorless, env) -> None:
    assert vectorless.status == "stored" and vectorless.media_id
    assert "found by what it shows" in vectorless.reason
    assert "slm media repair" in vectorless.reason
    assert env.store.vector_count("p1") == 0
    assert env.store.get_item(vectorless.media_id) is not None


def test_a_picture_with_its_vector_has_no_warning_reason(env) -> None:
    receipt = save(env)
    assert receipt.status == "stored" and receipt.reason == ""
    assert env.store.vector_count("p1") == 1


def test_the_index_mismatch_refusal_names_the_real_command(env, monkeypatch) -> None:
    assert save(env).status == "stored"
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")
    receipt = save(env, png("b"))
    assert receipt.status == "refused"
    assert "slm media repair" in receipt.reason and "dashboard" not in receipt.reason


def test_repair_embeds_the_pictures_that_have_no_vector(vectorless, env) -> None:
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert (report.missing, report.repaired, report.failed) == (1, 1, 0)
    assert report.rebuilt_index is False
    assert env.store.vector_count("p1") == 1


def test_repair_twice_does_nothing_the_second_time(vectorless, env) -> None:
    repair("p1", store=env.store, client=env.client, data_root=env.root)
    again = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert (again.missing, again.repaired) == (0, 0) and env.store.vector_count("p1") == 1


def test_a_dry_run_reports_and_changes_nothing(vectorless, env) -> None:
    report = repair("p1", dry_run=True, store=env.store, client=env.client, data_root=env.root)
    assert report.dry_run is True and report.missing == 1 and report.repaired == 0
    assert env.store.vector_count("p1") == 0


def test_a_picture_whose_file_is_gone_is_counted_not_guessed(vectorless, env) -> None:
    row = env.store.get_item(vectorless.media_id)
    (env.root / "media" / row["original_relpath"]).unlink()
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert (report.missing, report.repaired, report.skipped_no_file) == (1, 0, 1)


def test_a_failing_embed_is_counted_and_the_run_goes_on(vectorless, env, monkeypatch) -> None:
    def boom(paths, *, wait_cold=True):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(env.client, "embed_images", boom)
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert (report.repaired, report.failed) == (0, 1)


def test_a_cold_model_stops_the_run_with_a_try_again_reason(vectorless, env, monkeypatch) -> None:
    monkeypatch.setattr(env.client, "warm", False)
    monkeypatch.setattr(ingest, "_cold_wait_s", lambda remote=False: 0.0)
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert report.repaired == 0 and "starting" in report.reason


def test_repair_builds_a_new_index_when_the_image_model_changed(env, monkeypatch) -> None:
    assert save(env).status == "stored" and save(env, png("b")).status == "stored"
    before = env.store.active_space()["space_id"]
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")
    monkeypatch.setattr(env.client, "model_id", "fake:paired")
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert report.rebuilt_index is True and report.repaired == 2
    assert env.store.active_space()["space_id"] != before
    assert env.store.active_signature()["mode"] == "paired"
    assert save(env, png("c")).status == "stored"  # saving works again


def test_repair_only_updates_the_record_when_the_vectors_are_still_right(env, monkeypatch) -> None:
    assert save(env).status == "stored"
    before = env.store.active_space()["space_id"]
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")  # same image model: same vectors
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert report.rebuilt_index is True and report.repaired == 0 and report.missing == 0
    assert env.store.active_space()["space_id"] == before
    assert env.store.vector_count("p1") == 1 and env.store.active_signature()["mode"] == "paired"
    assert save(env, png("c")).status == "stored"


def test_a_dry_run_does_not_rebuild_the_index(env, monkeypatch) -> None:
    assert save(env).status == "stored"
    before = env.store.active_space()["space_id"]
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")
    report = repair("p1", dry_run=True, store=env.store, client=env.client, data_root=env.root)
    assert report.rebuilt_index is False and report.index_needs_rebuild is True
    assert env.store.active_space()["space_id"] == before and env.store.active_signature()["mode"] == "separate"


def test_a_failed_first_embed_leaves_the_old_index_in_place(env, monkeypatch) -> None:
    assert save(env).status == "stored"
    before = env.store.active_space()["space_id"]
    monkeypatch.setenv("SLM_MEDIA_SPACE_MODE", "paired")
    monkeypatch.setattr(env.client, "model_id", "fake:paired")

    def boom(paths, *, wait_cold=True):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(env.client, "embed_images", boom)
    report = repair("p1", store=env.store, client=env.client, data_root=env.root)
    assert report.rebuilt_index is False and report.failed == 1
    assert env.store.active_space()["space_id"] == before
