"""The picture channel under each space mode."""

from __future__ import annotations

import pytest

from superlocalmemory.retrieval.media_channel import MediaChannel
from superlocalmemory.runtimes.space_plan import SpacePlan

from ._media_support import FakeClient, MemoryDb, make_store, mid, unit

NOMIC = "nomic-ai/nomic-embed-text-v1.5"


def paired(text=NOMIC):
    return SpacePlan("paired", "nomic-ai/nomic-embed-vision-v1.5", "", 4, text, True, "test")


def separate():
    return SpacePlan("separate", "fake-model", "r1", 4, "", False, "default")


@pytest.fixture()
def world(tmp_path):
    db = MemoryDb()
    db.add("m1", "f1")
    store = make_store(tmp_path, [(mid(1), "image", "m1", unit(0))])
    yield db, store
    store.close()


def build(store, db, plan, *, text=None, client=None):
    calls = {"embedder": 0}

    def embedder():
        calls["embedder"] += 1
        return client

    ch = MediaChannel(lambda: store, embedder, db, text_query_vector=text, plan_factory=lambda: plan)
    return ch, calls


def sign(store, plan):
    store.ensure_active_space(plan.image_model, plan.image_revision, plan.dim, signature=plan.signature())


def test_paired_query_comes_from_the_text_function_and_never_the_worker(world):
    db, store = world
    sign(store, paired())
    asked = []
    client = FakeClient(unit(1))
    ch, calls = build(store, db, paired(), text=lambda q: asked.append(q) or unit(0), client=client)
    vector, state = ch.prepare("a red door", "default")
    assert (vector, state) == (unit(0), None) and asked == ["a red door"]
    assert client.calls == []  # the worker was never asked


def test_paired_cold_text_embedder_is_warming_not_a_worker_call(world):
    db, store = world
    sign(store, paired())
    client = FakeClient(unit(1))
    ch, _ = build(store, db, paired(), text=lambda q: None, client=client)
    assert ch.prepare("q", "default") == (None, "warming")
    assert client.calls == []


def test_paired_without_a_text_function_is_unavailable(world):
    db, store = world
    sign(store, paired())
    client = FakeClient(unit(1))
    ch, _ = build(store, db, paired(), text=None, client=client)
    assert ch.prepare("q", "default") == (None, "warming") and client.calls == []


def test_separate_uses_the_worker_as_before(world):
    db, store = world
    sign(store, separate())
    client = FakeClient(unit(0))
    ch, _ = build(store, db, separate(), text=lambda q: pytest.fail("text path used"), client=client)
    assert ch.prepare("q", "default") == (unit(0), None) and client.calls == [("q", 0.3)]


def test_legacy_space_without_a_signature_fits_separate_only(world):
    db, store = world  # make_store records no signature
    ch, _ = build(store, db, separate(), client=FakeClient(unit(0)))
    assert ch.is_active("default")
    ch2, _ = build(store, db, paired(), text=lambda q: unit(0), client=FakeClient(unit(0)))
    assert not ch2.is_active("default")


def test_incompatible_stored_signature_makes_the_channel_inactive(world):
    db, store = world
    sign(store, paired("some/other-model"))
    ch, _ = build(store, db, paired(), text=lambda q: unit(0), client=FakeClient(unit(0)))
    assert ch.is_active("default") is False
    assert ch.prepare("q", "default") == (None, None)


def test_no_plan_keeps_todays_behaviour(world):
    db, store = world
    ch = MediaChannel(lambda: store, lambda: FakeClient(unit(0)), db)
    assert ch.is_active("default") and ch.prepare("q", "default") == (unit(0), None)


def test_a_failing_plan_factory_turns_the_channel_off(world):
    db, store = world

    def boom():
        raise ValueError("not available in this build")

    ch = MediaChannel(lambda: store, lambda: FakeClient(unit(0)), db, plan_factory=boom)
    assert ch.is_active("default") is False


def test_query_cache_is_kept_apart_per_mode(world):
    db, store = world
    plans = [paired()]
    client = FakeClient(unit(1))
    ch = MediaChannel(lambda: store, lambda: client, db, text_query_vector=lambda q: unit(0),
                      plan_factory=lambda: plans[0])
    assert ch.query_vector("q") == unit(0)
    plans[0] = separate()
    assert ch.query_vector("q") == unit(1)
    plans[0] = paired()
    assert ch.query_vector("q") == unit(0)
    assert len(client.calls) == 1


def test_for_engine_passes_the_text_function(tmp_path):
    from superlocalmemory.retrieval import media_channel

    ch = media_channel.for_engine(MemoryDb(), text_query_vector=lambda q: unit(0))
    assert ch._text_query_vector("x") == unit(0)
