"""submit_memory: prepare each part by where it came from, then hand one request to the writer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from superlocalmemory.memory_core import ContentOrigin
from superlocalmemory.memory_core.submit import SaveReceipt, SaveRequest, submit_memory

KEY = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


class Runtime:
    def __init__(self, payload=None, exc=None):
        self.calls = []
        self.payload = payload or {"status": "queryable", "operation_id": "op1",
                                   "fact_ids": ["f1", "f2"], "memory_id": "m1"}
        self.exc = exc

    def remember(self, admission, actor, *, deadline_ms, accept_after_ms):
        self.calls.append((admission, actor, deadline_ms, accept_after_ms))
        if self.exc:
            raise self.exc
        return SimpleNamespace(payload=self.payload)


def req(**kw):
    base = dict(segments=(("my words", ContentOrigin.USER_TEXT),), profile_id="p1",
                source_type="media", trusted_actor_id="actor-1")
    base.update(kw)
    return SaveRequest(**base)


def cfg(pii=False):
    return SimpleNamespace(pii_redaction=pii)


def test_segments_are_prepared_by_origin_and_joined():
    rt = Runtime()
    out = submit_memory(rt, req(segments=((f"note {KEY}", ContentOrigin.USER_TEXT),
                                          (f"\n\n[Text in image]\nfound {KEY}", ContentOrigin.DERIVED_TEXT))),
                        config=cfg())
    admission, actor, deadline, accept = rt.calls[0]
    assert KEY in admission.content.split("[Text in image]")[0]
    assert KEY not in admission.content.split("[Text in image]")[1]
    assert out.secret_count == 1 and out.pii_count == 0
    assert admission.profile_id == "p1" and admission.source_type == "media"
    assert actor.principal_id == "actor-1" and actor.allowed_profiles == frozenset({"p1"})
    assert actor.allowed_scopes == frozenset({"personal"})
    assert isinstance(out, SaveReceipt) and out.status == "queryable"
    assert out.memory_id == "m1" and out.fact_ids == ("f1", "f2") and out.operation_id == "op1"
    assert 1 <= deadline <= 2000 and accept is not None


def test_pii_redaction_applies_to_every_segment_and_to_metadata_and_key():
    rt = Runtime()
    out = submit_memory(
        rt, req(segments=(("mail bob@example.com", ContentOrigin.USER_TEXT),
                          ("\nalso amy@example.org", ContentOrigin.DERIVED_TEXT)),
                tags="t", metadata={"who": "bob@example.com"}, idempotency_key="bob@example.com"),
        config=cfg(pii=True))
    admission = rt.calls[0][0]
    assert "@" not in admission.content and out.pii_count == 3
    assert admission.metadata["who"] == "[PII:EMAIL]" and admission.metadata["tags"] == "t"
    assert admission.idempotency_key.startswith("redacted:")


def test_reserved_keys_from_callers_are_stripped_but_trusted_ones_survive():
    rt = Runtime()
    submit_memory(rt, req(metadata={"_slm_source": "forged", "ok": 1},
                          trusted_metadata={"_slm_source": {"type": "media", "media_id": "x"}}), config=cfg())
    meta = rt.calls[0][0].metadata
    assert meta["_slm_source"] == {"type": "media", "media_id": "x"} and meta["ok"] == 1


def test_default_key_is_generated_and_accepted_status_has_no_memory_id():
    rt = Runtime(payload={"status": "accepted", "admission_id": "a"})
    out = submit_memory(rt, req(), config=cfg())
    assert rt.calls[0][0].idempotency_key and out.status == "accepted" and out.memory_id is None
    assert out.fact_ids == ()


def test_memory_id_is_found_from_the_first_fact_when_the_payload_has_none():
    class Db:
        def execute(self, sql, params=()):
            assert params == ("f1",)
            return [{"memory_id": "mem-from-fact"}]

    rt = Runtime(payload={"status": "queryable", "operation_id": "o", "fact_ids": ["f1"]})
    rt._db = Db()
    assert submit_memory(rt, req(), config=cfg()).memory_id == "mem-from-fact"


def test_writer_errors_propagate_and_empty_content_is_refused():
    with pytest.raises(RuntimeError):
        submit_memory(Runtime(exc=RuntimeError("boom")), req(), config=cfg())
    with pytest.raises(ValueError):
        submit_memory(Runtime(), req(segments=()), config=cfg())


def test_a_runtime_without_a_db_gives_no_memory_id_and_no_error():
    rt = Runtime(payload={"status": "queryable", "operation_id": "o", "fact_ids": ["f1"]})
    assert not hasattr(rt, "_db")
    assert submit_memory(rt, req(), config=cfg()).memory_id is None
