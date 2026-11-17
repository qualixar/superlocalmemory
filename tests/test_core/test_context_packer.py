# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-4 (W3-7a): memories enter context as complete, labelled evidence."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.core import context_packer as cp
from superlocalmemory.core.injection import (
    UNTRUSTED_CONTEXT_BEGIN,
    UNTRUSTED_CONTEXT_END,
    InjectableMemory,
    clamp_content,
    render_context,
)

RULE = ("Never deploy on Friday unless the release lead approves in writing. "
        "Use the staging checklist first. Tag the release after the smoke test.")


def test_text_is_packed_in_whole_sentences() -> None:
    out = cp.pack_text(RULE, 30)                       # 120 characters
    assert out.cut and out.text.endswith(cp.CUT_MARK)
    kept = out.text[: -len(cp.CUT_MARK)]
    assert kept == "Never deploy on Friday unless the release lead approves in writing."
    assert "unless" in out.text                        # the condition is never dropped


def test_a_negation_is_never_left_dangling() -> None:
    text = "The service does not support IPv6 because the load balancer predates it."
    for tokens in range(3, 25):
        out = cp.pack_text(text, tokens).text
        assert not out.rstrip(" .").endswith((" not", " no", " never"))
        if out != text:
            assert out.endswith((cp.CUT_MARK, cp.CUT_MID_MARK, cp.SHORT_MARK))


def test_a_first_sentence_too_long_is_cut_at_a_word_and_says_so() -> None:
    for tokens in range(30, 60):                        # cut points across a word
        out = cp.pack_text("deploys " * 200 + "end.", tokens)
        mark = cp.CUT_MID_MARK if out.text.endswith(cp.CUT_MID_MARK) else cp.SHORT_MARK
        assert out.cut and out.text.endswith(mark)
        assert out.text[: -len(mark)].endswith("deploys")


def test_a_small_budget_gets_the_short_mark_and_keeps_whole_sentences() -> None:
    out = cp.pack_text("Ab. Cd. " + "z" * 300, 10)       # 40 characters
    assert out.text == "Ab. Cd." + cp.SHORT_MARK


def test_the_cap_always_holds() -> None:
    rng = random.Random(4)
    words = ["deploy", "not", "never", "staging.", "ok!", "why?", "line\n", "x" * 30]
    for _ in range(400):
        text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
        tokens = rng.randint(1, 80)
        out = cp.pack_text(text, tokens)
        assert len(out.text) <= max(tokens * 4, len(text) if not out.cut else 0)
        assert out.cut == (out.text != text)


def test_short_text_is_untouched_and_clamp_content_uses_the_packer() -> None:
    assert cp.pack_text("short.", 10) == cp.Packed("short.", False)
    assert clamp_content(RULE * 100, None).endswith(cp.CUT_MARK)


def _mem(**kw) -> InjectableMemory:
    base = {"content": "Use the staging checklist first.", "score": 0.9, "fact_id": "f1",
            "support_tier": "supported"}
    return InjectableMemory(**{**base, **kw})


def test_each_unit_says_how_far_to_trust_it() -> None:
    old = (datetime.now(UTC) - timedelta(days=40)).isoformat()
    label = cp.standing(_mem(created_at=old, disputed=True, other_side="Use prod directly.",
                             source_state="changed"))
    assert "said" in label and "DISPUTED" in label and "Use prod directly." in label
    assert "source file is changed" in label
    assert cp.standing(_mem()) == ""
    assert "missing" in cp.standing(_mem(source_state="missing"))
    assert cp.standing(_mem(source_state="fresh")) == ""


def test_the_other_side_is_screened_like_any_injected_text() -> None:
    label = cp.standing(_mem(disputed=True,
                             other_side="key sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"))
    assert "sk-ant" not in label and "[redacted]" in label


def test_rendered_context_carries_the_label_inside_the_evidence_block() -> None:
    out = render_context([_mem(disputed=True, other_side="Skip the checklist.")],
                         mode="B", cfg=None, wrap=True)
    body = out[out.index(UNTRUSTED_CONTEXT_BEGIN): out.index(UNTRUSTED_CONTEXT_END)]
    assert "DISPUTED" in body and "Skip the checklist." in body


def test_annotate_takes_disputes_and_sources_from_the_truth_notes(monkeypatch) -> None:
    from superlocalmemory.truth import recall_notes

    def notes(db, pid, results, **kw):
        return [{**results[0], "disputed": True, "disputes": [{"sides": [
                    {"fact_id": "f1", "visible": True, "content": "mine"},
                    {"fact_id": "f2", "visible": True, "content": "the other"}]}],
                 "source": {"state": "changed"}},
                {**results[1]}]

    monkeypatch.setattr(recall_notes, "annotate", notes)
    a, b = cp.annotate([_mem(), _mem(fact_id="f9")], "/x/memory.db", "default")
    assert a.disputed and a.other_side == "the other" and a.source_state == "changed"
    assert not b.disputed and b.source_state == ""


def test_a_hidden_other_side_is_never_shown(monkeypatch) -> None:
    from superlocalmemory.truth import recall_notes

    monkeypatch.setattr(recall_notes, "annotate", lambda db, pid, results, **kw: [
        {**results[0], "disputed": True, "disputes": [{"sides": [
            {"fact_id": "f2", "visible": False, "content": None}]}]}])
    (a,) = cp.annotate([_mem()], "/x/memory.db", "default")
    assert a.disputed and a.other_side == ""
    assert "another memory says" not in cp.standing(a)


def test_annotate_fails_open(monkeypatch) -> None:
    from superlocalmemory.truth import recall_notes

    def boom(*a, **k):
        raise RuntimeError("no store")

    monkeypatch.setattr(recall_notes, "annotate", boom)
    mems = [_mem()]
    assert cp.annotate(mems, "/x", "default") is mems
    assert cp.annotate(mems, None, "default") is mems
    assert cp.annotate([], "/x", "default") == []


@pytest.mark.parametrize("text,expected", [
    ("One. Two! Three?", ["One.", " Two!", " Three?"]),
    ("a\nb", ["a\n", "b"]),
    ("v1.2 is out. Next.", ["v1.2 is out.", " Next."]),
])
def test_units(text, expected) -> None:
    assert cp.units(text) == expected


def test_units_never_lose_text() -> None:
    rng = random.Random(8)
    pieces = ["v1.2", "e.g.", "done.", "ok!", "why?", "\n", " ", "word", "3.14", "..."]
    for _ in range(500):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 40)))
        # every visible character survives, in order (whitespace-only pieces may go)
        assert "".join("".join(cp.units(text)).split()) == "".join(text.split())
