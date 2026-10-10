# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""A picture with no words is evidence only through the picture itself or the person's own note.

Found on the Mac end to end: "a photo of a red sports car on a beach", with no such photo saved,
returned four unrelated photos. Their media scores (0.46-0.50) were under the model's floor
(0.69); they passed on semantic 0.79 and entity 0.34, earned by the saving label
"[Image without text]" matching the word "photo", not by anything in the pictures.
"""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.media.labels import NO_TEXT, TEXT_MARKER
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult


def _fr(fid: str, **scores: float) -> FusionResult:
    return FusionResult(fact_id=fid, fused_score=0.7, channel_scores=scores)


def _facts(**contents: str) -> dict:
    return {fid: SimpleNamespace(content=text, pinned=False) for fid, text in contents.items()}


NOISE = dict(semantic=0.7977, spreading_activation=0.6463, entity_graph=0.343)


def test_a_wordless_picture_below_the_picture_floor_is_not_evidence():
    facts = _facts(photo=f"photo1.jpg\n\n{NO_TEXT}", bare=NO_TEXT)
    kept = RetrievalEngine._apply_evidence_floor(
        [_fr("photo", media=0.4577, **NOISE), _fr("bare", media=0.50, **NOISE)], facts, 0.60, 0.69)
    assert kept == []


def test_a_wordless_picture_that_matches_as_a_picture_is_kept():
    facts = _facts(car=NO_TEXT)
    kept = RetrievalEngine._apply_evidence_floor([_fr("car", media=0.72, **NOISE)], facts, 0.60, 0.69)
    assert [k.fact_id for k in kept] == ["car"]


def test_the_persons_own_note_still_counts_by_its_words():
    facts = _facts(note=f"my red car at the beach\n\n{NO_TEXT}")
    kept = RetrievalEngine._apply_evidence_floor(
        [_fr("note", media=0.40, bm25=2.1, **NOISE)], facts, 0.60, 0.69)
    assert [k.fact_id for k in kept] == ["note"]


def test_a_picture_with_readable_text_and_plain_memories_are_unchanged():
    facts = _facts(shot=f"{TEXT_MARKER}Recall Lab channel scores", plain="Staging DB is on port 5434")
    kept = RetrievalEngine._apply_evidence_floor(
        [_fr("shot", media=0.40, semantic=0.70), _fr("plain", semantic=0.70)], facts, 0.60, 0.69)
    assert [k.fact_id for k in kept] == ["shot", "plain"]


def test_a_pinned_wordless_picture_stays():
    facts = {"pin": SimpleNamespace(content=NO_TEXT, pinned=True)}
    kept = RetrievalEngine._apply_evidence_floor([_fr("pin", media=0.1, **NOISE)], facts, 0.60, 0.69)
    assert [k.fact_id for k in kept] == ["pin"]
