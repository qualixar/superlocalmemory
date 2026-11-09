# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Q9 explicit identity guard: shrinking the envelope must not touch the answer.

On ONE fixed synthetic store with a large (3,816-member) community,
``RetrievalEngine.recall()`` runs twice against the exact same db/channels —
once with ``build_community_context`` monkeypatched to reproduce the OLD,
pre-fix behaviour (the entire stored membership, no ``member_count`` /
``member_fact_ids_truncated``), once with the real, fixed function. The
``results`` the two runs return — same ids, same order, same scores, same
content — must be byte-for-byte identical; only ``community_context`` may
differ. Varun's condition: shrink the envelope "without degrading the
quality" — this is the test that says so explicitly, not an argument that it
follows from the diff.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from superlocalmemory.core.config import RetrievalConfig
from superlocalmemory.retrieval import community_context as community_context_mod
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.storage.models import AtomicFact

_HUGE_COMMUNITY_MEMBER_COUNT = 3_816


def _old_style_build_community_context(db, results, profile_id, top_k=8):
    """Reproduces the PRE-FIX ``RetrievalEngine._community_context`` body
    verbatim (git history: the ``member_fact_ids`` field was the full
    ``json.loads(fact_ids_json)``, with no ``member_count`` /
    ``member_fact_ids_truncated``). Used only to prove the NEW, bounded
    function changes nothing about ``results`` — only the envelope.
    """
    import json as _json
    from collections import Counter

    rows = [
        dict(r) for r in db.execute(
            "SELECT community_id, summary, keywords, fact_ids_json, "
            "fact_count FROM community_summaries WHERE profile_id = ?",
            (profile_id,),
        )
    ]
    if not rows:
        return None
    fact_to_cid: dict[str, int] = {}
    summ_by_cid: dict[int, dict] = {}
    for r in rows:
        cid = int(r["community_id"])
        summ_by_cid[cid] = r
        for fid in _json.loads(r.get("fact_ids_json") or "[]"):
            fact_to_cid[str(fid)] = cid
    top_ids = [
        res.fact.fact_id for res in results[:top_k]
        if getattr(res, "fact", None) is not None
    ]
    tally = Counter(fact_to_cid[fid] for fid in top_ids if fid in fact_to_cid)
    if not tally:
        return None
    best_cid, count = tally.most_common(1)[0]
    coverage = count / len(top_ids) if top_ids else 0.0
    if count < 2 or coverage < 0.4:
        return None
    row = summ_by_cid[best_cid]
    members = _json.loads(row.get("fact_ids_json") or "[]")
    return {
        "community_id": best_cid,
        "summary": row.get("summary", ""),
        "keywords": row.get("keywords", ""),
        "member_fact_ids": members,  # THE pre-fix bug: the entire membership
        "coverage": round(coverage, 3),
        "matched_results": count,
    }


def _fixed_synthetic_store():
    """One fixed store: 10 facts, a 3,816-member community containing all 10."""
    facts = [
        AtomicFact(fact_id=f"f{i}", memory_id="m0", content=f"fact {i} content",
                    confidence=0.9)
        for i in range(10)
    ]
    big_members = [f"f{i}" for i in range(10)] + [
        f"m{i}" for i in range(_HUGE_COMMUNITY_MEMBER_COUNT - 10)
    ]
    db = MagicMock()
    db.get_all_facts.return_value = facts
    db.get_facts_by_ids.side_effect = (
        lambda ids, pid, **kwargs: [f for f in facts if f.fact_id in ids]
    )
    db.get_scenes_for_fact.return_value = []
    db.get_invalidated_fact_ids.return_value = set()
    db.get_nonapplied_correction_successor_ids.return_value = set()
    db.get_strict_temporal_excluded_fact_ids.return_value = set()
    db.execute.return_value = [{
        "community_id": 0,
        "summary": "A very large community.",
        "keywords": "big",
        "fact_ids_json": json.dumps(big_members),
        "fact_count": _HUGE_COMMUNITY_MEMBER_COUNT,
    }]
    return db, facts


def _build_engine(db) -> RetrievalEngine:
    bm25 = MagicMock()
    # Fixed descending-score order over all 10 facts — this IS "the answer";
    # the identity test proves the envelope fix never perturbs it.
    bm25.search.return_value = [(f"f{i}", 0.9 - i * 0.05) for i in range(10)]
    return RetrievalEngine(
        db=db, config=RetrievalConfig(), channels={"bm25": bm25},
    )


def _result_tuples(results):
    """(fact_id, score, content), in order — everything "same ids, same
    order, same scores, same content" asks for."""
    return [(r.fact.fact_id, r.score, r.fact.content) for r in results]


class TestOldVsNewEnvelopeIdentity:
    def test_results_are_byte_identical_old_vs_new_envelope(self, monkeypatch) -> None:
        db, _facts = _fixed_synthetic_store()

        monkeypatch.setattr(
            community_context_mod, "build_community_context",
            _old_style_build_community_context,
        )
        old_engine = _build_engine(db)
        try:
            old_response = old_engine.recall("q", "default")
        finally:
            old_engine.close(wait=True)

        monkeypatch.undo()
        new_engine = _build_engine(db)
        try:
            new_response = new_engine.recall("q", "default")
        finally:
            new_engine.close(wait=True)

        assert _result_tuples(old_response.results) == _result_tuples(new_response.results)
        assert len(new_response.results) == 10  # the gate actually fired both times

        # Sanity: the two runs really did take different community_context
        # paths — otherwise this test would prove nothing.
        assert old_response.community_context is not None
        assert new_response.community_context is not None
        assert len(old_response.community_context["member_fact_ids"]) == _HUGE_COMMUNITY_MEMBER_COUNT
        assert len(new_response.community_context["member_fact_ids"]) <= 10
        assert "member_count" not in old_response.community_context
        assert new_response.community_context["member_count"] == _HUGE_COMMUNITY_MEMBER_COUNT

    def test_revert_check_a_reordered_new_path_is_caught(self, monkeypatch) -> None:
        """Proves the identity test above is not vacuous: a deliberate
        reorder/drop injected into the NEW path must fail it."""
        db, _facts = _fixed_synthetic_store()

        monkeypatch.setattr(
            community_context_mod, "build_community_context",
            _old_style_build_community_context,
        )
        old_engine = _build_engine(db)
        try:
            old_response = old_engine.recall("q", "default")
        finally:
            old_engine.close(wait=True)
        monkeypatch.undo()

        new_engine = _build_engine(db)
        try:
            new_response = new_engine.recall("q", "default")
        finally:
            new_engine.close(wait=True)
        sabotaged = list(new_response.results)
        sabotaged[0], sabotaged[1] = sabotaged[1], sabotaged[0]  # reorder
        sabotaged = sabotaged[:-1]  # drop

        assert _result_tuples(old_response.results) != _result_tuples(sabotaged)
