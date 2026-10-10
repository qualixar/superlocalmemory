"""A picture with no words of its own is not read by the text reranker."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.retrieval import media_rerank as mr
from superlocalmemory.retrieval.engine import RetrievalEngine
from superlocalmemory.retrieval.fusion import FusionResult


def fr(fid, score, **scores):
    return FusionResult(fact_id=fid, fused_score=score, channel_scores=scores)


def fact(fid, content):
    return SimpleNamespace(fact_id=fid, content=content)


class _Reranker:
    """Scores by a fixed table; anything not listed is a poor match."""

    def __init__(self, table):
        self.table, self.seen = table, []

    def rerank(self, query, candidates, top_k):
        self.seen += [f.fact_id for f, _ in candidates]
        return sorted(((f, self.table.get(f.fact_id, -5.0)) for f, _ in candidates),
                      key=lambda t: -t[1])


def engine(reranker):
    eng = RetrievalEngine.__new__(RetrievalEngine)
    eng._reranker = reranker
    return eng


FUSED = [fr("t1", 0.9, semantic=0.7), fr("img", 0.8, media=0.8), fr("t2", 0.7, bm25=1.0)]
FACTS = {"t1": fact("t1", "a note"), "img": fact("img", "[Image without text]"),
         "t2": fact("t2", "another note")}


def test_media_only_empty_text_keeps_its_fused_rank():
    eng = engine(_Reranker({"t1": 1.0, "t2": 9.0}))  # the text reranker would put t2 first
    out, applied, _ = eng._apply_reranker("q", list(FUSED), dict(FACTS))
    assert applied
    ids = [r.fact_id for r in out]
    assert ids.index("img") == 1  # the place fusion gave it
    assert ids[0] == "t2" and ids[2] == "t1"  # text candidates reordered as before
    assert "img" not in eng._reranker.seen


def test_scores_keep_the_order_the_rerank_chose():
    eng = engine(_Reranker({"t1": 1.0, "t2": 9.0}))
    out, _, _ = eng._apply_reranker("q", list(FUSED), dict(FACTS))
    by_score = [r.fact_id for r in sorted(out, key=lambda r: (-r.fused_score, r.fact_id))]
    assert by_score == [r.fact_id for r in out]


def test_text_candidates_are_unchanged_without_media():
    plain = [fr("t1", 0.9, semantic=0.7), fr("t2", 0.7, bm25=1.0)]
    eng = engine(_Reranker({"t1": 1.0, "t2": 9.0}))
    out, _, _ = eng._apply_reranker("q", plain, {k: FACTS[k] for k in ("t1", "t2")})
    assert [r.fact_id for r in out] == ["t2", "t1"]


def test_an_image_with_words_or_another_channel_is_reranked_normally():
    fused = [fr("t1", 0.9, semantic=0.7), fr("img", 0.8, media=0.8, bm25=0.4)]
    eng = engine(_Reranker({"t1": 1.0, "img": 9.0}))
    out, _, _ = eng._apply_reranker("q", fused, {"t1": FACTS["t1"], "img": fact("img", "a cat photo")})
    assert [r.fact_id for r in out] == ["img", "t1"]
    fused2 = [fr("t1", 0.9, semantic=0.7), fr("img", 0.8, media=0.8)]
    out2, _, _ = engine(_Reranker({"t1": 1.0, "img": 9.0}))._apply_reranker(
        "q", fused2, {"t1": FACTS["t1"], "img": fact("img", "a cat photo")})
    assert [r.fact_id for r in out2] == ["img", "t1"]  # words of its own: reranked


def test_strip_labels():
    assert mr.strip_labels("[Image without text]") == ""
    assert mr.strip_labels("[Text in image]\nSALE 50%") == "SALE 50%"
    assert mr.strip_labels("my cat\n\n[Text in image]\nmeow") == "my cat\n\nmeow"
    assert mr.strip_labels("plain") == "plain"


def test_label_only_needs_a_label():
    assert mr.is_label_only("[Image without text]")
    assert not mr.is_label_only("")
    assert not mr.is_label_only("[Text in image]\nhello")
