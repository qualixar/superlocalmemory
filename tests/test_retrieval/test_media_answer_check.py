"""The answer check does not pretend to judge a picture it cannot read."""

from __future__ import annotations

from types import SimpleNamespace

from superlocalmemory.core.answer_check_stage import run_answer_check
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import answerability as ans


def result(text, **scores):
    return SimpleNamespace(fact=SimpleNamespace(fact_id="f", content=text, profile_id="p",
                                                memory_kind=""), channel_scores=scores)


class _Judge:
    backend = "local"
    top_k = 3

    def __init__(self):
        self.seen = None

    def assess(self, query, documents, deadline=None, reuse=None):
        self.seen = [d.content for d in documents]
        return acs.JudgeOutcome(None, acs.STATUS_UNAVAILABLE)


def run(results):
    judge = _Judge()
    engine = SimpleNamespace(_sufficiency_judge=judge, _db=None)
    outcome = run_answer_check(engine, "what is in the picture?", SimpleNamespace(results=results))
    return outcome, judge


def test_all_pictures_without_text_are_unjudged():
    outcome, judge = run([result("[Image without text]", media=0.8),
                          result("[Image without text]", media=0.7)])
    assert (outcome.status, outcome.detail) == (acs.STATUS_SKIPPED, acs.DETAIL_MEDIA_UNJUDGED)
    assert judge.seen is None  # nothing was sent to the judge
    assert ans.answerability(outcome.status, outcome.detail, abstained=False, result_count=2) == (
        ans.UNJUDGED, "media_unjudged")


def test_mixed_results_are_judged_on_stripped_text():
    outcome, judge = run([result("[Image without text]", media=0.8),
                          result("[Text in image]\nopen 9 to 5", media=0.7)])
    assert outcome.detail != acs.DETAIL_MEDIA_UNJUDGED
    assert judge.seen == ["", "open 9 to 5"]


def test_ordinary_results_are_judged_as_before():
    outcome, judge = run([result("Paris is the capital", semantic=0.9)])
    assert judge.seen == ["Paris is the capital"]


def test_the_new_detail_is_a_known_word():
    assert acs.DETAIL_MEDIA_UNJUDGED in acs.ANSWER_CHECK_DETAILS
    assert acs.answer_check_note(acs.STATUS_SKIPPED, acs.DETAIL_MEDIA_UNJUDGED)
    assert "media_unjudged" in ans.REASONS
