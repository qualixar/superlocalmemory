# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""GitHub #150 with learned ranking switched on.

Learned ranking (``SLM_RANKING``) runs after retrieval and rewrites every
result's ranking score: the phase-2 heuristic from the displayed score, the
phase-3 model from its own features, the ensemble from bandit-weighted
channel scores. None of them knows the project, so a preference applied only
inside retrieval is undone by them - on exactly the stores that have
learning on. These tests run the issue's fixture under each mode, with a
seeded learning store (feedback rows, and for phase 3 a real LightGBM model
trained deterministically here), and check the project's memories still lead
and the order repeats.
"""

from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pytest

from tests.test_retrieval.test_project_scope_ranking import (  # noqa: F401 - fixture
    _PROJECT,
    PROJECT_PATH,
    _contents,
    _is_project,
    _recall,
)
from tests.test_retrieval.test_project_scope_ranking import (
    store as project_store,
)

# Register the borrowed fixture under its requested name without an unused
# import that Ruff mistakes for redefinition at every fixture argument.
store = project_store

QUERY = "what happens to an invoice"
MODES = ("off", "v1", "v2", "v2-ensemble")


def _trained_model(store) -> tuple[bytes, list[str]]:
    """A real LightGBM booster, trained deterministically here on THIS
    fixture's own retrieval features - rewarding the displayed score and
    keyword evidence - a plausible learned taste that knows nothing about
    projects, and that separates this fixture's memories.

    This used to train on an independent synthetic sample - x ~ uniform
    over a plausible-looking range (0.45-0.65 for the displayed score,
    0-5 for bm25, and so on) - rather than on values this fixture actually
    produces. That range is wide enough that the ten real values recall
    gives back for QUERY (a ~0.53-0.57 band - see the docstring on
    test_the_seeded_model_is_really_active for the measured numbers) are a
    handful of close points deep inside it. Histogram-based split-finding
    bins the training range into a fixed number of bins; whichever bin edges
    a platform's build happens to land on, nothing stops every one of those
    ten close points from landing in the SAME bin there even though they
    land in ten different leaves here - and when that happens every
    prediction for this fixture comes back identical.

    (The 0.3636741782759984 that CI measured for all ten scores in
    test_the_seeded_model_is_really_active was not that, though: it was a
    leaked LightGBM mock - see the guard at the top of this function.)

    Training directly on the rows this fixture's own (non-learning) recall
    produces - oversampled, each copy nudged by noise far smaller than the
    real gap between any two of them - puts all of the training mass inside
    the exact band being scored, so a split has to separate it wherever a
    platform's bin edges land, on every machine that runs this fixture.
    """
    import lightgbm as lgb

    from superlocalmemory.learning.features import FEATURE_NAMES, FeatureExtractor

    # tests/test_learning and tests/test_api mock lightgbm.train for their
    # own directories. When that mock leaked (session scope) into this one,
    # the "trained" model was the mock's fixed two-tree model, and every
    # score below came back 0.3636741782759984 on CI, where those
    # directories run first. Say so here rather than three asserts later.
    assert lgb.train.__module__ == "lightgbm.engine" and \
        lgb.Dataset.__module__ == "lightgbm.basic", (
            f"lightgbm is mocked here ({lgb.train!r}, {lgb.Dataset!r}): a "
            "test directory's LightGBM mock outlived that directory"
        )

    with pytest.MonkeyPatch.context() as mp:
        # Explicit, regardless of ambient state: these are the plain
        # (un-learned) features apply_v2_adaptive_ranking will itself
        # re-extract from the same results at test time.
        mp.setenv("SLM_RANKING", "off")
        plain = _recall(store, QUERY, limit=10).results

    rows: list[list[float]] = []
    labels: list[float] = []
    for r in plain:
        result = {
            "score": r.score,
            "cross_encoder_score": r.score,
            "trust_score": r.trust_score,
            "channel_scores": r.channel_scores or {},
            "fact": {"age_days": 0.0, "access_count": r.fact.access_count},
        }
        fv = FeatureExtractor.extract(result, {"query_type": "single_hop"})
        rows.append(fv.to_list())
        labels.append(10.0 * fv.features["cross_encoder_score"]
                      + 0.3 * fv.features["bm25_score"]
                      + 0.5 * fv.features["semantic_score"])

    distinct = sorted(set(round(v, 9) for v in labels))
    assert len(distinct) >= 4, (
        "the fixture no longer gives test_the_seeded_model_is_really_active "
        f"enough distinct per-memory features to train on: {labels!r}"
    )

    base = np.asarray(rows, dtype=np.float32)
    y_base = np.asarray(labels, dtype=np.float32)
    min_gap = min(b - a for a, b in zip(distinct, distinct[1:]))
    jitter_scale = min_gap / 20.0

    rng = np.random.RandomState(150)
    reps = 60
    x = np.concatenate([base] + [
        base + rng.normal(0.0, jitter_scale, size=base.shape).astype(np.float32)
        for _ in range(reps)
    ])
    y = np.concatenate([y_base] * (reps + 1))

    params = {"objective": "regression", "num_leaves": 15, "learning_rate": 0.2,
              "min_data_in_leaf": 5, "deterministic": True, "num_threads": 1,
              "seed": 150, "verbose": -1, "force_row_wise": True}
    booster = lgb.train(params, lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES)),
                        num_boost_round=40)

    # The model that is about to be persisted and shipped into the recall
    # path must really discriminate the rows it was built for - verified
    # here, not assumed, so a platform-specific collapse like the one above
    # fails loudly at fixture setup instead of as a confusing assertion deep
    # in a parametrized test.
    state = booster.model_to_string()
    # Predict with the model as recall will load it - from its saved text -
    # not with the in-memory booster: a stand-in booster can score the rows
    # one way and save a different model.
    predicted = lgb.Booster(model_str=state).predict(base)
    assert len(set(round(float(p), 6) for p in predicted)) > 3, (
        "freshly trained booster predicts a near-constant score for the "
        f"fixture's own memories: {predicted!r}"
    )

    return state.encode("utf-8"), list(FEATURE_NAMES)


@pytest.fixture()
def learning(store, monkeypatch):
    """A learning store past the phase-3 gate, with an active, verified model."""
    from superlocalmemory.learning import bandit_draw, model_cache
    from superlocalmemory.learning.bandit_cache import get_shared_cache
    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.storage.migrations import M002_model_state_history as m002

    # Recall is bound to this engine's store, independently of the env root.
    monkeypatch.setenv("SLM_DATA_DIR", str(store._config.base_dir))
    path = store._config.base_dir / "learning.db"
    # The production draw depends on an install secret. Pin the test key so
    # different temporary installs exercise the same arm and ranking bound.
    monkeypatch.setattr(bandit_draw, "_draw_key", lambda: b"project-ranking-fixture")
    get_shared_cache().clear()
    db = LearningDatabase(path)
    conn = sqlite3.connect(str(path))
    try:
        if not m002.verify(conn):
            conn.executescript(m002.DDL)
    finally:
        conn.close()
    for i in range(210):
        db.store_signal("default", f"q{i}", f"f{i}", "recall_hit", 1.0)
    state, names = _trained_model(store)
    db.persist_model(profile_id="default", state_bytes=state,
                     bytes_sha256=hashlib.sha256(state).hexdigest(),
                     feature_names=names, trained_on_count=210, metrics={})
    model_cache.invalidate()
    yield path
    model_cache.invalidate()
    get_shared_cache().clear()


def _top(store, monkeypatch, mode: str, **facets) -> list:
    monkeypatch.setenv("SLM_RANKING", mode)
    return _recall(store, QUERY, limit=10, **facets).results


def test_the_seeded_model_is_really_active(store, learning, monkeypatch) -> None:
    from superlocalmemory.core.recall_pipeline import _ReadOnlyLearningView
    from superlocalmemory.learning.model_cache import load_active

    view = _ReadOnlyLearningView(learning)
    assert view.count_signals("default") >= 200
    assert load_active(view, "default") is not None
    # Phase 3 really ranked: its scores are the model's, and they separate
    # the memories (a constant model would rank nothing).
    keys = [r.ranking_score for r in _top(store, monkeypatch, "v2")]
    heuristic = [r.ranking_score for r in _top(store, monkeypatch, "v1")]
    assert keys != heuristic and len(set(round(k, 6) for k in keys)) > 3


@pytest.mark.parametrize("mode", MODES)
def test_project_memories_lead_whatever_ranks_last(store, learning, monkeypatch, mode) -> None:
    plain = _contents_of(_top(store, monkeypatch, mode))
    after = _contents_of(_top(store, monkeypatch, mode, prefer_project=PROJECT_PATH))
    # Without the preference, trivia sits among the project's memories in
    # every mode - the order the preference has to win against.
    assert not all(_is_project(c) for c in plain[:5]), (mode, plain)
    # With it, the project's five lead, whatever ranked last.
    assert sorted(after[:5]) == sorted(_PROJECT), (mode, after)
    assert set(after) == set(plain), "the preference added or removed a memory"


@pytest.mark.parametrize("mode", MODES)
def test_the_order_repeats_under_learned_ranking(store, learning, monkeypatch, mode) -> None:
    runs = [_top(store, monkeypatch, mode, prefer_project=PROJECT_PATH) for _ in range(3)]
    shapes = [[r.fact.fact_id for r in run] for run in runs]
    assert shapes[0] == shapes[1] == shapes[2], mode


@pytest.mark.parametrize("mode", MODES)
def test_the_bound_holds_on_the_final_score(store, learning, monkeypatch, mode) -> None:
    """Whatever produced the final ranking score, a project memory only ever
    passes a memory whose score is less than 1.25x its own."""
    plain = _top(store, monkeypatch, mode)
    preferred = _top(store, monkeypatch, mode, prefer_project=PROJECT_PATH)
    from superlocalmemory.core.recall_pipeline import _rank_key

    utility = {r.fact.fact_id: -_rank_key(r)[0] for r in plain}
    order = [r.fact.fact_id for r in preferred]
    for a in order:
        for b in order:
            if a not in utility or b not in utility:
                continue
            a_first_now = order.index(a) < order.index(b)
            a_first_before = [r.fact.fact_id for r in plain].index(a) < \
                [r.fact.fact_id for r in plain].index(b)
            if a_first_now and not a_first_before:
                assert utility[a] * 1.25 > utility[b] - 1e-12, (mode, a, b)


def _contents_of(results) -> list[str]:
    return [r.fact.content for r in results]


def test_learning_fixture_seeds_the_store_recall_actually_uses(store, learning):
    from superlocalmemory.core.recall_pipeline import _engine_learning_db

    assert learning == _engine_learning_db(store._config)
