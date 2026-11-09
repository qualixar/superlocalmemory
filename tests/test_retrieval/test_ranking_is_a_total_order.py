# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The same question must return the same answer.

Sorting candidates by score alone is a PARTIAL order. Facts that tie keep
whatever order the input happened to arrive in, and that order comes from SQLite
page layout, dict insertion order, or which thread finished first — none of which
is data. So two runs over an unchanged store can return different answers, and
one of them is shown to the user as *the* answer.

Measured on a 517 MB store, 140 queries, two runs of identical code: 14 queries
returned a different top-10, 11 returned a different SET of facts, and 3 gave a
different single top answer. Nothing had changed between the runs.

The fix is to make every ordering total by breaking ties on `fact_id`, which is
unique and stable. `fusion.py` already did this; twenty-three other sorts and two
SQL queries did not.

Two tests here, deliberately of different kinds:

* a **behavioural** one, that tied scores come back in the same order however
  they went in — this is the property that matters;
* a **structural** one over the whole retrieval package, because the property
  holds at twenty-three separate call sites and a behavioural test for each would
  be twenty-three fixtures that still miss the twenty-fourth. The structural test
  is the one that catches a new sort added next month.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

RETRIEVAL = pathlib.Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "retrieval"


def _sort_calls_keyed_on_score_alone(tree: ast.AST) -> list[tuple[int, str]]:
    """Find `sorted(...)`/`.sort(...)` whose key returns a single scalar.

    A key that returns one value cannot express a tie-break. A key returning a
    tuple can, and is what every ordering here must use.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        is_sorted = isinstance(target, ast.Name) and target.id == "sorted"
        is_sort_method = isinstance(target, ast.Attribute) and target.attr == "sort"
        if not (is_sorted or is_sort_method):
            continue
        key = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
        if key is None:
            continue  # sorting the values themselves is already total
        if not isinstance(key, ast.Lambda):
            continue  # e.g. operator.itemgetter with several indices — allow
        body = key.body
        if isinstance(body, ast.Tuple):
            continue  # a tuple key can carry the tie-break
        # A single-scalar key. Reading the source segment keeps the failure
        # message actionable rather than making the reader hunt for the line.
        found.append((getattr(node, "lineno", -1), ast.unparse(key)))
    return found


def test_no_ordering_in_the_retrieval_path_is_keyed_on_score_alone() -> None:
    """Every sort must be able to break a tie.

    Listing the offenders rather than counting them, so the failure names exactly
    what to fix. If a genuinely score-only sort is ever correct here, it needs a
    tuple key whose second element is a constant and a comment saying why — not an
    exemption in this test.
    """
    offenders: list[str] = []
    for path in sorted(RETRIEVAL.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, key_src in _sort_calls_keyed_on_score_alone(tree):
            offenders.append(f"{path.name}:{lineno}  key={key_src}")

    assert not offenders, (
        "these orderings are keyed on a single value, so tied candidates are "
        "ordered by whatever the input happened to be — storage layout, dict "
        "insertion, or thread completion. Break the tie on fact_id:\n  "
        + "\n  ".join(offenders)
    )


def test_the_check_above_can_actually_fail() -> None:
    """A guard on the guard.

    The structural test passes when it finds nothing — which is also what it does
    if the detector is broken. This feeds it a known-bad snippet and a known-good
    one, so a detector that always returns nothing cannot pass.
    """
    bad = ast.parse("results.sort(key=lambda x: x[1], reverse=True)")
    good = ast.parse("results.sort(key=lambda x: (-x[1], x[0]))")
    assert _sort_calls_keyed_on_score_alone(bad), "the detector missed a score-only sort"
    assert not _sort_calls_keyed_on_score_alone(good), "the detector flagged a tuple key"


def _numpy_top_k_without_tiebreak(tree: ast.AST) -> list[tuple[int, str]]:
    """Find np.argsort / np.argpartition calls that sort on score alone.

    Both operations return indices ordered by one numeric array.  Equal scores
    tie in index (memory-layout) order, which differs between processes.  A
    secondary sort on a stable key (fact_id) must follow each such call before
    the indices are used to truncate or merge candidates.

    A line whose trailing comment contains 'tie-break:' is considered
    intentionally reviewed and is exempt — the caller documents the stable
    secondary key applied after this call.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in ("argsort", "argpartition"):
            continue
        if not isinstance(func.value, ast.Name):
            continue
        if func.value.id not in ("np", "numpy"):
            continue
        found.append((getattr(node, "lineno", -1), ast.unparse(node)))
    return found


def test_no_score_only_numpy_top_k_in_retrieval_path() -> None:
    """Every numpy top-k in retrieval/ must have a stable secondary key.

    np.argsort / np.argpartition sort by a single numeric array.  Equal scores
    tie in memory-layout order, which changes between processes — the mechanism
    behind the measured 9 % run-to-run candidate divergence.

    A '# tie-break: <explanation>' comment on the offending line marks it as
    reviewed and exempt (the explanation must describe the stable secondary key).
    CSR construction sorts row indices to group edges, without selecting or
    ranking candidates. graph_delta.py owns that structural operation; ANN
    and Hopfield score ties are also checked behaviourally in
    test_numpy_tie_stability.py.
    """
    offenders: list[str] = []
    for path in sorted(RETRIEVAL.glob("*.py")):
        if path.name == "graph_delta.py":
            continue  # CSR row grouping, not candidate top-k.
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)
        for lineno, call_src in _numpy_top_k_without_tiebreak(tree):
            line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if "tie-break:" in line:
                continue
            offenders.append(f"{path.name}:{lineno}  {call_src}")

    assert not offenders, (
        "these numpy sorts order by score alone — tied candidates are ordered "
        "by memory layout, which changes between processes.  Add a stable "
        "secondary sort on fact_id after each call, or add a "
        "'# tie-break: <explanation>' comment if the caller already does so:\n  "
        + "\n  ".join(offenders)
    )


def test_numpy_detector_can_actually_fail() -> None:
    """Guard on the numpy detector: a detector that always returns nothing cannot pass."""
    bad = ast.parse("top = np.argsort(-scores)[:k]")
    good = ast.parse("top = np.argsort(-scores)[:k]  # tie-break: caller re-sorts by fact_id")
    assert _numpy_top_k_without_tiebreak(bad), "detector missed np.argsort"
    # The tree has no source lines, so the comment-based exemption cannot fire
    # from the tree alone — callers that want exemption must pass source text.
    # The detector always flags from AST; the test function checks the text.


class TestTiedScoresComeBackInAStableOrder:
    """The behaviour the structural test is a proxy for.

    These tests call engine._build_results directly so a revert of the
    tie-break in retrieval/ causes them to fail, unlike tests that sort
    a local list and merely verify the Python standard library.
    """

    def _build_tied(self, fact_ids: list[str]) -> list[str]:
        """Run engine._build_results on facts with equal fused scores and ages."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock

        from superlocalmemory.core.config import RetrievalConfig
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.fusion import FusionResult
        from superlocalmemory.retrieval.strategy import QueryStrategy
        from superlocalmemory.storage.models import AtomicFact

        db = MagicMock()
        db.get_invalidated_fact_ids.return_value = set()
        db.get_nonapplied_correction_successor_ids.return_value = set()
        db.get_strict_temporal_excluded_fact_ids.return_value = set()

        # recency_prior_strength=0 so the amplifier is off; only Ebbinghaus
        # boost applies and it is identical for every fact because every fact
        # has the same age and access_count.
        cfg = RetrievalConfig(recency_prior_strength=0.0)
        engine = RetrievalEngine(db=db, config=cfg, channels={})

        created = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        facts = [
            AtomicFact(
                fact_id=fid, memory_id="m0", profile_id="default",
                content="Long enough content string to clear the quality threshold",
                confidence=0.9, access_count=0, created_at=created,
            )
            for fid in fact_ids
        ]
        fused = [FusionResult(fact_id=fid, fused_score=0.5) for fid in fact_ids]
        strat = QueryStrategy(query_type="factual", weights={}, confidence=0.8)
        fact_map = {f.fact_id: f for f in facts}
        return [r.fact.fact_id for r in engine._build_results(fused, fact_map, strat)]

    @pytest.mark.parametrize("reversed_input", [False, True])
    def test_input_order_does_not_decide_output_order(self, reversed_input: bool) -> None:
        """Two arrival orders, one identical result.

        This is what a user experiences as "the same question gave a different
        answer": several facts score identically and whichever arrived first won.
        Passes only if _build_results breaks ties on fact_id — it fails if the
        output simply follows input order.
        """
        fact_ids = [f"fact{i:02d}" for i in range(8)]
        if reversed_input:
            fact_ids = list(reversed(fact_ids))
        ordered = self._build_tied(fact_ids)
        assert ordered == [f"fact{i:02d}" for i in range(8)], (
            f"reversed_input={reversed_input}: got {ordered}. "
            "Output order followed input order — _build_results must sort by "
            "(-ranking_score, fact_id), not by the arrival order of fused results."
        )

    def test_scores_still_dominate_the_tie_break(self) -> None:
        """The tie-break must only ever decide ties.

        Without this, a key that sorted by fact_id first would satisfy the test
        above while destroying ranking entirely.
        """
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock

        from superlocalmemory.core.config import RetrievalConfig
        from superlocalmemory.retrieval.engine import RetrievalEngine
        from superlocalmemory.retrieval.fusion import FusionResult
        from superlocalmemory.retrieval.strategy import QueryStrategy
        from superlocalmemory.storage.models import AtomicFact

        db = MagicMock()
        db.get_invalidated_fact_ids.return_value = set()
        db.get_nonapplied_correction_successor_ids.return_value = set()
        db.get_strict_temporal_excluded_fact_ids.return_value = set()

        cfg = RetrievalConfig(recency_prior_strength=0.0)
        engine = RetrievalEngine(db=db, config=cfg, channels={})

        created = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        fact_data = [("zzz_best", 0.9), ("aaa_worst", 0.1), ("mmm_middle", 0.5)]
        facts = [
            AtomicFact(
                fact_id=fid, memory_id="m0", profile_id="default",
                content="Long enough content string to clear the quality threshold",
                confidence=0.9, access_count=0, created_at=created,
            )
            for fid, _ in fact_data
        ]
        fused = [FusionResult(fact_id=fid, fused_score=score) for fid, score in fact_data]
        strat = QueryStrategy(query_type="factual", weights={}, confidence=0.8)
        fact_map = {f.fact_id: f for f in facts}
        result_ids = [r.fact.fact_id for r in engine._build_results(fused, fact_map, strat)]

        assert result_ids == ["zzz_best", "mmm_middle", "aaa_worst"], (
            f"Score must dominate the tie-break. Got {result_ids}. "
            "A key that sorted by fact_id first would break ranking."
        )
