# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A question that names a version is not answered by another version.

Found on the author's store (9 Nov 2026): "SLM 4.2.15 release" ranked "SLM
4.2.0 released 13 Oct", "SLM 4.2.2 released" and five more 4.2.x notes above
current memories that name 4.2.15. The BM25 channel already gives a near-miss
version no credit; the semantic, Hopfield and entity channels see "4.2.0" and
"4.2.15" as the same thing, so the final order still put them first. A result
that names only OTHER versions now goes below every result that does not.
"""

from __future__ import annotations

from superlocalmemory.retrieval.version_scope import names_only_other_versions, prefer_named_versions
from superlocalmemory.storage.models import AtomicFact, RetrievalResult


def _r(content: str) -> RetrievalResult:
    return RetrievalResult(fact=AtomicFact(fact_id=content[:12], content=content), score=0.5)


def _order(query: str, contents: list[str]) -> list[str]:
    return [r.fact.content for r in prefer_named_versions([_r(c) for c in contents], query)]


def test_other_versions_go_below_the_asked_version_and_neutral_results():
    got = _order("SLM 4.2.15 release", [
        "SLM 4.2.0 released 13 Oct 2026",
        "SLM 4.2.2 released 13 Oct 2026",
        "Session close 9 Nov 2026: 4.2.15 and 4.2.16 shipped",
        "Releases go out after the full suite passes once.",
        "SLM v4.2.15 is on PyPI.",
    ])
    assert got == [
        "Session close 9 Nov 2026: 4.2.15 and 4.2.16 shipped",
        "Releases go out after the full suite passes once.",
        "SLM v4.2.15 is on PyPI.",
        "SLM 4.2.0 released 13 Oct 2026",
        "SLM 4.2.2 released 13 Oct 2026",
    ]


def test_a_prefix_version_is_not_the_asked_version():
    assert names_only_other_versions("SLM 4.2.1 released", ("4.2.15",))
    assert names_only_other_versions("SLM 4.2.150 released", ("4.2.15",))
    assert not names_only_other_versions("SLM 4.2.15 released", ("4.2.15",))
    assert not names_only_other_versions("Releases ship on Tuesdays", ("4.2.15",))


def test_a_query_without_a_version_changes_nothing():
    contents = ["SLM 4.2.0 released", "Releases ship on Tuesdays", "SLM 4.2.15 released"]
    results = [_r(c) for c in contents]
    assert prefer_named_versions(results, "how do releases ship") is results


def test_order_is_kept_inside_each_group_and_nothing_is_dropped():
    contents = [f"note {i} about SLM 4.1.{i}" for i in range(5)] + ["SLM 4.2.15 shipped"]
    got = _order("4.2.15", contents)
    assert got == ["SLM 4.2.15 shipped", *contents[:5]]


def test_the_recall_pipeline_applies_it_before_the_exact_hit_pin():
    import inspect

    from superlocalmemory.core import recall_pipeline

    source = inspect.getsource(recall_pipeline)
    assert source.index("version_scope.prefer_named_versions") < source.index(
        "_preserve_exact_lexical_evidence(response, query)")


def test_the_pool_rule_lets_a_memory_naming_the_version_past_the_cut():
    from types import SimpleNamespace as N

    from superlocalmemory.retrieval.version_scope import prefer_named_versions_in_pool

    facts = {f"n{i}": N(content=f"SLM 4.2.{i} released") for i in range(8)}
    facts["hit"] = N(content="Decision: the stub rule (4.2.15/4.2.16) covers bookkeeping.")
    pool = [N(fact_id=f"n{i}") for i in range(8)] + [N(fact_id="hit")]
    cut = prefer_named_versions_in_pool(pool, facts, "SLM 4.2.15 release")[:3]
    assert cut[0].fact_id == "hit"


def test_the_engine_applies_the_pool_rule_before_the_cut():
    import inspect

    from superlocalmemory.retrieval import engine

    source = inspect.getsource(engine)
    assert source.index("prefer_named_versions_in_pool(top, facts, query)") < source.index(
        "final_top = top[:effective_limit]")
