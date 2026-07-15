# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Every shipped HTTP mutation must consume identity and policy hooks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _function_block(source: str, function_name: str) -> str:
    start = source.index(f"async def {function_name}")
    next_route = source.find("\n@router.", start + 1)
    return source[start: next_route if next_route != -1 else None]


def test_route_authorization_uses_middleware_actor_and_wraps_hooks() -> None:
    from superlocalmemory.server.route_mutations import authorize_route_mutation

    engine = MagicMock()
    engine.profile_id = "default"
    request = SimpleNamespace(
        state=SimpleNamespace(authenticated_actor="capability:verified"),
        app=SimpleNamespace(
            state=SimpleNamespace(engine=engine, daemon_descriptor=None),
        ),
        headers={},
    )

    authorization = authorize_route_mutation(
        request,
        operation="update",
        source_agent_id="http-tier-pin",
        profile_id="work",
        fact_id="fact-1",
    )

    context = engine._hooks.run_pre.call_args.args[1]
    assert context["agent_id"] == "capability:verified"
    assert context["source_agent_id"] == "http-tier-pin"
    assert context["profile_id"] == "work"
    assert context["fact_id"] == "fact-1"

    authorization.complete()
    engine._hooks.run_post.assert_called_once_with("update", context)


def test_profile_and_tier_routes_cross_authenticated_policy_boundary() -> None:
    expected = {
        "profiles.py": (
            "switch_profile",
            "create_profile",
            "delete_profile",
        ),
        "tiers.py": (
            "evaluate_tiers_route",
            "pin_fact_route",
            "unpin_fact_route",
        ),
    }
    root = Path("src/superlocalmemory/server/routes")
    for filename, functions in expected.items():
        source = (root / filename).read_text(encoding="utf-8")
        for function_name in functions:
            block = _function_block(source, function_name)
            assert "authorize_route_mutation" in block, (
                filename,
                function_name,
            )
            assert ".complete()" in block, (filename, function_name)


def test_v3_memory_lifecycle_routes_cross_authenticated_policy_boundary() -> None:
    source = Path(
        "src/superlocalmemory/server/routes/v3_api.py"
    ).read_text(encoding="utf-8")
    for function_name in (
        "run_consolidation",
        "trigger_consolidation",
        "update_core_memory_block",
        "run_forgetting",
        "run_community_detection",
    ):
        block = _function_block(source, function_name)
        assert "authorize_route_mutation" in block, function_name
        assert ".complete()" in block, function_name


def test_fact_delete_and_edit_delegate_to_canonical_mutation_service() -> None:
    source = Path(
        "src/superlocalmemory/server/routes/memories.py"
    ).read_text(encoding="utf-8")

    delete_block = _function_block(source, "delete_memory")
    assert "delete_fact_authorized" in delete_block
    assert "DELETE FROM atomic_facts" not in delete_block

    edit_block = _function_block(source, "edit_memory")
    assert "update_fact_authorized" in edit_block
    assert "UPDATE atomic_facts SET content" not in edit_block
