"""Tests for Engine Lifecycle Hooks -- Task 7 of V3 build."""
import pytest

from superlocalmemory.core.hooks import HookRegistry


def test_pre_hook_runs():
    registry = HookRegistry()
    called = []
    registry.register_pre("store", lambda ctx: called.append(ctx["operation"]))
    registry.run_pre("store", {"operation": "store"})
    assert called == ["store"]


def test_pre_hook_can_reject():
    registry = HookRegistry()
    registry.register_pre("store", lambda ctx: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(PermissionError):
        registry.run_pre("store", {})


def test_post_hook_runs():
    registry = HookRegistry()
    called = []
    registry.register_post("store", lambda ctx: called.append("done"))
    registry.run_post("store", {})
    assert called == ["done"]


def test_post_hook_error_does_not_propagate():
    registry = HookRegistry()
    registry.register_post("store", lambda ctx: 1/0)
    registry.run_post("store", {})  # should not raise


def test_multiple_pre_hooks_run_in_order():
    registry = HookRegistry()
    order = []
    registry.register_pre("store", lambda ctx: order.append("first"))
    registry.register_pre("store", lambda ctx: order.append("second"))
    registry.run_pre("store", {})
    assert order == ["first", "second"]


def test_multiple_post_hooks_all_run():
    registry = HookRegistry()
    results = []
    registry.register_post("store", lambda ctx: results.append("a"))
    registry.register_post("store", lambda ctx: results.append("b"))
    registry.run_post("store", {})
    assert results == ["a", "b"]


def test_hooks_isolated_by_operation():
    registry = HookRegistry()
    store_calls = []
    recall_calls = []
    registry.register_pre("store", lambda ctx: store_calls.append(1))
    registry.register_pre("recall", lambda ctx: recall_calls.append(1))
    registry.run_pre("store", {})
    assert len(store_calls) == 1
    assert len(recall_calls) == 0


def test_clear_removes_all_hooks():
    registry = HookRegistry()
    called = []
    registry.register_pre("store", lambda ctx: called.append(1))
    registry.clear()
    registry.run_pre("store", {})
    assert len(called) == 0


def test_unknown_operation_is_noop():
    registry = HookRegistry()
    registry.run_pre("nonexistent", {})  # should not raise
    registry.run_post("nonexistent", {})  # should not raise
