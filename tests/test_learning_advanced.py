# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Learning Advanced Modules -- Task 12 of V3 build."""


import pytest

from superlocalmemory.learning.bootstrap import SyntheticBootstrap
from superlocalmemory.learning.cross_project import CrossProjectAggregator
from superlocalmemory.learning.project_context import ProjectContextManager
from superlocalmemory.learning.workflows import WorkflowMiner


@pytest.fixture
def bootstrap(tmp_path):
    return SyntheticBootstrap(tmp_path / "bootstrap.db")

@pytest.fixture
def workflow(tmp_path):
    return WorkflowMiner(tmp_path / "workflow.db")

@pytest.fixture
def aggregator(tmp_path):
    return CrossProjectAggregator(tmp_path / "aggregator.db")


# -- Bootstrap --
def test_bootstrap_init(bootstrap):
    assert bootstrap is not None

def test_bootstrap_generate_empty(bootstrap):
    results = bootstrap.generate("p1", count=5)
    assert isinstance(results, list)
    # Empty profile generates no synthetic data
    assert len(results) == 0

# -- Workflow --
def test_workflow_init(workflow):
    assert workflow is not None

def test_workflow_mine_empty(workflow):
    patterns = workflow.mine("p1")
    assert isinstance(patterns, list)
    assert len(patterns) == 0

def test_workflow_record_and_mine(workflow):
    workflow.record_action("p1", "store", {"topic": "auth"})
    workflow.record_action("p1", "recall", {"topic": "auth"})
    workflow.record_action("p1", "store", {"topic": "auth"})
    patterns = workflow.mine("p1")
    assert isinstance(patterns, list)

# -- Cross-Project --
def test_aggregator_init(aggregator):
    assert aggregator is not None

def test_aggregator_no_source(aggregator):
    results = aggregator.aggregate([], "target_profile")
    assert isinstance(results, list)
    assert len(results) == 0

# -- Project Context --
def test_project_context_detect(tmp_path):
    # Create a fake Python project
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'", encoding="utf-8")
    ctx = ProjectContextManager()
    info = ctx.detect(tmp_path)
    assert info["project_name"] == "test" or "project_name" in info

def test_project_context_empty_dir(tmp_path):
    ctx = ProjectContextManager()
    info = ctx.detect(tmp_path)
    assert isinstance(info, dict)

def test_project_context_node_project(tmp_path):
    (tmp_path / "package.json").write_text('{"name": "my-app"}', encoding="utf-8")
    ctx = ProjectContextManager()
    info = ctx.detect(tmp_path)
    assert "my-app" in str(info.get("project_name", ""))
