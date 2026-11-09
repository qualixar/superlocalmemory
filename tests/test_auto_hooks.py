# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for Auto-Capture & Auto-Recall — Task 16 of V3 build."""


from superlocalmemory.hooks.auto_capture import AutoCapture
from superlocalmemory.hooks.auto_recall import AutoRecall
from superlocalmemory.hooks.rules_engine import RulesEngine

# -- Auto Recall --


def test_auto_recall_disabled():
    recall = AutoRecall(engine=None, config={"enabled": False})
    assert recall.get_session_context() == ""


def test_auto_recall_no_engine():
    recall = AutoRecall(engine=None)
    assert recall.get_session_context() == ""


def test_auto_recall_enable_disable():
    recall = AutoRecall(engine=None)
    assert recall.enabled is True
    recall.disable()
    assert recall.enabled is False
    recall.enable()
    assert recall.enabled is True


# -- Auto Capture --


def test_capture_detects_decision():
    capture = AutoCapture(config={"min_confidence": 0.4})
    result = capture.evaluate("We decided to use PostgreSQL because it handles JSON well")
    assert result.capture is True
    assert result.category == "decision"


def test_capture_detects_bug():
    capture = AutoCapture(config={"min_confidence": 0.4})
    result = capture.evaluate("Fixed the authentication bug — the JWT token was expiring too fast")
    assert result.capture is True
    assert result.category == "bug"


def test_capture_detects_preference():
    capture = AutoCapture(config={"min_confidence": 0.4})
    result = capture.evaluate("I always use dark mode and never use light themes in editors")
    assert result.capture is True
    assert result.category == "preference"


def test_capture_skips_trivial():
    capture = AutoCapture()
    result = capture.evaluate("ok")
    assert result.capture is False


def test_capture_disabled():
    capture = AutoCapture(config={"enabled": False})
    result = capture.evaluate("We decided to use React for the frontend")
    assert result.capture is False


def test_capture_respects_category_toggle():
    capture = AutoCapture(config={"capture_decisions": False, "min_confidence": 0.4})
    result = capture.evaluate("We decided to use PostgreSQL")
    assert result.capture is False  # decisions disabled


# -- Rules Engine --


def test_rules_default():
    rules = RulesEngine()
    assert rules.should_recall("session_start") is True
    # 4.1.25: wired to the per-prompt hooks, which have always run by default.
    assert rules.should_recall("every_prompt") is True


def test_rules_should_capture():
    rules = RulesEngine()
    assert rules.should_capture("decision", 0.8) is True
    assert rules.should_capture("decision", 0.2) is False


def test_rules_disabled():
    rules = RulesEngine(config={"auto_capture": {"enabled": False}})
    assert rules.should_capture("decision", 0.9) is False


def test_rules_update():
    rules = RulesEngine()
    rules.update_rule("auto_recall", "on_every_prompt", False)
    assert rules.should_recall("every_prompt") is False


def test_rules_save_and_load(tmp_path):
    rules = RulesEngine()
    rules.update_rule("auto_capture", "min_confidence", 0.7)
    config_path = tmp_path / "config.json"
    rules.save(config_path)

    loaded = RulesEngine(config_path=config_path)
    assert loaded.get_capture_config()["min_confidence"] == 0.7


def test_rules_to_dict():
    rules = RulesEngine()
    d = rules.to_dict()
    assert "auto_recall" in d
    assert "auto_capture" in d
