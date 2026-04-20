# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.6

"""TDD tests for ``learning/labeler.py`` — integer-label mapping."""

from __future__ import annotations

import math

import pytest

from superlocalmemory.learning.labeler import label_for_row, label_gain


# ---------------------------------------------------------------------------
# §6.6 test_outcome_reward_mapping_buckets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reward,expected", [
    (0.95, 4),
    (0.90, 4),
    (0.89, 3),
    (0.60, 3),
    (0.59, 2),
    (0.30, 2),
    (0.29, 1),
    (0.01, 1),
    (0.00, 0),
    (-0.5, 0),
])
def test_outcome_reward_mapping_buckets(reward, expected):
    assert label_for_row({"outcome_reward": reward}) == expected


# ---------------------------------------------------------------------------
# §6.6 test_position_proxy_fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position,expected", [
    (0, 4),
    (1, 3),
    (2, 3),
    (3, 2),
    (4, 2),
    (5, 1),
    (9, 1),
    (10, 0),
    (99, 0),
])
def test_position_proxy_fallback(position, expected):
    assert label_for_row({"position": position}) == expected


def test_position_default_when_missing():
    # Missing position → defaults to 99 → label 0.
    assert label_for_row({}) == 0


def test_outcome_none_falls_to_position():
    assert label_for_row({"outcome_reward": None, "position": 0}) == 4


def test_outcome_nan_falls_to_position():
    assert label_for_row({"outcome_reward": math.nan, "position": 1}) == 3


def test_outcome_non_numeric_falls_to_position():
    assert label_for_row({"outcome_reward": "bad", "position": 0}) == 4


def test_position_non_numeric_returns_zero():
    assert label_for_row({"position": "??"}) == 0


# ---------------------------------------------------------------------------
# §6.6 test_label_gain_length_respected — max label < len(label_gain)
# ---------------------------------------------------------------------------


def test_label_gain_length_respected():
    gains = label_gain()
    assert len(gains) == 5
    # Enumerate all plausible labels, ensure none exceed gains length.
    labels = set()
    for r in [-1, 0, 0.1, 0.3, 0.6, 0.9, 1.0]:
        labels.add(label_for_row({"outcome_reward": r}))
    for p in range(0, 50):
        labels.add(label_for_row({"position": p}))
    assert max(labels) < len(gains)
    assert min(labels) >= 0
