# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory v3.4.58 — test mock infrastructure

"""LightGBM mock fixture for the unit test suite.

Unit tests MUST NOT run real LightGBM C code because:
  1. Real training calls into OpenMP which races with background threads
     (LanceDB Rust/tokio, sentence-transformers ONNX worker) and causes
     SIGSEGV on macOS ARM when multiple libomp.dylib runtimes are loaded.
  2. Real training downloads nothing, but does heavy computation that is
     irrelevant to the logic being tested.
  3. Tests must be deterministic — real LightGBM training has
     platform-dependent floating-point results.

Usage in conftest.py:
    from tests.fixtures.lgb_mock import MockBooster, MockDataset, mock_lgb_train

This module provides:
  - MOCK_MODEL_STR: a real, minimal, loadable LightGBM model string
    (generated once from a 40-row, 20-feature, 4-group lambdarank dataset)
  - MockBooster: drop-in replacement for lgb.Booster, loads MOCK_MODEL_STR
  - MockDataset: drop-in replacement for lgb.Dataset, stores metadata only
  - mock_lgb_train: drop-in replacement for lgb.train, returns MockBooster
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Pre-captured minimal valid lambdarank model (40 rows, 20 features, 2 trees)
# Generated with num_threads=1, num_leaves=4, num_boost_round=2.
# This string is loadable by lgb.Booster(model_str=MOCK_MODEL_STR) without
# any C extension calls — the Booster.load() path is fast deserialization.
# ---------------------------------------------------------------------------
MOCK_MODEL_STR = """\
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=19
objective=lambdarank
feature_names=f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19
feature_infos=[0.020071197301149368:0.96244728565216064] [0.051823537796735764:0.99005383253097534] [0.034388519823551178:0.96991437673568726] [0.014544665813446045:0.9093204140663147] [0.015456616878509521:0.96217256784439087] [0.0091970516368746758:0.98996025323867798] [0.045446380972862244:0.98600107431411743] [0.010837651789188385:0.96665483713150024] [0.0050615840591490269:0.96361994743347168] [0.046450413763523102:0.98688691854476929] [0.012154474854469299:0.9905051589012146] [0.01658782921731472:0.99971765279769897] [0.005522117018699646:0.99663680791854858] [0.081759035587310791:0.96264839172363281] [0.018110183998942375:0.98565047979354858] [0.026366975158452988:0.99296480417251587] [0.011353644542396069:0.93615478277206421] [0.023271935060620308:0.96502691507339478] [0.025419127196073532:0.98727613687515259] [0.040868617594242096:0.98621076345443726]
tree_sizes=554 557

Tree=0
num_leaves=4
num_cat=0
split_feature=16 9 7
split_gain=4.45101 1.77469 0.525
threshold=0.38445423543453222 0.71689605712890636 0.62463909387588512
decision_type=2 2 2
left_child=1 -1 -2
right_child=2 -3 -4
leaf_value=-0.1417257894911573 0.1867578616391439 0.15760163197079191 0.058431654886038242
leaf_weight=1.5560264773666861 0.82697125896811463 0.226966232061386 0.51881861686706543
leaf_count=14 12 4 10
internal_value=-5.77466e-09 -0.103623 0.137287
internal_weight=3.12878 1.78299 1.34579
internal_count=40 18 22
is_linear=0
shrinkage=0.1


Tree=1
num_leaves=4
num_cat=0
split_feature=7 17 0
split_gain=5.27497 4.86167 1.41937
threshold=0.22996278852224353 0.55949372053146373 0.11784553900361062
decision_type=2 2 2
left_child=2 -2 -1
right_child=1 -3 -4
leaf_value=-0.19146606596748472 -0.10402433619005323 0.065533447479454471 0.1769163166368545
leaf_weight=0.11157570779323567 4.3735236264765263 2.7570306956768036 1.6710619181394575
leaf_count=1 19 15 5
internal_value=-9.15577e-10 -0.0384648 0.153859
internal_weight=8.91319 7.13055 1.78264
internal_count=40 34 6
is_linear=0
shrinkage=0.1


end of trees

feature_importances:
f7=2
f0=1
f9=1
f16=1
f17=1

parameters:
[boosting: gbdt]
[objective: lambdarank]
[metric: ndcg]
[tree_learner: serial]
[device_type: cpu]
[num_iterations: 2]
[learning_rate: 0.1]
[num_leaves: 4]
[num_threads: 1]
[max_depth: 2]
[min_data_in_leaf: 1]
[label_gain: 0,1,2,3,4,5,6,7,8,9]
[verbosity: -1]

end of parameters

pandas_categorical:null
"""


class MockDataset:
    """Drop-in for lgb.Dataset. Stores metadata, never calls any C code."""

    def __init__(
        self,
        data=None,
        label=None,
        group=None,
        feature_name=None,
        free_raw_data=True,
        **kwargs,
    ) -> None:
        self._data = data
        self._label = label
        self._group = group
        self._feature_name = feature_name


class MockBooster:
    """Drop-in for lgb.Booster. Uses MOCK_MODEL_STR for serialization.

    predict() returns realistic floats (not zeros) derived from the
    real tree structure so shadow-test comparisons don't always tie.
    """

    def __init__(self, params=None, train_set=None, model_str=None, **kwargs) -> None:
        self._model_str = model_str or MOCK_MODEL_STR

    def predict(self, X, **kwargs) -> np.ndarray:
        # Deterministic scores based on feature sum — not zeros so
        # the shadow test gate (which compares NDCG of two boosters)
        # can observe a difference between a "good" and "bad" model.
        arr = np.asarray(X, dtype=np.float64)
        return (arr.sum(axis=1) * 0.01).astype(np.float64)

    def model_to_string(self, **kwargs) -> str:
        return self._model_str

    def num_trees(self) -> int:
        return 2

    def feature_name(self):
        return [f"f{i}" for i in range(20)]


def mock_lgb_train(params, train_set, num_boost_round=100, **kwargs) -> MockBooster:
    """Drop-in for lgb.train. Returns a MockBooster instantly, no C calls."""
    return MockBooster()
