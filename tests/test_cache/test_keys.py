# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Cache keys are validated values; the params hash ignores dict order."""

import pytest

from superlocalmemory.cache import CacheKey
from superlocalmemory.cache.keys import params_hash

SHA = "a" * 64


def test_a_valid_key_has_a_five_part_tuple():
    key = CacheKey(SHA, "pdf.render", "1", "m", "p")
    assert key.as_tuple() == (SHA, "pdf.render", "1", "m", "p")
    assert CacheKey(SHA, "x", "1").model_id == ""


@pytest.mark.parametrize("bad", ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65])
def test_a_bad_content_hash_is_refused(bad):
    with pytest.raises(ValueError):
        CacheKey(bad, "x", "1")


@pytest.mark.parametrize("deriver", ["", "UPPER", "has space", "a" * 65, "a/b"])
def test_a_bad_deriver_id_is_refused(deriver):
    with pytest.raises(ValueError):
        CacheKey(SHA, deriver, "1")


@pytest.mark.parametrize("version", ["", "v" * 65])
def test_a_bad_deriver_version_is_refused(version):
    with pytest.raises(ValueError):
        CacheKey(SHA, "x", version)


def test_params_hash_is_stable_across_dict_order():
    assert params_hash({"a": 1, "b": [1, 2]}) == params_hash({"b": [1, 2], "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})
    assert len(params_hash({})) == 64


def test_params_hash_rejects_values_that_are_not_json():
    with pytest.raises(ValueError):
        params_hash({"a": object()})
    with pytest.raises(ValueError):
        params_hash({"a": {1, 2}})
