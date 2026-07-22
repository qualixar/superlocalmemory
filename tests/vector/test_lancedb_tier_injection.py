# H8 (3.7.9): LanceDB tier updates must escape fact_id (no predicate injection).
from unittest.mock import MagicMock

from superlocalmemory.vector.lancedb_backend import LanceDBVectorBackend


def test_fact_predicate_doubles_single_quotes():
    assert LanceDBVectorBackend._fact_predicate("a'b") == "fact_id = 'a''b'"


def test_update_tier_escapes_fact_id():
    be = object.__new__(LanceDBVectorBackend)  # bypass __init__ — no real DB needed
    be._table = MagicMock()
    be.update_tier("evil' OR '1'='1", "hot")
    where = be._table.update.call_args.kwargs["where"]
    # single quotes doubled -> the payload stays inside the string literal (inert)
    assert where == "fact_id = 'evil'' OR ''1''=''1'"
    # balanced/doubled quotes: the payload cannot terminate the literal early
    assert where.count("'") % 2 == 0
