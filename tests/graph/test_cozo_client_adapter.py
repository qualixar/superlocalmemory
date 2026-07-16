"""PyCozo API compatibility tests that do not require its native wheel."""

from __future__ import annotations

from superlocalmemory.graph.cozo_backend import _CozoClientAdapter


class _LegacyClient:
    def __init__(self) -> None:
        self.imported = None
        self.closed = False

    def run(self, _script: str):
        return {"headers": ["id"], "rows": [["e1"]], "ok": True}

    def import_relations(self, payload):
        self.imported = payload

    def close(self) -> None:
        self.closed = True


class _ParameterizedClient(_LegacyClient):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def run(self, _script: str, params=None):
        self.params = params
        return {"headers": ["id"], "rows": [["e1"]], "ok": True}


def test_legacy_pycozo_rows_and_relation_import_are_normalized() -> None:
    legacy = _LegacyClient()
    adapter = _CozoClientAdapter(legacy)

    result = adapter.run("?[id] := *entity{id}")
    assert result.values.tolist() == [["e1"]]
    assert len(result) == 1

    adapter.put("entity", [{"id": "e1", "name": "Varun"}])
    assert legacy.imported == {
        "entity": {
            "headers": ["id", "name"],
            "rows": [["e1", "Varun"]],
        },
    }

    adapter.close()
    assert legacy.closed


def test_parameterized_queries_are_forwarded_without_string_interpolation() -> None:
    client = _ParameterizedClient()
    adapter = _CozoClientAdapter(client)

    adapter.run("?[id] := *entity{id}, id == $id", {"id": "not-a-query"})

    assert client.params == {"id": "not-a-query"}
