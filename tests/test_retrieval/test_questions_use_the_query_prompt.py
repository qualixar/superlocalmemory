"""A question is embedded with the embedder's question prompt when it has one."""

from __future__ import annotations

from unittest.mock import MagicMock

from superlocalmemory.retrieval.channel_registry import ChannelRegistry
from superlocalmemory.retrieval.query_embedding import QueryEmbedder, embed_as_query


class WithPrompts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed(self, text):
        self.calls.append(("document", text))
        return [1.0]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [2.0]


class Plain:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text):
        self.calls.append(text)
        return [3.0]


def test_embed_as_query_prefers_the_question_prompt():
    emb = WithPrompts()
    assert embed_as_query(emb, "q") == [2.0]
    assert emb.calls == [("query", "q")]


def test_every_other_embedder_is_called_exactly_as_before():
    plain = Plain()
    assert embed_as_query(plain, "q") == [3.0] and plain.calls == ["q"]
    mock = MagicMock()
    mock.embed.return_value = [4.0]
    assert embed_as_query(mock, "q") == [4.0]        # a mock does not pretend to have embed_query
    mock.embed.assert_called_once_with("q")


def test_the_recall_query_embedder_uses_it():
    emb = WithPrompts()
    vector, status = QueryEmbedder(lambda: emb).embed("what did I save?", wait_seconds=1.0)
    assert vector == [2.0] and status is None and emb.calls == [("query", "what did I save?")]


def test_channels_that_need_a_vector_get_the_question_vector():
    seen = []

    class Channel:
        def search(self, vector, profile_id, top_k):
            seen.append(vector)
            return [("f1", 0.9)]

    registry = ChannelRegistry()
    registry.register_channel("semantic", Channel(), needs_embedding=True)
    assert registry.run_all("q", "default", embedder=WithPrompts()) == {"semantic": [("f1", 0.9)]}
    assert seen == [[2.0]]
