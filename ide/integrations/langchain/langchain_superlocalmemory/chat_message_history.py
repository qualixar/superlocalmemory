#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V3 - LangChain Chat Message History

Implements LangChain's BaseChatMessageHistory backed by SuperLocalMemory V3's
local SQLite data root. Messages stay in your local SLM data root (optional SLM
providers, connectors, backup, and downloads have separate network behavior).

Usage:
    from langchain_superlocalmemory import SuperLocalMemoryChatMessageHistory

    history = SuperLocalMemoryChatMessageHistory(session_id="my-session")
    history.add_messages([HumanMessage(content="Hello")])
    print(history.messages)
"""
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)


# ---------------------------------------------------------------------------
def _data_root() -> Path:
    value = (
        os.environ.get("SLM_DATA_DIR")
        or os.environ.get("SL_MEMORY_PATH")
        or os.environ.get("SLM_HOME")
    )
    return Path(value).expanduser() if value else Path.home() / ".superlocalmemory"


# ``SLM_INSTALL_DIR`` is legacy installer metadata only. It is deliberately
# never added to ``sys.path``; the separately installed superlocalmemory package
# is the executable contract.

def _session_storage_id(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return f"langchain:{digest}"


# ---------------------------------------------------------------------------
# Message (de)serialization helpers
# ---------------------------------------------------------------------------

# Map from LangChain message type string to the concrete class used for
# deserialization. LangChain's own `messages_from_dict` handles this, but we
# keep a lookup for the fallback path in case the dict format diverges.

_MESSAGE_TYPE_MAP = {
    "human": HumanMessage,
    "ai": AIMessage,
    "system": SystemMessage,
    "function": FunctionMessage,
    "tool": ToolMessage,
}


def _serialize_message(message: BaseMessage) -> str:
    """Serialize a LangChain BaseMessage to a JSON string for SLM storage."""
    return json.dumps(message_to_dict(message), ensure_ascii=False)


def _deserialize_messages(dicts: List[dict]) -> List[BaseMessage]:
    """Deserialize a list of message dicts back to BaseMessage instances.

    Uses LangChain's ``messages_from_dict`` which handles all known message
    types including ``additional_kwargs``.
    """
    return messages_from_dict(dicts)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SuperLocalMemoryChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat message history backed by SuperLocalMemory V3.

    Each message is stored as an individual memory entry in the SLM SQLite
    database, tagged with the session ID for isolation.  The data lives in your
    local SLM data root and is queryable via any SLM access method (MCP, CLI,
    Skills, REST API).

    Parameters
    ----------
    session_id : str
        Unique identifier for the conversation session.  Messages from
        different session IDs are completely isolated.
    db_path : str or None
        Path to the SQLite database file.  Defaults to
        ``~/.superlocalmemory/memory.db``.
    """

    # Tag prefix used to isolate LangChain session messages inside SLM.
    _TAG_PREFIX = "langchain:session:"

    def __init__(self, session_id: str, db_path: Optional[str] = None) -> None:
        self.session_id = session_id
        self.db_path = db_path

        from langchain_superlocalmemory._v3_store import V3ChatStore

        store_path = Path(db_path) if db_path else _data_root() / "memory.db"
        self._store = V3ChatStore(store_path)
        self._storage_session_id = _session_storage_id(session_id)

    # -- property: messages ------------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore[override]
        """Return all messages for this session, ordered chronologically."""
        session_memories = self._store.list_session(self._storage_session_id)

        # Deserialize each memory's content back to a BaseMessage.
        message_dicts: List[dict] = []
        for mem in session_memories:
            try:
                parsed = json.loads(mem["content"])
                message_dicts.append(parsed)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip malformed entries silently -- they may be non-LangChain
                # memories that happen to share the tag pattern.
                continue

        if not message_dicts:
            return []

        return _deserialize_messages(message_dicts)

    # -- add_messages ------------------------------------------------------

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist messages through SuperLocalMemory V3 canonical ingestion.

        Each message becomes a separate memory entry tagged with the session
        identifier.  Importance is set to 3 (lower than typical user
        memories at 5) so LangChain history does not crowd out higher-value
        entries in search results.
        """
        session_tag = f"{self._TAG_PREFIX}{self.session_id}"

        for message in messages:
            serialized = _serialize_message(message)
            self._store.add(
                serialized,
                session_id=self._storage_session_id,
                metadata={
                    "integration": "langchain",
                    "chat_session_id": self.session_id,
                    "tags": ["langchain", session_tag],
                    "importance": 3,
                    "project_name": "langchain",
                },
            )

    # -- clear -------------------------------------------------------------

    def clear(self) -> None:
        """Remove all messages for this session from the store."""
        for mem in self._store.list_session(self._storage_session_id):
            self._store.delete(mem["id"])
