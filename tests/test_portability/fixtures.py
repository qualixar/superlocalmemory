# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Small made-up exports (generic text, no personal names) for every importer's tests."""

from __future__ import annotations

import json


def node(nid, parent, children, role=None, text=None, at=None, **extra):
    message = None
    if role is not None:
        message = {"id": nid, "author": {"role": role}, "create_time": at,
                   "content": {"content_type": extra.pop("content_type", "text"),
                               "parts": [text] if text is not None else []},
                   "metadata": extra.pop("metadata", {})}
    return {"id": nid, "message": message, "parent": parent, "children": children}


def chatgpt_conversation(cid="conv-1", created=1_700_000_000.0):
    """root -> sys -> q1 -> a1(old branch, regenerated) / a1b -> q2 -> a2 (current), plus hidden."""
    mapping = {
        "root": node("root", None, ["sys"]),
        "sys": node("sys", "root", ["q1"], "system", "You are a helpful assistant.", created),
        "q1": node("q1", "sys", ["a1", "a1b"], "user", "I prefer short answers with one example.",
                   created + 10),
        "a1": node("a1", "q1", [], "assistant", "DISCARDED long answer.", created + 20),
        "a1b": node("a1b", "q1", ["q2"], "assistant", "Understood, short answers with an example.",
                    created + 21),
        "q2": node("q2", "a1b", ["tool"], "user", "Our deploy day is Thursday.", created + 30),
        "tool": node("tool", "q2", ["a2"], "tool", "search results", created + 31),
        "a2": node("a2", "tool", ["a3"], "assistant", "Noted: deploys happen on Thursday.",
                   created + 40),
        "a3": node("a3", "a2", ["hid"], "assistant", "   ", created + 41),
        "hid": node("hid", "a3", [], "user", "hidden memory prompt", created + 42,
                    metadata={"is_visually_hidden_from_conversation": True}),
    }
    return {"id": cid, "title": "Example chat", "create_time": created,
            "update_time": created + 50, "current_node": "hid", "mapping": mapping}


def chatgpt_empty(cid="conv-empty"):
    return {"id": cid, "title": "Empty", "create_time": 1_700_100_000.0, "current_node": "r",
            "mapping": {"r": node("r", None, [])}}


def chatgpt_list():
    return [chatgpt_conversation(), chatgpt_empty()]


def claude_list():
    return [{
        "uuid": "11111111-0000-0000-0000-000000000001", "name": "Example chat",
        "created_at": "2026-01-15T10:30:00.000Z", "updated_at": "2026-01-15T10:35:00.000Z",
        "chat_messages": [
            {"uuid": "m-1", "sender": "human", "text": "My team meets on Mondays.",
             "created_at": "2026-01-15T10:30:00.000Z"},
            {"uuid": "m-2", "sender": "assistant", "text": "",
             "content": [{"type": "text", "text": "Monday meetings noted."},
                         {"type": "thinking", "thinking": "hidden"}],
             "created_at": "2026-01-15T10:30:05.000Z"},
            {"uuid": "m-3", "sender": "assistant", "text": "",
             "content": [{"type": "tool_use", "name": "x"}],
             "created_at": "2026-01-15T10:30:06.000Z"},
            {"uuid": "m-4", "sender": "human", "text": "  ", "created_at": "2026-01-15T10:31:00Z"},
        ]}, {"uuid": "22222222-0000-0000-0000-000000000002", "name": "Nothing",
             "created_at": "2026-02-01T00:00:00Z", "chat_messages": []}]


MEM0_ITEMS = [
    {"id": "00000000-0000-0000-0000-000000000001", "memory": "Prefers short answers.",
     "user_id": "user-1", "created_at": "2026-02-01T09:00:00Z", "categories": ["preferences"]},
    {"id": "00000000-0000-0000-0000-000000000002", "memory": "Deploys on Thursdays.",
     "created_at": "2026-02-02T09:00:00Z", "mystery_field": {"x": 1}},
    {"id": "00000000-0000-0000-0000-000000000003", "memory": "   ", "created_at": "2026-02-03T00:00:00Z"},
    {"id": "00000000-0000-0000-0000-000000000004", "memory": "Old fact that expired.",
     "expiration_date": "2020-01-01T00:00:00Z"},
    {"id": "00000000-0000-0000-0000-000000000005", "memory": "Replaced fact.",
     "replaced_by": "00000000-0000-0000-0000-000000000002"},
]


def letta_file():
    return {"created_at": "2026-03-01T08:00:00Z",
            "blocks": [{"label": "human", "value": "The user keeps a weekly review on Fridays.",
                        "limit": 20000, "read_only": False},
                       {"label": "persona", "value": "The agent answers in plain words.", "limit": 20000},
                       {"label": "scratch", "value": "   "}],
            "agents": [{"id": "agent-1", "name": "example-agent",
                        "memory_blocks": [{"label": "human",
                                           "value": "The user keeps a weekly review on Fridays."}],
                        "messages": [
                            {"id": "message-0", "role": "system", "created_at": "2026-03-01T08:00:00Z",
                             "content": [{"type": "text", "text": "System prompt."}]},
                            {"id": "message-1", "role": "user", "created_at": "2026-03-01T08:01:00Z",
                             "content": [{"type": "text", "text": "Remind me about the review."}]},
                            {"id": "message-2", "type": "assistant_message",
                             "created_at": "2026-03-01T08:01:05Z",
                             "content": [{"type": "text", "text": "Your review is on Friday."}]},
                            {"id": "message-3", "role": "tool", "content": []}]}],
            "tools": [{"name": "ignored"}]}


OKF_CONCEPT = """---
type: Reference
title: Release checklist
tags: [ops]
status: draft
---
Run the tests before every release.

## Rollback
Keep the previous build for one week.
"""
OKF_PLAIN = "---\ntype: Note\n---\nBackups run nightly at two.\n"
OKF_INDEX = "# Index\n- [Release checklist](release.md) - how we ship\n"

AGENTS_MD = """# Project rules
Use small functions.

## Tests
Run the suite before commit.

### Slow tests
Mark them slow.

```
# not a heading
```
"""
MEMORY_INDEX = "- [Feedback](feedback.md) - how I like replies\n- [Role](role.md) - what I do\n"
TOPIC_MD = "---\nname: feedback\ndescription: replies\n---\nKeep replies short and direct.\n"
CLAUDE_MEMORY_TXT = """Work context
- [2026-01-10] Runs a small research team on reliability.
- Prefers short answers.

A paragraph about tone: direct and calm,
continued on a second line.
"""


def dumps(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")
