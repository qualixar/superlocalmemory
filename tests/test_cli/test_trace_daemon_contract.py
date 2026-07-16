"""The CLI trace command must use the daemon-owned retrieval engine."""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import patch


def _args() -> Namespace:
    return Namespace(query="where is the project store", limit=10, json=True)


def test_trace_uses_daemon_and_preserves_channel_scores(capsys) -> None:
    from superlocalmemory.cli.commands import cmd_trace

    daemon_result = {
        "query": "where is the project store",
        "query_type": "factual",
        "retrieval_time_ms": 42.4,
        "results": [{
            "fact_id": "fact-1",
            "content": "Project Atlas uses SQLite.",
            "score": 0.9,
            "relevance_score": 0.9,
            "channel_scores": {"semantic": 0.8, "entity_graph": 1.0},
        }],
        "no_confident_match": False,
        "score_contract_version": "2",
        "calibration_status": "uncalibrated",
    }
    with patch(
        "superlocalmemory.cli.daemon.is_daemon_running", return_value=True,
    ), patch(
        "superlocalmemory.cli.daemon.daemon_request", return_value=daemon_result,
    ) as request:
        cmd_trace(_args())

    request.assert_called_once_with(
        "POST",
        "/api/v3/recall/trace",
        {"query": "where is the project store", "limit": 10},
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["data"]["results"][0]["channel_scores"] == {
        "semantic": 0.8,
        "entity_graph": 1.0,
    }
