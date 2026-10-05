# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The hosted answer check: one total deadline, nothing sent after it is
switched off, one client however many recalls share it, and one credential-only
memory never vetoes the rest.

Every server here is a local socket this test starts itself. Nothing leaves the
machine and every key is a fake.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import httpx
import pytest

from superlocalmemory.core.judge_keys import JudgeKeyStore
from superlocalmemory.retrieval import answer_check_status as acs
from superlocalmemory.retrieval import jev_judge, jev_transport
from superlocalmemory.retrieval.jev_judge import JEV_ENDPOINTS, JevSufficiencyJudge
from superlocalmemory.retrieval.jev_rerank import CHOICE_KEY
from superlocalmemory.retrieval.judge_recipe import JudgeDocument

FAKE_KEY = "sk-test-" + "m0N9b8V7" * 4
MODEL = JEV_ENDPOINTS["typesafe"][1]
#: A memory that is nothing but a (fake) credential: it redacts to nothing.
CREDENTIAL_ONLY = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"


@pytest.fixture
def key_store(tmp_path) -> JudgeKeyStore:
    store = JudgeKeyStore(slm_home=tmp_path)
    store.set_key("typesafe", FAKE_KEY)
    return store


def _answers(keys, *, choice: str | None = None) -> bytes:
    answers = {k: {"type": "noul", "noul": 0.9} for k in keys}
    if choice is not None:
        probs = {k: (0.9 if k == choice else 0.1 / (len(keys) - 1)) for k in keys}
        answers[CHOICE_KEY] = {"type": "choice", "choice": choice, "probabilities": probs}
    return json.dumps({"model": MODEL, "answers": answers, "usage": {}}).encode()


class _SlowServer:
    """A local HTTP server that answers correctly, but at its own pace."""

    def __init__(self, *, head_delay: float = 0.0, chunk: int = 0,
                 chunk_delay: float = 0.0) -> None:
        self.head_delay, self.chunk, self.chunk_delay = head_delay, chunk, chunk_delay
        self.requests = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn) -> None:
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                data += conn.recv(65536)
            head, _, body = data.partition(b"\r\n\r\n")
            length = int([h.split(b":")[1] for h in head.split(b"\r\n")
                          if h.lower().startswith(b"content-length")][0])
            while len(body) < length:
                body += conn.recv(65536)
            self.requests += 1
            keys = list(json.loads(body)["state"]["memories"])
            payload = _answers(keys)
            time.sleep(self.head_delay)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(payload)}\r\n\r\n".encode())
            step = self.chunk or len(payload)
            for i in range(0, len(payload), step):
                if self._stop.is_set():
                    return
                conn.sendall(payload[i:i + step])
                time.sleep(self.chunk_delay)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self) -> None:
        self._stop.set()
        self._sock.close()


@pytest.fixture
def slow_server(monkeypatch):
    servers = []

    def start(**kw) -> _SlowServer:
        server = _SlowServer(**kw)
        servers.append(server)
        monkeypatch.setitem(JEV_ENDPOINTS, "typesafe",
                            (f"http://127.0.0.1:{server.port}/v1/systemone", MODEL))
        return server

    yield start
    for server in servers:
        server.close()


# ---------------------------------------------------------------------------
# M-4: one total deadline, not one per phase
# ---------------------------------------------------------------------------

#: Scheduling allowance on top of a deadline the product itself enforces
#: (``HostedTransport.post`` waits ``deadline - now`` and abandons the rest).
#: Deliberately generous for a loaded host; each test below keeps its
#: regression several times further away than deadline + slack.
_DEADLINE_SLACK_S = 1.5


class TestOneTotalDeadline:
    def test_a_provider_that_trickles_its_reply_is_cut_off_at_the_deadline(
        self, key_store, slow_server,
    ) -> None:
        # The one-memory reply is 86 bytes; at 2 bytes every 0.2 s that is
        # ~8.6 s of trickle, every single read well inside any per-phase
        # timeout. (It was 8 bytes every 0.15 s: ~1.6 s, too close to the
        # deadline + slack below for the clock to tell the two apart.) That
        # ~8.6 s is what a per-phase deadline costs.
        timeout_s = 0.6
        slow_server(chunk=2, chunk_delay=0.2)
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    timeout_s=timeout_s)
        try:
            started = time.monotonic()
            outcome = judge.assess("q?", [JudgeDocument("the answer")])
            wall = time.monotonic() - started
        finally:
            judge.shutdown()
        assert wall < timeout_s + _DEADLINE_SLACK_S, (
            f"one hosted check took {wall:.1f}s against a {timeout_s}s timeout")
        assert outcome.verdict is None and outcome.status == acs.STATUS_UNAVAILABLE

    def test_the_recalls_deadline_wins_over_the_configured_timeout(
        self, key_store, slow_server,
    ) -> None:
        # The provider answers only after head_delay, well past the recall's
        # deadline + slack. Honouring the configured 5 s timeout instead would
        # get that answer: status JUDGED, not UNAVAILABLE, whatever the clock.
        recall_budget_s, head_delay_s = 0.3, 3.0
        slow_server(head_delay=head_delay_s)
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store, timeout_s=5.0)
        try:
            started = time.monotonic()
            outcome = judge.assess("q?", [JudgeDocument("the answer")],
                                   deadline=time.monotonic() + recall_budget_s)
            wall = time.monotonic() - started
        finally:
            judge.shutdown()
        assert outcome.status == acs.STATUS_UNAVAILABLE
        assert wall < recall_budget_s + _DEADLINE_SLACK_S < head_delay_s, (
            f"took {wall:.1f}s against a {recall_budget_s}s recall deadline")

    def test_a_prompt_provider_is_judged(self, key_store, slow_server) -> None:
        slow_server()
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store, timeout_s=2.0)
        try:
            outcome = judge.assess("q?", [JudgeDocument("the answer")])
        finally:
            judge.shutdown()
        assert outcome.status == acs.STATUS_JUDGED
        assert outcome.verdict.answer_confidence == pytest.approx(0.9)

    def test_a_deadline_already_passed_sends_nothing(self, key_store, slow_server) -> None:
        server = slow_server()
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store)
        try:
            outcome = judge.assess("q?", [JudgeDocument("the answer")],
                                   deadline=time.monotonic() - 1.0)
        finally:
            judge.shutdown()
        assert outcome.verdict is None
        time.sleep(0.1)
        assert server.requests == 0


# ---------------------------------------------------------------------------
# M-6 + MU-3: nothing sent after shutdown, and one client, always closed
# ---------------------------------------------------------------------------

class _CountingClient(httpx.Client):
    made: list = []
    build_delay = 0.0

    def __init__(self, *a, **kw) -> None:
        time.sleep(type(self).build_delay)
        super().__init__(*a, **kw)
        type(self).made.append(self)


@pytest.fixture
def counting_clients(monkeypatch):
    _CountingClient.made = []
    _CountingClient.build_delay = 0.0
    monkeypatch.setattr(jev_transport.httpx, "Client", _CountingClient)
    return _CountingClient


class TestNothingIsSentAfterShutdown:
    def test_a_shutdown_during_key_loading_stops_the_request(
        self, key_store, counting_clients,
    ) -> None:
        sent: list = []
        transport = httpx.MockTransport(lambda r: (sent.append(r), httpx.Response(
            200, content=_answers(list(json.loads(r.content)["state"]["memories"]))))[1])
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    transport=transport)
        real_load = key_store.load

        def load_then_switch_off(provider):
            key = real_load(provider)
            judge.shutdown()          # consent withdrawn while this recall was in flight
            return key

        key_store.load = load_then_switch_off
        outcome = judge.assess("q?", [JudgeDocument("the answer")])
        assert sent == [], "memories were sent after the check was switched off"
        assert outcome.verdict is None
        assert all(c.is_closed for c in counting_clients.made), "a client was left open"


class TestOneClientForEveryRecall:
    def test_two_first_recalls_at_once_build_one_client(
        self, key_store, counting_clients,
    ) -> None:
        counting_clients.build_delay = 0.2
        transport = httpx.MockTransport(lambda r: httpx.Response(
            200, content=_answers(list(json.loads(r.content)["state"]["memories"]))))
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    transport=transport)
        results: list = []
        threads = [threading.Thread(target=lambda: results.append(
            judge.assess("q?", [JudgeDocument("the answer")]))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        judge.shutdown()
        assert [o.status for o in results] == [acs.STATUS_JUDGED] * 2
        assert len(counting_clients.made) == 1, "two clients were built; one leaked"
        assert all(c.is_closed for c in counting_clients.made)


# ---------------------------------------------------------------------------
# F15: one credential-only memory is left out, never sent, never a veto
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self, choose_last: bool = True) -> None:
        self.bodies: list[dict] = []
        self.choose_last = choose_last
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request):
        body = json.loads(request.content)
        self.bodies.append(body)
        keys = list(body["state"]["memories"])
        choice = keys[-1] if CHOICE_KEY in body["questions"] else None
        return httpx.Response(200, content=_answers(keys, choice=choice))


def _docs(texts) -> list[JudgeDocument]:
    return [JudgeDocument(t) for t in texts]


class TestACredentialOnlyMemoryIsLeftOut:
    def test_the_reorder_runs_without_it_and_it_keeps_its_place(self, key_store) -> None:
        recorder = _Recorder()
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    transport=recorder.transport, rerank_k=20)
        texts = ["memory zero", "memory one", "memory two", "memory three",
                 CREDENTIAL_ONLY, "memory five"]
        outcome = judge.rerank_and_judge("which one?", _docs(texts))
        judge.shutdown()
        (body,) = recorder.bodies
        sent = list(body["state"]["memories"].values())
        assert all("ghp_" not in text and "[redacted]" not in text for text in sent)
        assert sent == ["memory zero", "memory one", "memory two", "memory three",
                        "memory five"], "the m-keys must name the remaining memories in order"
        # The provider chose the last key, m4 = "memory five" (original index 5).
        assert outcome.order[0] == 5
        assert outcome.order[4] == 4, "the credential-only memory left its place"
        assert sorted(outcome.order) == list(range(6))
        assert outcome.verdict is not None, "the shown top three were all judged"

    def test_a_shown_memory_that_was_left_out_withholds_the_verdict(self, key_store) -> None:
        """It might be the answer ("what is my token?"); a verdict that never read
        it must not say nothing answers."""
        recorder = _Recorder()
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    transport=recorder.transport, rerank_k=20)
        texts = ["memory zero", CREDENTIAL_ONLY, "memory two", "memory three"]
        outcome = judge.rerank_and_judge("which one?", _docs(texts))
        judge.shutdown()
        assert outcome.order is not None and outcome.order[1] == 1
        assert outcome.verdict is None
        assert outcome.status == acs.STATUS_SKIPPED

    def test_the_plain_check_still_never_sends_it(self, key_store) -> None:
        recorder = _Recorder()
        judge = JevSufficiencyJudge(provider="typesafe", key_store=key_store,
                                    transport=recorder.transport)
        outcome = judge.assess("which one?", _docs(["memory zero", CREDENTIAL_ONLY]))
        judge.shutdown()
        assert recorder.bodies == []
        assert outcome.verdict is None and outcome.status == acs.STATUS_SKIPPED


def test_the_transport_module_is_what_the_judge_uses() -> None:
    assert jev_judge.HostedTransport is jev_transport.HostedTransport
