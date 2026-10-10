"""The media worker script, run for real as a separate process in fake mode."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "runtimes" / "multimodal_worker.py"


class Proc:
    def __init__(self, tmp_path, env_extra=None):
        import os
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        env.update(env_extra or {})
        self.p = subprocess.Popen([sys.executable, "-I", str(WORKER)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  cwd=str(tmp_path), env=env)
        self.n = 0

    def ask(self, cmd, **kw):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "cmd": cmd, **kw}) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        assert reply["id"] == self.n
        return reply


@pytest.fixture
def worker(tmp_path):
    w = Proc(tmp_path)
    yield w
    w.p.kill()
    w.p.wait()
    for f in (w.p.stdin, w.p.stdout, w.p.stderr):
        f.close()


@pytest.fixture
def loaded(worker):
    assert worker.ask("load", model="fake:768")["ok"]
    return worker


def norm(v):
    return sum(x * x for x in v) ** 0.5


def test_script_imports_nothing_from_the_package():
    text = WORKER.read_text(encoding="utf-8")
    assert "superlocalmemory" not in text.replace("SuperLocalMemory", "").split('"""')[2]


def test_ping_before_and_after_load(worker):
    first = worker.ask("ping")
    assert first["ok"] and first["loaded"] is False and first["device"] == "cpu"
    assert worker.ask("load", model="fake:768") == {"id": 2, "ok": True, "dim": 768}
    after = worker.ask("ping")
    assert after["loaded"] is True and after["model"] == "fake:768"


def test_text_vectors_are_768_unit_deterministic_and_distinct(loaded, tmp_path):
    a = loaded.ask("embed_text", texts=["alpha", "beta"], prompt="Document")["vectors"]
    assert len(a) == 2 and all(len(v) == 768 for v in a)
    assert all(abs(norm(v) - 1) < 1e-6 for v in a)
    assert a[0] != a[1]
    again = Proc(tmp_path)
    again.ask("load", model="fake:768")
    try:
        assert again.ask("embed_text", texts=["alpha", "beta"], prompt="SearchQuery")["vectors"] == a
    finally:
        again.p.kill()
        again.p.communicate()


def test_image_vector_from_file_bytes(loaded, tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"pixels")
    g = tmp_path / "b.bin"
    g.write_bytes(b"other pixels")
    vecs = loaded.ask("embed_image", paths=[str(f), str(g)])["vectors"]
    assert len(vecs) == 2 and len(vecs[0]) == 768 and vecs[0] != vecs[1]
    assert abs(norm(vecs[0]) - 1) < 1e-6


def test_embedding_before_load_is_refused(worker):
    assert worker.ask("embed_text", texts=["x"])["ok"] is False


def test_limits_are_refused_without_echoing_content(loaded, tmp_path):
    assert not loaded.ask("embed_text", texts=["x"] * 65, prompt="Document")["ok"]
    assert not loaded.ask("embed_text", texts=["y" * 8001], prompt="Document")["ok"]
    assert not loaded.ask("embed_text", texts=["x"], prompt="Other")["ok"]
    secret = tmp_path / "SECRET-NAME-123.txt"
    big = tmp_path / "big.bin"
    with open(big, "wb") as fh:
        fh.truncate(26 * 1024 * 1024)
    for paths in ([str(secret)], [str(tmp_path)], [str(big)], [str(big)] * 17, "nope"):
        reply = loaded.ask("embed_image", paths=paths)
        assert reply["ok"] is False
        assert "SECRET-NAME" not in json.dumps(reply)
    secret.write_text("TOP-SECRET-CONTENT")
    assert loaded.ask("embed_image", paths=[str(secret)])["ok"]


def test_unknown_command_and_sleep_only_in_fake_mode(worker):
    assert worker.ask("nope")["ok"] is False
    assert worker.ask("sleep", seconds=0)["ok"] is False  # not loaded in fake mode yet
    worker.ask("load", model="fake:768")
    assert worker.ask("sleep", seconds=0)["ok"] is True


def test_garbage_line_gets_an_error_not_a_crash(worker):
    worker.p.stdin.write("not json\n")
    worker.p.stdin.flush()
    assert json.loads(worker.p.stdout.readline())["ok"] is False
    assert worker.ask("ping")["ok"]


def test_quit_exits_zero(worker):
    assert worker.ask("quit")["ok"]
    assert worker.p.wait(timeout=10) == 0


def test_eof_on_stdin_exits(worker):
    worker.p.stdin.close()
    assert worker.p.wait(timeout=10) == 0


def test_real_model_ids_need_the_managed_env(worker):
    reply = worker.ask("load", model="some/not-installed-model")
    assert reply["ok"] is False and "traceback" not in json.dumps(reply).lower()
