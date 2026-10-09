import os
import sys
from pathlib import Path

import pytest


class StubEnv:
    """The few things the worker client needs from a managed environment."""

    def __init__(self, root: Path, state: str = "ready"):
        self.root = Path(root)
        self._state = state

    def python(self) -> Path:
        return Path(sys.executable)

    def weights_dir(self) -> Path:
        return self.root / "weights"

    def status(self):
        from superlocalmemory.runtimes.managed_env import EnvStatus
        return EnvStatus(state=self._state, progress=1.0, step="")


@pytest.fixture
def stub_env(tmp_path):
    return StubEnv(tmp_path)


@pytest.fixture
def client_factory(stub_env):
    made = []

    def make(**kw):
        from superlocalmemory.runtimes.worker_client import MediaWorkerClient
        kw.setdefault("model_id", "fake:768")
        kw.setdefault("revision", "")
        c = MediaWorkerClient(stub_env, **kw)
        made.append(c)
        return c

    yield make
    for c in made:
        c.stop()
