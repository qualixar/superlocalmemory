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


@pytest.fixture(scope="session")
def image_python(tmp_path_factory):
    """An interpreter with Pillow and ImageHash, for the real image tests.

    ``SLM_TEST_IMAGE_PYTHON`` names a ready one; otherwise a throw-away venv is built once.
    """
    import subprocess

    ready = os.environ.get("SLM_TEST_IMAGE_PYTHON")
    if ready:
        return Path(ready)
    root = tmp_path_factory.mktemp("imgvenv")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(root / "v")], check=True, capture_output=True, timeout=300)
        py = root / "v" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "pillow==12.3.0", "ImageHash==4.3.2"],
                       check=True, capture_output=True, timeout=900)
    except (subprocess.SubprocessError, OSError) as exc:
        pytest.skip(f"could not build an image test environment (no network?): {type(exc).__name__}")
    return py
