import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def pdf_python(tmp_path_factory):
    """An interpreter with pypdfium2 and Pillow, for the real PDF tests.

    ``SLM_TEST_PDF_PYTHON`` names a ready one; otherwise a throw-away venv is built once.
    """
    ready = os.environ.get("SLM_TEST_PDF_PYTHON")
    if ready:
        return Path(ready)
    root = tmp_path_factory.mktemp("pdfvenv")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(root / "v")], check=True, capture_output=True, timeout=300)
        py = root / "v" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(py), "-m", "pip", "install", "-q", "pypdfium2==5.14.0", "pillow==12.3.0"],
                       check=True, capture_output=True, timeout=900)
    except (subprocess.SubprocessError, OSError) as exc:
        pytest.skip(f"could not build a PDF test environment (no network?): {type(exc).__name__}")
    return py
