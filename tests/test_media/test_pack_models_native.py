# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The two real models through their native ``transformers`` classes (WS-8). NOT run in a normal build.

These need the pinned weights on this machine, so they run only when the folders are named:

    SLM_TEST_PACK_CAPTIONS_DIR=<folder with florence-community/Florence-2-base at its pinned revision>
    SLM_TEST_PACK_DOCUMENTS_DIR=<folder with PaddlePaddle/PaddleOCR-VL-1.6 at its pinned revision>
    SLM_TEST_PACK_PYTHON=<python of the managed media environment, transformers 5.19.0>

and are skipped otherwise. What they check is exactly what the build could not: that the native classes load
these weights without remote code, and what comes out. UNVERIFIED until the lead runs them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from superlocalmemory.runtimes import media_packs as mp
from superlocalmemory.runtimes import pack_models

pytestmark = pytest.mark.native

PYTHON = os.environ.get("SLM_TEST_PACK_PYTHON") or sys.executable
RUNNER = """
import json, sys
sys.path.insert(0, sys.argv[1])
import pack_models as pm
role, model_dir, image = sys.argv[2:5]
if role == "caption":
    out = {"caption": pm.caption(pm.load_caption(model_dir, "cpu"), image)}
else:
    out = pm.parse_document(pm.load_document(model_dir, "cpu"), image)
print(json.dumps(out))
"""


def _folder(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        pytest.skip(f"{name} is not set to a folder with the model files")
    return Path(value)


def _picture(tmp_path: Path, text: str) -> Path:
    PIL = pytest.importorskip("PIL.Image")
    from PIL import ImageDraw

    img = PIL.new("RGB", (640, 200), "white")
    ImageDraw.Draw(img).text((20, 80), text, fill="black")
    path = tmp_path / "page.png"
    img.save(path)
    return path


def _run(role: str, model_dir: Path, image: Path) -> dict:
    here = Path(pack_models.__file__).parent
    done = subprocess.run([PYTHON, "-I", "-c", RUNNER, str(here), role, str(model_dir), str(image)],
                          capture_output=True, text=True, timeout=1800, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_florence_loads_natively_and_describes_a_picture(tmp_path):
    folder = _folder("SLM_TEST_PACK_CAPTIONS_DIR")
    out = _run("caption", folder, _picture(tmp_path, "INVOICE 42"))
    assert isinstance(out["caption"], str) and 0 < len(out["caption"]) <= pack_models.MAX_CAPTION_CHARS


def test_paddleocr_vl_1_6_loads_in_the_native_class_and_reads_a_page(tmp_path):
    folder = _folder("SLM_TEST_PACK_DOCUMENTS_DIR")
    out = _run("document", folder, _picture(tmp_path, "INVOICE 42 TOTAL DUE"))
    assert out["engine"] == "paddleocr_vl" and "42" in out["text"]
    assert out["regions"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_the_pinned_revisions_are_what_the_folders_must_hold():
    assert mp.CAPTIONS.revision == "00921df66db728a9ceb750f5eca43e5c203a2051"
    assert len(mp.DOCUMENTS.revision) == 40
