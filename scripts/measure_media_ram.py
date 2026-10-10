#!/usr/bin/env python3
"""Measure the picture worker's memory for each loadout (the RAM gate).

Starts the worker with the full loadout and with the text-only loadout (role
"text"), and reads its resident size (RSS, children included) idle after the
load, during N picture embeds, and during batch-16 long-text embeds. Prints one
JSON object ``{loadout: {idle_mb, peak_mb, ...}}`` and, as the last line, a
verdict against ``--budget-mb``. Runs on any OS (needs psutil)::

    python scripts/measure_media_ram.py [--images 32] [--budget-mb 4500]
    python scripts/measure_media_ram.py --fake      # the fake model, for tests

Picture samples are made with Pillow when it is available (here, or in the media
environment); otherwise the picture step is skipped with a note. Exit code: 0
within budget, 1 over budget, 2 when nothing could be measured.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil  # noqa: E402

MB = 1024 * 1024
BATCH = 16
LOADOUTS = {"full": "", "text": "text"}
_PNG_SNIPPET = (
    "import sys\nfrom PIL import Image\nout=sys.argv[1]\nn=int(sys.argv[2])\n"
    "for i in range(n):\n"
    "    img=Image.effect_noise((896,896),64+i).convert('RGB')\n"
    "    img.save('%s/pic-%d.png'%(out,i))\n")


def rss_mb(pid: int) -> float:
    """Memory of a process and its children, in MB; 0 when it is gone.

    The shared reader: physical footprint on macOS (RSS under-reports there once memory
    is compressed), RSS elsewhere.
    """
    from superlocalmemory.infra.proc_memory import tree_memory_mb

    return tree_memory_mb(pid)


class Sampler:
    """Watches one process in a thread and remembers the largest size seen."""

    def __init__(self, pid_of, interval_s: float = 0.05) -> None:
        self._pid_of, self._interval = pid_of, interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample()
            time.sleep(self._interval)

    def sample(self) -> float:
        pid = self._pid_of()
        now = rss_mb(pid) if pid else 0.0
        with self._lock:
            self._peak = max(self._peak, now)
        return now

    def reset(self) -> None:
        with self._lock:
            self._peak = 0.0

    @property
    def peak(self) -> float:
        self.sample()
        with self._lock:
            return self._peak

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(2)


def make_pictures(directory: Path, count: int, python: str | None, fake: bool) -> tuple[list[Path], str]:
    """Sample pictures and a note; an empty list means the picture step is skipped."""
    for interpreter in filter(None, (sys.executable, python)):
        done = subprocess.run([interpreter, "-I", "-c", _PNG_SNIPPET, str(directory), str(count)],
                              capture_output=True, timeout=300)
        if done.returncode == 0:
            return sorted(directory.glob("pic-*.png")), ""
    if fake:  # the fake model only hashes bytes
        paths = []
        for i in range(count):
            path = directory / f"pic-{i}.bin"
            path.write_bytes(os.urandom(2048))
            paths.append(path)
        return paths, "Pillow not found; used raw sample files (fake model only)"
    return [], "Pillow not found in this Python or the media environment; picture step skipped"


def long_texts(count: int = BATCH) -> list[str]:
    sentence = "The quarterly report covers revenue, hiring plans, and the migration schedule for every region. "
    # As long as the worker accepts (8,000 characters).
    return [(f"Document {i}. " + sentence * 80)[:7900] for i in range(count)]


def measure(client: Any, pictures: list[Path], rounds: int) -> dict[str, Any]:
    """Load, settle, then run the picture and long-text work; sizes in MB."""
    result: dict[str, Any] = {"picture_peak_mb": None, "text_peak_mb": None}
    with Sampler(lambda: client.pid) as sampler:
        client.embed_texts(["warm up"], prompt="Document")  # first request loads the model
        time.sleep(0.5)
        result["idle_mb"] = round(sampler.sample(), 1)
        if pictures and client.role != "text":
            sampler.reset()
            for i in range(0, len(pictures), BATCH):
                client.embed_images(pictures[i:i + BATCH])
            result["picture_peak_mb"] = round(sampler.peak, 1)
        sampler.reset()
        texts = long_texts()
        for _ in range(rounds):
            client.embed_texts(texts, prompt="Document")
        result["text_peak_mb"] = round(sampler.peak, 1)
        result["peak_mb"] = round(max(sampler.peak, result["idle_mb"], result["picture_peak_mb"] or 0,
                                      result["text_peak_mb"] or 0), 1)
    return result


class _FakeEnv:
    """Just enough of the managed environment for the fake model."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def python(self) -> str:
        return sys.executable

    def weights_dir(self) -> Path:
        return self.root / "weights"


def _real_env() -> tuple[Any, str, str]:
    from superlocalmemory.runtimes import media_models
    from superlocalmemory.runtimes.media_env import media_env

    env = media_env()
    if env.status().state != "ready":
        raise SystemExit("The media environment is not installed. Run: slm media enable")
    return env, media_models.EG2_REPO, media_models.EG2_REVISION


def run(args: argparse.Namespace) -> dict[str, Any]:
    from superlocalmemory.runtimes.worker_client import MediaWorkerClient

    scratch = Path(tempfile.mkdtemp(prefix="slm-ram-"))
    try:
        if args.fake:
            os.environ["SLM_TEST_ISOLATION"] = "1"
            env, model, revision = _FakeEnv(scratch), "fake:768", ""
        else:
            env, model, revision = _real_env()
            model = args.model or model
        pictures, note = make_pictures(scratch, args.images, str(env.python()), args.fake)
        report: dict[str, Any] = {}
        for name in args.loadouts:
            client = MediaWorkerClient(env, model_id=model, revision=revision, role=LOADOUTS[name],
                                       idle_s=3600.0, rss_limit_mb=0, request_timeout_s=900.0,
                                       load_timeout_s=900.0)
            try:
                report[name] = measure(client, pictures, args.text_rounds)
            finally:
                client.stop()
            if note:
                report[name]["note"] = note
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def verdict(report: dict[str, Any], budget_mb: float) -> tuple[bool, str]:
    worst_name, worst = max(report.items(), key=lambda kv: kv[1]["peak_mb"])
    ok = worst["peak_mb"] <= budget_mb
    return ok, (f"{'PASS' if ok else 'FAIL'}: worst peak {worst['peak_mb']:.0f} MB ({worst_name}) "
                f"{'within' if ok else 'over'} the {budget_mb:.0f} MB budget")


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--fake", action="store_true", help="use the fake model (for tests)")
    p.add_argument("--images", type=int, default=32, help="pictures to embed (default 32)")
    p.add_argument("--text-rounds", type=int, default=3, help="batch-16 long-text rounds (default 3)")
    p.add_argument("--budget-mb", type=float, default=4500.0, help="memory budget for the verdict")
    p.add_argument("--model", default="", help="model repo id (default: the picture model)")
    p.add_argument("--loadouts", nargs="+", choices=sorted(LOADOUTS), default=["full", "text"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    report = run(args)
    if not report:
        print("nothing was measured", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    ok, line = verdict(report, args.budget_mb)
    print(line)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
