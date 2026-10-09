"""Image operations in the worker: strip metadata, thumbnail, hash, text. Real Pillow in a throw-away venv."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[2] / "src" / "superlocalmemory" / "runtimes" / "multimodal_worker.py"
OPS = WORKER.with_name("media_image_ops.py")

MAKE = r'''
import json, os, random, sys
from PIL import Image, PngImagePlugin
d = sys.argv[1]
def noisy(w, h, seed=1):
    rnd = random.Random(seed)
    img = Image.frombytes("RGB", (w, h), bytes(rnd.getrandbits(8) for _ in range(w * h * 3)))
    return img
def scene(kind):
    from PIL import ImageDraw
    rnd = random.Random(7 if kind == "a" else 99)
    img = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x, y = rnd.randrange(0, 100), rnd.randrange(0, 70)
        draw.rectangle([x, y, x + rnd.randrange(10, 40), y + rnd.randrange(10, 30)],
                       fill=tuple(rnd.randrange(0, 256) for _ in range(3)))
    return img
exif = Image.Exif()
exif[0x010F] = "Acme"; exif[0x0110] = "Cam 1"; exif[0x0112] = 6
exif.get_ifd(0x8769)[0x9003] = "2020:01:02 03:04:05"
gps = exif.get_ifd(0x8825)
gps[1] = "N"; gps[2] = (12.0, 34.0, 56.0); gps[3] = "E"; gps[4] = (78.0, 9.0, 1.0)
img = scene("a").resize((40, 20))
img.save(d + "/gps.jpg", exif=exif)
meta = PngImagePlugin.PngInfo(); meta.add_text("Comment", "secret-note-xyz")
scene("a").save(d + "/text.png", pnginfo=meta)
scene("a").save(d + "/a.png")
scene("a").save(d + "/a_re.jpg", quality=60)
scene("b").save(d + "/b.png")
noisy(4000, 3000).save(d + "/noisy.jpg", quality=90)
scene("a").save(d + "/x.bmp"); scene("a").save(d + "/x.tiff")
scene("a").save(d + "/x.webp"); scene("a").save(d + "/x.gif")
open(d + "/x.svg", "w").write("<svg xmlns='http://www.w3.org/2000/svg'><rect width='4' height='4'/></svg>")
data = open(d + "/a.png", "rb").read(); open(d + "/cut.png", "wb").write(data[: len(data) // 2])
Image.new("1", (10000, 6000)).save(d + "/huge.png")
'''

INSPECT = r'''
import json, sys
from PIL import Image
out = {}
for p in sys.argv[1:]:
    with Image.open(p) as im:
        im.load()
        out[p] = {"size": list(im.size), "format": im.format, "exif": len(im.getexif()),
                  "gps": bool(im.getexif().get_ifd(0x8825)), "info": sorted(k for k in im.info if k != "icc_profile")}
print(json.dumps(out))
'''


def run_py(py, code, *args):
    done = subprocess.run([str(py), "-c", code, *map(str, args)], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture(scope="session")
def samples(image_python, tmp_path_factory):
    d = tmp_path_factory.mktemp("samples")
    run_py(image_python, MAKE, d)
    return d


class Proc:
    def __init__(self, python, cwd):
        env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        self.p = subprocess.Popen([str(python), "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, cwd=str(cwd), env=env)
        self.n = 0

    def ask(self, cmd, **kw):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "cmd": cmd, **kw}) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        assert reply["id"] == self.n
        return reply

    def close(self):
        self.p.kill()
        self.p.wait()
        for f in (self.p.stdin, self.p.stdout, self.p.stderr):
            f.close()


@pytest.fixture
def real(image_python, tmp_path):
    w = Proc(image_python, tmp_path)
    yield w
    w.close()


@pytest.fixture
def plain(tmp_path):
    w = Proc(sys.executable, tmp_path)
    yield w
    w.close()


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


def prep(w, src, out):
    return w.ask("prepare_image", path=str(src), out_dir=str(out))


def test_ops_module_imports_nothing_from_the_package():
    text = OPS.read_text(encoding="utf-8")
    assert "superlocalmemory" not in text.split('"""')[2].replace("SuperLocalMemory", "")
    assert len(text.splitlines()) < 300


def test_jpeg_location_and_all_metadata_are_stripped(real, samples, out_dir, image_python):
    r = prep(real, samples / "gps.jpg", out_dir)
    assert r["ok"], r
    assert (r["width"], r["height"]) == (20, 40)  # orientation 6 turns 40x20 upright
    assert r["mime"] == "image/jpeg" and r["stored_ext"] == ".jpg"
    assert r["exif"] == {"DateTimeOriginal": "2020:01:02 03:04:05", "Make": "Acme", "Model": "Cam 1",
                         "Orientation": 6}
    assert "GPS" not in json.dumps(r).upper().replace("GPS.JPG", "")
    seen = json.loads(run_py(image_python, INSPECT, r["stored_path"]))[r["stored_path"]]
    assert seen["exif"] == 0 and seen["gps"] is False and seen["size"] == [20, 40]
    raw = Path(r["stored_path"]).read_bytes()
    assert b"Exif" not in raw and b"Acme" not in raw


def test_png_text_chunks_are_stripped(real, samples, out_dir, image_python):
    r = prep(real, samples / "text.png", out_dir)
    assert r["ok"] and r["mime"] == "image/png" and r["stored_ext"] == ".png"
    assert b"secret-note-xyz" not in Path(r["stored_path"]).read_bytes()
    seen = json.loads(run_py(image_python, INSPECT, r["stored_path"]))[r["stored_path"]]
    assert "Comment" not in seen["info"] and r["exif"] == {}


def test_webp_and_gif_are_accepted_gif_becomes_png(real, samples, out_dir):
    w = prep(real, samples / "x.webp", out_dir)
    g = prep(real, samples / "x.gif", out_dir)
    assert w["ok"] and w["mime"] == "image/webp" and w["stored_ext"] == ".webp"
    assert g["ok"] and g["stored_ext"] == ".png" and g["stored_path"].endswith(".png")


def test_thumbnail_is_small_webp(real, samples, out_dir, image_python):
    r = prep(real, samples / "noisy.jpg", out_dir)
    assert r["ok"] and (r["width"], r["height"]) == (4000, 3000)
    thumb = Path(r["thumb_path"])
    assert thumb.stat().st_size <= 32_768 and thumb.read_bytes()[8:12] == b"WEBP"
    seen = json.loads(run_py(image_python, INSPECT, thumb))[str(thumb)]
    assert max(seen["size"]) <= 320


def test_phash_is_stable_and_tells_images_apart(real, samples, out_dir, tmp_path):
    first = prep(real, samples / "a.png", out_dir)["phash"]
    again = Proc(real.p.args[0], tmp_path)
    try:
        assert prep(again, samples / "a.png", out_dir)["phash"] == first
    finally:
        again.close()
    assert len(first) == 16 and int(first, 16) >= 0
    dist = lambda x, y: bin(int(x, 16) ^ int(y, 16)).count("1")  # noqa: E731
    assert dist(first, prep(real, samples / "a_re.jpg", out_dir)["phash"]) <= 4
    assert dist(first, prep(real, samples / "b.png", out_dir)["phash"]) > 4


@pytest.mark.parametrize("name", ["x.bmp", "x.tiff", "x.svg", "cut.png", "huge.png"])
def test_refusals_leave_nothing_behind(real, samples, out_dir, name):
    r = prep(real, samples / name, out_dir)
    assert r["ok"] is False and str(samples) not in json.dumps(r)
    assert list(out_dir.iterdir()) == []
    assert real.ask("ping")["ok"]


def test_directory_missing_and_oversize_are_refused(real, samples, out_dir, tmp_path):
    big = tmp_path / "big.png"
    with open(big, "wb") as fh:
        fh.truncate(26 * 1024 * 1024)
    for src in (tmp_path, tmp_path / "nope.png", big):
        assert prep(real, src, out_dir)["ok"] is False
    assert real.ask("prepare_image", path=str(samples / "a.png"), out_dir=str(tmp_path / "missing"))["ok"] is False
    assert real.ask("prepare_image", path=str(samples / "a.png"), out_dir=str(samples / "a.png"))["ok"] is False
    assert list(out_dir.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_written_files_are_private_with_random_names(real, samples, out_dir):
    r1, r2 = prep(real, samples / "a.png", out_dir), prep(real, samples / "a.png", out_dir)
    paths = [r1["stored_path"], r1["thumb_path"], r2["stored_path"], r2["thumb_path"]]
    assert len(set(paths)) == 4
    for p in paths:
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600 and Path(p).parent == out_dir
        assert "a.png" not in Path(p).name


def test_fake_ocr_reads_the_sidecar(real, samples, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"x")
    assert real.ask("load", model="fake:768")["ok"]
    assert real.ask("ocr_image", path=str(img)) == {"id": 2, "ok": True, "engine": "fake", "text": ""}
    (tmp_path / "shot.png.ocr.txt").write_text("hello text", encoding="utf-8")
    r = real.ask("ocr_image", path=str(img))
    assert r["engine"] == "fake" and r["text"] == "hello text"
    assert real.ask("ocr_image", path=str(tmp_path / "gone.png"))["ok"] is False


def test_without_an_engine_the_answer_is_none(real, samples):
    r = real.ask("ocr_image", path=str(samples / "a.png"))
    assert r["ok"] and r["engine"] == "none" and r["text"] == ""


def test_without_pillow_the_worker_says_so_and_keeps_serving(plain, samples, out_dir):
    plain.ask("load", model="fake:768")
    r = prep(plain, samples / "a.png", out_dir)
    assert r["ok"] is False and r["error"] == "pillow unavailable"
    assert plain.ask("ping")["ok"] and plain.ask("embed_text", texts=["x"])["ok"]
    assert plain.ask("ocr_image", path=str(samples / "a.png"))["engine"] in ("fake", "none")


def test_client_methods(client_factory, stub_env, samples, out_dir, image_python, monkeypatch):
    monkeypatch.setattr(type(stub_env), "python", lambda self: image_python)
    c = client_factory()
    r = c.prepare_image(samples / "gps.jpg", out_dir)
    assert r["width"] == 20 and "ok" not in r and "id" not in r
    assert c.ocr_image(samples / "a.png")["engine"] == "fake"
    from superlocalmemory.runtimes.worker_client import MediaWorkerError, MediaWorkerWarming
    with pytest.raises(MediaWorkerError):
        c.prepare_image(samples / "x.bmp", out_dir)
    cold = client_factory()
    with pytest.raises(MediaWorkerWarming):
        cold.prepare_image(samples / "a.png", out_dir, wait_cold=False)
    with pytest.raises(MediaWorkerWarming):
        cold.ocr_image(samples / "a.png", wait_cold=False)
