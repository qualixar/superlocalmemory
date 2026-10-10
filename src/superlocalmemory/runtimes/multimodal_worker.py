# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Image-and-text embedding worker: one process, one model, JSON lines.

Run by file path inside the managed environment (``python -I multimodal_worker.py``),
which does not have superlocalmemory installed: this file imports nothing from it
and, in fake mode, needs only the standard library.

Every request may carry an ``"id"``; the reply echoes it.

  {"cmd": "ping"}   -> {"ok": true, "loaded": bool, "model": "...", "device": "cpu|mps"}
  {"cmd": "load", "model": "<repo | folder | fake:768>", "revision": "...",
   "hf_home": "...", "device": "auto|cpu", "role": "|image|text", "max_pixels": 0}
                    -> {"ok": true, "dim": 768} | {"ok": false, "error": "..."}
  {"cmd": "embed_text", "texts": [str, ...<=64], "prompt": "SearchQuery|Document"}
                    -> {"ok": true, "vectors": [[...], ...]}
  {"cmd": "embed_image", "paths": [str, ...<=16]}
                    -> {"ok": true, "vectors": [[...], ...]}
  {"cmd": "prepare_image", "path": "...", "out_dir": "..."}
                    -> {"ok": true, "mime", "width", "height", "exif", "phash",
                        "stored_path", "stored_ext", "thumb_path"}
  {"cmd": "ocr_image", "path": "..."}
                    -> {"ok": true, "engine": "apple_vision|rapidocr|none|fake", "text": "..."}
  {"cmd": "quit"}

Role ``image`` loads a vision-only model: it embeds images, never text. Role ``text`` loads
only the text tower of a text+image model: it embeds text, never images. ``max_pixels`` (> 0)
shrinks larger pictures before they are embedded. A fake model
does both whatever the role.

Fake mode (``SLM_MEDIA_WORKER_FAKE=1`` or a model id ``fake:<dim>``) answers with
deterministic unit vectors derived from a hash of the input, so the protocol can be
exercised anywhere. ``sleep`` (fake mode only) stands in for a hung model.

stdout carries the protocol and nothing else. Errors never contain file contents.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import signal
import sys
import threading
import time

# Set before any torch import: CPU only, no tokenizer thread storms.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_TEXTS = 64
MAX_TEXT_CHARS = 8_000
MAX_PATHS = 16
# One forward pass holds at most this much: measured on a 24 GB Mac, batch-16 long texts
# peaked at 12 GB and batch-16 pictures at 7.3 GB, against ~3.9 GB for one picture.
TEXT_PASS_CHARS = 4_000
TEXT_PASS_MAX = 8
IMAGE_PASS = 2
MAX_FILE_BYTES = 25 * 1024 * 1024
PROMPTS = ("SearchQuery", "Document")
FAKE_DIM = 768
_WATCHDOG_S = 2.0

_STATE: dict = {"model": None, "name": "", "device": "cpu", "fake": False, "dim": 0}


class _Invalid(Exception):
    """A request that breaks a limit; the text is safe to send back."""


# -- fake mode -----------------------------------------------------------------

def _fake_vector(data: bytes, dim: int) -> list[float]:
    digest = hashlib.sha256(data).digest()
    raw: list[float] = []
    block = 0
    while len(raw) < dim:
        raw.extend(b - 127.5 for b in hashlib.sha256(digest + block.to_bytes(4, "big")).digest())
        block += 1
    raw = raw[:dim]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# -- real mode -----------------------------------------------------------------

def _pick_device(wanted: str) -> str:
    if wanted != "auto" or sys.platform != "darwin":
        return "cpu"
    try:
        torch = importlib.import_module("torch")
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - no usable torch backend means CPU
        return "cpu"


def _load_real(model: str, revision: str, hf_home: str, device: str, role: str = "", max_pixels: int = 0) -> int:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        if os.path.isdir(hf_home) and os.listdir(hf_home):
            os.environ["HF_HUB_OFFLINE"] = "1"
    # Loaded by name: this file runs only inside the managed environment, never in the daemon.
    torch = importlib.import_module("torch")
    SentenceTransformer = importlib.import_module("sentence_transformers").SentenceTransformer

    chosen = _pick_device(device)
    kwargs = {"revision": revision} if revision and not os.path.isdir(model) else {}
    # The sound tower is never loaded; the text-only loadout drops the picture tower as well.
    towers = {"vision_config": None, "audio_config": None} if role == "text" else {"audio_config": None}
    # CPU with sdpa attention and fp32 is mandatory: default kwargs were ~40x slower.
    loaded = SentenceTransformer(model, device=chosen, **kwargs, config_kwargs=towers,
                                 model_kwargs={"attn_implementation": "sdpa", "dtype": torch.float32})
    _STATE.update(model=loaded, device=chosen, text_only=role == "text", max_pixels=max_pixels,
                  dim=int(loaded.get_sentence_embedding_dimension() or 0))
    return _STATE["dim"]


def _load_vision(model: str, revision: str, hf_home: str, device: str) -> int:
    """A vision-only model (the image tower of a text+image pair). Needs the managed environment.

    Untested against the real weights: whether this loader is right is checked on a machine with the model.
    """
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        if os.path.isdir(hf_home) and os.listdir(hf_home):
            os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        kwargs = {"revision": revision} if revision and not os.path.isdir(model) else {}
        chosen = _pick_device(device)
        processor = transformers.AutoImageProcessor.from_pretrained(model, **kwargs)
        loaded = transformers.AutoModel.from_pretrained(model, trust_remote_code=True, **kwargs)
        loaded = loaded.to(chosen).eval()
    except Exception:  # noqa: BLE001 - library or weights missing, or the loader does not fit this model
        raise _Invalid("the image model could not be loaded here; it needs the image environment "
                       "with its model files") from None
    _STATE.update(model=(loaded, processor, torch), device=chosen, vision=True)
    return int(getattr(getattr(loaded, "config", None), "hidden_size", 0) or 0)


def _encode_vision(paths: list[str]) -> list[list[float]]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 100_000_000
    model, processor, torch = _STATE["model"]
    images = []
    try:
        for path in paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        with torch.no_grad():
            inputs = processor(images, return_tensors="pt").to(_STATE["device"])
            hidden = model(**inputs).last_hidden_state[:, 0]
        return _normalise(hidden.float().cpu().tolist())
    finally:
        for img in images:
            img.close()


def _passes(sizes: list[int], budget: int, max_items: int) -> list[tuple[int, int]]:
    """Consecutive ``(start, end)`` groups whose sizes sum to at most ``budget`` (one item may
    exceed it alone) and that hold at most ``max_items`` each."""
    groups: list[tuple[int, int]] = []
    start, used = 0, 0
    for i, size in enumerate(sizes):
        if i > start and (used + size > budget or i - start >= max_items):
            groups.append((start, i))
            start, used = i, 0
        used += size
    if sizes:
        groups.append((start, len(sizes)))
    return groups


def _release_memory() -> None:
    """Hand cached accelerator memory back after a request, so the peak does not stay resident."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing a cache is best effort
        pass


def _normalise(rows) -> list[list[float]]:
    out = []
    for row in rows:
        vec = [float(x) for x in row]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


def _shrunk(img, max_pixels: int):
    """``img`` scaled down (aspect kept) to at most ``max_pixels``; unchanged when 0 or already small."""
    width, height = img.size
    if max_pixels <= 0 or width * height <= max_pixels:
        return img
    scale = math.sqrt(max_pixels / (width * height))
    return img.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def _encode_images(paths: list[str]) -> list[list[float]]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 100_000_000
    out: list[list[float]] = []
    for start, end in _passes([1] * len(paths), IMAGE_PASS, IMAGE_PASS):
        images = []
        try:
            for path in paths[start:end]:
                with Image.open(path) as img:
                    images.append(_shrunk(img.convert("RGB"), int(_STATE.get("max_pixels") or 0)))
            out.extend(_normalise(_STATE["model"].encode(images, normalize_embeddings=True,
                                                         show_progress_bar=False)))
        finally:
            for img in images:
                img.close()
    return out


# -- commands ------------------------------------------------------------------

def _need_loaded() -> None:
    if not _STATE["name"]:
        raise _Invalid("model is not loaded")


def _cmd_ping(_: dict) -> dict:
    return {"loaded": bool(_STATE["name"]), "model": _STATE["name"], "device": _STATE["device"]}


def _pixel_cap(req: dict) -> int:
    value = req.get("max_pixels")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _cmd_load(req: dict) -> dict:
    model = req.get("model")
    if not isinstance(model, str) or not model:
        raise _Invalid("model is required")
    role = req.get("role") or ""
    if role not in ("", "image", "text"):
        raise _Invalid("unknown role")
    fake = model.startswith("fake:") or os.environ.get("SLM_MEDIA_WORKER_FAKE") == "1"
    _STATE.update(vision=False, text_only=False)
    if fake:
        try:
            dim = int(model.split(":", 1)[1]) if model.startswith("fake:") else FAKE_DIM
        except ValueError:
            raise _Invalid("bad fake model") from None
        if not 1 <= dim <= 4096:
            raise _Invalid("bad fake model")
        _STATE.update(model=None, device="cpu", dim=dim)
    elif role == "image":
        _STATE["dim"] = _load_vision(model, str(req.get("revision") or ""), str(req.get("hf_home") or ""),
                                     str(req.get("device") or "auto"))
    else:
        _STATE["dim"] = _load_real(model, str(req.get("revision") or ""), str(req.get("hf_home") or ""),
                                   str(req.get("device") or "auto"), role, _pixel_cap(req))
    _STATE.update(name=model, fake=fake)
    return {"dim": _STATE["dim"]}


def _cmd_embed_text(req: dict) -> dict:
    _need_loaded()
    texts, prompt = req.get("texts"), req.get("prompt", "Document")
    if (not isinstance(texts, list) or not 0 < len(texts) <= MAX_TEXTS or prompt not in PROMPTS
            or any(not isinstance(t, str) or len(t) > MAX_TEXT_CHARS for t in texts)):
        raise _Invalid("invalid texts")
    if _STATE["fake"]:
        return {"vectors": [_fake_vector(t.encode("utf-8"), _STATE["dim"]) for t in texts]}
    if _STATE.get("vision"):
        raise _Invalid("this model embeds images only")
    vectors: list[list[float]] = []
    for start, end in _passes([len(t) for t in texts], TEXT_PASS_CHARS, TEXT_PASS_MAX):
        rows = _STATE["model"].encode(texts[start:end], prompt_name=prompt, normalize_embeddings=True,
                                      show_progress_bar=False)
        vectors.extend(_normalise(rows))
    _release_memory()
    return {"vectors": vectors}


def _check_paths(paths) -> list[str]:
    if not isinstance(paths, list) or not 0 < len(paths) <= MAX_PATHS or any(not isinstance(p, str) for p in paths):
        raise _Invalid("invalid paths")
    for path in paths:
        try:
            if not os.path.isfile(path) or os.path.getsize(path) > MAX_FILE_BYTES:
                raise _Invalid("a file is missing, not a regular file, or too large")
        except OSError:
            raise _Invalid("a file is missing, not a regular file, or too large") from None
    return paths


def _cmd_embed_image(req: dict) -> dict:
    _need_loaded()
    paths = _check_paths(req.get("paths"))
    if _STATE["fake"]:
        vectors = []
        for path in paths:
            with open(path, "rb") as fh:
                vectors.append(_fake_vector(fh.read(), _STATE["dim"]))
        return {"vectors": vectors}
    if _STATE.get("text_only"):
        raise _Invalid("this model embeds text only")
    if _STATE.get("vision"):
        vectors = [v for s, e in _passes([1] * len(paths), IMAGE_PASS, IMAGE_PASS)
                   for v in _encode_vision(paths[s:e])]
    else:
        vectors = _encode_images(paths)
    _release_memory()
    return {"vectors": vectors}


def _image_ops():
    """``media_image_ops.py`` next to this file, loaded by path (-I leaves our directory off sys.path)."""
    if "ops" not in _STATE:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media_image_ops.py")
        spec = importlib.util.spec_from_file_location("media_image_ops", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STATE["ops"] = module
    return _STATE["ops"]


def _cmd_prepare_image(req: dict) -> dict:
    path, out_dir = req.get("path"), req.get("out_dir")
    if not isinstance(path, str) or not isinstance(out_dir, str):
        raise _Invalid("invalid paths")
    _check_paths([path])
    if not os.path.isdir(out_dir):
        raise _Invalid("out_dir is not a directory")
    try:
        return _image_ops().prepare_image(path, out_dir)
    except ImportError:
        raise _Invalid("pillow unavailable") from None
    except ValueError as exc:  # the module's own messages: no paths, no contents
        raise _Invalid(f"ValueError: {exc}") from None


def _cmd_ocr_image(req: dict) -> dict:
    path = req.get("path")
    if not isinstance(path, str):
        raise _Invalid("invalid paths")
    _check_paths([path])
    if _STATE["fake"]:
        try:
            with open(path + ".ocr.txt", encoding="utf-8") as fh:
                return {"engine": "fake", "text": fh.read(MAX_TEXT_CHARS * 8)}
        except OSError:
            return {"engine": "fake", "text": ""}
    return _image_ops().ocr_image(path)


def _cmd_sleep(req: dict) -> dict:
    if not _STATE["fake"]:
        raise _Invalid("unknown command")
    time.sleep(min(float(req.get("seconds", 0)), 600.0))
    return {}


_COMMANDS = {"ping": _cmd_ping, "load": _cmd_load, "embed_text": _cmd_embed_text,
             "embed_image": _cmd_embed_image, "prepare_image": _cmd_prepare_image,
             "ocr_image": _cmd_ocr_image, "sleep": _cmd_sleep}


def handle(req: dict) -> dict:
    """One request to one reply. Never raises, never echoes file contents."""
    reply: dict
    try:
        fn = _COMMANDS.get(req.get("cmd")) if isinstance(req, dict) else None
        if fn is None or (req["cmd"] == "sleep" and not _STATE["fake"]):
            raise _Invalid("unknown command")
        reply = {"ok": True, **fn(req)}
    except _Invalid as exc:
        reply = {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a failed request must not end the worker
        reply = {"ok": False, "error": f"request failed ({type(exc).__name__})"}
    if isinstance(req, dict) and "id" in req:
        reply["id"] = req["id"]
    return reply


# -- process plumbing ----------------------------------------------------------

def _watch_parent() -> None:
    parent = os.getppid()
    while True:
        time.sleep(_WATCHDOG_S)
        if os.getppid() != parent:
            os._exit(0)


def main() -> int:
    import warnings

    # Library deprecation notices are for their developers; on stderr they read as daemon warnings.
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        threading.Thread(target=_watch_parent, daemon=True, name="parent-watchdog").start()
    out = sys.stdout
    sys.stdout = sys.stderr  # stray library prints must not corrupt the protocol
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
        except ValueError:
            req = None
        if isinstance(req, dict) and req.get("cmd") == "quit":
            out.write(json.dumps({"ok": True, **({"id": req["id"]} if "id" in req else {})}) + "\n")
            out.flush()
            return 0
        out.write(json.dumps(handle(req)) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
