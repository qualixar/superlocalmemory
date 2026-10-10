# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Image operations for the media worker: strip, thumbnail, hash, read text.

Loaded by path from ``multimodal_worker.py`` inside the managed environment, which does
not have superlocalmemory installed: this file imports nothing from it. Pillow and
ImageHash are imported inside the functions, so the module loads without them.

Nothing here reads location data: only four plain camera fields leave the original.
Error text never contains a path or file contents.
"""

from __future__ import annotations

import importlib
import io
import os
import secrets
import sys

MAX_PIXELS = 50_000_000
_FORMATS = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"), "WEBP": ("image/webp", ".webp"),
            "GIF": ("image/png", ".png")}  # a GIF keeps its first frame, stored as PNG
_EXIF_FIELDS = {0x010F: "Make", 0x0110: "Model", 0x0112: "Orientation"}
_EXIF_IFD, _DATE_ORIGINAL = 0x8769, 0x9003
_THUMB_FALLBACK_EDGE = 240


def _write_private(directory: str, ext: str, data: bytes) -> str:
    """A new file with a random name, owner-only, created exclusively."""
    path = os.path.join(directory, secrets.token_hex(16) + ext)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    return path


def _camera_fields(exif) -> dict:
    """The four fields that are kept; GPS and everything else is never read."""
    out: dict = {}
    date = exif.get_ifd(_EXIF_IFD).get(_DATE_ORIGINAL)
    if date:
        out["DateTimeOriginal"] = str(date).strip()
    for tag, name in _EXIF_FIELDS.items():
        value = exif.get(tag)
        if value not in (None, ""):
            out[name] = int(value) if name == "Orientation" else str(value).strip()
    return out


def _open_checked(src: str):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        img = Image.open(src)
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("not a supported image") from None
    if img.format not in _FORMATS:
        img.close()
        raise ValueError("unsupported image format")
    if img.width * img.height > MAX_PIXELS:
        img.close()
        raise ValueError("image has too many pixels")
    return img


def _load_upright(img):
    from PIL import Image, ImageOps

    try:
        if img.format == "GIF":
            img.seek(0)
        img.load()
        exif = img.getexif()
        fields = _camera_fields(exif)
        icc = img.info.get("icc_profile")
        upright = ImageOps.exif_transpose(img)
    except (OSError, SyntaxError, ValueError, EOFError, Image.DecompressionBombError):
        raise ValueError("image is damaged") from None
    return upright, fields, icc


def _encode_original(upright, fmt: str, icc) -> bytes:
    """Re-encode without any metadata block; colour profile only."""
    img = upright
    if fmt == "JPEG":
        img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
        kwargs = {"quality": 92}
    elif fmt == "WEBP":
        kwargs = {"quality": 92}
    else:  # PNG, and GIF stored as PNG
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB") if img.mode not in ("RGB", "RGBA", "L", "LA") else img
        kwargs = {}
    if icc and fmt != "GIF":
        kwargs["icc_profile"] = icc
    buf = io.BytesIO()
    img.save(buf, "PNG" if fmt == "GIF" else fmt, **kwargs)
    return buf.getvalue()


def _encode_thumb(upright, edge: int, max_bytes: int) -> bytes:
    from PIL import Image

    thumb = upright.convert("RGBA" if upright.mode in ("RGBA", "LA", "P") and "transparency" in upright.info
                            or upright.mode in ("RGBA", "LA") else "RGB")
    thumb.thumbnail((edge, edge), Image.LANCZOS)
    data = b""
    for quality in (80, 70, 60, 50, 40, 30):
        buf = io.BytesIO()
        thumb.save(buf, "WEBP", quality=quality)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    thumb.thumbnail((_THUMB_FALLBACK_EDGE, _THUMB_FALLBACK_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, "WEBP", quality=30)
    return buf.getvalue()


def prepare_image(src: str, out_dir: str, *, thumb_edge: int = 320, thumb_max_bytes: int = 32_768) -> dict:
    """Strip, thumbnail and hash an image; write both results into ``out_dir``.

    Raises ValueError for anything that is not a plain PNG, JPEG, WEBP or GIF.
    On any failure nothing is left in ``out_dir``.
    """
    imagehash = importlib.import_module("imagehash")
    written: list[str] = []
    img = _open_checked(src)
    try:
        fmt = img.format
        upright, fields, icc = _load_upright(img)
        try:
            mime, ext = _FORMATS[fmt]
            result = {"mime": mime, "width": upright.width, "height": upright.height, "exif": fields,
                      "phash": str(imagehash.phash(upright.convert("RGB")))}
            written.append(_write_private(out_dir, ext, _encode_original(upright, fmt, icc)))
            result.update(stored_path=written[0], stored_ext=ext)
            written.append(_write_private(out_dir, ".webp", _encode_thumb(upright, thumb_edge, thumb_max_bytes)))
            result["thumb_path"] = written[1]
            return result
        finally:
            upright.close()
    except BaseException:
        for path in written:
            try:
                os.unlink(path)
            except OSError:
                pass
        raise
    finally:
        img.close()


# -- text in images ------------------------------------------------------------

def _apple_vision(path: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        vision = importlib.import_module("Vision")
        foundation = importlib.import_module("Foundation")
    except ImportError:
        return None
    handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(
        foundation.NSURL.fileURLWithPath_(path), None)
    request = vision.VNRecognizeTextRequest.alloc().init()
    request.setUsesLanguageCorrection_(True)
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return ""
    lines = [obs.topCandidates_(1)[0].string() for obs in (request.results() or []) if obs.topCandidates_(1)]
    return "\n".join(lines)


def _rapidocr(path: str) -> str | None:
    """Only when the package and its model files are already installed; never downloads."""
    try:
        module = importlib.import_module("rapidocr")
    except ImportError:
        return None
    models = os.path.join(os.path.dirname(module.__file__), "models")
    if not os.path.isdir(models) or not any(n.endswith(".onnx") for n in os.listdir(models)):
        return None
    result = module.RapidOCR()(path)
    texts = getattr(result, "txts", None)
    return "\n".join(texts) if texts else ""


def ocr_image(path: str, *, engine: str = "auto") -> dict:
    """Read the text in an image: ``{"engine": "apple_vision|rapidocr|none", "text": str}``."""
    candidates = {"apple_vision": _apple_vision, "rapidocr": _rapidocr}
    for name in (candidates if engine == "auto" else [engine] if engine in candidates else []):
        text = candidates[name](path)
        if text is not None:
            return {"engine": name, "text": text}
    return {"engine": "none", "text": ""}
