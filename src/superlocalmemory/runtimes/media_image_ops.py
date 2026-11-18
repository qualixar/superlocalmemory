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

import hashlib
import importlib
import io
import logging
import os
import secrets
import sys
from types import ModuleType

logger = logging.getLogger(__name__)
MAX_PIXELS = 50_000_000
_FORMATS = {"PNG": ("image/png", ".png"), "JPEG": ("image/jpeg", ".jpg"), "WEBP": ("image/webp", ".webp"),
            "GIF": ("image/png", ".png")}  # a GIF keeps its first frame, stored as PNG
_EXIF_FIELDS = {0x010F: "Make", 0x0110: "Model", 0x0112: "Orientation"}
_EXIF_IFD, _DATE_ORIGINAL = 0x8769, 0x9003
_THUMB_FALLBACK_EDGE = 240

# Pinned rapidocr==3.10.0 wheel models; explicit paths bypass its downloader.
_RAPIDOCR_MODELS = {
    "Det": ("PP-OCRv6_det_small.onnx",
            "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "Cls": ("ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
    "Rec": ("PP-OCRv6_rec_small.onnx",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
}


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
    if date := exif.get_ifd(_EXIF_IFD).get(_DATE_ORIGINAL):
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


def prepare_image(src: str, out_dir: str, *, thumb_edge: int = 320, thumb_max_bytes: int = 32_768,
                  classify: bool = True) -> dict:
    """Strip, thumbnail and hash an image; write both results into ``out_dir``.

    ``classify=False`` skips the subject classifier (a rasterised document page).
    Raises ValueError unless the image is a plain PNG, JPEG, WEBP or GIF; on any
    failure nothing is left in ``out_dir``.
    """
    imagehash = importlib.import_module("imagehash")
    written: list[str] = []
    img = _open_checked(src)
    try:
        fmt = img.format
        upright, fields, icc = _load_upright(img)
        try:
            mime, ext = _FORMATS[fmt]
            subjects = _apple_subjects(src) if classify else ""
            result = {"mime": mime, "width": upright.width, "height": upright.height, "exif": fields,
                      "phash": str(imagehash.phash(upright.convert("RGB"))),
                      "visual_summary": visual_summary(upright) + subjects}
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
    request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
    request.setAutomaticallyDetectsLanguage_(True)
    request.setUsesLanguageCorrection_(True)
    ok, _err = handler.performRequests_error_([request], None)
    if not ok:
        return ""
    lines = [obs.topCandidates_(1)[0].string() for obs in (request.results() or []) if obs.topCandidates_(1)]
    return "\n".join(lines)


def _apple_subjects(path: str) -> str:
    """Use the installed offline Vision classifier for a bounded subject caption.

    Apple's VNClassifyImageRequest supplies identifier/confidence observations:
    https://developer.apple.com/documentation/vision/vnclassifyimagerequest
    No downloads; unsupported hosts retain the measured pixel summary and OCR.
    """
    if sys.platform != "darwin":
        return ""
    try:
        vision = importlib.import_module("Vision")
        foundation = importlib.import_module("Foundation")
        handler = vision.VNImageRequestHandler.alloc().initWithURL_options_(
            foundation.NSURL.fileURLWithPath_(path), None)
        request = vision.VNClassifyImageRequest.alloc().init()
        ok, _ = handler.performRequests_error_([request], None)
        if not ok:
            return ""
        subjects = [str(obs.identifier()).replace('_', ' ') for obs in (request.results() or [])
                    if float(obs.confidence()) >= .8][:3]
        return " Image classifier subjects: " + ", ".join(subjects) + "." if subjects else ""
    except Exception as exc:  # noqa: BLE001 - any binding failure: a save never fails for a caption
        logger.warning("image classifier unavailable (%s); no subjects", type(exc).__name__)
        return ""


def _local_ocr_params(module: ModuleType) -> dict[str, str] | None:
    """Resolve the complete verified model set without invoking the backend."""
    models = os.path.join(os.path.dirname(module.__file__), "models")
    params = {}
    for role, (filename, expected) in _RAPIDOCR_MODELS.items():
        path = os.path.join(models, filename)
        try:
            with open(path, "rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
        except OSError:
            logger.warning("RapidOCR model %s missing or unreadable; OCR unavailable", role)
            continue
        if actual != expected:
            logger.warning("RapidOCR model %s corrupt; OCR unavailable", role)
            continue
        params[f"{role}.model_path"] = path
    return params if len(params) == len(_RAPIDOCR_MODELS) else None


def _rapidocr(path: str, found: list | None = None) -> str | None:
    """Verified local models only; an unavailable backend never blocks save. ``found`` gets (box, text) pairs."""
    try:
        module = importlib.import_module("rapidocr")
        params = _local_ocr_params(module)
        if params is None:
            return None
        result = module.RapidOCR(params=params)(path)
        texts = getattr(result, "txts", None)
        if found is not None and texts and getattr(result, "boxes", None) is not None:
            found.extend(zip(list(result.boxes), texts))
        return "\n".join(texts) if texts else ""
    except ImportError:
        logger.warning("RapidOCR unavailable: package or inference dependency missing")
    except Exception as exc:  # noqa: BLE001 - optional inference must preserve save
        logger.warning("RapidOCR unavailable (%s); OCR skipped", type(exc).__name__)
    return None


def ocr_image(path: str, *, engine: str = "auto", found: list | None = None) -> dict:
    """Read the text in an image: ``{"engine": "apple_vision|rapidocr|none", "text": str}``; ``found``: see rapidocr."""
    candidates, empty = {"apple_vision": _apple_vision, "rapidocr": _rapidocr}, None
    for name in (candidates if engine == "auto" else [engine] if engine in candidates else []):
        text = candidates[name](path, found) if found is not None and name == "rapidocr" else candidates[name](path)
        if text is not None:
            result = {"engine": name, "text": text}
            if text.strip() or engine != "auto":
                return result
            if empty is None:
                empty = result
    return empty or {"engine": "none", "text": ""}


_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
    "red": (255, 0, 0), "green": (0, 128, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
}


def visual_summary(image):
    """Describe measured dimensions and dominant colors of a small RGB sample."""
    sample = image.convert("RGB")
    try:
        sample.thumbnail((32, 32))
        counts = dict.fromkeys(_COLORS, 0)
        for pixel in sample.getdata():
            color = min(_COLORS, key=lambda key: sum(
                (a - b) ** 2 for a, b in zip(pixel, _COLORS[key])))
            counts[color] += 1
        total = sum(counts.values())
        colors = sorted(counts, key=lambda key: (-counts[key], key))[:3]
        measured = ", ".join(f"{color} ({round(100 * counts[color] / total)}%)"
                             for color in colors if counts[color])
        return f"Image {image.width} by {image.height} pixels. Dominant pixel colors: {measured}."
    finally:
        sample.close()
