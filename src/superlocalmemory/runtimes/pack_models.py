# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The two optional picture models of the multimodal pack (WS-8, D-19), for the media worker.

* ``caption``: Florence-2 (``florence-community/Florence-2-base``, MIT) writes a short description
  of a picture. Loaded with the native ``Florence2ForConditionalGeneration`` class.
* ``document``: PaddleOCR-VL reads a page image into text. Loaded with the native
  ``PaddleOCRVLForConditionalGeneration`` class.

Both classes ship in the pinned ``transformers==5.19.0``; ``trust_remote_code`` is never set, so
the Python files some model repositories carry are never run. Loaded by path from
``multimodal_worker.py`` inside the managed environment, which does not have superlocalmemory
installed: this file imports nothing from it. Heavy libraries are imported inside the functions.

The real classes and the real weights are NOT exercised by the unit tests (no model is
downloaded in a build): ``tests/test_media/test_pack_models_native.py`` runs them when the
weights are present on the machine and is skipped otherwise.

Nothing here logs picture contents; error text never carries a path or the recognised words.
"""

from __future__ import annotations

import importlib
import re

CAPTION_TASK = "<CAPTION>"
CAPTION_MAX_NEW_TOKENS = 64
CAPTION_BEAMS = 3
DOCUMENT_PROMPT = "OCR:"
DOCUMENT_MAX_NEW_TOKENS = 1024
MAX_CAPTION_CHARS = 400
MAX_PAGE_CHARS = 64_000
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_text(text: str, limit: int) -> str:
    """Words only: control characters removed, white space collapsed, cut at ``limit``."""
    return re.sub(r"[ \t]+", " ", _CONTROL.sub("", str(text or ""))).strip()[:limit]


def _open_rgb(path: str):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 50_000_000
    with Image.open(path) as img:
        return img.convert("RGB")


# -- boxes ---------------------------------------------------------------------

MAX_REGIONS = 400


def picture_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def normalised_regions(pairs, size) -> list[dict]:
    """One ``{"text", "bbox": [x0, y0, x1, y1]}`` per ``(box, text)`` pair, the box as fractions (0..1) of the
    picture's width and height, so a thumbnail and the original agree. A box that cannot be read is left out;
    nothing here raises."""
    width, height = size
    out: list[dict] = []
    if not width or not height:
        return out
    for box, text in list(pairs)[:MAX_REGIONS]:
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            x0, x1 = max(0.0, min(xs) / width), min(1.0, max(xs) / width)
            y0, y1 = max(0.0, min(ys) / height), min(1.0, max(ys) / height)
        except (TypeError, ValueError, IndexError):
            continue
        if x1 > x0 and y1 > y0 and str(text).strip():
            out.append({"text": str(text), "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]})
    return out


# -- captions ------------------------------------------------------------------

def load_caption(model_dir: str, device: str):
    """``(model, processor, torch)`` for Florence-2 from a local folder (no network, no remote code)."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = transformers.Florence2ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32)
    return model.to(device).eval(), processor, torch


def caption(state, path: str) -> str:
    model, processor, torch = state
    image = _open_rgb(path)
    try:
        inputs = processor(text=CAPTION_TASK, images=image, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=CAPTION_MAX_NEW_TOKENS,
                                 num_beams=CAPTION_BEAMS, do_sample=False)
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    finally:
        image.close()
    return clean_text(text, MAX_CAPTION_CHARS)


# -- document pages --------------------------------------------------------------

def load_document(model_dir: str, device: str):
    """``(model, processor, torch)`` for PaddleOCR-VL from a local folder (no network, no remote code)."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = transformers.PaddleOCRVLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32)
    return model.to(device).eval(), processor, torch


def parse_document(state, path: str) -> dict:
    """The text of one page image. The model reads the whole page in one pass, so the one region
    it can honestly report is the page itself (layout boxes need the separate layout model, which
    this pack does not load)."""
    model, processor, torch = state
    image = _open_rgb(path)
    try:
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": DOCUMENT_PROMPT}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=DOCUMENT_MAX_NEW_TOKENS)
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], ids)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    finally:
        image.close()
    text = _CONTROL.sub("", text).strip()[:MAX_PAGE_CHARS]
    return {"engine": "paddleocr_vl", "text": text,
            "regions": [{"text": text[:400], "bbox": [0.0, 0.0, 1.0, 1.0], "label": "page"}] if text else []}
