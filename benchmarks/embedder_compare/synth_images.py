"""Deterministic synthetic test images (Pillow + DejaVu fonts).

CONTENT below is the single source of truth: golden/media_queries.jsonl
queries are written from this table, never from model output.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = (1280, 720)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

# name -> (kind, payload). Names become doc ids "img:syn_<name>".
CONTENT: dict[str, tuple[str, dict]] = {
    "slide_q3": ("slide", {"title": "Q3 2026 Results", "lines": [
        "Revenue: $4.2M (+18% QoQ)", "Gross margin: 71%", "Net new customers: 212", "Churn: 2.1%"]}),
    "slide_launch": ("slide", {"title": "Launch Plan Spring", "lines": [
        "Apr: Beta invites", "May: Public preview", "Jun: General availability"]}),
    "slide_hiring": ("slide", {"title": "Hiring Plan FY27", "lines": [
        "Engineers: 6", "Designers: 2", "Support staff: 3"]}),
    "slide_okr": ("slide", {"title": "Team Objectives", "lines": [
        "Reduce p95 latency to 180 ms", "Ship offline mode", "Raise test coverage to 85%"]}),
    "slide_agenda": ("slide", {"title": "Meeting Agenda", "lines": [
        "1. Budget review", "2. Vendor contracts", "3. Office move"]}),
    "bar_signups": ("bars", {"title": "Monthly Sign-ups", "data": [
        ("Jan", 120), ("Feb", 180), ("Mar", 260), ("Apr", 210)]}),
    "bar_regions": ("bars", {"title": "Revenue by Region", "data": [
        ("North", 40), ("South", 25), ("East", 55), ("West", 30)]}),
    "line_temp": ("line", {"title": "Server Temperature (C)", "data": [
        ("00h", 41), ("04h", 38), ("08h", 52), ("12h", 70), ("14h", 78), ("18h", 60), ("22h", 45)]}),
    "line_latency": ("line", {"title": "Weekly Latency (ms)", "data": [
        ("Mon", 210), ("Tue", 190), ("Wed", 175), ("Thu", 240), ("Fri", 160), ("Sat", 120), ("Sun", 110)]}),
    "pie_browsers": ("pie", {"title": "Browser Share", "data": [
        ("Chrome", 62), ("Safari", 20), ("Firefox", 8), ("Other", 10)]}),
    "dialog_conn": ("dialog", {"title": "Connection failed", "lines": [
        "Could not reach database host db-eu-2", "(timeout after 30 s)"], "buttons": ["Retry", "Cancel"]}),
    "dialog_disk": ("dialog", {"title": "Disk almost full", "lines": [
        "Volume /data is 97% full.", "Free up space to continue."], "buttons": ["Clean up", "Ignore"]}),
    "dialog_perm": ("dialog", {"title": "Permission denied", "lines": [
        "User deploy-bot cannot write to", "/var/releases"], "buttons": ["OK"]}),
    "diagram_pipeline": ("boxes", {"title": "Data Pipeline", "boxes": ["Ingest", "Clean", "Embed", "Index"]}),
    "diagram_auth": ("boxes", {"title": "Request Path", "boxes": ["Browser", "Gateway", "Auth Service", "Database"]}),
    "diagram_org": ("tree", {"title": "Org Chart", "root": "Director", "kids": ["Engineering", "Operations", "Finance"]}),
    "scene_sunset": ("scene", {"sky": ((250, 150, 60), (90, 40, 120)), "sun": (640, 430, 110, (255, 220, 60)),
                               "hills": [(30, 30, 50), (60, 40, 70)]}),
    "scene_forest": ("scene", {"sky": ((120, 190, 240), (210, 235, 250)), "sun": (1100, 120, 60, (255, 250, 200)),
                               "hills": [(20, 90, 40), (10, 60, 30)]}),
    "scene_night": ("scene", {"sky": ((5, 10, 40), (30, 40, 90)), "sun": (300, 140, 70, (240, 240, 230)),
                              "hills": [(15, 20, 25), (25, 30, 40)]}),
    "receipt_cafe": ("receipt", {"shop": "Maple Cafe", "items": [
        ("Latte", "4.50"), ("Croissant", "3.20")], "total": "7.70"}),
    "receipt_hardware": ("receipt", {"shop": "Pine Hardware", "items": [
        ("Wood screws", "6.40"), ("Drill bit", "12.90")], "total": "19.30"}),
    "code_python": ("code", {"lines": [
        "def fibonacci(n):", "    a, b = 0, 1", "    for _ in range(n):", "        a, b = b, a + b", "    return a"]}),
    "code_sql": ("code", {"lines": [
        "SELECT name, total", "FROM orders", "WHERE total > 500", "ORDER BY total DESC;"]}),
    "table_shifts": ("table", {"title": "Shift Schedule", "rows": [
        ("Name", "Day", "Shift"), ("Asha", "Mon", "Early"), ("Jonas", "Tue", "Late"), ("Lena", "Wed", "Night")]}),
}


def _font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def _canvas(bg=(255, 255, 255)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", SIZE, bg)
    return im, ImageDraw.Draw(im)


def _slide(p: dict) -> Image.Image:
    im, d = _canvas()
    d.rectangle([0, 0, SIZE[0], 20], fill=(30, 80, 160))
    d.text((60, 60), p["title"], fill="black", font=_font(48, bold=True))
    for i, line in enumerate(p["lines"]):
        d.text((90, 190 + i * 70), "- " + line, fill=(40, 40, 40), font=_font(34))
    return im


def _axes(d: ImageDraw.ImageDraw, title: str) -> tuple[int, int, int, int]:
    d.text((60, 30), title, fill="black", font=_font(40, bold=True))
    box = (140, 130, 1180, 620)
    d.line([box[0], box[1], box[0], box[3], box[2], box[3]], fill="black", width=3)
    return box


def _bars(p: dict) -> Image.Image:
    im, d = _canvas()
    x0, y0, x1, y1 = _axes(d, p["title"])
    top = max(v for _, v in p["data"])
    slot = (x1 - x0) // len(p["data"])
    for i, (label, v) in enumerate(p["data"]):
        h = int((y1 - y0 - 40) * v / top)
        bx = x0 + i * slot + slot // 4
        d.rectangle([bx, y1 - h, bx + slot // 2, y1], fill=(50, 110, 200))
        d.text((bx, y1 - h - 34), str(v), fill="black", font=_font(26))
        d.text((bx, y1 + 10), label, fill="black", font=_font(26))
    return im


def _line(p: dict) -> Image.Image:
    im, d = _canvas()
    x0, y0, x1, y1 = _axes(d, p["title"])
    top = max(v for _, v in p["data"])
    step = (x1 - x0 - 60) // (len(p["data"]) - 1)
    pts = [(x0 + 30 + i * step, y1 - int((y1 - y0 - 40) * v / top)) for i, (_, v) in enumerate(p["data"])]
    d.line(pts, fill=(200, 60, 40), width=5)
    for (x, y), (label, v) in zip(pts, p["data"]):
        d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(200, 60, 40))
        d.text((x - 20, y - 40), str(v), fill="black", font=_font(24))
        d.text((x - 24, y1 + 10), label, fill="black", font=_font(24))
    return im


def _pie(p: dict) -> Image.Image:
    im, d = _canvas()
    d.text((60, 30), p["title"], fill="black", font=_font(40, bold=True))
    colors = [(50, 110, 200), (230, 140, 30), (60, 160, 90), (150, 150, 150)]
    total, start = sum(v for _, v in p["data"]), -90.0
    for i, (label, v) in enumerate(p["data"]):
        end = start + 360.0 * v / total
        d.pieslice([120, 130, 560, 570], start, end, fill=colors[i])
        d.rectangle([700, 200 + i * 80, 740, 240 + i * 80], fill=colors[i])
        d.text((760, 200 + i * 80), f"{label} {v}%", fill="black", font=_font(32))
        start = end
    return im


def _dialog(p: dict) -> Image.Image:
    im, d = _canvas((200, 205, 215))
    d.rectangle([260, 160, 1020, 560], fill=(245, 245, 245), outline=(80, 80, 80), width=3)
    d.rectangle([260, 160, 1020, 220], fill=(170, 40, 40))
    d.text((290, 170), "! " + p["title"], fill="white", font=_font(34, bold=True))
    for i, line in enumerate(p["lines"]):
        d.text((300, 270 + i * 50), line, fill="black", font=_font(30))
    for i, b in enumerate(p["buttons"]):
        x = 700 + i * 160 if len(p["buttons"]) > 1 else 860
        d.rectangle([x, 470, x + 140, 530], fill=(220, 220, 220), outline="black", width=2)
        d.text((x + 20, 482), b, fill="black", font=_font(28))
    return im


def _boxes(p: dict) -> Image.Image:
    im, d = _canvas()
    d.text((60, 30), p["title"], fill="black", font=_font(40, bold=True))
    n, w = len(p["boxes"]), 220
    gap = (SIZE[0] - 120 - n * w) // (n - 1)
    for i, label in enumerate(p["boxes"]):
        x = 60 + i * (w + gap)
        d.rectangle([x, 300, x + w, 420], fill=(225, 235, 250), outline=(30, 80, 160), width=4)
        d.text((x + 14, 345), label, fill="black", font=_font(25, bold=True))
        if i < n - 1:
            d.line([x + w, 360, x + w + gap, 360], fill="black", width=4)
            d.polygon([(x + w + gap, 360), (x + w + gap - 18, 348), (x + w + gap - 18, 372)], fill="black")
    return im


def _tree(p: dict) -> Image.Image:
    im, d = _canvas()
    d.text((60, 30), p["title"], fill="black", font=_font(40, bold=True))
    d.rectangle([490, 140, 790, 220], fill=(225, 235, 250), outline=(30, 80, 160), width=4)
    d.text((540, 165), p["root"], fill="black", font=_font(30, bold=True))
    for i, kid in enumerate(p["kids"]):
        x = 120 + i * 400
        d.line([640, 220, x + 140, 400], fill="black", width=3)
        d.rectangle([x, 400, x + 280, 480], fill=(250, 240, 220), outline=(160, 100, 20), width=4)
        d.text((x + 24, 425), kid, fill="black", font=_font(28))
    return im


def _scene(p: dict) -> Image.Image:
    im, d = _canvas()
    (r0, g0, b0), (r1, g1, b1) = p["sky"]
    for y in range(SIZE[1]):
        t = y / (SIZE[1] - 1)
        d.line([0, y, SIZE[0], y], fill=(int(r0 + (r1 - r0) * t), int(g0 + (g1 - g0) * t), int(b0 + (b1 - b0) * t)))
    sx, sy, sr, sc = p["sun"]
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=sc)
    back, front = p["hills"]
    d.polygon([(0, 720), (0, 500), (300, 380), (600, 520), (900, 400), (1280, 540), (1280, 720)], fill=back)
    d.polygon([(0, 720), (0, 600), (400, 520), (800, 620), (1280, 560), (1280, 720)], fill=front)
    return im


def _receipt(p: dict) -> Image.Image:
    im, d = _canvas((235, 235, 235))
    d.rectangle([380, 30, 900, 690], fill="white", outline=(150, 150, 150), width=2)
    d.text((440, 70), p["shop"], fill="black", font=_font(38, bold=True))
    for i, (item, price) in enumerate(p["items"]):
        d.text((440, 190 + i * 60), item, fill="black", font=_font(30, mono=True))
        d.text((760, 190 + i * 60), price, fill="black", font=_font(30, mono=True))
    d.line([440, 400, 860, 400], fill="black", width=2)
    d.text((440, 420), "TOTAL", fill="black", font=_font(32, bold=True, mono=True))
    d.text((760, 420), p["total"], fill="black", font=_font(32, bold=True, mono=True))
    return im


def _code(p: dict) -> Image.Image:
    im, d = _canvas((30, 32, 40))
    d.rectangle([0, 0, SIZE[0], 50], fill=(50, 52, 62))
    for i, line in enumerate(p["lines"]):
        d.text((50, 110 + i * 60), line, fill=(180, 230, 160), font=_font(36, mono=True))
    return im


def _table(p: dict) -> Image.Image:
    im, d = _canvas()
    d.text((60, 30), p["title"], fill="black", font=_font(40, bold=True))
    for r, row in enumerate(p["rows"]):
        y = 150 + r * 90
        for c, cell in enumerate(row):
            x = 100 + c * 360
            d.rectangle([x, y, x + 360, y + 90], outline="black", width=2,
                        fill=(225, 235, 250) if r == 0 else None)
            d.text((x + 20, y + 25), cell, fill="black", font=_font(30, bold=(r == 0)))
    return im


_RENDER = {"slide": _slide, "bars": _bars, "line": _line, "pie": _pie, "dialog": _dialog,
           "boxes": _boxes, "tree": _tree, "scene": _scene, "receipt": _receipt,
           "code": _code, "table": _table}


def generate(out_dir: Path) -> list[str]:
    """Write every synthetic image as <out_dir>/<name>.png; return the names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (kind, payload) in CONTENT.items():
        _RENDER[kind](payload).save(out_dir / f"{name}.png", optimize=False)
    return list(CONTENT)


if __name__ == "__main__":
    print("\n".join(generate(Path(sys.argv[1] if len(sys.argv) > 1 else "synth_out"))))
