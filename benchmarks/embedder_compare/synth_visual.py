"""Purely visual synthetic images: shapes and colours only, no text anywhere."""
from __future__ import annotations

from PIL import Image, ImageDraw

SIZE = (1280, 720)


def _bicycle(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 560], fill=(40, 90, 190))
    d.rectangle([0, 560, 1280, 720], fill=(120, 120, 125))
    red = (200, 30, 35)
    for cx in (430, 850):
        d.ellipse([cx - 130, 420, cx + 130, 680], outline=red, width=14)
    d.line([430, 550, 620, 330, 850, 550], fill=red, width=14)
    d.line([620, 330, 560, 550, 430, 550], fill=red, width=14)
    d.line([620, 330, 790, 330, 850, 550], fill=red, width=14)
    d.line([760, 300, 830, 300], fill=(20, 20, 20), width=14)
    d.line([540, 310, 600, 330], fill=(20, 20, 20), width=18)


def _sailboat(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 420], fill=(190, 220, 245))
    d.rectangle([0, 420, 1280, 720], fill=(15, 40, 110))
    d.polygon([(380, 500), (900, 500), (820, 590), (460, 590)], fill=(130, 80, 40))
    d.line([640, 160, 640, 500], fill=(70, 50, 30), width=10)
    d.polygon([(640, 170), (640, 480), (820, 480)], fill=(250, 250, 250))
    d.polygon([(620, 220), (620, 480), (470, 480)], fill=(235, 235, 235))
    d.polygon([(640, 160), (700, 180), (640, 200)], fill=(245, 205, 40))


def _snowman(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 440], fill=(170, 195, 220))
    d.rectangle([0, 440, 1280, 720], fill=(245, 248, 252))
    for x in (120, 260, 1020, 1160):
        d.polygon([(x, 150), (x - 90, 470), (x + 90, 470)], fill=(20, 80, 45))
        d.rectangle([x - 14, 470, x + 14, 520], fill=(90, 60, 30))
    for cy, r in ((560, 130), (400, 95), (280, 65)):
        d.ellipse([640 - r, cy - r, 640 + r, cy + r], fill=(255, 255, 255), outline=(150, 160, 175), width=4)
    d.rectangle([585, 180, 695, 232], fill=(190, 25, 30))
    d.rectangle([565, 228, 715, 244], fill=(190, 25, 30))
    d.polygon([(640, 280), (730, 292), (640, 304)], fill=(240, 120, 20))
    for x in (615, 665):
        d.ellipse([x - 7, 258, x + 7, 272], fill=(20, 20, 20))


def _cone(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 720], fill=(95, 95, 100))
    d.rectangle([0, 0, 1280, 250], fill=(150, 160, 170))
    d.polygon([(640, 130), (500, 560), (780, 560)], fill=(245, 120, 20))
    d.polygon([(600, 260), (680, 260), (700, 330), (580, 330)], fill=(250, 250, 250))
    d.polygon([(560, 400), (720, 400), (735, 470), (545, 470)], fill=(250, 250, 250))
    d.rectangle([450, 560, 830, 610], fill=(235, 110, 15))


def _house(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 420], fill=(150, 205, 245))
    d.rectangle([0, 420, 1280, 720], fill=(70, 150, 60))
    d.rectangle([300, 300, 700, 560], fill=(245, 215, 70))
    d.polygon([(260, 300), (500, 150), (740, 300)], fill=(185, 40, 35))
    d.rectangle([460, 420, 540, 560], fill=(110, 70, 40))
    d.rectangle([350, 350, 420, 420], fill=(170, 215, 245))
    d.rectangle([800, 360, 850, 600], fill=(100, 65, 35))
    d.ellipse([720, 160, 930, 400], fill=(30, 120, 40))


def _umbrella(d: ImageDraw.ImageDraw) -> None:
    d.rectangle([0, 0, 1280, 300], fill=(160, 215, 245))
    d.rectangle([0, 300, 1280, 440], fill=(30, 185, 195))
    d.rectangle([0, 440, 1280, 720], fill=(238, 215, 160))
    d.line([640, 200, 640, 640], fill=(90, 60, 40), width=14)
    for i, color in enumerate(((235, 200, 40), (200, 35, 40)) * 2):
        d.pieslice([340, 120, 940, 420], 180 + i * 45, 225 + i * 45, fill=color)


DRAWERS = {"bicycle": _bicycle, "sailboat": _sailboat, "snowman": _snowman,
           "cone": _cone, "house": _house, "umbrella": _umbrella}


def render(name: str) -> Image.Image:
    im = Image.new("RGB", SIZE, (255, 255, 255))
    DRAWERS[name](ImageDraw.Draw(im))
    return im
