#!/usr/bin/env python3
"""
Generate a book cover from the center crop of the Unsplash photo.
Overlays title and author text in a classic academic book style.
"""

import os

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_IMAGE = os.path.join(REPO_ROOT, "peter-bryan-KORGgJI1odA-unsplash.jpg")
OUT_IMAGE = os.path.join(REPO_ROOT, "docs", "_static", "cover.jpg")

TITLE_LINE1 = "The Watchmaker's Guide"
TITLE_LINE2 = "to Population Genetics"
SUBTITLE = "Build every algorithm from first principles"
AUTHOR = "Kevin Korfmann"
COVER_W = 2550  # 8.5 in at 300 dpi
COVER_H = 3300  # 11 in at 300 dpi

FONTS = "/System/Library/Fonts/Supplemental"


def font(name, size):
    path = os.path.join(FONTS, name)
    if os.path.isfile(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def center_text(draw, y, text, fnt, fill):
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    draw.text(((COVER_W - tw) // 2, y), text, font=fnt, fill=fill)
    return bbox[3] - bbox[1]


def main():
    Image.MAX_IMAGE_PIXELS = 100_000_000

    print("Loading source image...")
    src = Image.open(SRC_IMAGE)
    sw, sh = src.size

    target_ratio = COVER_W / COVER_H
    if sw / sh > target_ratio:
        crop_h = sh
        crop_w = int(sh * target_ratio)
    else:
        crop_w = sw
        crop_h = int(sw / target_ratio)

    x0 = (sw - crop_w) // 2
    y0 = (sh - crop_h) // 2
    print(f"Cropping center: ({x0},{y0}) -> ({x0+crop_w},{y0+crop_h})")
    cropped = src.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    cover = cropped.resize((COVER_W, COVER_H), Image.LANCZOS)

    # Darken for text readability
    dark = Image.new("RGB", (COVER_W, COVER_H), (0, 0, 0))
    cover = Image.blend(cover, dark, 0.4)

    draw = ImageDraw.Draw(cover)

    cream = (235, 225, 205)
    gold = (200, 180, 140)
    light = (210, 200, 180)

    # Top rule
    rule_y = 360
    draw.line([(250, rule_y), (COVER_W - 250, rule_y)], fill=gold, width=3)

    # Title (Didot — elegant serif)
    title_font = font("Didot.ttc", 130)
    y = rule_y + 80
    h = center_text(draw, y, TITLE_LINE1, title_font, cream)
    y += h + 20
    center_text(draw, y, TITLE_LINE2, title_font, cream)
    y += h + 60

    # Subtitle (Futura — clean geometric sans)
    sub_font = font("Futura.ttc", 48)
    center_text(draw, y, SUBTITLE, sub_font, light)
    y += 100

    # Bottom rule below subtitle
    draw.line([(250, y), (COVER_W - 250, y)], fill=gold, width=3)

    # Author near bottom (Baskerville — classic book feel)
    author_font = font("Baskerville.ttc", 72)
    center_text(draw, COVER_H - 380, AUTHOR, author_font, cream)

    # Thin bottom rule
    draw.line([(250, COVER_H - 280), (COVER_W - 250, COVER_H - 280)], fill=gold, width=2)

    os.makedirs(os.path.dirname(OUT_IMAGE), exist_ok=True)
    cover.save(OUT_IMAGE, quality=95)
    print(f"Cover saved: {OUT_IMAGE}")


if __name__ == "__main__":
    main()
