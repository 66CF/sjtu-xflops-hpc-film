"""Flat blue substrate and neutral white ink, with no animated texture.

Every zero-ink pixel maps to exactly BLUE at every time. Antialiasing comes
from the glyph coverage itself; there is no grain, bloom or channel shift.
"""
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLUE = (20, 31, 244)
WHITE = (250, 250, 250)
MENLO = "/System/Library/Fonts/Menlo.ttc"


def finish(ink: Image.Image, t: float) -> Image.Image:
    """Apply a time-independent blue/white lookup table."""
    if ink.mode != "L":
        raise ValueError("finish expects a PIL image in mode 'L'")
    coverage = np.arange(256, dtype=np.float32)[:, None] / 255
    palette = np.rint(np.asarray(BLUE) + coverage *
                     (np.asarray(WHITE)-np.asarray(BLUE))).astype(np.uint8)
    return Image.fromarray(palette[np.asarray(ink)])


@lru_cache(maxsize=192)
def mono_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Menlo TTC index 0 is Regular; index 1 is Bold (verified on this host).

    Bold better matches reference terminal/reverse captions. Keep Regular
    for small ASCII surface detail, where Bold can fill counters at 10–15 px.
    """
    return ImageFont.truetype(MENLO, max(6, int(size)), index=int(bold))


@lru_cache(maxsize=8192)
def cached_glyph(ch: str, size: int, rotation: int = 0,
                 bold: bool = False, inverted: bool = False) -> Image.Image:
    """Optional centered L-mode glyph, suitable for rotation and compositing.

    ``inverted=True`` makes a white tile with a blue/zero-ink glyph. Use the
    returned image as a luminance layer; the tile itself needs an opaque mask
    so its dark glyph is not treated as transparency. The caller owns layout.
    """
    f = mono_font(size, bold)
    box = f.getbbox(ch, anchor="mm")
    padx, pady = max(2, round(size * 0.15)), max(2, round(size * 0.1))
    w, h = box[2] - box[0] + padx * 2, box[3] - box[1] + pady * 2
    layer = Image.new("L", (w, h), 255 if inverted else 0)
    ImageDraw.Draw(layer).text((padx - box[0], pady - box[1]), ch,
                              font=f, fill=0 if inverted else 255, anchor="mm")
    if rotation:
        layer = layer.rotate(rotation, resample=Image.Resampling.BICUBIC,
                             expand=True, fillcolor=0)
    return layer
