"""Subtle display/film material for the blue-and-white ASCII film.

``finish(ink, t)`` takes an L-mode ink image (0 = blue, 255 = white),
returns an RGB image of the same size, and never changes its input.
It is deterministic, seekable, and needs only Pillow and NumPy.

Measured from the supplied reference, rather than a generic CRT preset:
* Clean blue at 5.5 s has RGB standard deviation of about 1.2 levels in
  the quiet lower field and 3.3 levels in the more textured top field.
* Top-field horizontal first-difference SD is about 2.7 levels, versus
  0.7 vertically: fine, irregular vertical striation is the main texture.
* White lettering stays crisp. There are no conspicuous horizontal scan
  bars, vignette, lens distortion, or large neon halos to reproduce.

The reference changes blue hue between scenes. This helper deliberately
keeps the production's selected blue (20, 31, 244) and neutral white 250.
The material is synthesized; no reference pixels or audio are reused.
"""

from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BLUE = (20, 31, 244)
WHITE = (250, 250, 250)
MENLO = "/System/Library/Fonts/Menlo.ttc"


def _stretch_noise(rng, width, height, cell_x, cell_y):
    """A zero-mean unit-SD field with independently controlled grain axes."""
    small = rng.normal(size=(max(2, math.ceil(height / cell_y)),
                             max(2, math.ceil(width / cell_x)))).astype(np.float32)
    field = np.asarray(Image.fromarray(small).resize(
        (width, height), Image.Resampling.BICUBIC), dtype=np.float32).copy()
    field -= field.mean()
    field /= max(float(field.std()), 1e-6)
    return field


@lru_cache(maxsize=3)
def _fields(width, height):
    rng = np.random.default_rng(240510)
    # Irregular streaks have narrow width and long, varying vertical extent.
    # The second scale breaks up the mechanically uniform "striped wallpaper"
    # appearance of a repeated sine-wave scanline effect.
    fine = _stretch_noise(rng, width, height, 1.8, 95)
    long = _stretch_noise(rng, width, height, 6.0, 290)
    grain = fine * 0.87 + long * 0.35
    grain -= grain.mean()
    grain /= grain.std()
    other = _stretch_noise(rng, width, height, 3.2, 155)

    # A restrained, spatially varying strength matches the source's quieter
    # lower field. It changes texture amplitude, not the base blue exposure.
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    strength = 1.12 + 2.4 * np.exp(-yy * 3.8)
    micro = rng.normal(0, 1, (height, width)).astype(np.float32)
    return grain, other, strength, micro


def finish(ink: Image.Image, t: float) -> Image.Image:
    """Map grayscale ink to the restrained blue/white display material.

    At 1920x1080 the irregular vertical field has roughly 1.2–3.3/255
    amplitude. Ink edges get a 1–2 px, low-energy bloom and less than 0.05 px
    effective RGB separation. White tiles remain nearly neutral and smooth.
    From 19.5 to 20.5 seconds the substrate settles to three percent of its
    texture strength, leaving the closing typography on nearly flat blue.
    No displacement is applied, so independent captions never drift.
    """
    if ink.mode != "L":
        raise ValueError("finish expects a PIL image in mode 'L'")
    width, height = ink.size
    a = np.asarray(ink, dtype=np.float32) / np.float32(255.0)

    # Edge energy only. Broad empty fields and solid white tiles remain flat
    # in exposure; bloom does not fog the whole image or close small counters.
    narrow = np.asarray(ink.filter(ImageFilter.GaussianBlur(0.45)),
                        dtype=np.float32) / np.float32(255.0)
    halo = np.asarray(ink.filter(ImageFilter.GaussianBlur(1.35)),
                      dtype=np.float32) / np.float32(255.0)
    a = np.clip(a * 0.95 + narrow * 0.05 + halo * (1.0 - a) * 0.035,
                0.0, 1.0)

    grain, other, strength, micro = _fields(width, height)
    # Stable long streaks plus very small, temporally changing fine grain.
    # The two independent fields vary slowly, without a pulsing exposure.
    phase = math.sin(float(t) * 0.73) * 0.16
    streaks = (grain * math.cos(phase) + other * math.sin(phase)) * strength
    frame = math.floor(float(t) * 24.0)
    fine_grain = np.roll(micro, (frame * 37 % height, frame * 53 % width),
                         axis=(0, 1)) * 0.45
    texture = streaks + fine_grain
    # The reference's quiet ending drops the active display texture. Ease the
    # entire substrate down without changing blue exposure or typographic ink.
    quiet = min(1.0, max(0.0, (float(t) - 19.5) / 1.0))
    quiet = quiet * quiet * (3.0 - 2.0 * quiet)
    texture *= 1.0 - 0.97 * quiet
    # Most texture belongs to the blue substrate. Very light white-text
    # grain avoids sterile vector edges without dirtying the reverse labels.
    texture *= 1.0 - a * 0.94

    # Fractional-pixel LCD-like edge bleed, intentionally below the level of
    # visible RGB glitch. Edge padding avoids a seam on either side.
    left = np.empty_like(a)
    left[:, 0] = a[:, 0]
    left[:, 1:] = a[:, :-1]
    right = np.empty_like(a)
    right[:, -1] = a[:, -1]
    right[:, :-1] = a[:, 1:]
    channels = (a * 0.958 + left * 0.042,
                a * 0.979 + right * 0.021,
                a)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    for c, coverage in enumerate(channels):
        value = BLUE[c] + (WHITE[c] - BLUE[c]) * coverage + texture
        rgb[:, :, c] = np.clip(np.rint(value), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


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
