"""Reverse-video labels that assemble from character cells and decode in place.

The timing was measured from consecutive Coinbase reference frames, rather than
from a generic glitch preset.  The reveal has three distinct operations: isolated
inverse cells, a growing solid strip, then the last wrong symbols resolving.  The
readable hold never changes position, typeface, size, text, or opacity.

``draw_label`` is a drop-in replacement for render_director.timed_label.  It draws
into that renderer's L ink image, or into an RGB image for standalone previews.
``label_state`` exposes the exact printed cells for inspection or obstacle masks.
"""
from functools import lru_cache
import math

from PIL import ImageDraw, ImageFont


BLUE = (20, 31, 244)
MONO = '/System/Library/Fonts/Menlo.ttc'
SYMBOLS = '#*%$+<>/[]_!?01='
REFERENCE_FPS = 24000 / 1001


@lru_cache(maxsize=64)
def _font(size):
    return ImageFont.truetype(MONO, max(5, int(size)), index=1)


def _noise(index, step, seed):
    """Small deterministic hash; no process-random hash or mutable RNG state."""
    v = (index * 374761393 + step * 668265263 + seed * 1442695041) & 0xffffffff
    v = ((v ^ (v >> 13)) * 1274126177) & 0xffffffff
    return ((v ^ (v >> 16)) & 0xffffffff) / 4294967296.


def _ease(x):
    x = min(1., max(0., x))
    return x * x * (3. - 2. * x)


def label_state(txt, xy, t, start, end, size=29, *, reveal=None,
                retire=None, seed=0, direction=None, fps=24):
    """Return a deterministic set of inverse/normal cells at fixed text anchors.

    xy is the centre of the *complete* label.  ``reveal`` and ``retire`` are
    durations, not timestamps.  Explicit reveal/retire overrides allow several
    labels to join on one visual beat.  No frame-to-frame simulation state is
    needed, so the renderer may request arbitrary or out-of-order frames.
    """
    duration = float(end - start)
    if not txt or duration <= 0 or t < start or t >= end:
        return {'visible': False, 'phase': 'hidden', 'cells': (), 'rect': None}
    n = len(txt)
    size = int(round(size))
    advance = size * .75
    width = round(advance * n) + 16
    height = round(size * 1.19)
    x0 = round(xy[0] - width / 2)
    y0 = round(xy[1] - height / 2)
    reveal = min(.458, duration * .32) if reveal is None else max(0., float(reveal))
    retire = min(.167, duration * .10) if retire is None else max(0., float(retire))
    # Even an unusually short caller interval gets a genuine readable hold.
    budget = duration * .65
    if reveal + retire > budget:
        scale = budget / (reveal + retire)
        reveal *= scale
        retire *= scale
    elapsed = float(t - start)
    if direction is None:
        direction = 'left' if txt.startswith('<<') else 'right'
    if direction not in ('left', 'right', 'center'):
        raise ValueError('direction must be left, right, or center')
    seed = int(seed) + sum((i + 1) * ord(c) for i, c in enumerate(txt))

    # Only the decoding clock is stepped. Position and the final reading hold
    # stay exact, independent of the output movie's frame rate.
    step = max(0, int(math.floor(elapsed * fps + 1e-7)))
    if reveal and elapsed < reveal:
        phase = 'decode'
        progress = elapsed / reveal
        cover = .075 + .925 * _ease(progress / .72)
    elif retire and t > end - retire:
        phase = 'release'
        progress = (t - (end - retire)) / retire
        cover = 1 - _ease(progress)
    else:
        phase = 'hold'
        progress = 1.
        cover = 1.

    cells = []
    for i, final in enumerate(txt):
        # Slot rank chooses strip growth, but it never shifts the glyph slot.
        rank = ((n - i - .5) / n if direction == 'left' else
                abs((i + .5) / n - .5) * 2 if direction == 'center' else
                (i + .5) / n)
        left = x0 if i == 0 else x0 + 8 + round(i * advance)
        right = x0 + width if i == n - 1 else x0 + 8 + round((i + 1) * advance)
        top = y0
        inverse = True
        char = final
        locked = phase == 'hold'
        if phase == 'decode':
            if rank > cover:
                continue
            # A slight irregularity in settlement order avoids an ordinary
            # typewriter. Once a slot resolves it never becomes wrong again.
            lock_at = .38 + .50 * rank + .10 * _noise(i, 0, seed)
            locked = progress >= lock_at or progress >= .985
            if not locked:
                tick = step // 2  # reference substitutions often hold 2 frames
                char = SYMBOLS[int(_noise(i, tick + 1, seed) * len(SYMBOLS))]
                # The first two/three beats are isolated inverse cells, as in
                # the reference. Their short pre-read offset is one text row.
                # Decoded letters are always on the final baseline.
                if progress < .21:
                    top += round(height * (.72 if (i + seed) % 2 else -.72))
            # The early islands join by whole cells, not by fading or scaling.
            if progress < .23 and (i + seed) % 3 == 1:
                continue
        elif phase == 'release':
            # The bar unsets into inverse glyph cells before they switch back
            # to ordinary white ink and vanish. No whole-label fade/pop.
            removal = .18 + .76 * _noise(i, 4, seed)
            if progress > removal:
                continue
            locked = progress < .22
            if not locked:
                char = SYMBOLS[int(_noise(i, step // 2 + 1, seed) * len(SYMBOLS))]
                inverse = progress < removal - .17
        cells.append({
            'index': i, 'char': char, 'locked': locked, 'inverse': inverse,
            'rect': (int(left), int(top), int(right), int(top + height)),
            'xy': (x0 + 8 + i * advance, top - size * .055),
        })
    return {'visible': True, 'phase': phase, 'cells': tuple(cells),
            'rect': (x0, y0, x0 + width, y0 + height),
            'size': size, 'reveal': reveal, 'retire': retire,
            'readable_from': start + reveal, 'readable_until': end - retire}


def draw_label(im, txt, xy, t, start, end, size=29, *, font=None,
               reveal=None, retire=None, seed=0, direction=None, fps=24,
               foreground=None, background=None):
    """Draw the label and return its state.

    ``font`` may be a Pillow font or a ``font(size)`` callback.  L images use
    255 for the white strip and 0 for its blue-cutout letters; RGB images use
    white and BLUE.  This keeps the existing pure-blue finishing pipeline.
    """
    state = label_state(txt, xy, t, start, end, size, reveal=reveal,
                        retire=retire, seed=seed, direction=direction, fps=fps)
    if not state['visible']:
        return state
    f = _font(state['size']) if font is None else font(state['size']) if callable(font) else font
    foreground = (255 if im.mode == 'L' else (255, 255, 255)) if foreground is None else foreground
    background = (0 if im.mode == 'L' else BLUE) if background is None else background
    d = ImageDraw.Draw(im)
    # Paint all white cells first, otherwise the next cell's white rectangle
    # could cover overhanging ink from the previous letter.
    for cell in state['cells']:
        if cell['inverse']:
            x0, y0, x1, y1 = cell['rect']
            d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=foreground)
    for cell in state['cells']:
        d.text(cell['xy'], cell['char'], font=f,
               fill=background if cell['inverse'] else foreground)
    return state


# Import under this name to replace existing calls without another wrapper.
timed_label = draw_label
