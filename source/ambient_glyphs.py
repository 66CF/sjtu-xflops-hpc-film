"""Continuous secondary type for the knot and the quiet closing passage.

The repeating word field is independent editorial typography, while closing
remnants retain the exact identity, size, ink, position and velocity of real
fluid tracers. No layer is randomly regenerated on a seek and no entire
layer fades out. Every printed mark enters or leaves through a frame edge.

Integration (all calls use the director's L-mode ink image)::

    draw_background(im, t, batch=batch)  # before physical_flow / subjects
    physical_flow(..., exclude_ids=exclusion_ids(t))

When the knot is on screen, combine its source_ids with exclusion_ids(t).
The knot background itself does not replace the noncaptured fluid tracers.
"""
from functools import lru_cache
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1920, 1080
FLOW_PATH = ROOT / 'work/physics-v5/flow.npz'
MENLO = '/System/Library/Fonts/Menlo.ttc'
REMNANT_START = 19.10
FLOW_HANDOFF = 19.75


def _smooth(x):
    u = np.clip(x, 0., 1.)
    return u*u*(3.-2.*u)


@lru_cache(maxsize=128)
def _font(size):
    return ImageFont.truetype(MENLO, max(6, int(size)), index=1)


@lru_cache(maxsize=8192)
def _tile(ch, size, angle):
    w, h = math.ceil(size*.73), math.ceil(size*1.25)
    sp = Image.new('L', (w, h))
    ImageDraw.Draw(sp).text((w/2, -size*.04), str(ch), font=_font(size),
                           fill=255, anchor='ma')
    return sp.rotate(angle, Image.Resampling.BICUBIC, expand=True) if angle else sp


def _print(im, xy, size, glyph, alpha=1., angle=None, bold=True):
    """Director-compatible fallback, including its glyph anchor convention."""
    n = len(xy)
    sizes = np.broadcast_to(size, (n,))
    alphas = np.broadcast_to(alpha, (n,))
    angles = np.zeros(n) if angle is None else np.broadcast_to(angle, (n,))
    for p, s, ch, a, rot in zip(xy, sizes, glyph, alphas, angles):
        if not (-90 < p[0] < WIDTH+90 and -90 < p[1] < HEIGHT+90):
            continue
        if ch == ' ' or a <= .025:
            continue
        sp = _tile(str(ch), int(np.clip(np.rint(s), 7, 84)),
                   int(np.rint(rot/4)*4)%360)
        im.paste(round(float(np.clip(a, 0, 1))*255),
                 (round(p[0]-sp.width/2), round(p[1]-sp.height/2)), sp)


@lru_cache(maxsize=1)
def _word_material():
    """Intact words form columns, with empty channels between those columns."""
    positions, glyphs, lane, phase = [], [], [], []
    phrases = ('PARALLEL', 'COMPUTE', 'AI_INFRA')
    for col in range(-1, 10):
        phrase = phrases[col % len(phrases)]
        for row in range(-6, 41):
            center_x = 80. + 230.*col
            center_y = row*34. + 13.*(col % 2)
            for j, ch in enumerate(phrase):
                positions.append((center_x+(j-(len(phrase)-1)/2)*12.1,
                                  center_y))
                glyphs.append(ch)
                lane.append(col)
                phase.append(row)
    return (np.asarray(positions), np.asarray(glyphs),
            np.asarray(lane), np.asarray(phase))


def word_field_state(t):
    """A fixed type material bent by a travelling current, never respawned."""
    if not 7.20 <= t <= 10.10:
        return None
    base, glyph, lane, row = _word_material()
    x, y = base[:, 0].copy(), base[:, 1].copy()
    # Different row phases make a passing wave, rather than a rigid slide.
    y -= 82.*(t-8.25)
    x += 37.*np.sin(y*.0046 + t*1.1) + 14.*np.sin(row*.29-t*1.7)
    y += 10.*np.sin(lane*.92+t*1.3)
    # The type bends around the volume: leave room for its white silhouette
    # without erasing letters or fading a circular mask through the print.
    dx, dy = x-960., (y-530.)*1.13
    radius = np.hypot(dx, dy)
    enlarged = np.sqrt(radius*radius+390.*390.)
    ratio = enlarged/np.maximum(radius, 1.)
    x = 960.+dx*ratio
    y = 530.+dy*ratio/1.13
    # A right-to-left arrival takes place behind the departing chip. Each row
    # follows the same eased current with a small delay, maintaining its ink.
    arrival = _smooth((t-7.22-.055*np.sin(row*.27)-.013*lane)/1.03)
    x += 2440.*(1.-arrival)
    # The final updraft carries whole word rows out as the terminal returns.
    # The clearing front stays above the next terminal's rising first row.
    lift_time = np.maximum(0., t-9.20-.025*np.sin(lane))
    y -= 9200.*lift_time*lift_time
    alpha = np.where(lane % 3 == 0, .56, .42)
    angle = np.zeros(len(x))
    return dict(xy=np.column_stack((x, y)), size=np.full(len(x), 20.),
                glyph=glyph, alpha=alpha, angle=angle)


def _sample(data, t, key):
    ts = data['times']
    nearest=int(np.argmin(abs(ts-t)))
    if abs(float(ts[nearest])-t)<2e-5:
        return data[key][nearest]
    i = int(np.clip(np.searchsorted(ts, t, side='right')-1, 0, len(ts)-2))
    u = float(np.clip((t-ts[i])/(ts[i+1]-ts[i]), 0., 1.))
    return data[key][i]*(1.-u)+data[key][i+1]*u


def _fixed_visible(data):
    ids = np.arange(len(data['group']))
    try:
        from glyph_material import visible_glyphs
        return np.asarray(visible_glyphs(data['group']), dtype=bool)
    except (ImportError, AttributeError, TypeError):
        # Matches the director's time-independent print choice. All terminal
        # letters survive; the high-density inlet/foreground are sampled once.
        g = data['group']
        hashed = ((ids.astype(np.uint64)*2654435761)%4294967296)%3 != 0
        return np.where(g == 0, True, np.where(g == 2, ids % 4 == 0,
                        np.where(g == 3, True, hashed)))


@lru_cache(maxsize=2)
def _remnant_data(path, mtime_ns):
    """Select neighboring real particles, not new random closing confetti."""
    with np.load(path) as z:
        data = {key: z[key] for key in
                ('times', 'xy', 'velocity', 'angle', 'size', 'glyph',
                 'group', 'birth')}
    p = _sample(data, FLOW_HANDOFF, 'xy').astype(np.float64)
    v = _sample(data, FLOW_HANDOFF, 'velocity').astype(np.float64)
    angle = _sample(data, FLOW_HANDOFF, 'angle').astype(np.float64)
    angular_velocity = (angle-_sample(data, FLOW_HANDOFF-1/24, 'angle').astype(np.float64))*24
    ids = np.arange(len(p))
    # These same marks are visible in the preceding decimated stair field.
    allowed = ((data['group'] <= 1) & (ids % 4 == 0) & _fixed_visible(data) &
               (data['birth'] <= REMNANT_START) & (data['glyph'] != ' ') &
               (p[:, 0] > 75.) & (p[:, 0] < WIDTH-75.) &
               (p[:, 1] > 65.) & (p[:, 1] < HEIGHT-65.))
    # Keep the future closing message's full rectangle clear, including the
    # short initial inertial glide. The eight regions make deliberate clumps.
    allowed &= ((p[:, 0] < 680.) | (p[:, 0] > 1580.) |
                (p[:, 1] < 340.) | (p[:, 1] > 870.))
    regions = ((270., 210.), (590., 450.), (1700., 230.), (1700., 740.),
               (520., 940.), (1390., 955.), (1000., 160.), (270., 770.))
    end_times = (23.05, 24.60, 22.70, 24.35, 22.25, 23.55, 24.05, 21.70)
    selected, forces, drifts, cluster_index = [], [], [], []
    used = set()
    for group_index, (region, end_time) in enumerate(zip(regions, end_times)):
        eligible = ids[allowed & ~np.isin(ids, list(used))]
        if not len(eligible):
            continue
        dist = np.linalg.norm(p[eligible]-np.asarray(region), axis=1)
        nearest = eligible[np.argsort(dist)[:7]]
        # Use a compact patch of the real field, even if its precise location
        # changes after a new GPU bake. Never move particles to a design grid.
        anchor = p[nearest[0]]
        close = eligible[np.linalg.norm(p[eligible]-anchor, axis=1) < 135.]
        near_dist = np.linalg.norm(p[close]-anchor, axis=1)
        chosen = close[np.argsort(near_dist)[:7]]
        if len(chosen) < 2:
            chosen = nearest[:3]
        used.update(map(int, chosen))
        center = p[chosen].mean(axis=0)
        mean_v = v[chosen].mean(axis=0)
        # Choose the nearest edge; a small tangential drift keeps the patch
        # alive while a steadily growing outward force drains it off screen.
        edges = np.array((center[0], WIDTH-center[0], center[1], HEIGHT-center[1]))
        edge = int(np.argmin(edges))
        direction = np.array(((-1., 0.), (1., 0.), (0., -1.), (0., 1.)))[edge]
        drift = direction*(20.+group_index*2.)
        tangent = np.array((-direction[1], direction[0]))
        drift += tangent*(-12.+4.*(group_index%5))
        life = end_time-FLOW_HANDOFF
        decay_integral = (1.-math.exp(-7.*life))/7.
        endpoint = center + mean_v*decay_integral + drift*(life-decay_integral)
        needed = max(0., edges[edge]+110.-float(np.dot(endpoint-center, direction)))
        ramp_force = direction*(6.*needed/(life**3))
        selected.extend(chosen)
        forces.extend([ramp_force]*len(chosen))
        drifts.extend([drift]*len(chosen))
        cluster_index.extend([group_index]*len(chosen))
    selected = np.asarray(selected, dtype=int)
    trajectory={key:data[key][:,selected] for key in ('xy','velocity','angle')}
    trajectory['times']=data['times']
    return dict(ids=selected, xy=p[selected], velocity=v[selected],
                angle=angle[selected], angular_velocity=angular_velocity[selected],
                size=data['size'][selected],
                glyph=data['glyph'][selected], force=np.asarray(forces),
                drift=np.asarray(drifts), cluster=np.asarray(cluster_index),
                trajectory=trajectory)


def _remnants():
    if not FLOW_PATH.exists():
        return None
    return _remnant_data(str(FLOW_PATH), FLOW_PATH.stat().st_mtime_ns)


def exclusion_ids(t):
    """IDs now printed here instead of in the fading late physical-flow pass."""
    if t < REMNANT_START:
        return np.empty(0, dtype=int)
    data = _remnants()
    return data['ids'] if data is not None else np.empty(0, dtype=int)


def remnant_state(t):
    if t < REMNANT_START:
        return None
    data = _remnants()
    if data is None:
        return None
    alpha=.27+.73*float(_smooth((t-REMNANT_START)/.70))
    if t<=FLOW_HANDOFF:
        # While other physical letters remain on screen, keep these IDs in
        # the same collision-solved trajectory. Changing their forces early
        # would let them cross the still-visible neighbouring flow bodies.
        return dict(xy=_sample(data['trajectory'],t,'xy'),size=data['size'],
                    glyph=data['glyph'],alpha=alpha,
                    angle=_sample(data['trajectory'],t,'angle'),
                    ids=data['ids'],cluster=data['cluster'])
    dt = t-FLOW_HANDOFF
    drag = (1.-math.exp(-7.*dt))/7.
    p = data['xy'] + data['velocity']*drag + data['drift']*(dt-drag)
    p = p + data['force']*(dt**3/6.)
    # Small coherent transverse motion from a decaying wake: zero position
    # AND zero velocity at the handoff, so it cannot introduce a visible snap.
    tangent = np.column_stack((-data['drift'][:, 1], data['drift'][:, 0]))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.)
    wake = (1.-math.exp(-3.*dt))**2*np.sin(dt*1.45+data['cluster']*.2)
    p += tangent*(7.*wake[:, None])
    # Retain the initial angle and damp angular motion; no font/material swap.
    angle = data['angle'] + data['angular_velocity']*(1.-math.exp(-5.*dt))/5.
    # The preceding distant fluid layer prints at .27 contrast. Preserve it
    # exactly at transfer; as the busy stage clears these few marks come forward.
    return dict(xy=p, size=data['size'], glyph=data['glyph'], alpha=alpha,
                angle=angle, ids=data['ids'], cluster=data['cluster'])


def draw_background(im, t, batch=None, text=None):
    """Print behind subjects into an L-mode canvas; return transferred IDs."""
    if im.mode != 'L':
        raise ValueError('ambient glyphs expect an L-mode ink image')
    render = batch or _print
    for state in (word_field_state(t), remnant_state(t)):
        if state is not None:
            render(im, state['xy'], state['size'], state['glyph'],
                   state['alpha'], state['angle'], bold=True)
    return exclusion_ids(t)
