"""One visible RUN cursor, shared by the picture and the fluid boundary.

Coordinates are 1920x1080 screen pixels, Y down. Rotation is radians in the
same coordinate system (negative is an upward tilt). The cursor does not
fade into view: it crosses the left edge, travels through the terminal, and
leaves through the right edge. All interpolation has continuous velocity.
"""
from __future__ import annotations

import numpy as np

START = 3.04
END = 4.52
SIZE = (220.0, 116.0)
_TIMES = np.array([3.04, 3.50, 4.02, 4.52])
_POSITIONS = np.array([[-180., 940.], [570., 730.],
                       [1360., 400.], [2210., 210.]])
_VELOCITIES = np.array([[650., -145.], [1645., -565.],
                        [1575., -545.], [1880., -255.]])


def _position(t):
    t = float(np.clip(t, START, END))
    i = int(np.clip(np.searchsorted(_TIMES, t, side="right")-1, 0, 2))
    dt = _TIMES[i+1]-_TIMES[i]
    u = (t-_TIMES[i])/dt
    p = ((2*u**3-3*u**2+1)*_POSITIONS[i]
         +(u**3-2*u**2+u)*dt*_VELOCITIES[i]
         +(-2*u**3+3*u**2)*_POSITIONS[i+1]
         +(u**3-u**2)*dt*_VELOCITIES[i+1])
    v = ((6*u*u-6*u)*_POSITIONS[i]/dt
         +(3*u*u-4*u+1)*_VELOCITIES[i]
         +(-6*u*u+6*u)*_POSITIONS[i+1]/dt
         +(3*u*u-2*u)*_VELOCITIES[i+1])
    return p, v


def impact_state(t):
    """Return rigid rectangle center/size/rotation and boundary velocities.

    ``corners`` is a clockwise 4x2 polygon; ``velocity`` is px/s and
    ``angular_velocity`` rad/s. Use this geometry for both drawing and masks.
    ``label`` is fixed independent typography, never particle pool glyphs.
    """
    center, velocity = _position(t)
    rotation = float(np.arctan2(velocity[1], velocity[0])*.58)
    eps = 1/1200
    va = _position(t-eps)[1]
    vb = _position(t+eps)[1]
    omega = float((np.arctan2(vb[1],vb[0])-np.arctan2(va[1],va[0]))*.58/(2*eps))
    c, s = np.cos(rotation), np.sin(rotation)
    rot = np.array([[c,-s],[s,c]])
    corners = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])*SIZE
    return dict(active=START <= t <= END, center=center.astype(np.float32),
                size=np.array(SIZE,np.float32), rotation=rotation,
                velocity=velocity.astype(np.float32), angular_velocity=omega,
                corners=(corners@rot.T+center).astype(np.float32),
                label="RUN", font_size=51)


def impact_mask(t, resolution=(512,288)):
    """Return ``mask[H,W]`` and rigid ``velocity[H,W,2]`` in px/s.

    Sampling uses pixel centers across the complete 1920x1080 frame. Merge
    this mask with the film's other solids; the cursor wins on overlap.
    """
    w, h = map(int, resolution)
    state = impact_state(t)
    mask = np.zeros((h,w),bool)
    velocity = np.zeros((h,w,2),np.float32)
    if not state['active']:
        return mask, velocity
    xx, yy = np.meshgrid((np.arange(w)+.5)*1920/w,
                         (np.arange(h)+.5)*1080/h)
    ox = xx-state['center'][0]
    oy = yy-state['center'][1]
    c, s = np.cos(state['rotation']), np.sin(state['rotation'])
    mask = (abs(c*ox+s*oy)<=SIZE[0]/2)&(abs(-s*ox+c*oy)<=SIZE[1]/2)
    velocity[mask,0] = state['velocity'][0]-state['angular_velocity']*oy[mask]
    velocity[mask,1] = state['velocity'][1]+state['angular_velocity']*ox[mask]
    return mask, velocity
