"""Thick ASCII platforms with shared visible faces and solid collision masks."""
import math
import numpy as np
from PIL import Image, ImageDraw
from stair_choreography import PLATFORMS

THICKNESS = 72.
DEPTH = (45., -28.)
STAGE = ((230., 970., 250.),) + PLATFORMS


def _smooth(x):
    x = float(np.clip(x, 0., 1.))
    return x*x*(3-2*x)


def platform_faces(t):
    rise = 68. * (1-_smooth((t-17.05)/.16))
    for x, y, w in STAGE:
        y += rise
        dx, dy = DEPTH
        top = [(x,y), (x+w,y), (x+w+dx,y+dy), (x+dx,y+dy)]
        front = [(x,y), (x+w,y), (x+w,y+THICKNESS), (x,y+THICKNESS)]
        side = [(x+w,y), (x+w+dx,y+dy),
                (x+w+dx,y+dy+THICKNESS), (x+w,y+THICKNESS)]
        yield x,y,w,(top,front,side)


def platform_mask(t, xx, yy):
    """One projected solid union, sampled on the solver's screen grid."""
    mask = np.zeros_like(xx, dtype=bool)
    if not 17.05 <= t < 19.95:
        return mask
    for x,y,w,_ in platform_faces(t):
        # Top and right side share a skewed depth coordinate.
        mask |= (xx>=x)&(xx<=x+w)&(yy>=y)&(yy<=y+THICKNESS)
        dx = np.clip(xx-(x+w),0,DEPTH[0])
        side_top = y + DEPTH[1]*dx/DEPTH[0]
        mask |= ((xx>=x+w)&(xx<=x+w+DEPTH[0])&
                 (yy>=side_top)&(yy<=side_top+THICKNESS))
        depth = (y-yy)/(-DEPTH[1])
        mask |= ((depth>=0)&(depth<=1)&(xx>=x+DEPTH[0]*depth)&
                 (xx<=x+w+DEPTH[0]*depth))
    return mask


def draw_platforms(im,t,batch):
    """Opaque blue faces with persistent white character masonry."""
    if not 17.05 <= t < 19.95:
        return
    alpha=_smooth((t-17.05)/.12)*(1-_smooth((t-19.55)/.4))
    for n,(x,y,w,faces) in enumerate(platform_faces(t)):
        cover=Image.new('L',im.size)
        d=ImageDraw.Draw(cover)
        for polygon in faces:d.polygon(polygon,fill=round(255*alpha))
        im.paste(0,(0,0),cover)
        xy=[];glyph=[];light=[]
        # Tight oblique hatch on the top provides a clear landing surface.
        for row in range(3):
            q=(row+.35)/3
            for col in range(math.ceil(w/14)):
                px=x+col*14+7+DEPTH[0]*q
                xy.append((px,y+DEPTH[1]*q-2));glyph.append('/');light.append(1.)
        # A four-row front face gives the platforms weight and readable depth.
        for row in range(4):
            for col in range(math.ceil(w/14)):
                xy.append((x+7+col*14,y+11+row*17))
                glyph.append('=#+'[(col//3+row+n)%3]);light.append(.88 if row<2 else .65)
        for col in range(3):
            q=(col+.5)/3
            for row in range(4):
                xy.append((x+w+DEPTH[0]*q,y+DEPTH[1]*q+10+row*17))
                glyph.append(':');light.append(.60)
        batch(im,np.asarray(xy),18,np.asarray(glyph),np.asarray(light)*alpha,bold=True)
