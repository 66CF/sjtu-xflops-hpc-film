#!/usr/bin/env python3
"""Perspective ASCII solids for the kinetic Xflops film.

Public functions ``accelerator`` and ``tensor_cube`` accept time (seconds), a
screen-space center, pixel/world-unit scale, and XYZ Euler angles in radians.
They return the same glyph dictionary: xy, size, angle, alpha, depth, glyph.
The optional ``hull`` is a clockwise projected silhouette for erasing the
particle field behind the opaque device before drawing its ASCII material.

The front is local -Z; rotation=(0, 0, 0) is the readable front view.  Coordinates
are sampled on actual extruded faces, advected continuously in surface UV, lit
by transformed normals, ray-occluded by opaque layers, then projected.  Time
does not secretly rotate the object: the caller owns its complete choreography.
"""
from functools import lru_cache
import math
import numpy as np

_CHARS = np.array(list("01/+*:<>=#%X"))
_CAMERA_DISTANCE = 7.0


def _rotation(r):
    rx, ry, rz = r
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx),
                             math.cos(ry), math.sin(ry),
                             math.cos(rz), math.sin(rz))
    x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return z @ y @ x


def _smooth(a, b, x):
    q = np.clip((x-a)/(b-a), 0, 1)
    return q*q*(3-2*q)


def _box(center, half, kind, spacing=(.054, .081)):
    """Return face templates, never edges represented by flat target points."""
    surfaces = []
    for axis in range(3):
        uv = [q for q in range(3) if q != axis]
        # A vertical face keeps its long dimension finely sampled. Thin faces
        # retain at least two rows so a flip has tangible thickness.
        nx = max(2, int(2*half[uv[0]]/spacing[0]))
        ny = max(2, int(2*half[uv[1]]/spacing[1]))
        u, v = np.meshgrid((np.arange(nx)+.5)/nx,
                           (np.arange(ny)+.5)/ny)
        for sign in (-1, 1):
            surfaces.append(dict(c=np.array(center,dtype=float), h=np.array(half,dtype=float),
                                 axis=axis, uv=uv, sign=sign, kind=kind,
                                 u=u.ravel(), v=v.ravel(), nx=nx, ny=ny))
    return surfaces


@lru_cache(maxsize=2)
def _geometry(kind):
    surfaces, solids = [], []
    if kind == "accelerator":
        boxes = [
            # A broad socket lip, a thick package, a raised central die.
            ((0, 0, .095), (1.35, 1.35, .14), "socket"),
            ((0, 0, -.095), (1.25, 1.25, .125), "package"),
            ((0, 0, -.285), (.69, .69, .095), "die"),
        ]
        for c, h, material in boxes:
            surfaces += _box(c, h, material)
            solids.append((np.array(c)-h, np.array(c)+h))
        # Sixteen separate physical connector pins along each edge.  Their
        # ends and side faces remain visible through a full 3D turnover.
        for a in range(2):
            for sign in (-1, 1):
                for lane, v in enumerate(np.linspace(-1.18, 1.18, 17)):
                    c = [0., 0., .105]; h = [.041, .041, .047]
                    c[a] = sign*1.48; c[1-a] = v; h[a] = .175
                    surfaces += _box(c, h, "pin", (.059, .072))
        extent = (1.66, 1.66, .39)
    else:
        surfaces += _box((0, 0, 0), (.83, .83, .83), "tensor", (.054, .074))
        solids.append((np.array([-.83]*3), np.array([.83]*3)))
        extent = (.83, .83, .83)
    return surfaces, solids, np.array(extent)


def _texture(s, p, u, v):
    """A sparse, designed ASCII material with clear blue negative space."""
    kind, axis, sign = s["kind"], s["axis"], s["sign"]
    n = len(u)
    if kind == "pin":
        return np.full(n, .98), np.full(n, 5, dtype=int)
    if kind == "tensor":
        # Bright outer frame and 3D lattice; face centers retain breathing room.
        edge = np.maximum(abs(u*2-1), abs(v*2-1))
        grid = np.minimum(abs(((u*4+.5) % 1)-.5),
                          abs(((v*4+.5) % 1)-.5))
        a = _smooth(.84, .93, edge)
        a = np.maximum(a, .91*(1-_smooth(.025, .072, grid)))
        index = (np.floor(u*34)+np.floor(v*29)*3).astype(int)%11
        return a, index
    if axis != 2:
        # Long rim marks follow the real side face, emphasising its thickness.
        a = np.full(n, .82 if kind == "socket" else .94)
        index = (np.floor(u*37)+np.floor(v*11)).astype(int)%11
        return a, index
    x, y = p[:, 0], p[:, 1]
    r = np.maximum(abs(x), abs(y))
    if kind == "die" and sign < 0:
        # A cut-out X mark built from moving material, never overlaid text.
        outer = _smooth(.56, .62, r)
        diagonal = np.minimum(abs(x-y), abs(x+y))
        cross = (1-_smooth(.095, .15, diagonal))*(1-_smooth(.48, .56, r))
        # The almost empty gap makes the X read at speed.
        a = np.maximum(.025, np.maximum(.92*outer, .99*cross))
        index = np.where(cross>.3, 9, 4 + (np.floor(u*37).astype(int)%2))
        return a, index
    if kind == "package" and sign < 0:
        # Double etched square rim, with radial PCB traces on the substrate.
        border = _smooth(1.15, 1.20, r)
        inner = 1-_smooth(.015, .042, abs(r-.87))
        # Trace lines turn at right angles; animated UV samples pass through
        # these fixed physical paths instead of sliding a completed picture.
        nearest = np.minimum(abs((x*5+.5)%1-.5), abs((y*5+.5)%1-.5))
        traces = (1-_smooth(.035, .09, nearest))*_smooth(.75, .86, r)
        a = np.maximum(.028, np.maximum(.96*border, np.maximum(.72*inner, .47*traces)))
        index = (np.floor(u*79)+np.floor(v*53)*3).astype(int)%11
        return a, index
    # Socket and back are designed outlines with crossing buses, so the prop
    # never becomes an opaque white slab as it turns away from the camera.
    h = max(s["h"][0], s["h"][1])
    border = _smooth(h-.11, h-.045, r)
    horizontal = 1-_smooth(.02, .065, abs(y))
    vertical = 1-_smooth(.02, .065, abs(x))
    a = np.maximum(.028, np.maximum(.90*border, .50*np.maximum(horizontal, vertical)))
    index = (np.floor(u*71)+np.floor(v*47)*5).astype(int)%11
    return a, index


def _convex_hull(points):
    p = sorted(set(map(tuple, np.round(points, 3))))
    if len(p) < 3:
        return np.array(p)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower = []
    for a in p:
        while len(lower)>1 and cross(lower[-2],lower[-1],a)<=0: lower.pop()
        lower.append(a)
    upper=[]
    for a in reversed(p):
        while len(upper)>1 and cross(upper[-2],upper[-1],a)<=0: upper.pop()
        upper.append(a)
    return np.asarray(lower[:-1]+upper[:-1])


def _make(kind, t, center, scale, rotation):
    surfaces, solids, extent = _geometry(kind)
    R = _rotation(rotation)
    camera = np.array([0., 0., -_CAMERA_DISTANCE]) @ R
    points, normals, tangents, alphas, chars = [], [], [], [], []
    for face in surfaces:
        axis, uv, sign = face["axis"], face["uv"], face["sign"]
        normal = np.zeros(3); normal[axis] = sign
        # Convex plane facing test in local coordinates, before all expensive
        # material work. Perspective camera location is transformed correctly.
        plane = face["c"].copy(); plane[axis] += sign*face["h"][axis]
        if np.dot(normal, camera-plane) <= 0:
            continue
        # Flow is continuous in physical UV and has per-lane differential speed.
        # t=0 gives a nearly ordered face; any later t has shear and curl.
        u0, v0 = face["u"], face["v"]
        u = (u0 + t*(.019+.008*np.sin(v0*9)) +
             .009*np.sin(v0*14+t*.83)) % 1
        v = (v0 + .006*np.sin(u0*12+t*.67)) % 1
        p = np.tile(face["c"], (len(u),1))
        p[:,axis] += sign*face["h"][axis]
        p[:,uv[0]] += (2*u-1)*face["h"][uv[0]]
        p[:,uv[1]] += (2*v-1)*face["h"][uv[1]]
        a, ch = _texture(face, p, u, v)
        # A crisp object is made from visible white glyphs and real blue gaps.
        # Low-luminance interior samples previously read as a wireframe haze.
        keep = a > .30
        if not np.any(keep): continue
        p=p[keep];a=np.minimum(1.,a[keep]*1.20+.09);ch=ch[keep]
        tangent=np.zeros(3);tangent[uv[0]]=1
        points.append(p); alphas.append(a); chars.append(ch)
        normals.append(np.tile(normal,(len(p),1)))
        tangents.append(np.tile(tangent,(len(p),1)))
    p=np.concatenate(points); n=np.concatenate(normals)
    tangent=np.concatenate(tangents); alpha=np.concatenate(alphas)
    glyph=np.concatenate(chars)
    # Each point is checked against the opaque layered body. This is exact
    # segment/slab ray occlusion, including at the nearly edge-on poses.
    ray = p-camera
    inv = np.divide(1.,ray,out=np.full_like(ray,1e15),where=abs(ray)>1e-12)
    visible = np.ones(len(p),dtype=bool)
    for lo,hi in solids:
        near=(lo-camera)*inv;far=(hi-camera)*inv
        enter=np.max(np.minimum(near,far),axis=1)
        leave=np.min(np.maximum(near,far),axis=1)
        hidden=(leave>=np.maximum(enter,0))&(enter<.9999)&(leave>0)
        visible &= ~hidden
    p=p[visible];n=n[visible];tangent=tangent[visible]
    alpha=alpha[visible];glyph=glyph[visible]
    q=np.einsum("ij,kj->ik",p,R)
    z=q[:,2]+_CAMERA_DISTANCE
    fac=scale*_CAMERA_DISTANCE/z
    xy=np.column_stack([center[0]+q[:,0]*fac,center[1]-q[:,1]*fac])
    world_n=np.einsum("ij,kj->ik",n,R)
    # Camera-side light keeps all visible faces readable, with sides subdued.
    light=np.array([-.34,.47,-.81]);light/=np.linalg.norm(light)
    shade=.85+.15*np.maximum(0,np.einsum("ij,j->i",world_n,light))
    alpha=np.clip(alpha*shade,0,1)
    tq=np.einsum("ij,kj->ik",p+tangent*.01,R)
    tfac=scale*_CAMERA_DISTANCE/(tq[:,2]+_CAMERA_DISTANCE)
    tx= center[0]+tq[:,0]*tfac
    ty= center[1]-tq[:,1]*tfac
    angle=np.degrees(np.arctan2(ty-xy[:,1],tx-xy[:,0]))
    angle=(angle+90)%180-90
    # Glyphs retain the typographic field's upright reading direction. Their
    # surface tangent causes a small bank, never unreadable full text rotation.
    angle=np.clip(angle*.22,-18,18)
    size=np.clip(fac*.079,8.5,58)
    # A sparse screen-space Z buffer resolves projected pin overlaps.  Points
    # on the same physical plane are not culled simply for being neighbours.
    cell=max(2.5,scale*.012)
    bins=np.floor(xy/cell).astype(int)
    keys=bins[:,0].astype(np.int64)*100000+bins[:,1]
    _, unique=np.unique(keys,return_inverse=True)
    depths=np.full(int(unique.max())+1,np.inf)
    np.minimum.at(depths,unique,z)
    visible=z<=depths[unique]+.035
    order=np.argsort(-z[visible],kind="stable")
    ii=np.where(visible)[0][order]
    # Hull follows actual sampled geometry (including independent pin ends).
    # It is intentionally absent any decorative frame or fixed title banner.
    hull=_convex_hull(xy[ii])
    return dict(xy=xy[ii],size=size[ii],angle=angle[ii],alpha=alpha[ii],
                depth=z[ii],glyph=_CHARS[glyph[ii]],hull=hull,
                world_rotation=R)


def accelerator(t, center=(960,540), scale=300, rotation=(0,0,0)):
    """Layered HPC accelerator with four rows of pins and a central X die."""
    return _make("accelerator",float(t),center,float(scale),rotation)


def tensor_cube(t, center=(960,540), scale=300, rotation=(0,0,0)):
    """An opaque, physically rotating lattice cube for the early field burst."""
    return _make("tensor",float(t),center,float(scale),rotation)


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image, ImageDraw, ImageFont
    out=Path(__file__).resolve().parents[1]/"work"/"kinetic-object"
    out.mkdir(parents=True,exist_ok=True)
    fontpath="/System/Library/Fonts/Menlo.ttc"
    poses=[("front",(0,0,0)),("oblique",(.30,.63,-.17)),
           ("edge",(.13,1.50,-.12)),("back",(.25,2.70,.17))]
    thumbs=[]
    for name,pose in poses:
        state=accelerator(6.7,rotation=pose,scale=225)
        im=Image.new("RGB",(1920,1080),(25,18,246));d=ImageDraw.Draw(im)
        fonts={}
        for j in range(len(state["xy"])):
            x,y=state["xy"][j]; fs=max(5,round(state["size"][j]))
            if fs not in fonts:fonts[fs]=ImageFont.truetype(fontpath,fs)
            a=state["alpha"][j]
            color=tuple(round(b+(w-b)*a) for b,w in zip((25,18,246),(250,250,255)))
            d.text((x,y),str(state["glyph"][j]),font=fonts[fs],fill=color,anchor="mm")
        d.text((70,50),f"{name.upper()}  |  {len(state['xy'])} GLYPHS",
               font=ImageFont.truetype(fontpath,25),fill="white")
        im.save(out/f"{name}.png")
        thumbs.append(im.resize((960,540)))
        print(name,len(state["xy"]))
    contact=Image.new("RGB",(1920,1080))
    for i,im in enumerate(thumbs):contact.paste(im,((i%2)*960,(i//2)*540))
    contact.save(out/"poses.jpg")
