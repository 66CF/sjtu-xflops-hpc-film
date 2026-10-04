#!/usr/bin/env python3
"""Analytic opaque 3D solids, sampled on a regular camera raster for ASCII.

No surface point clouds or wireframe points are returned.  Each screen sample
casts a perspective ray and stores its closest physical surface, Lambert/Blinn
illumination, and occupancy. The caller chooses its own ASCII luminance ramp.

Public API:
    sample_runner(t, rotation=(0, -.28, -.02), resolution=(160, 90))
    sample_chip(t, rotation=(.18, .48, -.13), resolution=(160, 90))
    sample_worker(t, rotation=(0, .68, 0), resolution=(80, 90))

``resolution`` is (width, height). Results contain ``luma`` (H,W float32, 0..1),
``mask`` (H,W bool), ``depth`` (H,W float32; inf outside), and ``normal`` (H,W,3).
Camera: perspective at (0,0,-7), world Y up, front/local visible face is -Z.
The fixed aperture is y=[-2.25,2.25]; x has the same pixel aspect.  Thus geometry
never automatically rescales when its silhouette changes during a run or flip.
Euler rotation angles are XYZ radians, applied to the complete posed object.
"""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np


def _rotation(angles):
    x,y,z=angles
    cx,sx,cy,sy,cz,sz=math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                     [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],
                     [-sy,cy*sx,cy*cx]],dtype=np.float32)


def _transform(points,matrix):
    return np.einsum('...j,ij->...i',points,matrix,optimize=False)


@lru_cache(maxsize=12)
def _camera(resolution):
    w,h=resolution
    xx,yy=np.meshgrid((np.arange(w,dtype=np.float32)+.5)/w,
                      (np.arange(h,dtype=np.float32)+.5)/h)
    d=np.stack([(xx-.5)*4.5*w/h,(.5-yy)*4.5,np.full_like(xx,7)],axis=-1).reshape(-1,3)
    d/=np.sqrt(np.sum(d*d,axis=1))[:,None]
    return np.array([0,0,-7],dtype=np.float32),d


@lru_cache(maxsize=12)
def _orthographic_camera(resolution):
    """Grounded stage camera: parallel rays preserve exact planted soles."""
    w,h=resolution
    xx,yy=np.meshgrid((np.arange(w,dtype=np.float32)+.5)/w,
                     (np.arange(h,dtype=np.float32)+.5)/h)
    origin=np.stack([(xx-.5)*4.5*w/h,(.5-yy)*4.5,np.full_like(xx,-7)],axis=-1).reshape(-1,3)
    rays=np.broadcast_to(np.array([0,0,1],dtype=np.float32),origin.shape)
    return origin,rays


def _ellipsoid(center,radii,material=.95,basis=None,tag=0):
    return ('ellipsoid',np.asarray(center,dtype=np.float32),
            np.asarray(radii,dtype=np.float32),
            np.eye(3,dtype=np.float32) if basis is None else np.asarray(basis,dtype=np.float32),
            float(material),int(tag))


def _box(center,half,material=.9,basis=None,tag=0):
    return ('box',np.asarray(center,dtype=np.float32),
            np.asarray(half,dtype=np.float32),
            np.eye(3,dtype=np.float32) if basis is None else np.asarray(basis,dtype=np.float32),
            float(material),int(tag))


def _bone(a,b,width,material=.93,depth=None):
    """A muscular capsule: rounded elongated ellipsoid + spherical joints."""
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    delta=b-a;length=np.linalg.norm(delta);direction=delta/max(1e-8,length)
    helper=np.array([0,0,1],dtype=np.float32)
    if abs(direction[2])>.95:helper=np.array([1,0,0],dtype=np.float32)
    x=np.cross(direction,helper);x/=np.linalg.norm(x)
    z=np.cross(x,direction)
    B=np.stack([x,direction,z],axis=1)
    return _ellipsoid((a+b)/2,(width,length*.58,width if depth is None else depth),material,B)


def _runner_primitives(t):
    """Alternating foot recovery and arm opposition over a .60-second stride."""
    phase=t*math.tau/.60
    bob=.055*math.cos(phase*2-.40)
    hip=np.array([-.12,-.20+bob,0.],dtype=np.float32)
    shoulder=np.array([.22,.83+bob,0.],dtype=np.float32)
    torso_twist=.15*math.sin(phase+.35)
    pelvis_twist=-.105*math.sin(phase+.35)
    Rt=_rotation((0,torso_twist,-.17))
    Rp=_rotation((0,pelvis_twist,-.11))
    # Leaning, tapered torso rather than disconnected spheres.  Athletic
    # clothing is represented only by material luminance, not drawn contours.
    p=[_ellipsoid(hip+np.array([.04,.13,0]),(.26,.31,.21),.77,Rp),
       _ellipsoid((hip+shoulder)/2+np.array([0,.02,0]),(.245,.55,.215),.93,Rt),
       _ellipsoid(shoulder+np.array([-.055,-.12,0]),(.295,.35,.25),.96,Rt)]
    neck=np.array([.29,1.08+bob,0.]);head=np.array([.36,1.43+bob,0.])
    p += [_bone(neck,head-[.025,.15,0],.105,.94),
          _ellipsoid(head,(.225,.295,.215),.96,_rotation((0,.12,-.13)))]
    # Side separation is genuine Z geometry: near and far elbows, hips and
    # shoes exchange occlusion naturally under a three-quarter view.
    for side in (-1,1):
        q=phase+(0 if side<0 else math.pi)
        h=hip+_transform(np.array([0,0,side*.175]),Rp)
        thigh=.96*math.sin(q)+.12
        flex=.48+1.18*(.5-.5*math.sin(q-.46))
        knee=h+np.array([.70*math.sin(thigh),-.70*math.cos(thigh),side*.035])
        shank=thigh-flex
        ankle=knee+np.array([.695*math.sin(shank),-.695*math.cos(shank),-.018*side])
        p += [_bone(h,knee,.157,.84,depth=.150),
              _ellipsoid(knee,(.117,.126,.115),.90),
              _bone(knee,ankle,.105,.94,depth=.097),
              _ellipsoid(ankle,(.082,.098,.083),.89)]
        # A substantial forefoot and small heel make the running silhouette
        # read instantly, including the tucked rear-foot recovery.
        shoe_angle=-.16-.45*(.5-.5*math.sin(q))
        foot=ankle+np.array([.095*math.cos(shoe_angle),-.057+.095*math.sin(shoe_angle),0])
        p.append(_ellipsoid(foot,(.235,.110,.132),.83,_rotation((0,0,shoe_angle))))
        # Opposed arm swing, flexed elbow, hands on a long inertial arc.
        s=shoulder+_transform(np.array([-.03,-.01,side*.266]),Rt)
        arm=-.91*math.sin(q)-.12
        elbow=s+np.array([.53*math.sin(arm),-.53*math.cos(arm),side*.053])
        bend=1.68+.18*math.sin(q+.7)
        fore=arm+bend
        wrist=elbow+np.array([.48*math.sin(fore),-.48*math.cos(fore),-side*.075])
        p += [_ellipsoid(s,(.152,.172,.148),.96),
              _bone(s,elbow,.111,.95,depth=.103),
              _ellipsoid(elbow,(.093,.10,.092),.94),
              _bone(elbow,wrist,.084,.94,depth=.079),
              _ellipsoid(wrist+np.array([.025,.025,0]),(.102,.125,.088),.94,
                         _rotation((0,0,-fore*.35)))]
    return p


@lru_cache(maxsize=1)
def _chip_primitives():
    p=[_box((0,0,.075),(1.32,1.32,.16),.80,tag=1),
       _box((0,0,-.095),(1.23,1.23,.13),.94,tag=2),
       _box((0,0,-.285),(.79,.79,.145),.95,tag=3)]
    # Raised die rim is physical relief: four rails above the substrate.
    for side in (-1,1):
        p.append(_box((side*.89,0,-.238),(.045,.935,.05),.90,tag=4))
        p.append(_box((0,side*.89,-.238),(.845,.045,.05),.90,tag=4))
    for axis in (0,1):
        for sign in (-1,1):
            for lane in np.linspace(-1.17,1.17,13):
                center=[0.,0.,.10];half=[.047,.047,.065]
                center[axis]=sign*1.47;center[1-axis]=lane;half[axis]=.19
                p.append(_box(center,half,.94,tag=5))
    return p


def _worker_primitives(t):
    """A running compute process: a thick X die with arms, gloves and shoes."""
    phase=float(t)*math.tau/.60
    bob=.056*math.cos(phase*2-.3)
    bodycenter=np.array([0,.40+bob,0.],dtype=np.float32)
    bodyrot=_rotation((-.15,.11*math.sin(phase+.2),.035*math.sin(phase)))
    chipscale=.55
    # Reuse the complete physical layered chip. Its square front is the
    # character identity; there is intentionally no human face or human head.
    p=[]
    for shape,c,h,B,material,tag in _chip_primitives():
        p.append((shape,bodycenter+_transform(c*chipscale,bodyrot),h*chipscale,
                  np.einsum('ij,jk->ik',bodyrot,B),material,tag))
    for side in (-1,1):
        q=phase+(0 if side<0 else math.pi)
        # Forward in the character's own space is -Z. Hip widths in X make
        # both legs appear attached correctly to the bottom of the processor.
        hip=bodycenter+_transform(np.array([side*.34,-.755,.02]),bodyrot)
        thigh=.93*math.sin(q)+.10
        flex=.40+1.18*(.5-.5*math.sin(q-.44))
        knee=hip+np.array([side*.025,-.51*math.cos(thigh),-.51*math.sin(thigh)])
        calf=thigh-flex
        ankle=knee+np.array([-side*.018,-.50*math.cos(calf),-.50*math.sin(calf)])
        p += [_ellipsoid(hip,(.105,.11,.10),.88),
              _bone(hip,knee,.123,.91),
              _ellipsoid(knee,(.10,.112,.102),.96),
              _bone(knee,ankle,.102,.96),
              _ellipsoid(ankle,(.082,.09,.083),.97)]
        shoe_angle=.10+.42*(.5-.5*math.sin(q))
        foot=ankle+np.array([0,-.058,-.105])
        p.append(_ellipsoid(foot,(.186,.134,.30),.97,_rotation((shoe_angle,0,0))))
        shoulder=bodycenter+_transform(np.array([side*.865,.27,.02]),bodyrot)
        arm=-.85*math.sin(q)-.07
        elbow=shoulder+np.array([side*.055,-.39*math.cos(arm),-.39*math.sin(arm)])
        fore=arm+1.73+.12*math.sin(q+.5)
        wrist=elbow+np.array([-side*.01,-.37*math.cos(fore),-.37*math.sin(fore)])
        p += [_ellipsoid(shoulder,(.095,.11,.103),.91),
              _bone(shoulder,elbow,.074,.90),
              _ellipsoid(elbow,(.086,.090,.087),.94),
              _bone(elbow,wrist,.073,.96),
              # Glove body and thumb merge into one opaque silhouette.
              _ellipsoid(wrist+np.array([0,.024,-.02]),(.133,.155,.122),1.),
              _ellipsoid(wrist+np.array([-side*.102,.01,-.028]),(.063,.08,.07),1.)]
    # Fixed enlargement, shared by all animation phases. This gives a 3.3-unit
    # high character in the 4.5-unit camera opening, without silhouette fitting.
    overall=1.20
    offset=np.array([0,.12,0],dtype=np.float32)
    p=[(shape,c*overall+offset,h*overall,B,material,tag) for shape,c,h,B,material,tag in p]
    pattern=(bodycenter*overall+offset,bodyrot,chipscale*overall)
    return p,pattern


def _render(primitives,rotation,resolution,kind,pattern=None,orthographic=False):
    resolution=tuple(map(int,resolution));w,h=resolution
    origin,world_rays=(_orthographic_camera if orthographic else _camera)(resolution)
    R=_rotation(rotation)
    # Transform the fixed camera into object space once, then intersect all
    # physical primitives there. t remains the true distance along a unit ray.
    ray=_transform(world_rays,R.T)
    eye=_transform(origin,R.T)
    count=len(ray)
    distance=np.full(count,np.inf,dtype=np.float32)
    normals=np.zeros((count,3),dtype=np.float32)
    material=np.zeros(count,dtype=np.float32)
    tags=np.zeros(count,dtype=np.int16)
    for shape,center,radii,B,albedo,tag in primitives:
        # Primitive basis columns define its local physical axes.
        o=_transform(eye-center,B.T)
        d=_transform(ray,B.T)
        if shape=='ellipsoid':
            invr=1/radii
            od=o*invr;dd=d*invr
            aa=np.sum(dd*dd,axis=1)
            bb=np.sum(dd*od,axis=1)
            cc=np.sum(od*od,axis=-1)-1
            disc=bb*bb-aa*cc
            candidate=disc>=0
            if not candidate.any():continue
            tt=np.full(count,np.inf,dtype=np.float32)
            idx=np.flatnonzero(candidate)
            tt[idx]=(-bb[idx]-np.sqrt(disc[idx]))/aa[idx]
            closer=candidate&(tt>0)&(tt<distance)
            idx=np.flatnonzero(closer)
            if not len(idx):continue
            hit=(o[idx] if o.ndim==2 else o)+tt[idx,None]*d[idx]
            nn=hit/(radii*radii)
            nn/=np.sqrt(np.sum(nn*nn,axis=1))[:,None]
            nn=_transform(nn,B)
        else:
            invd=np.divide(1.,d,out=np.full_like(d,1e15),where=abs(d)>1e-10)
            t1=(-radii-o)*invd;t2=(radii-o)*invd
            near=np.minimum(t1,t2);far=np.maximum(t1,t2)
            entry=np.max(near,axis=1);leave=np.min(far,axis=1)
            closer=(leave>=entry)&(entry>0)&(entry<distance)
            idx=np.flatnonzero(closer)
            if not len(idx):continue
            tt=entry
            axis=np.argmax(near[idx],axis=1)
            nn=np.zeros((len(idx),3),dtype=np.float32)
            nn[np.arange(len(idx)),axis]=-np.sign(d[idx,axis])
            nn=_transform(nn,B)
        distance[idx]=tt[idx];normals[idx]=nn;material[idx]=albedo;tags[idx]=tag
    mask=np.isfinite(distance)
    world_n=_transform(normals,R)
    light=np.array([-.48,.66,-.88],dtype=np.float32);light/=np.linalg.norm(light)
    lambert=np.maximum(0,np.sum(world_n*light,axis=1))
    half=light-world_rays
    half/=np.sqrt(np.sum(half*half,axis=1))[:,None]
    spec=np.maximum(0,np.sum(world_n*half,axis=1))**23
    facing=np.clip(-np.sum(world_n*world_rays,axis=1),0,1)
    rim=(1-facing)**2
    value=material*(.23+.69*lambert)+.15*spec+.075*rim
    feature=np.zeros(count,dtype=np.uint8)
    if kind in ('chip','worker'):
        # Solid die remains occupied even where its etched X is dark. The X
        # therefore emerges through the caller's intensity-to-glyph ramp.
        point=eye+np.where(mask,distance,0)[:,None]*ray
        if pattern is not None:
            center,B,scale=pattern
            point=_transform(point-center,B.T)/scale
            face_n=_transform(normals,B.T)
        else:face_n=normals
        top=(tags==3)&(face_n[:,2]<-.9)
        x,y=point[:,0],point[:,1]
        diag=np.minimum(abs(x-y),abs(x+y))
        cross=(diag<.195)&(np.maximum(abs(x),abs(y))<.59)
        rimdie=(np.maximum(abs(x),abs(y))>.68)
        value[top]*=np.where(cross[top]|rimdie[top],1.12,.18)
        value[top&cross]=np.maximum(value[top&cross],.92)
        feature[top&cross]=1
        feature[top&rimdie]=2
        pcb=(tags==2)&(face_n[:,2]<-.9)
        nearest=np.minimum(abs((x*4+.5)%1-.5),abs((y*4+.5)%1-.5))
        value[pcb]*=np.where(nearest[pcb]<.058,.97,.71)
        if kind=='worker':
            limb=(tags==0)&mask
            value[limb]=np.maximum(value[limb],.51)
            feature[limb]=3
    value=np.where(mask,np.clip(value,0,1),0).astype(np.float32)
    return dict(luma=value.reshape(h,w),mask=mask.reshape(h,w),
                feature=feature.reshape(h,w),
                depth=distance.reshape(h,w),normal=world_n.reshape(h,w,3),
                extent=(4.5*w/h,4.5),camera_distance=7.)


def sample_runner(t,rotation=(0.,-.28,-.02),resolution=(160,90)):
    """Muscular runner with a continuous .60-second full articulated stride."""
    return _render(_runner_primitives(float(t)),rotation,resolution,'runner')


def sample_chip(t,rotation=(.18,.48,-.13),resolution=(160,90)):
    """Opaque thick processor, raised die and 52 separate physical pins."""
    return _render(_chip_primitives(),rotation,resolution,'chip')


def sample_worker(t,rotation=(0.,.68,0.),resolution=(80,90)):
    """Running chip character with continuously articulated limbs and gloves."""
    primitives,pattern=_worker_primitives(float(t))
    return _render(primitives,rotation,resolution,'worker',pattern)


if __name__=='__main__':
    from PIL import Image,ImageDraw
    import time
    out=Path(__file__).resolve().parents[1]/'work'/'solid-v5'
    out.mkdir(parents=True,exist_ok=True)
    thumbs=[]
    samples=[('run-000',sample_runner,0,(0,-.38,-.02)),
             ('run-075',sample_runner,.075,(0,-.38,-.02)),
             ('run-150',sample_runner,.150,(0,-.38,-.02)),
             ('run-225',sample_runner,.225,(0,-.38,-.02)),
             ('chip-front',sample_chip,0,(0,0,0)),
             ('chip-side',sample_chip,0,(.34,.67,-.17)),
             ('chip-edge',sample_chip,0,(.08,1.49,-.13)),
             ('chip-back',sample_chip,0,(.15,2.73,.08))]
    for name,fn,t,rotation in samples:
        s=fn(t,rotation,resolution=(320,180))
        im=Image.fromarray(np.rint(s['luma']*255).astype(np.uint8)).resize((960,540),Image.Resampling.NEAREST).convert('RGB')
        ImageDraw.Draw(im).text((20,18),name,fill='white')
        im.save(out/f'{name}.png');thumbs.append(im.resize((640,360)))
    sheet=Image.new('RGB',(2560,720))
    for j,im in enumerate(thumbs):sheet.paste(im,((j%4)*640,(j//4)*360))
    sheet.save(out/'solid-poses.jpg')
    worker_sheet=Image.new('RGB',(1920,1080))
    for j in range(8):
        s=sample_worker(j*.075,resolution=(160,180))
        im=Image.fromarray(np.rint(s['luma']*255).astype(np.uint8)).resize((480,540),Image.Resampling.NEAREST).convert('RGB')
        ImageDraw.Draw(im).text((20,20),f'worker {j*.075:.3f}',fill='white')
        im.save(out/f'worker-{j:02d}.png')
        worker_sheet.paste(im,((j%4)*480,(j//4)*540))
    worker_sheet.save(out/'worker-poses.jpg')
    for fn in (sample_runner,sample_chip,sample_worker):
        start=time.perf_counter()
        for j in range(30):s=fn(j/24,rotation=(.13,j*.09,-.05))
        print(fn.__name__,f'{(time.perf_counter()-start)/30:.4f} s/frame',int(s['mask'].sum()),'occupied samples')
