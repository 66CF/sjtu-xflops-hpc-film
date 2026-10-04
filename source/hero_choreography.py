"""One physical chip unfolds into a runner, folds up, then accelerates away.

There is no replacement object at the former 6.173 s scene boundary.  The
worker's layered processor is retained; the limbs slide into that processor
while its centre, perspective scale and spin follow continuous curves.

Public API:
    hero_state(t) -> x, y, width, height, rotation, cell, visible
    sample_hero(t, resolution=(110, 120)) -> the corresponding opaque surfaces

Print at ``(x, y)`` with canvas ``(height * resolution[0] / resolution[1],
height)`` and constant ``cell=25``.  Use this same sampler for collision masks.
After 7.03 s position and scale follow the existing rightward chip exit.
The chip completes its somersault before that point, settles elastically,
and keeps its X face legible as it accelerates right.
"""
import math
import numpy as np
from ascii_solids import (_worker_primitives, _chip_primitives, _render,
                          _rotation, _transform)
from chip_choreography import chip_state, CHIP_END
from render_kinetic import curve

HERO_START = 4.65
FOLD_START = 5.70
FOLD_END = 6.62
EXIT_START = 7.03
CELL = 25


def _smooth(x):
    x=float(np.clip(x,0,1))
    return x*x*(3-2*x)


def _hermite(a,b,va,vb,t,start,end):
    u=float(np.clip((t-start)/(end-start),0,1));dt=end-start
    return ((2*u**3-3*u*u+1)*a+(u**3-2*u*u+u)*dt*va
            +(-2*u**3+3*u*u)*b+(u**3-u*u)*dt*vb)


def _incoming(t):
    x=curve(t,[(4.65,2210),(4.76,1620),(4.93,1320),(5.10,1080),
               (5.28,930),(5.65,890),(6.17,845)])
    h=curve(t,[(4.65,640),(4.95,890),(5.2,1050),(5.7,1030),(6.17,1070)])
    rot=(0.,.62+.14*math.sin(t*3),.015*math.sin(t*8))
    return x,h,rot


def _pose_time(t):
    # The running stride brakes, then stays frozen while the limbs retract.
    # Integral of 1 - smooth(u) gives continuous velocity at both ends.
    duration=.24;u=float(np.clip((t-FOLD_START)/duration,0,1))
    return FOLD_START+duration*(u-u**3+.5*u**4)


def _shape_scale(t):
    return .66+.34*_smooth((t-FOLD_START)/(FOLD_END-FOLD_START))


def _exit_rotation(t):
    # Acceleration carries a little forward lean, rather than another flip
    # that would hide the chip edge-on immediately before its exit.
    u=_smooth((t-EXIT_START)/.72)
    return (math.tau-.14+.42*u,-.24-.28*u,-.09-.16*u)


def hero_state(t):
    t=float(t)
    if t<=FOLD_START:
        x,h,rot=_incoming(t);y=545.
    elif t<EXIT_START:
        x0,h0,r0=_incoming(FOLD_START)
        ep=chip_state(EXIT_START)
        dt=1e-5
        vx0=(_incoming(FOLD_START+dt)[0]-_incoming(FOLD_START-dt)[0])/(2*dt)
        x=_hermite(x0,ep['x'],vx0,0.,t,FOLD_START,EXIT_START)
        y=_hermite(545.,ep['y'],0.,0.,t,FOLD_START,EXIT_START)
        # Compensate for the processor's local expansion from .66 to 1.0:
        # its visible scale shrinks monotonically, with no size swell or pop.
        effective=_hermite(h0*.66,ep['height'],0.,0.,t,FOLD_START,EXIT_START)
        h=effective/_shape_scale(t)
        r1=_exit_rotation(EXIT_START)
        vr1=(0.,0.,0.)
        vr0=(0.,.42*math.cos(FOLD_START*3),.12*math.cos(FOLD_START*8))
        # Anticipation, one fast somersault, then an inertial overshoot and
        # damped settle. The face is readable again before acceleration.
        if t<6.03:
            pitch=_hermite(0.,-.14,0.,0.,t,FOLD_START,6.03)
        elif t<6.66:
            pitch=_hermite(-.14,math.tau+.11,0.,2.2,t,6.03,6.66)
        else:
            pitch=_hermite(math.tau+.11,r1[0],2.2,0.,t,6.66,EXIT_START)
        rot=(pitch,
             _hermite(r0[1],r1[1],vr0[1],vr1[1],t,FOLD_START,EXIT_START),
             _hermite(r0[2],r1[2],vr0[2],vr1[2],t,FOLD_START,EXIT_START))
    else:
        state=chip_state(t)
        x,y,h=state['x'],state['y'],state['height']
        rot=_exit_rotation(t)
    return dict(x=float(x),y=float(y),width=float(h)*110/120,height=float(h),
                rotation=tuple(rot),cell=CELL,
                visible=HERO_START<=t<CHIP_END,
                phase='run' if t<FOLD_START else 'fold' if t<FOLD_END else 'chip')


def _folding_primitives(t):
    pose=_pose_time(t)
    primitives,pattern=_worker_primitives(pose)
    center,basis,scale=pattern
    u=_smooth((t-FOLD_START)/(FOLD_END-FOLD_START))
    phase=pose*math.tau/.60
    body_angles=np.array([-.15,.11*math.sin(phase+.2),.035*math.sin(phase)])
    next_basis=_rotation(body_angles*(1-u))
    next_center=center*(1-u)
    next_scale=_shape_scale(t)
    result=[];chip_count=len(_chip_primitives())
    for i,(shape,c,r,B,material,tag) in enumerate(primitives):
        local=_transform(c-center,basis.T)/scale
        local_basis=basis.T@B
        local_radius=r/scale
        if i>=chip_count:
            ordinal=i-chip_count;side=-1 if ordinal<12 else 1
            is_arm=ordinal%12>=6
            # Arms fold first; shoes follow into the lower edge of the die.
            start=5.74 if is_arm else 5.84
            end=6.26 if is_arm else 6.44
            fold=_smooth((t-start)/(end-start))
            destination=np.array([side*(1.03 if is_arm else .5),
                                  .30 if is_arm else -.91,.08])
            local=local*(1-fold)+destination*fold
            local_radius=local_radius*(1-.985*fold)
            # Fully retracted parts are inside the opaque substrate, so
            # removing their internal surfaces cannot change the silhouette.
            if fold>=1:continue
        result.append((shape,next_center+_transform(local*next_scale,next_basis),
                       local_radius*next_scale,next_basis@local_basis,material,tag))
    return result,(next_center,next_basis,next_scale)


def sample_hero(t,resolution=(110,120)):
    t=float(t);path=hero_state(t)
    if t<=FOLD_START:
        primitives,pattern=_worker_primitives(t)
        state=_render(primitives,path['rotation'],resolution,'worker',pattern)
    elif t<FOLD_END:
        primitives,pattern=_folding_primitives(t)
        state=_render(primitives,path['rotation'],resolution,'worker',pattern)
    else:
        state=_render(_chip_primitives(),path['rotation'],resolution,'chip')
    # A broad fill light keeps the back of the flipping chip occupied by
    # visible glyphs. Without it the dark rear surface crosses the ASCII
    # ramp's blank threshold and misleadingly appears to disappear.
    floor=.48*_smooth((t-FOLD_START)/.25)
    state['luma']=np.where(state['mask'],np.maximum(state['luma'],floor),0.)
    return state
