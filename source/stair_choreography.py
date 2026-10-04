#!/usr/bin/env python3
"""Purposeful right-facing staircase performance, shared by drawing and masks.

Public API:
    staircase_state(t) -> dict with center, height, platforms, phase, feet_screen
    sample_climber(t, resolution=(110,120)) -> ascii_solids surface sample

Draw the sample at state['center'], sized (height*w/h, height). The camera is
orthographic for this grounded shot: each shoe's bottom is the specified
platform y, with no perspective drift. Local forward is -Z, so NEGATIVE yaw
makes the hero face screen-right. No run cycle is used during a jump.

Landing audio/contact times: 17.70, 18.33, 18.96 seconds. Final motion stays on
the upper platform, moving right; the hero clears the right edge by ~19.70.
"""
from pathlib import Path
import math
import numpy as np
from ascii_solids import (_render, _chip_primitives, _rotation, _transform,
                          _ellipsoid, _bone)

LANDING_TIMES = (17.70, 18.33, 18.96)
PLATFORMS = ((570., 875., 325.), (960., 730., 325.), (1350., 585., 1050.))
# launch, land, launch x, landing x, old floor, new floor
JUMPS = ((17.28, 17.70, 327., 610., 970., 875.),
         (17.91, 18.33, 634., 1000., 875., 730.),
         (18.54, 18.96, 1024., 1390., 730., 585.))
GRAVITY = 4100.  # pixels/sec^2; flight has an actual parabola and apex pause


def smooth(x):
    x=float(np.clip(x,0,1))
    return x*x*(3-2*x)


def mix(a,b,u):
    return a+(b-a)*u


def hermite(a,b,va,vb,u,duration):
    u=float(np.clip(u,0,1))
    return ((2*u**3-3*u*u+1)*a+(u**3-2*u*u+u)*duration*va
            +(-2*u**3+3*u*u)*b+(u**3-u*u)*duration*vb)


def _exit_x(t):
    # Grounded acceleration, then constant rightward travel. No upward exit.
    if t <= 19.08:
        u=(t-18.96)/.12
        return hermite(1390,1414,871.43,0,u,.12)
    u=(t-19.08)/.64
    return hermite(1414,2160,0,1450,u,.64)


def _exit_foot(t,side):
    """A foot stays fixed through stance, then recovers to its next contact."""
    period=.20
    phase=(t-19.08)/period+(0 if side<0 else .5)
    cycle=math.floor(phase)
    u=phase-cycle
    start=19.08+(cycle-(0 if side<0 else .5))*period
    # Mid-stance body position is the fixed support anchor for this step.
    contact=_exit_x(start+period*.25)+(20 if side>0 else -20)
    if cycle==0:
        contact=1390.+(-23. if side<0 else 23.)
    if u < .50:
        return contact,585.,True
    next_contact=_exit_x(start+period*1.25)+(20 if side>0 else -20)
    q=(u-.5)*2
    return mix(contact,next_contact,smooth(q)),585.-25*math.sin(math.pi*q),False


def staircase_state(t):
    """Screen-space grounded pose and stage geometry, deterministic at any t."""
    t=float(t)
    height=mix(285.,380.,smooth((t-17.05)/.23))
    scale=height/4.5
    yaw=mix(.60,-.92,smooth((t-17.05)/.18))
    floor=970.
    x=307.
    crouch=.0
    lean=.045
    arm=(-.05,.20)
    feet=np.array([[284.,970.],[330.,970.]],dtype=float)
    planted=np.array([True,True])
    phase='turn'
    jump_index=-1
    if t < JUMPS[0][0]:
        # Turn in place, then load the legs. Shoes remain on the same marks.
        if t<17.18:
            x=mix(307,300,smooth((t-17.05)/.13))
        else:
            x=hermite(300,327,0,(610-327)/.42,(t-17.18)/.10,.10)
        crouch=.32*math.sin(math.pi*smooth((t-17.05)/.23))
        if t>17.19:phase='anticipation' if t<17.25 else 'push_off'
        lean=.045+.17*smooth((t-17.15)/.13)
        arm=(-.60,-.18)
    else:
        for j,(launch,land,x0,x1,y0,y1) in enumerate(JUMPS):
            next_launch=JUMPS[j+1][0] if j+1<len(JUMPS) else 19.08
            if launch<=t<land:
                jump_index=j;phase='flight';planted[:]=False
                u=(t-launch)/(land-launch);duration=land-launch
                x=mix(x0,x1,u)
                floor=mix(y0,y1,u)-.5*GRAVITY*duration*duration*u*(1-u)
                tuck=math.sin(math.pi*u)
                crouch=.04*tuck
                support_x=307. if j==0 else JUMPS[j-1][3]
                support_offset=(support_x-x0)*(1-smooth(u))
                feet=np.array([[x+support_offset-23.-10*tuck,floor-.64*scale*tuck],
                               [x+support_offset+23.+12*tuck,floor-.34*scale*tuck]])
                lean=mix(.21,.06,smooth(u))
                # Opposed reach is a single jump gesture, not air-walking.
                start_arm=(-.60,-.18) if j==0 else (-.56,-.16)
                arm=tuple(mix(start_arm[k],(1.12,-.62)[k],smooth(u/.28)) for k in range(2))
                if u>.65:
                    arm=tuple(mix(arm[k],(.75,.15)[k],smooth((u-.65)/.35)) for k in range(2))
                break
            if land<=t<next_launch:
                jump_index=j;phase='landing';floor=y1
                dt=t-land
                feet=np.array([[x1-23.,y1],[x1+23.,y1]])
                # 3 frames of compression/rebound are readable before push.
                if dt<.12:
                    crouch=.27*math.sin(math.pi*dt/(.12 if j==2 else .16))
                    x=hermite(x1,x1+24,(x1-x0)/(land-launch),0,dt/.12,.12)
                    lean=.06+.10*math.sin(math.pi*dt/.12)
                    arm=(.75,.15)
                elif j+1<len(JUMPS):
                    phase='anticipation' if dt<.17 else 'push_off'
                    u=(dt-.12)/(next_launch-land-.12)
                    x=hermite(x1+24,x1+24,-230,
                              (JUMPS[j+1][3]-JUMPS[j+1][2])/.42,
                              u,next_launch-land-.12)
                    crouch=(.27/math.sqrt(2))*(1-smooth(u))
                    lean=mix(.06,.21,smooth(u))
                    arm=tuple(mix((.75,.15)[k],(-.56,-.16)[k],smooth(u)) for k in range(2))
                else:
                    x=x1+24;crouch=0;lean=.06;arm=(.1,.2)
                break
        else:
            phase='exit';floor=585.;x=_exit_x(t);yaw=-1.02
            cycle=(t-19.08)/.20*math.tau
            crouch=.18*smooth((t-19.08)/.10)+.045*(1-math.cos(cycle*2))
            lean=.12
            footstates=[_exit_foot(t,side) for side in (-1,1)]
            feet=np.array([f[:2] for f in footstates])
            planted=np.array([f[2] for f in footstates])
            gait_arm=(.60*math.sin(cycle),-.60*math.sin(cycle))
            arm=tuple(mix((.75,.15)[k],gait_arm[k],smooth((t-19.08)/.10)) for k in range(2))
    body_y=floor-(1.95-crouch)*scale
    center=np.array([x,body_y+.58*scale])
    # Camera/world coordinates retain feet fixed in screen space even while
    # the torso shifts through landing compression and push-off.
    sole_world=np.column_stack(((feet[:,0]-center[0])/scale,
                                (center[1]-feet[:,1])/scale,
                                [-.26,.26]))
    return dict(time=t,center=tuple(center),height=height,yaw=yaw,lean=lean,
                body_center=np.array([0.,.58,0.]),feet_screen=feet,
                soles=sole_world,planted=planted,arm_angles=arm,
                phase=phase,jump_index=jump_index,platforms=PLATFORMS,
                landing_times=LANDING_TIMES,baseline=floor,
                visible=t<19.75,cell=max(9,round(height/31)))


def _two_bone(a,b,length_a,length_b,bend):
    """Analytic two-bone IK; knee bend follows the forward local direction."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    delta=b-a;distance=np.linalg.norm(delta)
    direction=delta/max(distance,1e-9)
    # Distances are validated below. Tiny extension covers raster-scale soles
    # without a disconnected ankle when the support reaches the toe-off end.
    total=length_a+length_b
    if distance>total:
        length_a*=distance/total;length_b*=distance/total
    projection=(length_a*length_a-length_b*length_b+distance*distance)/(2*max(distance,1e-9))
    height=math.sqrt(max(0,length_a*length_a-projection*projection))
    bend=np.asarray(bend,dtype=float)
    bend-=direction*np.dot(bend,direction)
    bend/=max(np.linalg.norm(bend),1e-9)
    return a+direction*projection+bend*height


def _climber_primitives(state):
    bodycenter=state['body_center']
    yaw=state['yaw']
    bodyrot=_rotation((0,0,-state['lean'])) @ _rotation((0,yaw,0))
    chipscale=.66
    p=[]
    for shape,c,h,B,material,tag in _chip_primitives():
        p.append((shape,bodycenter+_transform(c*chipscale,bodyrot),h*chipscale,
                  np.einsum('ij,jk->ik',bodyrot,B),material,tag))
    forward=np.array([-math.sin(yaw),0.,-math.cos(yaw)])
    shoe_basis=_rotation((0,yaw,0))
    for index,side in enumerate((-1,1)):
        hip=bodycenter+_transform(np.array([side*.408,-.906,.024]),bodyrot)
        sole=state['soles'][index]
        shoe_center=sole+np.array([0.,.15,0.])
        ankle=shoe_center-forward*.10+np.array([0.,.07,0.])
        knee=_two_bone(hip,ankle,.615,.605,forward+np.array([0,.10,0]))
        p += [_ellipsoid(hip,(.126,.132,.12),.88),
              _bone(hip,knee,.148,.91),_ellipsoid(knee,(.12,.134,.122),.96),
              _bone(knee,ankle,.122,.96),_ellipsoid(ankle,(.098,.108,.100),.97),
              _ellipsoid(shoe_center,(.225,.15,.36),.97,shoe_basis)]
        shoulder=bodycenter+_transform(np.array([side*1.038,.324,.024]),bodyrot)
        arm=state['arm_angles'][index]
        elbow=shoulder+_transform(np.array([side*.066,-.468*math.cos(arm),-.468*math.sin(arm)]),_rotation((0,yaw,0)))
        fore=arm+1.70
        wrist=elbow+_transform(np.array([-side*.012,-.444*math.cos(fore),-.444*math.sin(fore)]),_rotation((0,yaw,0)))
        p += [_ellipsoid(shoulder,(.114,.132,.124),.91),
              _bone(shoulder,elbow,.089,.90),_ellipsoid(elbow,(.103,.108,.104),.94),
              _bone(elbow,wrist,.088,.96),
              _ellipsoid(wrist+np.array([0,.029,-.024]),(.160,.186,.146),1.),
              _ellipsoid(wrist+np.array([-side*.122,.012,-.034]),(.076,.096,.084),1.)]
    return p,(bodycenter,bodyrot,chipscale)


def sample_climber(t,resolution=(110,120)):
    """Chip/gloves/shoes identity with planted feet and intentional jump pose."""
    state=staircase_state(t)
    primitives,pattern=_climber_primitives(state)
    result=_render(primitives,(0,0,0),resolution,'worker',pattern,orthographic=True)
    result['choreography']=state
    return result


def build_qa():
    from PIL import Image,ImageDraw
    out=Path(__file__).resolve().parents[1]/'work'/'stair-qa'
    out.mkdir(parents=True,exist_ok=True)
    moments=[17.05,17.18,17.28,17.48,17.70,17.78,17.91,18.12,
             18.33,18.41,18.54,18.74,18.96,19.04,19.24,19.52]
    sheet=Image.new('RGB',(1920,1080),(20,31,244))
    report=[]
    for i,t in enumerate(moments):
        state=staircase_state(t);sample=sample_climber(t,resolution=(176,192))
        frame=Image.new('RGB',(1920,1080),(20,31,244));draw=ImageDraw.Draw(frame)
        for x,y,w in PLATFORMS:
            draw.line((x,y,x+w,y),fill='white',width=7)
        a=Image.fromarray(np.uint8(np.clip(sample['luma']*255,0,255)))
        h=state['height'];w=h*176/192
        a=a.resize((round(w),round(h)),Image.Resampling.BICUBIC)
        frame.paste('white',(round(state['center'][0]-w/2),round(state['center'][1]-h/2)),a)
        draw.text((60,45),f"{t:.2f}s  {state['phase']}  yaw {state['yaw']:.2f}",fill='white',stroke_width=1)
        frame.save(out/f'{t:.2f}.png')
        sheet.paste(frame.resize((480,270)),((i%4)*480,(i//4)*270))
        report.append({'time':t,'phase':state['phase'],'center':state['center'],
                       'feet':state['feet_screen'].tolist(),'planted':state['planted'].tolist()})
    sheet.save(out/'contact.jpg',quality=95)
    import json
    (out/'poses.json').write_text(json.dumps(report,indent=2))
    print(out/'contact.jpg')


if __name__=='__main__':
    build_qa()
