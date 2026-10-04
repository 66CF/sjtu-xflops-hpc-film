#!/usr/bin/env python3
"""Reference-led motion: scrolling terminal, advected glyphs, real 3D props.

The first ten seconds follow measured event times in the supplied reference.
Text, fluid glyphs and live 3D surfaces are intentionally separate renderers.
"""
from pathlib import Path
from functools import lru_cache
import argparse, math, json, subprocess, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS,DURATION=1920,1080,24,30
BLUE=(20,31,244);WHITE=(244,248,255)
MONO='/System/Library/Fonts/Menlo.ttc'
RNG=np.random.default_rng(5102026)
PAL=[]
for i in range(256):PAL.extend([round(a+(b-a)*i/255) for a,b in zip(BLUE,WHITE)])

def clamp(x):return np.clip(x,0.,1.)
def smooth(x):x=clamp(x);return x*x*(3-2*x)
def mix(a,b,k):return a*(1-k)+b*k

def curve(t,keys):
    """Shape-preserving cubic Hermite curves, with per-channel key times."""
    arr=np.asarray(keys,dtype=float);tt=arr[:,0];vv=arr[:,1:]
    if t<=tt[0]:v=vv[0]
    elif t>=tt[-1]:v=vv[-1]
    else:
        slopes=np.diff(vv,axis=0)/np.diff(tt)[:,None]
        tangent=np.zeros_like(vv);tangent[0]=slopes[0];tangent[-1]=slopes[-1]
        for j in range(1,len(tt)-1):
            same=slopes[j-1]*slopes[j]>0
            tangent[j]=np.where(same,2*slopes[j-1]*slopes[j]/(slopes[j-1]+slopes[j]+1e-12),0)
        k=np.searchsorted(tt,t)-1;dt=tt[k+1]-tt[k];u=(t-tt[k])/dt
        v=(2*u**3-3*u*u+1)*vv[k]+(u**3-2*u*u+u)*dt*tangent[k]+(-2*u**3+3*u*u)*vv[k+1]+(u**3-u*u)*dt*tangent[k+1]
    return float(v[0]) if len(v)==1 else v

@lru_cache(maxsize=256)
def font(n):return ImageFont.truetype(MONO,max(6,n))

@lru_cache(maxsize=80000)
def sprite(ch,size,angle=0):
    f=font(size);w=math.ceil(size*.75);h=math.ceil(size*4/3)
    im=Image.new('L',(w,h),0);ImageDraw.Draw(im).text((w/2,0),str(ch),font=f,fill=255,anchor='ma')
    if angle:im=im.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    return im

def batch(im,xy,size,glyph,alpha=1.,angle=None,depth=None):
    xy=np.asarray(xy);n=len(xy)
    if n==0:return
    sz=np.broadcast_to(size,(n,));aa=np.broadcast_to(alpha,(n,))
    ang=np.zeros(n) if angle is None else np.broadcast_to(angle,(n,))
    codes=np.broadcast_to(glyph,(n,))
    keep=np.where((xy[:,0]>-100)&(xy[:,0]<W+100)&(xy[:,1]>-100)&(xy[:,1]<H+100)&(aa>.018)&(codes!=' '))[0]
    if depth is not None:keep=keep[np.argsort(np.asarray(depth)[keep])[::-1]]
    sizes=np.clip(np.rint(sz),7,78).astype(int);angles=(np.rint(ang/3)*3).astype(int)%360;values=np.clip(aa*255,0,255).astype(int)
    for i in keep:
        sp=sprite(str(codes[i]),int(sizes[i]),int(angles[i]))
        im.paste(int(values[i]),(round(xy[i,0]-sp.width/2),round(xy[i,1]-sp.height/2)),sp)

def plain(im,txt,xy,size=27,value=255,anchor='la'):
    ImageDraw.Draw(im).text(xy,txt,font=font(size),fill=round(value),anchor=anchor)

INTRO=[
    '/*','A new era of compute.', '',
    '    ...error... Serial mode is not enough.',
    'A bigger question has been detected.',
    'Multiple minds required.', '',
    'An upgrade is in progress...', '',
    'Ready >>',
    'Set System Mode to: PARALLEL',
    '{ Initialize a new generation of computing }',
    '    ...from one core to collective intelligence...',
    '    ...breaking through the next bottleneck...',
    'The world is ready for the next discovery.',
    '    >> Accelerate?', 'Sure.',
    '    // Set mode: XFLOPS.',
    'Open possibilities. Build the infrastructure.', '',
    'HPC system updating...', 'LOG(DEBUG): Executing <<', '',
    '#include <mpi.h>',
    'int main(int argc, char **argv) {',
    '    MPI_Init(&argc, &argv);',
    '    // Initialize the compute fabric',
    '    launch_parallel_kernels();',
    '    synchronize_workers();',
    '    return discover_what_is_next();',
    '}', '',
    '// Start the main loop',
    'while (residual > tolerance) {',
    '    exchange_boundaries();',
    '    iterate();',
    '}', '',
    '>>', 'Initializing system upgrade protocol...',
    'Activating: SJTU Xflops / Shanghai Jiao Tong University',
]
TASKS=['Mapping compute kernels to available execution units',
       'Scheduling independent tasks across the cluster',
       'Preparing high bandwidth interconnect routes',
       'Synchronizing collective communication across all ranks',
       'Retrieving memory pages for the next working set',
       'Building a new pipeline for distributed training',
       'Activating all-reduce gradient synchronization',
       'Removing unnecessary serialization from the hot path',
       'Overlapping memory transfers with active computation',
       'Restoring numerical stability across boundary conditions',
       'Creating the next simulation from a better question',
       'Checking shared state and solver convergence',
       'Committing a new generation of parallel potential',
       'Releasing the next wave of discovery']
LOGS=INTRO.copy()
for k in range(150):
    msg=TASKS[k%len(TASKS)]+'...'
    detail=f' rank={k%64:02d} / block={1024*(k%8+1):04d} / phase={k%9+1:02d} / active'
    LOGS.append((msg+detail)[:89].ljust(90)+['[OK]','[OK]','[>>]'][k%3])

def pretear_alpha(xy,t):
    # The reference opens small holes in late terminal rows before those rows
    # release into the larger wind field. This precedes bulk advection.
    strength=float(smooth((t-2.57)/.20)*(1-smooth((t-3.12)/.24)))
    hole=np.zeros(len(xy))
    for cx,cy,rx,ry in [(.42*W,.74*H,260,63),(.53*W,.87*H,385,60)]:
        r=np.sqrt(((xy[:,0]-cx)/rx)**2+((xy[:,1]-cy)/ry)**2)
        hole=np.maximum(hole,1-smooth((r-.75)/.25))
    return 1-hole*strength

def terminal_line(im,text,x,y,size=27,t=None):
    # A common baseline and explicit tracking match the wide terminal grid.
    if not text:return
    advance=size*19.8/27
    xy=np.array([[x+(i+.5)*advance,y+size*2/3] for i in range(len(text))])
    batch(im,xy,size,np.array(list(text)),pretear_alpha(xy,t) if t is not None else 1.)

SCROLL_KEYS=[(0,0),(2.25,0),(2.34,.8),(2.42,1.57),(2.51,1.97),(2.59,3.55),(2.67,2.77),(2.84,2.77),(2.92,.8),(3.01,.4),(3.17,.8),(3.26,.4),(3.43,.4),(3.51,0),(4.8,0)]
ST=np.arange(0,5.01,.001)
SV=np.array([curve(t,SCROLL_KEYS) for t in ST])*H
SI=np.cumsum(SV)*.001
def scroll(t):return math.floor(float(np.interp(t,ST,SI))/36)*36
ROW_KEYS=[(0,0),(.2,1),(.45,2),(1.45,2),(1.70,5),(1.95,8),(2.12,13),(2.25,19),(2.34,29),(3.8,29)]

def terminal(im,t,second=False):
    if not second:
        shift=scroll(t);rowmax=int(curve(t,ROW_KEYS)+shift/36)
        for row in range(min(rowmax,len(LOGS))):
            y=40+row*36-shift
            if -36<y<H:terminal_line(im,LOGS[row],42,y,27,t=t)
    else:
        local=t-9.45
        y0=curve(t,[(9.45,1060),(9.64,510),(9.90,70),(10.25,40),(11.5,-470),(12,-970)])
        lines=['Ready >>','Set System Mode to: DISTRIBUTED',
               '{ Synchronizing the next generation of intelligence }',
               'GLOBAL COMPUTE ZONES:','',
               '>>> SYSTEM UPDATE: XFLOPS / AI INFRASTRUCTURE',
               'ModuleLoaded: TrainingEngine.cpp       RoutingOptimizer: ENABLED',
               'Autolayer: DATA_PARALLEL               GradientSync: ALL_REDUCE',
               'Interconnect: ACTIVE                  WorkerState: READY','',
               '// SYSTEM MESSAGE - Running background loop',
               'TensorShardingPlan: Resolved',
               'Checkpoint: Consistent',
               'CollectiveGroup: Synchronized',
               'Optimizer.step() -> NEXT_ITERATION','',
               '#include <xflops/collective.h>',
               'int main() {',
               '    launch_training_pipeline();',
               '    synchronize_gradients();',
               '    return next_breakthrough();','}', '',
               '>>> COMPUTE ONLINE - OPEN - COLLABORATIVE - PARALLEL']
        for k in range(50):lines.append(LOGS[40+k])
        limit=int(max(0,local)*85)+8
        for row,line in enumerate(lines[:limit]):
            y=y0+row*36
            if -40<y<H:
                if t<11.35:terminal_line(im,line,42,y,26)
                elif line:
                    k=np.arange(len(line));dt=t-11.35
                    xx=42+(k+.5)*26*19.8/27
                    phase=xx*.004+row*.3
                    xy=np.column_stack((xx+dt*dt*(1000+500*np.sin(phase)),y+26*2/3+dt*dt*(-850+450*np.cos(phase))))
                    batch(im,xy,26,np.array(list(line)),1-smooth(dt/.65),dt*70*np.sin(phase))

def prepare_blowout():
    """Integrate velocity. Never interpolate particles toward a target model."""
    start=3.03;dt=1/120;ts=np.arange(start,5.451,dt)
    shift=scroll(start);first=int(shift/36)-2
    xy=[];chars=[]
    for row in range(max(first,0),max(first,0)+58):
        for col,ch in enumerate(LOGS[row]):
            if ch==' ':continue
            xy.append((42+(col+.5)*19.8,40+row*36-shift+18));chars.append(ch)
    p0=np.asarray(xy);n=len(p0);noise=RNG.random(n);p=p0.copy();vel=np.zeros_like(p);angle=np.zeros(n)
    seeds=np.array([[.08*W,.04*H],[.39*W,.06*H],[.30*W,.62*H],[.59*W,.58*H],[.98*W,.4*H]])
    dd=np.linalg.norm(p0[:,None]-seeds[None],axis=-1).min(axis=1)
    release=3.04+.30*np.clip(dd/650,0,1)+.10*noise+.58*(np.clip(p0[:,0]/W,0,1)**2)
    release+=np.where((np.arange(n)//25)%11==0,.35,0)
    attached=np.ones(n,dtype=bool);states=[];spin=RNG.uniform(-1,1,n)
    signs=np.array([1,-1,-1,1,-1])
    for it,t in enumerate(ts):
        attached= t<release
        p[attached]=p0[attached]-np.array([0,scroll(t)-shift])
        active=~attached
        amp=curve(t,[(3.03,0),(3.25,.45),(3.6,1.05),(4.0,1.5),(4.45,1.8),(5.4,1.7)])
        delta=p[:,None]-seeds[None]
        r2=np.sum(delta*delta,axis=-1)
        fall=np.exp(-r2/(310**2))
        tangent=np.stack([-delta[:,:,1],delta[:,:,0]],axis=-1)
        vortex=(tangent*(fall*signs)[...,None]*2.5).sum(axis=1)
        outward=(delta*fall[...,None]*1.7).sum(axis=1)
        wind=np.stack([100*np.sin(p[:,1]*.005+t)-90,70*np.sin(p[:,0]*.004-t*.8)],axis=1)
        target=amp*(vortex+outward+wind)
        late=float(smooth((t-4.30)/.65))
        target[:,0]=mix(target[:,0],-420+140*np.sin(p[:,1]*.007+t),late*.65)
        vel[active]+=(target[active]-vel[active])*min(1,dt*8)
        p[active]+=vel[active]*dt
        angle[active]+=spin[active]*amp*115*dt
        if it%5==0:
            states.append((p.copy(),angle.copy(),np.where(attached,1.,np.clip(1-.14*(t-release),.7,1))))
    return dict(p=np.array([x[0] for x in states],dtype=np.float32),a=np.array([x[1] for x in states],dtype=np.float32),b=np.array([x[2] for x in states],dtype=np.float32),glyph=np.asarray(chars),start=start,step=dt*5)

BLOW=None
def blowout(im,t):
    global BLOW
    if BLOW is None:
        BLOW=prepare_blowout()
    k=(t-float(BLOW['start']))/float(BLOW['step']);i=int(np.clip(math.floor(k),0,len(BLOW['p'])-2));u=np.clip(k-i,0,1)
    p=mix(BLOW['p'][i],BLOW['p'][i+1],u);a=mix(BLOW['a'][i],BLOW['a'][i+1],u)
    strength=1-smooth((t-4.65)/.65)
    size=27+3*np.sin(np.arange(len(p))*.14)*smooth((t-3.4)/.8)
    batch(im,p,size,BLOW['glyph'],strength*BLOW['b'][i]*pretear_alpha(p,t),a)

# Each row is a material streamline; material is advected through a continuously
# deforming sheet.  It has no target positions and never morphs into a headline.
ORNG=np.random.default_rng(618)
OU,OV=np.meshgrid(np.arange(-2100,2100,24.),np.arange(-1350,1350,35.))
OUID=np.arange(OU.size);OV=OV.ravel();OU=OU.ravel()
OJ=ORNG.uniform(-1,1,(len(OU),3))
OGLYPH=np.array(list('00110101+-><=/*#:%'))[ORNG.integers(0,17,len(OU))]
OKEY=ORNG.random(len(OU))>.16

def ocean(im,t,opacity=1.,mode='normal'):
    # Integrated flow speed has an impulse at the diagonal-sheet cut. The
    # material follows curved lane tangents, with independent depth parallax.
    tau=t-4.2
    travel=410*tau+110*tau*tau/(1+abs(tau))
    if t>7.63:travel+=430*(1-math.exp(-(t-7.63)*5))
    u=(OU-travel+2100)%4200-2100
    v=OV+OJ[:,1]*9
    if 8.62<=t<9.67:return
    early=4.2<t<6.173;sheet=7.633<=t<8.62
    if 8.43<t<8.62:opacity*=1-smooth((t-8.43)/.19)
    bend_strength=.45 if early else (.23 if sheet else 1.)
    bending=bend_strength*(70*np.sin(u*.0031+v*.003+tau*.9)+44*np.sin(u*.006-v*.004-tau*.6))
    z=(.08 if sheet else .24)*np.sin(u*.0024+v*.0015+tau*.3)+(.04 if sheet else .20)*np.cos(v*.004+tau*.6)
    perspective=1/(1+z)
    x=u*perspective+W/2
    y=(v+bending)*perspective+H/2
    angles=-np.degrees(np.arctan(bend_strength*(.217*np.cos(u*.0031+v*.003+tau*.9)+.264*np.cos(u*.006-v*.004-tau*.6))))
    # A pressure pocket displaces the streamlines around the approaching prop.
    if 4.35<t<7.633:
        pocket=curve(t,[(4.35,0),(4.75,.78),(5.15,1),(6.12,1),(6.18,1),(7.5,1),(7.633,.8)])
        cx=curve(t,[(4.35,1.10*W),(4.76,.83*W),(5.12,.50*W),(5.7,.44*W),(6.12,.42*W),(6.18,.20*W),(7.633,.20*W)])
        cy=.51*H
        dx=(x-cx)/1.48;dy=y-cy;r=np.hypot(dx,dy)+.1
        # The divergence is radial near the object and curls around its rim.
        radius=curve(t,[(4.35,120),(5,340),(6.1,370),(6.18,380),(7.633,370)])*pocket
        theta=np.arctan2(dy,dx)+.50*pocket*np.exp(-r/650)*np.sin(t*.7+r*.003)
        nr=np.sqrt(r*r+radius*radius)
        x=cx+nr*np.cos(theta)*1.48;y=cy+nr*np.sin(theta)
        angles+=np.degrees(.4*pocket*np.exp(-r/650)*np.sin(t*.7+r*.003))
    if 7.633<=t<8.62:
        # Source cut to a close, oblique material sheet. Camera then opens it.
        ang=-.53;dx=x-W/2;dy=y-H/2
        x=W/2+dx*math.cos(ang)-dy*math.sin(ang)
        y=H/2+dx*math.sin(ang)+dy*math.cos(ang)
        angles+=30
        radius=curve(t,[(7.633,0),(7.80,70),(8.0,200),(8.20,280),(8.47,200),(8.62,180)])
        dx=x-W/2;dy=(y-H/2)*1.25;r=np.hypot(dx,dy)+.1
        nr=np.sqrt(r*r+radius*radius)
        gather=float(smooth((t-8.18)/.29))
        nr=mix(nr,250+140*np.tanh((r-600)/1000),gather)
        twist=.7*np.exp(-r/700)*smooth((t-7.7)/.6)
        th=np.arctan2(dy,dx)+twist
        x=W/2+nr*np.cos(th);y=H/2+nr*np.sin(th)/1.25
        angles+=np.degrees(twist)
    xy=np.column_stack((x,y))
    size=(26+5*OJ[:,2])*perspective
    alpha=(.85+.14*OJ[:,0])*opacity*OKEY
    glyph=OGLYPH
    if early:
        pattern=np.array(list('01_<>#//----++++***==/0101_____'))
        glyph=pattern[(OUID+OUID//175*7)%len(pattern)]
        # Streaked readable material and empty lanes, rather than uniform noise.
        alpha*=np.where(((OUID%175)+(OUID//175)*5)%37<26,1.,.12)
    elif sheet:
        pattern=np.array(list('///01////10////01///<>////'))
        glyph=pattern[OUID%len(pattern)]
        alpha*=np.where(OUID//175%9==0,.15,1.)
        alpha*=mix(1.,np.where(OUID%3==0,1.,.025),gather)
        size*=mix(1.,.75,gather)
    if 12<t<19.8:
        # Open a quiet channel around the cabinets and tensor while the near
        # plane continues flowing past. This preserves their silhouette.
        hero_x=curve(t,[(12,960),(15,900),(15.8,760),(17.7,780),(18.6,930),(19.8,1100)])
        envelope=float(smooth((t-12)/.6)*(1-smooth((t-19)/.8)))
        clearance=np.exp(-((x-hero_x)/560)**2-((y-530)/430)**2)
        alpha*=1-.35*clearance*envelope
    batch(im,xy,size,glyph,alpha,angles,z)
    # Near plane fragments are a different population, with much larger type
    # and higher velocity. Their curved flight crosses the camera aperture.
    ids=np.arange(125);phase=ids*2.399963
    xx=((ids*83.41-t*840+120*np.sin(phase+t*.8))%2600)-340
    yy=((ids*169.37+t*170+180*np.sin(phase+t*1.2))%1600)-260
    if 4.35<t<7.633:
        dd=(xx-cx)**2/(560**2)+(yy-cy)**2/(390**2)
        visibility=smooth((dd-.7)/.7)
    elif 7.633<t<9.6:
        dd=((xx-W/2)/550)**2+((yy-H/2)/410)**2
        visibility=smooth((dd-.6)/.8)
    else:visibility=1.
    foreground_fade=1-smooth((t-8.18)/.22) if 8.18<t<9.67 else 1.
    batch(im,np.column_stack((xx,yy)),44+15*np.sin(phase)**2,
          np.array(list('01*/<>+=#'))[ids%9],opacity*.9*visibility*foreground_fade,
          40*np.sin(phase+t*.8))

def caption(im,text,t,start,settle,end,center=(1530,540),size=29,break_end=None):
    """Independent typography. A readable hold precedes per-letter release."""
    if t<start:return
    if break_end is None:break_end=end+.30
    if t>=break_end:return
    f=font(size);advance=f.getlength('M');width=round(advance*len(text)+24);height=round(size*1.30)
    x,y=center;x-=width/2;y-=height/2
    if t<=end:
        a=float(smooth((t-start)/.09))
        ImageDraw.Draw(im).rectangle((x,y,x+width,y+height),fill=round(255*a))
        if t<settle:
            reveal=int(len(text)*smooth((t-start)/(settle-start)))
            noise='<>01/*#+';txt=text[:reveal]+''.join(noise[(i*11+int(t*28))%8] if c!=' ' else ' ' for i,c in enumerate(text[reveal:]))
        else:txt=text
        ImageDraw.Draw(im).text((x+12,y+height/2),txt,font=f,fill=0,anchor='lm')
    else:
        dt=t-end
        for i,ch in enumerate(text):
            if ch==' ':continue
            lag=(i%3)*.025;u=max(0,dt-lag);vx=(i-len(text)/2)*145
            xx=x+12+(i+.5)*advance+vx*u+60*math.sin(i)*u*u
            yy=y+height/2-650*u*u+math.cos(i*2)*210*u
            batch(im,np.array([[xx,yy]]),size,ch,1-smooth(dt/(break_end-end)),(i%2*2-1)*210*u)

def draw_prop(im,state,alpha=1.):
    if len(state['hull'])>=3:
        ImageDraw.Draw(im).polygon([tuple(p) for p in state['hull']],fill=0)
    batch(im,state['xy'],state['size'],state['glyph'],state['alpha']*alpha,state['angle'],state['depth'])

def first_prop(im,t):
    from kinetic_geometry import tensor_cube
    x=curve(t,[(4.65,1.10),(4.76,.83),(4.93,.67),(5.10,.54),(5.28,.46),(5.65,.43),(6.14,.38)])*W
    y=.52*H+38*math.sin((t-4.65)*6)
    size=curve(t,[(4.65,150),(4.85,250),(5.15,340),(5.4,315),(5.75,355),(6.14,315)])
    tau=t-4.65
    # Strong impulse followed by a long braking tail; ongoing tumbling never
    # freezes at the end of the entry translation.
    phase=2.3*(1-math.exp(-tau*2.3))+.95*tau
    pose=(.55+.56*math.sin(tau*5),-.7+phase,.16*math.sin(tau*6))
    state=tensor_cube(t,center=(x,y),scale=size,rotation=pose)
    draw_prop(im,state)
    # A second, much smaller tensor turns independently through the foreground.
    satellite=tensor_cube(t+.7,center=(x+290*math.cos(tau*4),y-270*math.sin(tau*4)),scale=50+16*math.sin(tau*4),rotation=(tau*2,tau*3,-tau))
    draw_prop(im,satellite)

def flipping_chip(im,t):
    from kinetic_geometry import accelerator
    x=curve(t,[(6.173,.035),(6.27,.09),(6.35,.235),(6.43,.30),(6.52,.35),(6.60,.40),(6.77,.42),(7.633,.42)])*W
    diameter=curve(t,[(6.173,.34),(6.27,.43),(6.35,.36),(6.43,.20),(6.52,.16),(6.60,.09),(6.77,.145),(6.94,.15),(7.02,.15),(7.19,.17),(7.44,.23),(7.633,.28)])*H
    # The two face-to-edge flips use real model pitch; apparent depth scale is
    # separate from the view-facing area, just as measured in the reference.
    pitch=curve(t,[(6.173,-.50),(6.27,0),(6.52,1.48),(6.77,math.pi),(7.02,4.71),(7.27,6.28),(7.633,6.50)])
    pose=(pitch,.16*math.sin((t-6.2)*5),-.1+.14*math.sin((t-6.2)*6))
    state=accelerator(t,center=(x,.5*H+12*math.sin(t*6)),scale=diameter/3.45,rotation=pose)
    draw_prop(im,state)

OPH,OTH=np.meshgrid(np.linspace(.05,math.pi-.05,31),np.linspace(0,2*math.pi,71,endpoint=False))
OPH=OPH.ravel();OTH=OTH.ravel()

def orb(im,t):
    from kinetic_geometry import _rotation
    tau=t-8.47;emerge=float(smooth((t-8.40)/.14))
    r=1+.20*np.sin(OTH*3+OPH*4+tau*2)+.11*np.cos(OPH*9-tau*3)
    p=np.column_stack((np.sin(OPH)*np.cos(OTH),np.cos(OPH),np.sin(OPH)*np.sin(OTH)))*r[:,None]
    p=np.einsum('ij,kj->ik',p,_rotation((tau*1.2,tau*2.5,.3+tau*.7)))
    scale=curve(t,[(8.40,185),(8.47,220),(8.60,260),(8.87,255),(9.20,280),(9.38,280),(9.65,440)])
    burst=max(0,t-9.36)
    if burst:
        p*=1+burst*burst*32
        p[:,1]-=burst*burst*20
    z=p[:,2];pers=4.3/(4.3+z)
    xy=np.column_stack((W/2+p[:,0]*scale*pers,H*.52+p[:,1]*scale*pers))
    alpha=(.4+.55*clamp((1.5-z)/2.7))*(1-smooth(burst/.31))*emerge
    chars=np.array(list('0011/*+#<>'))[np.arange(len(p))%10]
    batch(im,xy,(19+5*np.sin(OTH*3))*pers,chars,alpha*.6,np.degrees(OTH*.17+tau*.2),z)
    ids=np.arange(340);ph=np.arccos(1-2*(ids+.5)/len(ids));th=ids*2.399963
    rr=1.04+.21*np.sin(th*3+ph*4+tau*2)
    q=np.column_stack((np.sin(ph)*np.cos(th),np.cos(ph),np.sin(ph)*np.sin(th)))*rr[:,None]
    q=np.einsum('ij,kj->ik',q,_rotation((tau*1.2,tau*2.5,.3+tau*.7)))
    q*=1+burst*burst*32;q[:,1]-=burst*burst*20
    pp=4.3/np.maximum(1.3,4.3+q[:,2])
    xy=np.column_stack((W/2+q[:,0]*scale*pp,H*.52+q[:,1]*scale*pp))
    near=clamp((.65-q[:,2])/.55)*(1-smooth(burst/.31))*emerge
    batch(im,xy,(35+12*np.sin(th)**2)*pp,np.array(list('0123456789'))[ids%10],near,np.degrees(th*.23+tau),q[:,2])
    # Readable orbit ribbons travel in real 3D; near and far halves are drawn
    # at different scales, independently of the central surface rotation.
    for lane in range(3):
        a=np.linspace(0,2*math.pi,165,endpoint=False)+tau*(1.6+lane*.24)
        q=np.column_stack((1.68*np.cos(a),.12*np.sin(a*3+lane),1.68*np.sin(a)))
        q=np.einsum('ij,kj->ik',q,_rotation((.55+lane*.47,.2+tau*.3,lane*.95)))
        fac=4.3/(4.3+q[:,2]);s=scale*(1+burst*burst*25)
        xy=np.column_stack((W/2+q[:,0]*s*fac,H*.52+q[:,1]*s*fac-burst*burst*4200))
        text=np.array(list(('ALL_REDUCE>>01//TENSOR_PARALLEL>>10/'*7)[:165]))
        batch(im,xy,20*fac,text,(.65+.28*clamp(-q[:,2]))*(1-smooth(burst/.31))*emerge,20*np.cos(a+lane),q[:,2])

def frame(t):
    im=Image.new('L',(W,H),0)
    if t<3.03:terminal(im,t)
    elif t<4.55:blowout(im,t)
    elif t<5.30:
        ocean(im,t,float(smooth((t-4.55)/.4)))
        blowout(im,t)
        if t>=4.65:first_prop(im,t)
    elif t<6.173:
        ocean(im,t);first_prop(im,t)
    elif t<7.633:
        ocean(im,t);flipping_chip(im,t)
    elif t<8.40:ocean(im,t)
    elif t<9.67:
        ocean(im,t,float(1-smooth((t-9.20)/.47)))
        if t>=9.45:terminal(im,t,True)
        orb(im,t)
    elif t<12:
        if t>11.35:ocean(im,t,float(smooth((t-11.35)/.65)))
        terminal(im,t,True)
    else:
        from kinetic_later import draw_later
        draw_later(im,t)
    caption(im,'<<HPC IS...',t,5.18,5.48,6.34,(1540,530),29,6.62)
    caption(im,'>>FASTER>>',t,7.75,7.98,8.36,(960,530),30,8.55)
    caption(im,'--AI INFRA--',t,9.76,9.96,10.87,(870,550),29,11.13)
    # An indexed palette preserves crisp glyph antialiasing with exact two-tone
    # styling; no artificial camera drift is applied to readable captions.
    im.putpalette(PAL)
    return im.convert('RGB')

def render(start=0,end=DURATION,path=None,width=W):
    path=Path(path or ROOT/'work/kinetic-silent.mp4');path.parent.mkdir(exist_ok=True,parents=True)
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
         '-vf',f'scale={width}:-2:out_color_matrix=bt709:out_range=tv','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',
         '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);st=time.time();count=round((end-start)*FPS)
    try:
        for i in range(count):
            proc.stdin.write(frame(start+i/FPS).tobytes())
            if i%24==0:print(f'{start+i/FPS:.2f}s | frame {i}/{count} | {time.time()-st:.1f}s',flush=True)
    finally:proc.stdin.close()
    rc=proc.wait()
    if rc:raise RuntimeError(f'ffmpeg exited {rc}')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=float,default=0);ap.add_argument('--end',type=float,default=DURATION)
    ap.add_argument('--output');ap.add_argument('--width',type=int,default=W);ap.add_argument('--stills',nargs='*',type=float)
    args=ap.parse_args()
    if args.stills is not None:
        dest=ROOT/'work/kinetic-stills';dest.mkdir(parents=True,exist_ok=True)
        for t in args.stills:frame(t).save(dest/f'{t:05.2f}.png');print(t,flush=True)
    else:render(args.start,args.end,args.output,args.width)
