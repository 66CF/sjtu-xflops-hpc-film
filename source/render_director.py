#!/usr/bin/env python3
"""A program learns to run together: character-led ASCII motion film."""
from pathlib import Path
from functools import lru_cache
import math,argparse,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import render_kinetic as old

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS,DURATION=1920,1080,24,30
curve,smooth,mix=old.curve,old.smooth,old.mix
MONO='/System/Library/Fonts/Menlo.ttc'
PHYSICS=None
KNOT=None

def physical_flow(im,t,opacity=1.,decimate=1,exclude_ids=None):
    global PHYSICS
    if PHYSICS is None:
        candidates=[ROOT/'work/physics-v5/flow.npz',ROOT/'work/physics-v5/physics-qa.npz']
        found=next((p for p in candidates if p.exists()),None)
        if found is None:return False
        z=np.load(found);PHYSICS={k:z[k] for k in z.files};z.close()
    data=PHYSICS;times=data['times']
    if t<times[0] or t>times[-1]+.005:return False
    i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2));u=float(np.clip((t-times[i])/(times[i+1]-times[i]),0,1))
    xy=mix(data['xy'][i],data['xy'][i+1],u)
    velocity=mix(data['velocity'][i],data['velocity'][i+1],u)
    angle=mix(data['angle'][i],data['angle'][i+1],u)
    # Cache samples interpolate an integrated simulation, never design targets.
    group=data['group'];ids=np.arange(len(group));size=data['size'].copy()
    birth=data.get('birth',np.where(group==3,data['release']-.025,0.))
    alive=(t>=birth)&((ids%decimate==0)|(group==3))
    aa=opacity*alive.astype(float)
    if exclude_ids is not None:aa[exclude_ids]=0
    # The force field is deliberately richer than the visible print. A stable
    # material sample keeps curls legible without forming a solid white wall.
    material=float(smooth((t-4.15)/.6))
    pinned=data['attached'][i]&(group==0)
    material=material*(~pinned)
    fine=(group==0)|(group==1)
    visible=((ids*2654435761)%4294967296)%3!=0
    visible=np.where(group==2,ids%4==0,visible)
    visible=np.where(group==3,True,visible)
    aa*=1-material*(~visible)
    size*=np.where(fine,1-.22*material,np.where(group==2,1-.16*material,1.))
    # Once the lead worker takes the stairs, the released worker surfaces
    # recede visually. Their integrated collision/wake motion is unchanged.
    focus=float(smooth((t-17.22)/.42))
    aa*=np.where(group==3,1-.48*focus,1.)
    coarse=group==2
    size[coarse]=mix(size[coarse],np.minimum(size[coarse],26.),focus)
    bold=t<4.45
    blur=(np.linalg.norm(velocity,axis=1)>420)&(aa>.01)
    if blur.any():
        batch(im,xy[blur]-velocity[blur]/120,size[blur],data['glyph'][blur],aa[blur]*.15,angle[blur],bold=bold)
    if bold:
        batch(im,xy,size,data['glyph'],aa,angle,bold=True)
    else:
        batch(im,xy[~pinned],size[~pinned],data['glyph'][~pinned],aa[~pinned],angle[~pinned],bold=False)
        batch(im,xy[pinned],size[pinned],data['glyph'][pinned],aa[pinned],angle[pinned],bold=True)
    return True

@lru_cache(maxsize=256)
def face(size,bold=True):return ImageFont.truetype(MONO,max(5,int(size)),index=1 if bold else 0)

@lru_cache(maxsize=80000)
def tile(ch,size,angle=0,inverse=False,bold=True):
    w=math.ceil(size*.73);h=math.ceil(size*1.25)
    im=Image.new('L',(w,h),255 if inverse else 0)
    ImageDraw.Draw(im).text((w/2,-size*.04),str(ch),font=face(size,bold),fill=0 if inverse else 255,anchor='ma')
    if angle:im=im.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
    return im

def batch(im,xy,size,glyph,alpha=1.,angle=None,depth=None,bold=True):
    xy=np.asarray(xy);n=len(xy)
    if n==0:return
    sz=np.broadcast_to(size,(n,));aa=np.broadcast_to(alpha,(n,));codes=np.broadcast_to(glyph,(n,))
    ang=np.zeros(n) if angle is None else np.broadcast_to(angle,(n,))
    keep=np.where((xy[:,0]>-90)&(xy[:,0]<W+90)&(xy[:,1]>-90)&(xy[:,1]<H+90)&(aa>.025)&(codes!=' '))[0]
    if depth is not None:keep=keep[np.argsort(np.asarray(depth)[keep])[::-1]]
    sz=np.clip(np.rint(sz),7,84).astype(int);ang=(np.rint(ang/4)*4).astype(int)%360;v=np.clip(aa*255,0,255).astype(int)
    for j in keep:
        sp=tile(str(codes[j]),int(sz[j]),int(ang[j]),False,bold)
        im.paste(int(v[j]),(round(xy[j,0]-sp.width/2),round(xy[j,1]-sp.height/2)),sp)

def text(im,txt,xy,size=30,fill=255,anchor='la',bold=True):
    ImageDraw.Draw(im).text(xy,txt,font=face(size,bold),fill=int(fill),anchor=anchor)

old.batch=batch
old.font=lambda n:face(n,True)
# Terminal rows stay intact until the visible physical cursor strikes them.
old.pretear_alpha=lambda xy,t:np.ones(len(xy))
# Language supports the physical joke: a compute process literally starts running.
old.INTRO[1]='A program can run.'
old.INTRO[3]='    ...error... One core can only go so far.'
old.INTRO[4]='A bigger question has been detected.'
old.INTRO[5]='More than one mind required.'
old.INTRO[14]='The next breakthrough is a team effort.'
old.LOGS[:len(old.INTRO)]=old.INTRO

def label(im,txt,xy,size=29,alpha=1.):
    f=face(size);advance=size*.75;w=round(advance*len(txt))+16;h=round(size*1.19)
    sp=Image.new('L',(w,h),255);d=ImageDraw.Draw(sp)
    for i,ch in enumerate(txt):d.text((8+i*advance,-size*.055),ch,font=f,fill=0)
    mask=Image.new('L',sp.size,round(alpha*255))
    im.paste(sp,(round(xy[0]-w/2),round(xy[1]-h/2)),mask)

def timed_label(im,txt,xy,t,start,end,size=29):
    if start<t<end:
        a=float(smooth((t-start)/.10)*(1-smooth((t-end+.12)/.12)))
        label(im,txt,xy,size,a)

def solid(im,state,center=(960,540),canvas=(950,950),cell=20,alpha=1.):
    """Print lighting as type density, not a translucent wire-frame."""
    lum=np.asarray(state['luma']);mask=np.asarray(state['mask'])
    cw=int(cell*.68);ch=cell
    nx=max(6,round(canvas[0]/cw));ny=max(6,round(canvas[1]/ch))
    lum=np.asarray(Image.fromarray(np.uint8(np.clip(lum,0,1)*255)).resize((nx,ny),Image.Resampling.BILINEAR))/255
    cov=Image.fromarray(np.uint8(mask)*255).resize((nx,ny),Image.Resampling.BILINEAR)
    ma=np.asarray(cov)>110
    features=np.asarray(Image.fromarray(state.get('feature',np.zeros_like(mask,dtype=np.uint8))).resize((nx,ny),Image.Resampling.NEAREST))
    x0=round(center[0]-nx*cw/2);y0=round(center[1]-ny*ch/2)
    cover=cov.resize((nx*cw,ny*ch),Image.Resampling.NEAREST)
    if alpha<1:cover=cover.point(lambda a:round(a*alpha))
    im.paste(0,(x0,y0),cover)
    ramp=np.array(list(' .:-=+*#%MW'))
    for iy,ix in np.argwhere(ma):
        val=float(np.clip((lum[iy,ix]-.29)/.48,0,1))
        level=int(np.clip(val*10,0,10))
        code=ramp[level]
        if code==' ':continue
        ft=features[iy,ix]
        inv=(ft==1) or (ft==3 and val>.90)
        if inv:code='01X#MW'[(ix+iy*3)%6]
        if ft==1:code='X'
        # High lights become a reverse-video cell; intermediate shades use
        # larger/smaller ink coverage while remaining emphatically white.
        sp=tile(str(code),round(cell*.98),0,inv,True)
        if inv:
            block=Image.new('L',(cw,ch),255)
            block.paste(sp,((cw-sp.width)//2,(ch-sp.height)//2))
            if alpha<1:im.paste(block,(x0+ix*cw,y0+iy*ch),Image.new('L',block.size,round(alpha*255)))
            else:im.paste(block,(x0+ix*cw,y0+iy*ch))
        else:im.paste(round(255*alpha),(x0+ix*cw+(cw-sp.width)//2,y0+iy*ch+(ch-sp.height)//2),sp)

# A designed stream material: intact repeated symbol phrases, sparse lanes,
# and deep blue cavities. No all-purpose random particle wallpaper.
U,V=np.meshgrid(np.arange(-1600,1600,20.),np.arange(-950,950,30.))
U=U.ravel();V=V.ravel();IDS=np.arange(len(U));ROW=IDS//160;COL=IDS%160
PATTERNS=[list('.....-----+++++<<0001100>>+++***##'),
          list('________//0011//////<<<<>>>+'),
          list('01XX**##>>><<===+++....'),
          list('/////1/////0/////1/////0/')]

def stream(im,t,phase='run',opacity=1.):
    speed=curve(t,[(4.4,180),(4.75,950),(5.1,360),(6.17,460),(6.6,630),(7.6,570),(8.0,950)])
    # Per-row advection bends along a wave. The character identity belongs to
    # its material lane, while patches repeat coherently along that lane.
    travel=(t-4.4)*410+90*math.sin((t-4.4)*2)
    x=(U-travel+1600)%3200-1600
    y=V.copy()
    wave=45*np.sin(x*.004+y*.005+t*.8)+24*np.cos(x*.008-y*.004-t)
    yy=y+wave
    scale=1+.16*np.sin(x*.002+y*.003+t*.4)
    xx=x*scale+960;yy=yy*scale+540
    angles=6*np.cos(x*.004+y*.005+t*.8)
    g=np.empty(len(x),dtype='<U1')
    for lane,p in enumerate(PATTERNS):
        choose=ROW%4==lane;a=np.array(p)
        g[choose]=a[(COL[choose]+ROW[choose]*3)%len(p)]
    alpha=np.ones(len(x))*.94*opacity
    sizes=np.full(len(x),26.)*scale
    if phase=='run':
        # Horizontally streaked background and a receding ground plane make
        # the runner's direction and contact with a surface legible.
        cx=curve(t,[(4.4,2200),(4.75,1570),(5.05,1100),(5.3,890),(6.17,860)])
        hole=((xx-cx)/410)**2+((yy-570)/430)**2
        alpha*=smooth((hole-.62)/.4)
        alpha*=np.where((ROW%9<4)|((COL+ROW*3)%43<15),1.,.025)
        # The back rows remain level; the floor is made of perspectival lines.
        ground=yy>840
        angles[ground]=18+12*np.sin(xx[ground]*.001)
        g[ground]=np.array(list('/01_'))[(COL[ground]+ROW[ground])%4]
    elif phase=='chip':
        dx=(xx-390)/1.48;dy=yy-540;r=np.hypot(dx,dy)+.1
        th=np.arctan2(dy,dx)+.12*np.sin(r*.004+t)
        nr=np.sqrt(r*r+345**2)
        xx=390+1.48*nr*np.cos(th);yy=540+nr*np.sin(th)
        alpha*=np.where((ROW%12<8)|((COL+3*ROW)%36<15),1.,.15)
        sizes*=.9
        # Three coarse foreground fragments peel away from the tiny backdrop.
        patch=((COL+ROW*2)%52<7)&(ROW%18<5)
        sizes[patch]*=1.7;g[patch]=np.array(list('10X+'))[ROW[patch]%4]
    else:
        # A coherent slash sheet tilts through the camera and gathers tightly.
        a=-.50;dx=xx-960;dy=yy-540
        xx=960+dx*math.cos(a)-dy*math.sin(a);yy=540+dx*math.sin(a)+dy*math.cos(a)
        g=np.array(PATTERNS[3])[IDS%len(PATTERNS[3])];angles+=28
        dx=xx-960;dy=(yy-540)*1.15;r=np.hypot(dx,dy)+.1
        radius=curve(t,[(7.63,0),(7.95,180),(8.15,255),(8.47,170)])
        rr=np.sqrt(r*r+radius*radius)
        gather=float(smooth((t-8.18)/.28))
        rr=mix(rr,220+90*np.tanh((r-500)/800),gather)
        th=np.arctan2(dy,dx)+.32*gather+.18*np.sin(r*.004+t)*gather
        xx=960+rr*np.cos(th);yy=540+rr*np.sin(th)/1.15
        alpha*=mix(1.,np.where(IDS%3==0,1.,0),gather)
        sizes*=mix(1.,.8,gather)
    batch(im,np.column_stack((xx,yy)),sizes,g,alpha,angles)

def hero_worker(im,t):
    from ascii_solids import sample_worker
    cx=curve(t,[(4.65,2210),(4.76,1620),(4.93,1320),(5.10,1080),(5.28,930),(5.65,890),(6.17,845)])
    h=curve(t,[(4.65,640),(4.95,890),(5.2,1050),(5.7,1030),(6.17,1070)])
    state=sample_worker(t,rotation=(.0,.62+.14*math.sin(t*3),.015*math.sin(t*8)),resolution=(110,120))
    solid(im,state,(cx,545),(h*110/120,h),cell=25)

def chip(im,t):
    from ascii_solids import sample_chip
    from chip_choreography import chip_state
    path=chip_state(t);diameter=path['height']
    state=sample_chip(t,rotation=path['rotation'],resolution=(100,100))
    solid(im,state,(path['x'],path['y']),(diameter,diameter),cell=max(12,round(diameter/26)))

def opening_impact(im,t):
    from opening_impact import impact_state
    state=impact_state(t)
    if not state['active']:return
    w,h=map(round,state['size'])
    plaque=Image.new('L',(w,h),255)
    ImageDraw.Draw(plaque).text((w/2,h/2-3),state['label'],font=face(state['font_size']),fill=0,anchor='mm')
    mask=Image.new('L',(w,h),255)
    rotation=-math.degrees(state['rotation'])
    plaque=plaque.rotate(rotation,Image.Resampling.BICUBIC,expand=True)
    mask=mask.rotate(rotation,Image.Resampling.BICUBIC,expand=True)
    x,y=state['center'];im.paste(plaque,(round(x-plaque.width/2),round(y-plaque.height/2)),mask)

def load_knot():
    global KNOT
    if KNOT is None:
        path=ROOT/'work/physics-v5/knot.npz'
        if path.exists():
            z=np.load(path);KNOT={k:z[k] for k in z.files};z.close()
    return KNOT

def data_knot(im,t):
    load_knot()
    if KNOT is not None:
        ts=KNOT['times'];i=int(np.clip(np.searchsorted(ts,t)-1,0,len(ts)-2))
        u=float(np.clip((t-ts[i])/(ts[i+1]-ts[i]),0,1))
        p=mix(KNOT['xyz'][i],KNOT['xyz'][i+1],u)
        vel=mix(KNOT['velocity'][i],KNOT['velocity'][i+1],u)
        ang=mix(KNOT['angle'][i],KNOT['angle'][i+1],u)
        fac=1200/(1200+p[:,2]);xy=np.column_stack((960+p[:,0]*fac,540+p[:,1]*fac))
        continuous='source_ids' in KNOT
        alpha=1. if continuous else float(smooth((t-8.39)/.14))
        depthfade=.70+.30*np.clip((320-p[:,2])/600,0,1)
        if continuous:
            # All nodes are the exact printed glyphs captured by the fluid.
            # Keep brightness/size at handoff, then depth changes naturally.
            depthfade=1-.30*np.clip(p[:,2]/600,0,1)
            rate=-1200*vel[:,2]/(1200+p[:,2])**2
            screen_v=vel[:,:2]*fac[:,None]+p[:,:2]*rate[:,None]
            blur=np.linalg.norm(screen_v,axis=1)>420
            batch(im,xy[blur]-screen_v[blur]/120,KNOT['size'][blur]*fac[blur],KNOT['glyph'][blur],depthfade[blur]*.15,ang[blur],p[blur,2],bold=False)
            batch(im,xy,KNOT['size']*fac,KNOT['glyph'],depthfade,ang,p[:,2],bold=False)
        else:
            ids=np.arange(len(p));keep=(ids%3==0)|(ids%17==0)
            batch(im,xy[keep],KNOT['size'][keep]*fac[keep]*.92,KNOT['glyph'][keep],alpha*depthfade[keep],ang[keep],p[keep,2],bold=False)
        return
    # Irregular rotating type sculpture. No longitude/latitude wire grid.
    from kinetic_geometry import _rotation
    n=680;i=np.arange(n);u=i*2.399963;v=np.arccos(1-2*(i+.5)/n);tau=t-8.44
    r=1+.23*np.sin(3*u+4*v+tau*3)+.12*np.cos(7*v-tau*2)
    p=np.column_stack((np.cos(u)*np.sin(v),np.cos(v),np.sin(u)*np.sin(v)))*r[:,None]
    R=_rotation((.45+tau*1.6,tau*2.6,tau*.5));p=np.einsum('ij,kj->ik',p,R)
    burst=max(0,t-9.3);p*=1+burst*burst*27;p[:,1]-=burst*burst*16
    depth=p[:,2]+4.3;fac=4.3/np.maximum(depth,.8)
    scale=curve(t,[(8.40,190),(8.58,255),(9.05,240),(9.30,270),(9.65,380)])
    xy=np.column_stack((960+p[:,0]*scale*fac,535+p[:,1]*scale*fac))
    vis=smooth((t-8.4)/.15)*(1-smooth(burst/.36))
    letters=np.array(list('0123456789<>+*'))[i%14]
    sizes=(18+24*((i%9)/8)**2)*fac
    aa=(.45+.55*np.clip((1.5-p[:,2])/2,0,1))*vis
    batch(im,xy,sizes,letters,aa,u*180/math.pi+t*40,depth)
    # One clean, oblique word orbit forms the silhouette's readable outer edge.
    txt='ALL_REDUCE//01/TENSOR_PARALLEL//10/'*4
    j=np.arange(len(txt));theta=j/len(txt)*math.tau+tau*2
    q=np.column_stack((1.45*np.cos(theta),.08*np.sin(theta*3),1.45*np.sin(theta)))
    q=np.einsum('ij,kj->ik',q,_rotation((.75,.2,tau*.3)))
    f=4.3/(q[:,2]+4.3)
    xy=np.column_stack((960+q[:,0]*scale*f,535+q[:,1]*scale*f-burst*burst*3700))
    batch(im,xy,22*f,np.array(list(txt)),vis*.9,18*np.cos(theta),q[:,2])

def command_fork(im,t):
    if t<10.55:
        old.terminal(im,t,True)
    else:
        # Terminal rows are lifted away, leaving one actionable command.
        alpha=1-float(smooth((t-10.55)/.45))
        if alpha>0:
            layer=Image.new('L',(W,H));old.terminal(layer,t,True)
            im.paste(layer,(0,-round(smooth((t-10.55)/.45)*250)),layer.point(lambda v:round(v*alpha)))
    if 10.6<t<12.1:
        prefix='$ mpirun -np ';suffix=' ./next_breakthrough'
        count='1' if t<11.30 else '6';f=face(33)
        total=f.getlength(prefix+count+suffix);x=(W-total)/2;y=430
        a=float(smooth((t-10.6)/.15)*(1-smooth((t-11.8)/.30)))
        text(im,prefix,(x,y),33,255*a)
        xx=x+f.getlength(prefix)
        if t>10.94:label(im,count,(xx+f.getlength(count)/2,y+17),33,a)
        else:text(im,count,(xx,y),33,255*a)
        text(im,suffix,(xx+f.getlength(count),y),33,255*a)

def word_weave(im,t,opacity):
    # Language becomes material for one specific shot, then clears completely.
    size=22;advance=13.5;phrase='PARALLEL>   ';period=len(phrase)*advance
    for row in range(37):
        y=8+row*30
        drift=((t-12)*105+row*27)%period
        for col in range(-1,14):
            x=col*period-drift
            band=.9 if row%5<3 else .45
            text(im,phrase,(x,y),size,round(255*opacity*band))

def workers(im,t):
    from ascii_solids import sample_worker
    # A single process forks to six, then camera pullback reveals a 4x6 quilt.
    zoom=curve(t,[(11.0,1),(11.65,1),(12.3,.70),(13.1,.51),(14.0,.46),(15.8,.46),(16.7,.46),(17.4,.46)])
    opening=float(smooth((t-11.15)/.7))
    reveal=float(smooth((t-12.25)/.85))
    weave_op=curve(t,[(11,0),(12,.05),(12.7,.30),(13.4,.20),(15,.13),(16.6,.05),(17.2,0)])
    if PHYSICS is None:word_weave(im,t,weave_op)
    # Local phases remain distinct while arrival times converge at a barrier.
    columns,rows=6,4;arrivals=[]
    for row in range(rows):
        for col in range(columns):
            idx=row*columns+col
            if idx==18 and t>=17.05:continue
            if row>0 and reveal<=0:continue
            birth=11.05+col*.07 if row==0 else 12.25+(row-1)*.13+col*.025
            opacity=float(smooth((t-birth)/.28))
            if idx!=18:opacity*=1-float(smooth((t-(16.61+idx*.009))/.24))
            if opacity<=0:continue
            rawx=(col-2.5)*580;rawy=(row-1.5)*505
            x=960+rawx*zoom*opening
            y=535+rawy*zoom*(.50+.50*reveal)
            settle=16.12 if idx==23 else 14.65+(idx%6)*.10+(idx//6)*.055
            arrivals.append(settle)
            run_t=t+.063*idx
            if t>settle:run_t=settle+.063*idx+math.sin((t-settle)*9)*.026
            # A small lateral stride finishes at each worker's local sync line.
            step=curve(t,[(birth,-110),(birth+.4,0),(14.6,0),(settle,32),(17.2,32)]) if settle>birth+.4 else 0
            x+=step*zoom
            if idx==23 and t>14.1:
                x+=curve(t,[(14.1,530),(14.8,505),(15.2,400),(15.65,175),(16.12,0),(17.2,0)])
            h=620*zoom
            state=sample_worker(run_t,rotation=(0,.60,.0),resolution=(64,74))
            solid(im,state,(x,y),(h*64/74,h),cell=max(9,round(19*zoom)),alpha=opacity)
            # Independent input tape is visibly consumed by each active core.
            baseline=y+h*.34
            for k in range(11):
                dx=((k*24-(t-birth)*130)%264)-170
                if t>=settle and k>3:continue
                batch(im,np.array([[x+dx*zoom,baseline+24*zoom]]),max(10,20*zoom),'01'[k%2],opacity*.75)
            text(im,'________',(x-95*zoom,baseline+22*zoom),max(10,23*zoom),round(opacity*240))
            if t>16.15:
                flash=float(smooth((t-16.15)/.08)*(1-smooth((t-16.48)/.12)))
                if flash>0:label(im,'X',(x,y-h*.055),max(10,round(35*zoom)),flash)
    timed_label(im,'WAITING FOR RANK 23_',(525,1010),t,15.18,16.10,24)
    timed_label(im,'ALL_REDUCE: READY',(525,1010),t,16.14,16.8,24)
    if t>16.18 and PHYSICS is None:
        # Output leaves the workers along curved tributaries, joining one
        # stream underneath them. They produce the next scene's stepping path.
        a=float(smooth((t-16.18)/.32))
        for branch in range(6):
            j=np.arange(75);s=((j/75+(t-16.18)*.80)%1)
            x0=300+branch*264
            xx=(1-s)**2*x0+2*(1-s)*s*960+s*s*1710
            yy=(1-s)**2*740+2*(1-s)*s*1020+s*s*800
            batch(im,np.column_stack((xx,yy)),18,np.array(list('01_+'))[j%4],a*.95)

def takeoff(im,t):
    from stair_choreography import staircase_state,sample_climber
    path=staircase_state(t)
    opacity=float(smooth((t-17.05)/.13)*(1-smooth((t-19.55)/.40)))
    f=face(32)
    for x,y,w in ((230.,970.,250.),)+path['platforms']:
        # The printed underscore top is the actual sole contact plane.
        txt='_'*math.ceil(w/f.getlength('_'));box=f.getbbox(txt)
        line=Image.new('L',(round(w),box[3]-box[1]))
        ImageDraw.Draw(line).text((-box[0],-box[1]),txt,font=f,fill=255)
        im.paste(round(255*opacity),(round(x),round(y)),line)
    if path['visible']:
        h=path['height']
        solid(im,sample_climber(t,resolution=(110,120)),path['center'],(h*110/120,h),cell=path['cell'])
        # The coarse ASCII sample can miss the tangent of an ellipsoid sole.
        # Give only a planted foot a compact flat tread at its actual contact
        # plane; airborne shoes remain entirely controlled by the 3D pose.
        if t>=17.18:
            d=ImageDraw.Draw(im)
            for (x,y),planted in zip(path['feet_screen'],path['planted']):
                if planted:
                    d.polygon([(x-11,y-6),(x+8,y-7),(x+15,y-3),(x+15,y-1),(x-11,y-1)],fill=255)
                    d.line((x-8,y-4,x+3,y-4),fill=0,width=1)
    timed_label(im,'NEXT >',(1580,750),t,18.12,19.22,30)


def ending(im,t):
    # Reference-style silence: a handful of remnants, small considered type,
    # and a large clean field. The last frame is not a title-slide composition.
    residual=1-float(smooth((t-19.1)/.65))
    if residual>0:physical_flow(im,t,residual,4)
    if t<19.95:takeoff(im,t)
    if 20.0<t<24.8:
        text(im,'System > Upgraded',(870,490),30)
        items=[('Compute is;',21.0),('Parallel',21.7),('Connected',22.25),('Open',22.8),('HPC / AI INFRA',23.35)]
        for n,(s,born) in enumerate(items):
            if t>born:
                count=min(len(s),int((t-born)*35))
                text(im,s[:count],(1050,555+n*36),29)
    elif 24.8<=t<27.65:
        s='The next breakthrough starts here.'
        text(im,s[:min(len(s),int((t-24.8)*40))],(960,540),30,anchor='mm')
    elif t>=27.65:
        s='SJTU Xflops';part=s[:min(len(s),int((t-27.65)*24))]
        if part:label(im,part,(960,528),47)
        if t>28.4:text(im,'HPC / AI INFRA',(960,595),25,anchor='mm')

def ink_frame(t):
    im=Image.new('L',(W,H))
    if t<3.03:old.terminal(im,t)
    elif t<4.62:
        if not physical_flow(im,t):old.blowout(im,t)
        opening_impact(im,t)
    elif t<6.173:
        if not physical_flow(im,t):
            if t<5.18:old.blowout(im,t)
            stream(im,t,'run',float(smooth((t-4.60)/.38)))
        hero_worker(im,t)
        timed_label(im,'<<HPC IS...',(1360,725),t,5.22,6.173,29)
    elif t<7.80:
        if not physical_flow(im,t):stream(im,t,'chip')
        chip(im,t)
        timed_label(im,'<<HPC IS...',(1360,725),t,5.22,6.45,29)
        timed_label(im,'>>FASTER>>',(960,525),t,7.78,8.38,29)
    elif t<8.5:
        if not physical_flow(im,t):stream(im,t,'sheet')
        timed_label(im,'>>FASTER>>',(960,525),t,7.78,8.38,29)
    elif t<9.67:
        knot=load_knot()
        if knot is not None and 'source_ids' in knot:
            physical_flow(im,t,exclude_ids=knot['source_ids'])
        elif t<8.75:
            a=float(1-smooth((t-8.44)/.31))
            if not physical_flow(im,t,a):stream(im,t,'sheet',a)
        if t>9.45:old.terminal(im,t,True)
        data_knot(im,t)
    elif t<11.0:
        command_fork(im,t)
        timed_label(im,'--AI INFRA--',(860,560),t,9.76,10.5,29)
    elif t<17.4:
        physical_flow(im,t,1.,4)
        workers(im,t)
        command_fork(im,t)
        if t>=17.05:takeoff(im,t)
    elif t<19.75:
        physical_flow(im,t,float(1-smooth((t-19.1)/.65)),4)
        takeoff(im,t)
    else:ending(im,t)
    return im

def frame(t):
    from film_material import finish
    return finish(ink_frame(t),t)

def render(start=0,end=30,output=None,width=1920):
    path=Path(output or ROOT/'work/director-silent.mp4');path.parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','24','-i','-',
         '-vf',f'scale={width}:-2:out_color_matrix=bt709:out_range=tv','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p',
         '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-movflags','+faststart',str(path)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE);st=time.time()
    try:
        for i in range(round((end-start)*24)):
            p.stdin.write(frame(start+i/24).tobytes())
            if i%24==0:print(f'{start+i/24:.2f}s / {time.time()-st:.1f}s elapsed',flush=True)
    finally:p.stdin.close()
    if p.wait():raise RuntimeError('Encoding failed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--start',type=float,default=0);a.add_argument('--end',type=float,default=30)
    a.add_argument('--output');a.add_argument('--width',type=int,default=1920);a.add_argument('--stills',nargs='*',type=float)
    ar=a.parse_args()
    if ar.stills is not None:
        dest=ROOT/'work/director-stills';dest.mkdir(parents=True,exist_ok=True)
        for t in ar.stills:frame(t).save(dest/f'{t:05.2f}.png');print(t,flush=True)
    else:render(ar.start,ar.end,ar.output,ar.width)
