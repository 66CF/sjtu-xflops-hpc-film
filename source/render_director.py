#!/usr/bin/env python3
"""A program learns to run together: character-led ASCII motion film."""
from pathlib import Path
from functools import lru_cache
import math,argparse,subprocess,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import render_kinetic as old
from glyph_material import visible_glyphs

ROOT=Path(__file__).resolve().parents[1]
W,H,FPS,DURATION=1920,1080,24,30
curve,smooth,mix=old.curve,old.smooth,old.mix
MONO='/System/Library/Fonts/Menlo.ttc'
PHYSICS=None

def physical_flow(im,t,opacity=1.,decimate=1,exclude_ids=None):
    global PHYSICS
    if PHYSICS is None:
        candidates=[ROOT/'work/physics-v5/flow.npz',ROOT/'work/physics-v5/physics-qa.npz']
        found=next((p for p in candidates if p.exists()),None)
        if found is None:return False
        z=np.load(found);PHYSICS={k:z[k] for k in z.files};z.close()
    data=PHYSICS;times=data['times']
    if t<times[0] or t>times[-1]+.005:return False
    nearest=int(np.argmin(abs(times-t)))
    if abs(float(times[nearest])-t)<2e-5:
        # Float32 cache timestamps have sub-microsecond rounding error. A
        # rendered 24fps frame must use its audited state exactly, including
        # integer glyph placement and four-degree rotation quantization.
        xy=data['xy'][nearest];angle=data['angle'][nearest]
    else:
        i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2));u=float(np.clip((t-times[i])/(times[i+1]-times[i]),0,1))
        xy=mix(data['xy'][i],data['xy'][i+1],u)
        angle=mix(data['angle'][i],data['angle'][i+1],u)
    # Cache samples interpolate an integrated simulation, never design targets.
    group=data['group'];ids=np.arange(len(group));size=data['size'].copy()
    birth=data.get('birth',np.where(group==3,data['release']-.025,0.))
    alive=(t>=birth)&visible_glyphs(group)
    aa=opacity*alive.astype(float)
    if exclude_ids is not None:aa[exclude_ids]=0
    if t>=19.10:
        from ambient_glyphs import exclusion_ids
        aa[exclusion_ids(t)]=0
    # The camera's attention shifts to the worker grid after the terminal.
    # Distance is expressed through contrast, never a change of font or size.
    rear=float(smooth((t-8.55)/1.65))
    aa*=np.where(group<3,1-.73*rear,1.)
    # A moving letter keeps its original size and Menlo Bold face. Release,
    # scene boundaries and obstacle contact never change its printed identity.
    # Once the lead worker takes the stairs, the released worker surfaces
    # recede visually. Their integrated collision/wake motion is unchanged.
    focus=float(smooth((t-17.22)/.42))
    aa*=np.where(group==3,1-.48*focus,1.)
    # One printed body per simulated glyph. A second exposure copy would
    # visually penetrate neighbours even when the physical bodies separate.
    batch(im,xy,size,data['glyph'],aa,angle,bold=True)
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
        im.paste(int(v[j]),(round(float(xy[j,0])-sp.width/2),round(float(xy[j,1])-sp.height/2)),sp)

def text(im,txt,xy,size=30,fill=255,anchor='la',bold=True):
    ImageDraw.Draw(im).text(xy,txt,font=face(size,bold),fill=int(fill),anchor=anchor)

old.batch=batch
old.font=lambda n:face(n,True)
# Terminal rows stay intact until the visible physical cursor strikes them.
old.pretear_alpha=lambda xy,t:np.ones(len(xy))
# Language supports the physical joke: a compute process literally starts running.

def label(im,txt,xy,size=29,alpha=1.):
    f=face(size);advance=size*.75;w=round(advance*len(txt))+16;h=round(size*1.19)
    sp=Image.new('L',(w,h),255);d=ImageDraw.Draw(sp)
    for i,ch in enumerate(txt):d.text((8+i*advance,-size*.055),ch,font=f,fill=0)
    mask=Image.new('L',sp.size,round(alpha*255))
    im.paste(sp,(round(xy[0]-w/2),round(xy[1]-h/2)),mask)

def timed_label(im,txt,xy,t,start,end,size=29):
    from label_scramble import draw_label
    draw_label(im,txt,xy,t,start,end,size=size,font=face,
               reveal=min(.458,(end-start)*.42))

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

def hero_worker(im,t):
    from hero_choreography import hero_state,sample_hero
    path=hero_state(t)
    solid(im,sample_hero(t),(path['x'],path['y']),
          (path['width'],path['height']),cell=path['cell'])

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

def workers(im,t):
    from ascii_solids import sample_worker
    from worker_choreography import worker_state
    from cluster_scene import draw_trays,draw_missing_rank,draw_late_rank
    draw_trays(im,t,batch)
    draw_missing_rank(im,t,text)
    draw_late_rank(im,t,solid)
    for idx in range(24):
        path=worker_state(idx,t)
        if not path['visible']:continue
        x,y,h=path['x'],path['y'],path['height']
        opacity=path['opacity'];zoom=path['zoom']
        state=sample_worker(path['pose'],rotation=path['rotation'],resolution=(64,74))
        solid(im,state,(x,y),(path['width'],h),cell=path['cell'],alpha=opacity)
        baseline=y+h*.34
        for k in range(11):
            dx=((k*24-(t-11.05-idx%6*.07)*130)%264)-170
            if t>=path['settle'] and k>3:continue
            batch(im,np.array([[x+dx*zoom,baseline+24*zoom]]),10,'01'[k%2],opacity*.75)
        text(im,'________',(x-95*zoom,baseline+22*zoom),11,round(opacity*240))
        if t>16.15:
            flash=float(smooth((t-16.15)/.08)*(1-smooth((t-16.48)/.12)))
            if flash>0:label(im,'X',(x,y-h*.055),16,flash)
    timed_label(im,'WAITING FOR RANK 23_',(960,535),t,14.30,16.10,42)
    timed_label(im,'ALL_REDUCE: READY',(960,535),t,16.14,17.20,42)

def takeoff(im,t):
    from stair_choreography import staircase_state,sample_climber
    from stair_platforms import draw_platforms
    path=staircase_state(t)
    draw_platforms(im,t,batch)
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
        timed_label(im,'SJTU Xflops',(960,528),t,27.65,31.,47)
        if t>28.4:text(im,'HPC / AI INFRA',(960,595),25,anchor='mm')

def ink_frame(t):
    im=Image.new('L',(W,H))
    from ambient_glyphs import draw_background
    from cluster_scene import draw_background as cluster_background,draw_cluster
    # The ending keeps its inherited physical remnants. The old knot's
    # arriving word sheet is replaced by the cluster's own material current.
    if t>=19.1:draw_background(im,t,batch=batch)
    if 7.30<=t<14.0:cluster_background(im,t,batch,text)
    if t<3.03:old.terminal(im,t)
    elif t<4.62:
        physical_flow(im,t)
        opening_impact(im,t)
    elif t<7.30:
        physical_flow(im,t)
        hero_worker(im,t)
        timed_label(im,'<<HPC IS...',(1360,725),t,5.22,6.45,29)
    elif t<14.0:
        physical_flow(im,t)
        draw_cluster(im,t,solid,batch,label,text)
        timed_label(im,'>>FASTER>>',(620,760),t,7.38,8.58,38)
        timed_label(im,'AI INFRA',(960,535),t,10.05,12.20,42)
    elif t<17.4:
        physical_flow(im,t)
        workers(im,t)
        if t>=17.05:takeoff(im,t)
    elif t<19.75:
        physical_flow(im,t,float(1-smooth((t-19.1)/.65)))
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
