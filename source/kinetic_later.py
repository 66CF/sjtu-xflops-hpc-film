#!/usr/bin/env python3
"""Later movement of the Xflops film: rooms, orbits, commands, signature.

The opening's fluid character sheet remains the physical connective material.
Live camera geometry and editorial type have independent renderers and timing.
``draw_later`` receives a 1920x1080 L image, where 0 is cobalt and 255 is white.
"""
from functools import lru_cache
import math
import numpy as np
from PIL import Image, ImageDraw
import kinetic_geometry as kg

W,H=1920,1080
_C=np.array(list("01+*=/<>:#"))


def _ease(x):
    x=np.clip(x,0,1)
    return x*x*(3-2*x)


def _rot(x,y,z):
    cx,sx,cy,sy,cz,sz=math.cos(x),math.sin(x),math.cos(y),math.sin(y),math.cos(z),math.sin(z)
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                     [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],
                     [-sy,cy*sx,cy*cx]])


def _mm(p,R):
    return np.einsum('ij,kj->ik',p,R)


def _highlight(im,r,text,center,t,start,end,size=40):
    """Stationary, independently typeset reversed label, drawn by a fast wipe."""
    a=float(_ease((t-start)/.17)*(1-_ease((t-end)/.16)))
    if a<=0:return
    f=r.font(size); box=f.getbbox(text)
    width=math.ceil(f.getlength(text))+28;height=box[3]-box[1]+18
    layer=Image.new('L',(width,height),255)
    ImageDraw.Draw(layer).text((14,9-box[1]),text,font=f,fill=0)
    reveal=float(_ease((t-start)/.21))
    left=round(center[0]-width/2);top=round(center[1]-height/2)
    mask=Image.new('L',layer.size,0)
    ImageDraw.Draw(mask).rectangle((0,0,round(width*reveal),height),fill=round(a*255))
    im.paste(layer,(left,top),mask)


@lru_cache(maxsize=1)
def _rack_template():
    """A cabinet has eight distinct node sleds, side vents and a top plate."""
    p=[];n=[];alpha=[];glyph=[]
    # Sparse front material, designed in physical space rather than a flat icon.
    xx,yy=np.meshgrid(np.arange(-.82,.821,.092),np.arange(-1.95,1.951,.120))
    xx=xx.ravel();yy=yy.ravel()
    slot=((yy+1.95)/.47)%1
    edges=(abs(xx)>.735)|(abs(yy)>1.88)
    rail=(slot<.18)|(slot>.88)
    led=(xx<-.47)&(slot>.35)&(slot<.57)
    vent=(xx>.05)&(np.mod(xx+.015,.24)<.082)&(slot>.30)&(slot<.70)
    keep=edges|rail|led|vent
    pos=np.column_stack([xx[keep],yy[keep],np.full(sum(keep),-.64)])
    p.append(pos);n.append(np.tile((0,0,-1),(len(pos),1)))
    alpha.append(np.where(edges[keep],1.,np.where(rail[keep],.98,np.where(led[keep],1.,.68))))
    glyph.append(np.where(led[keep],0,np.where(rail[keep],4,7)))
    for sign in (-1,1):
        yy,zz=np.meshgrid(np.arange(-1.95,1.951,.126),np.arange(-.64,.641,.108))
        yy=yy.ravel();zz=zz.ravel()
        edge=(abs(yy)>1.84)|(abs(zz)>.55)
        vents=(np.mod(yy+1.95,.39)<.09)&(abs(zz)<.45)
        keep=edge|vents
        pos=np.column_stack([np.full(sum(keep),sign*.82),yy[keep],zz[keep]])
        p.append(pos);n.append(np.tile((sign,0,0),(len(pos),1)))
        alpha.append(np.where(edge[keep],.95,.67));glyph.append(np.full(len(pos),5))
    xx,zz=np.meshgrid(np.arange(-.82,.821,.103),np.arange(-.64,.641,.112))
    xx=xx.ravel();zz=zz.ravel();keep=(abs(xx)>.71)|(abs(zz)>.54)|((np.mod(xx,.29)<.045)&(abs(zz)<.35))
    pos=np.column_stack([xx[keep],np.full(sum(keep),1.95),zz[keep]])
    p.append(pos);n.append(np.tile((0,1,0),(len(pos),1)))
    alpha.append(np.full(len(pos),.89));glyph.append(np.full(len(pos),2))
    return np.concatenate(p),np.concatenate(n),np.concatenate(alpha),np.concatenate(glyph)


def _racks(im,r,t,opacity):
    local=t-12
    # The room approaches, briefly settles, then a dolly accelerates through it.
    camz=r.curve(t,[(12,-17),(12.65,-12.2),(13.25,-10),(14.2,-8.7),(14.8,-5.5),(15.45,-.2),(16.0,5.7)])
    yaw=r.curve(t,[(12,-.22),(12.8,-.09),(13.65,.08),(14.5,.055),(15.8,-.22)])
    roll=r.curve(t,[(12,.065),(13.2,-.025),(14.4,-.01),(15.8,.12)])
    R=_rot(-.06,yaw,roll)
    base,normal,a,g=_rack_template()
    output=[]
    # Two rows of physical cabinets. Every depth level moves with a different
    # perspective speed as the camera enters the central aisle.
    for bank in (-1,1):
        for level in range(3):
            center=np.array([bank*(2.40+.13*level),0.,level*3.30])
            p=base.copy()
            # Packets travel across the actual front rails, staying on the slab.
            front=normal[:,2]<-.5
            drift=.018*np.sin(local*2.6+p[:,1]*4+level)
            p[front,0]+=drift[front]
            p+=center
            q=_mm(p,R);nn=_mm(normal,R)
            q[:,2]-=camz
            facing=np.einsum('ij,ij->i',q,nn)<0
            keep=facing&(q[:,2]>.7)
            q=q[keep];nn=nn[keep];aa=a[keep];gg=g[keep]
            if len(q)==0:continue
            factor=1060/q[:,2]
            xy=np.column_stack([960+q[:,0]*factor,545-q[:,1]*factor])
            size=np.clip(factor*.146,10.5,49)
            light=.86+.14*np.maximum(0,-nn[:,2])
            pulse=.88+.12*np.sin(q[:,1]*5-local*5+level)
            corners=np.array([[x,y,z] for x in (-.82,.82) for y in (-1.95,1.95) for z in (-.64,.64)])+center
            corners=_mm(corners,R);corners[:,2]-=camz
            corners=corners[corners[:,2]>.65]
            cf=1060/corners[:,2]
            hull=kg._convex_hull(np.column_stack([960+corners[:,0]*cf,545-corners[:,1]*cf]))
            output.append((xy,size,_C[gg],aa*opacity*light*pulse,q[:,2],hull))
    if output:
        # These are opaque enclosures. Painter order and per-cabinet clearing
        # prevent rear cabinet fronts or the sea from showing through a door.
        for xy,sz,ch,aa,zz,hull in sorted(output,key=lambda s:np.mean(s[4]),reverse=True):
            if len(hull)>2:
                mask=Image.new('L',im.size,0)
                ImageDraw.Draw(mask).polygon([tuple(v) for v in hull],fill=round(opacity*255))
                im.paste(0,(0,0),mask)
            r.batch(im,xy,sz,ch,aa,depth=zz)
    # A floor stream enters from depth and slides under the racks, making the
    # physical flight visible even while the prominent caption is stationary.
    j=np.arange(560);seed=(j*.61803398875)%1
    z=1+(j%56)*.46
    zz=1+((z-local*r.curve(t,[(12,.7),(13.5,1.3),(14.7,3.1),(16,5.2)]))%26)
    x=(seed-.5)*4.15
    factor=1000/zz
    xy=np.column_stack([960+x*factor,560+2.22*factor])
    r.batch(im,xy,np.clip(factor*.079,8,27),_C[j%10],opacity*.48,depth=zz)


def _orbits(im,r,t,opacity):
    # Angular speed starts as an impulse, brakes for recognition, then winds up
    # for the next command. The axis itself precesses rather than merely spins.
    phase=r.curve(t,[(15.0,-1.3),(15.55,.05),(16.0,1.17),(16.75,1.62),(17.5,2.01),(18.3,3.10),(19.6,5.72)])
    cy=r.curve(t,[(15,780),(15.65,530),(16.4,540),(18.5,545),(19.65,85)])
    cx=r.curve(t,[(15,620),(15.8,760),(17.7,780),(18.6,930),(19.65,1100)])
    scale=r.curve(t,[(15,58),(15.7,205),(16.25,228),(17.35,211),(18.3,252),(19.65,505)])
    turn=(.42+.20*math.sin(phase*.73),phase*.72,-.20+.19*math.sin(phase))
    core=kg.tensor_cube(t,center=(cx,cy),scale=scale,rotation=turn)
    R=_rot(.33,phase*.12,-.18)
    arrays=[]
    for ring in range(3):
        count=190;j=np.arange(count);u=(j/count*math.tau+phase*(.84+ring*.29))
        radius=1.52+.12*np.sin(u*3+phase+ring)
        p=np.column_stack([radius*np.cos(u),radius*np.sin(u),.10*np.sin(u*4-phase)])
        if ring==1:p=p[:,[0,2,1]]
        if ring==2:p=p[:,[2,0,1]]
        p=_mm(p,R)
        z=p[:,2]+7
        f=scale*7/z
        xy=np.column_stack([cx+p[:,0]*f,cy-p[:,1]*f])
        glyph=_C[(j*3+ring)%10]
        size=np.clip(f*.075,10,31)
        a=opacity*(.64+.31*(.5+.5*np.sin(u*2-phase)))
        arrays.append((xy,size,glyph,a,z))
    xy=np.concatenate([x[0] for x in arrays]);size=np.concatenate([x[1] for x in arrays]);glyph=np.concatenate([x[2] for x in arrays]);alpha=np.concatenate([x[3] for x in arrays]);z=np.concatenate([x[4] for x in arrays])
    behind=z>7
    r.batch(im,xy[behind],size[behind],glyph[behind],alpha[behind],depth=z[behind])
    # Dissolve opaque clearing gradually together with the approaching object.
    if opacity>.15:
        mask=Image.new('L',im.size,0)
        ImageDraw.Draw(mask).polygon([tuple(v) for v in core['hull']],fill=round(opacity*255))
        im.paste(0,(0,0),mask)
    r.batch(im,core['xy'],core['size'],core['glyph'],core['alpha']*opacity,core['angle'],core['depth'])
    r.batch(im,xy[~behind],size[~behind],glyph[~behind],alpha[~behind],depth=z[~behind])


_COMMANDS=[
    ('$ xflops run --distributed',0.00),
    ('> Linking minds. Scaling possibility.',.48),
    ('',.80),
    ('[1/4] PROFILE',.90),
    ('      Find the bottleneck.',1.05),
    ('      Trace kernels. Inspect memory. Measure communication.',1.19),
    ('[2/4] OPTIMIZE',1.48),
    ('      Fuse. Tile. Overlap. Repeat.',1.60),
    ('      More useful work. Less waiting.',1.74),
    ('[3/4] SCALE',2.04),
    ('      One node -> many nodes -> one shared challenge.',2.18),
    ('      ALL_REDUCE: synchronized',2.28),
    ('      PIPELINE:   running',2.39),
    ('[4/4] DISCOVER',2.64),
    ('      Train the model. Simulate the future.',2.80),
    ('',2.95),
    ('> COMPUTE BEYOND LIMITS_',3.07),
]


def _commands(im,r,t):
    local=t-19.15
    if local<0:return
    # A vertical entrance is finished before keyboard reading begins. New
    # lines then arrive in discrete terminal cadence, not constantly drifting.
    origin=r.curve(t,[(19.15,1210),(19.50,130),(19.69,76),(22.05,76),(22.35,28),(22.8,-100),(23.35,-235)])
    wind=float(_ease((t-23.15)/1.05))
    xy=[];size=[];glyph=[];alpha=[];angle=[]
    for row,(line,born) in enumerate(_COMMANDS):
        age=local-born
        if age<0:continue
        n=min(len(line),max(0,int(age*145)))
        line=line[:n]
        y=origin+row*45
        if t<23.15:
            if -40<y<1120:r.plain(im,line,(65,y),29,255)
        else:
            # Sentence material releases into a curling sheet after reading.
            # Different lanes release at different times; no target positions.
            for col,ch in enumerate(line):
                if ch==' ':continue
                age=max(0,t-23.15-row*.014-col*.0013)
                vel=age*age
                x=65+(col+.5)*17.46
                phase=x*.006+row*.32
                dx=vel*(330+155*math.sin(phase))
                dy=vel*(-240+170*math.cos(phase+.9*age))
                # Match the baseline of plain(..., anchor='la') at release.
                xy.append((x+dx,y+19+dy));size.append(29+5*age)
                glyph.append(ch);alpha.append(max(0,1-age*.58))
                angle.append(age*42*math.sin(phase))
    if xy:r.batch(im,xy,size,glyph,alpha,angle)
    if 19.69<t<22.3 and int(t*3)%2==0:
        # The cursor belongs to the active line, which is set by birth time.
        eligible=[(i,line,born) for i,(line,born) in enumerate(_COMMANDS) if local>=born]
        if eligible:
            row,line,born=eligible[-1];n=min(len(line),int((local-born)*145))
            r.plain(im,'_', (65+n*17.46,origin+row*45),29)


def draw_later(im,t):
    """Draw the complete 12–30 second continuation into an L image."""
    import render_kinetic as r
    ocean_op=r.curve(t,[(12,1),(12.6,.66),(13.5,.41),(14.6,.62),(15.6,.74),
                        (17.3,.46),(18.4,.66),(19.5,.23),(20.2,.055),
                        (22.6,.075),(23.3,.46),(24.5,.65),(25.8,.33),
                        (27.0,.38),(28.1,.11),(30,.075)])
    r.ocean(im,t,opacity=ocean_op,mode='normal')
    if 12<=t<16.1:
        opacity=float(_ease((t-12)/.48)*(1-_ease((t-15.1)/.82)))
        _racks(im,r,t,opacity)
        _highlight(im,r,'HPC.',(960,430),t,13.12,14.65,51)
    if 15<=t<19.8:
        opacity=float(_ease((t-15)/.66)*(1-_ease((t-18.95)/.65)))
        _orbits(im,r,t,opacity)
        _highlight(im,r,'AI INFRA.',(1470,503),t,16.27,18.48,42)
        if 16.7<t<18.35:
            value=190*_ease((t-16.7)/.25)*(1-_ease((t-18.15)/.2))
            r.plain(im,'TRAIN. CONNECT. SCALE.',(1230,560),21,value)
    if 19.15<=t<24.55:
        _commands(im,r,t)
    if 24.03<t<27.25:
        _highlight(im,r,'COMPUTE',(960,435),t,24.03,26.83,83)
        _highlight(im,r,'BEYOND LIMITS',(960,574),t,24.55,26.86,75)
        # A tiny command prefix ties the independent hero typography back to
        # its originating terminal without making all letters pooled particles.
        a=_ease((t-24.6)/.3)*(1-_ease((t-26.67)/.23))
        r.plain(im,'>>',(515,430),34,210*a)
        r.plain(im,'_', (1400,602),34,210*a)
    if t>=27:
        # Deliberately sharp punctuation event, followed by an undisturbed hold.
        _highlight(im,r,'SJTU Xflops',(960,468),t,27.00,31,88)
        _highlight(im,r,'HPC TEAM',(960,595),t,27.34,31,38)
        a=float(_ease((t-27.6)/.27))
        r.plain(im,'SHANGHAI JIAO TONG UNIVERSITY',(960,682),23,210*a,anchor='mm')
        r.plain(im,'> READY FOR THE NEXT CHALLENGE',(960,803),23,190*a,anchor='mm')
        if int((t-27)*2.5)%2==0:r.plain(im,'_', (1192,803),23,190*a,anchor='mm')
