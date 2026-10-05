"""One accelerator docks, powers a cluster and unfolds into the same workers.

The 7.30–14.00 sequence is one continuous projected stage. The camera reveals
existing workstations; neither processor identities nor the supporting trays
are replaced at the worker handoff. Geometry supplies the fluid colliders.
"""
import math
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw
from ascii_solids import (_render,_chip_primitives,_worker_primitives,
                          _rotation,_transform,sample_chip,sample_worker)
from hero_choreography import hero_state,sample_hero
from worker_choreography import worker_state

START=7.30
DOCK_TIME=8.56
NETWORK_TIME=10.20
DEPLOY_TIME=12.20
FOOT_CONTACT_TIME=13.15
RUN_TIME=13.35
HANDOFF=14.00
PRIMARY=6
MISSING_RANK=23
LATE_ENTRY=14.42
LATE_DOCK=15.32
LATE_UNFOLD=16.10
LATE_HANDOFF=16.12
NODE_HEIGHT=620*.46
CHIP_HEIGHT=NODE_HEIGHT*.66
PRIMARY_CENTER=np.array([293.,418.85-37.5])


def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*(3.-2*x)


def smoother(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(x*(x*6-15)+10)


def lerp(a,b,u):return np.asarray(a)+(np.asarray(b)-np.asarray(a))*u


def hermite(a,b,va,vb,t,start,end):
    u=float(np.clip((t-start)/(end-start),0,1));d=end-start
    return ((2*u**3-3*u*u+1)*np.asarray(a)+(u**3-2*u*u+u)*d*np.asarray(va)
            +(-2*u**3+3*u*u)*np.asarray(b)+(u**3-u*u)*d*np.asarray(vb))


def camera(t):
    # A restrained lateral track leads into a single slow pullback.
    pull=float(smoother((t-8.64)/1.56))
    zoom=2.30-1.30*pull
    slotx=float(hermite(2210.,1120.,-170.,0.,t,START,DOCK_TIME))
    sloty=532.
    focus=lerp((slotx,sloty),PRIMARY_CENTER,pull)
    return zoom,np.asarray(focus)-PRIMARY_CENTER*zoom


def project(points,t):
    zoom,shift=camera(t)
    return np.asarray(points)*zoom+shift


def node_center(index):
    row,col=divmod(index,6)
    return np.array([960+(col-2.5)*580*.46,535+(row-1.5)*505*.46])


def activation_time(index):
    row,col=divmod(index,6)
    return DOCK_TIME+.15+.11*col+.11*abs(row-1)


def _deploy(index,t):
    if index==MISSING_RANK:
        return float(smoother((t-LATE_DOCK)/(LATE_UNFOLD-LATE_DOCK)))
    delay=(index%6)*.017+(index//6)*.019
    return float(smoother((t-(DEPLOY_TIME+delay))/.96))


def _pose(index,t):
    if index==MISSING_RANK:
        return float(hermite(LATE_DOCK+.063*index,LATE_HANDOFF+.063*index,
                             0.,1.,t,LATE_DOCK,LATE_HANDOFF))
    # Smoothly match the original running phase and its velocity at 14 s.
    if t<=RUN_TIME:return RUN_TIME+.063*index
    return float(hermite(RUN_TIME+.063*index,HANDOFF+.063*index,
                         0.,1.,t,RUN_TIME,HANDOFF))


@lru_cache(maxsize=4)
def _idle_chip(resolution):
    return sample_chip(0,rotation=(0,.60,0),resolution=resolution)


def _node_sample(index,t,resolution=(64,74)):
    u=_deploy(index,t)
    if u<=0:return _idle_chip(tuple(resolution))
    if u>=1:return sample_worker(_pose(index,t),rotation=(0,.60,0),resolution=resolution)
    pose=_pose(index,t);full,pattern=_worker_primitives(pose)
    center,basis,scale=pattern
    q=.66+.34*u
    # Compensated aperture: the processor face does not inflate as legs unfold.
    body=center*u/q
    B=_rotation(np.array([-.15,.11*math.sin(pose*math.tau/.60+.2),
                          .035*math.sin(pose*math.tau/.60)])*u)
    chipscale=scale/q
    output=[];count=len(_chip_primitives())
    for j,(shape,c,h,R,mat,tag) in enumerate(full):
        local=_transform(c-center,basis.T)/scale
        local_R=basis.T@R
        radius=h/scale
        if j>=count:
            ordinal=j-count;side=-1 if ordinal<12 else 1;arm=ordinal%12>=6
            origin=np.array([side*(1.02 if arm else .49),.30 if arm else -.91,.075])
            extension=float(smoother((u-(.06 if arm else 0.))/.87))
            local=lerp(origin,local,extension)
            radius*=.025+.975*extension
            if extension<.001:continue
        output.append((shape,body+_transform(local*chipscale,B),radius*chipscale,
                       B@local_R,mat,tag))
    return _render(output,(0,.60,0),resolution,'worker',(body,B,chipscale))


def _primary_path(t):
    start=hero_state(START);dt=1e-4
    before,after=hero_state(START-dt),hero_state(START+dt)
    v=((after['x']-before['x'])/(2*dt),(after['y']-before['y'])/(2*dt))
    if t<7.86:
        center=hermite((start['x'],start['y']),(1320.,529.),v,(0.,0.),t,START,7.86)
    else:
        center=hermite((1320.,529.),(1120.,532.),(0.,0.),(0.,0.),t,7.86,DOCK_TIME)
    # A short compression overshoot reads as a latch touching a rigid socket.
    if 8.40<t<DOCK_TIME:
        center[0]+=8*math.sin((t-8.40)/(DOCK_TIME-8.40)*math.pi)**2
    h=float(hermite(start['height'],CHIP_HEIGHT*2.3,0.,0.,t,START,DOCK_TIME))
    rot0=np.asarray(start['rotation']).copy();rot0[0]-=math.tau
    rot=lerp(rot0,(0,.60,0),float(smoother((t-START)/(DOCK_TIME-START))))
    return center,h,tuple(rot)


def node_state(index,t,with_sample=True):
    zoom,_=camera(t);u=_deploy(index,t)
    c=node_center(index).copy();c[1]-=37.5*(1-u)
    if index==MISSING_RANK:
        c[0]+=(worker_state(index,t)['x']-node_center(index)[0])*u
    center=project(c,t)
    h=NODE_HEIGHT*(.66+.34*u)*zoom
    cell=round(11*zoom-2*u)
    res=(64,74)
    if index==PRIMARY and t<DOCK_TIME:
        center,h,rotation=_primary_path(t);res=(110,120)
        sample=sample_chip(t,rotation=rotation,resolution=res) if with_sample else None
        if abs(t-START)<1e-8:sample=sample_hero(START,resolution=res) if with_sample else None
        cell=25
    else:
        sample=_node_sample(index,t,resolution=res) if with_sample else None
    if index==MISSING_RANK and t<LATE_DOCK:
        center[0]=float(hermite(2200.,node_center(index)[0],-650.,0.,t,LATE_ENTRY,LATE_DOCK))
        flight=float(np.clip((t-LATE_ENTRY)/(LATE_DOCK-LATE_ENTRY),0,1))
        center[1]+=12*math.sin(math.pi*flight)**2
    # Rack LEDs power up in an ordered cascade, with the face itself retained.
    live=float(smooth((t-activation_time(index))/.16))
    contrast=1. if index==PRIMARY or t>=12.2 else .34+.66*live
    return dict(id=index,center=np.asarray(center),height=h,canvas=(h*res[0]/res[1],h),
                state=sample,cell=max(9,cell),contrast=contrast,deploy=u,
                visible=index!=MISSING_RANK or t>=LATE_ENTRY)


def _box_faces(x,y,w,h,depth=25.):
    # Projected solid front/top/right faces. These are opaque, never wireframe.
    dx,dy=depth,-depth*.62
    return [(np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)]),'front'),
            (np.array([(x,y),(x+dx,y+dy),(x+w+dx,y+dy),(x+w,y)]),'top'),
            (np.array([(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)]),'side')]


def tray_faces(index,t):
    c=node_center(index);x=c[0]-54.;y=c[1]+NODE_HEIGHT*.34+11.04
    if t>=14:
        path=worker_state(index,t);x+=path['x']-c[0]
    retire=float(smooth((t-(16.61+index*.009))/.24)) if index!=18 else float(smooth((t-17.0)/.18))
    y+=retire*165.
    return [(project(p,t) if t<14 else p,face) for p,face in _box_faces(x,y,110.,10.,16.)],1-retire


def structural_faces(t):
    retract=float(smoother((t-12.20)/1.42))
    faces=[]
    for col in range(6):
        x=node_center(col)[0]
        # Split rails visibly withdraw sideways; trays and processors stay put.
        for side in (-1,1):
            railx=x+side*108-9+side*retract*2100
            # Real ventilation gaps below each processor connect every bay
            # to the lateral current. A continuous projected wall would trap
            # finite letters in sealed pockets as the camera pulls back.
            for row in range(4):
                for poly,kind in _box_faces(railx,52+row*505*.46,18,150,40):
                    faces.append((project(poly,t),kind,f'rail-{col}-{side}-{row}'))
        # The rack crown and base retract vertically after the side guides.
        for sy in (52,1024):
            shift=(-1 if sy<100 else 1)*retract*1350
            for poly,kind in _box_faces(x-108,sy+shift,234,16,40):
                faces.append((project(poly,t),kind,f'cap-{col}-{sy}'))
    for index in range(24):
        center=node_center(index)+[0,-37.5]
        for side in (-1,1):
            x=center[0]+side*79-7+side*retract*2100
            for poly,kind in _box_faces(x,center[1]-23,14,46,19):
                faces.append((project(poly,t),kind,f'socket-{index}-{side}'))
    return faces


def _face_print(im,points,kind,t,batch,alpha=1.,size=14):
    if alpha<=.025:return
    points=np.asarray(points)
    if points[:,0].max()<0 or points[:,0].min()>1920 or points[:,1].max()<0 or points[:,1].min()>1080:return
    low=np.maximum(np.floor(points.min(axis=0)).astype(int)-2,[0,0])
    high=np.minimum(np.ceil(points.max(axis=0)).astype(int)+3,[1920,1080])
    if np.any(high<=low):return
    mask=Image.new('L',tuple(high-low));ImageDraw.Draw(mask).polygon(
        [tuple(p-low) for p in points],fill=round(255*alpha))
    im.paste(0,tuple(low),mask)
    # Sample the physical face's two parametric directions with persistent
    # glyphs. A denser cap and a quieter side give volume without gray panels.
    a,b,c,d=points
    nx=max(1,math.ceil(np.linalg.norm(b-a)/11));ny=max(1,math.ceil(np.linalg.norm(d-a)/14))
    nx=min(nx,500);ny=min(ny,180)
    uu,vv=np.meshgrid((np.arange(nx)+.5)/nx,(np.arange(ny)+.5)/ny)
    p=a+(b-a)*uu.ravel()[:,None]+(d-a)*vv.ravel()[:,None]
    ids=np.arange(len(p));chars=np.array(list('=+##'))[ids%4] if kind=='front' else np.full(len(p),'/' if kind=='top' else ':')
    light={'front':.72,'top':1.,'side':.48}[kind]
    batch(im,p,size,chars,alpha*light,bold=True)


def draw_trays(im,t,batch):
    if t<7.30 or t>=17.2:return
    for index in range(24):
        faces,alpha=tray_faces(index,t)
        for points,kind in faces:_face_print(im,points,kind,t,batch,alpha,size=13)


def _network_paths(t):
    paths=[]
    for row in range(4):
        for col in range(5):
            idx=row*6+col
            a=node_center(idx)+[0,-37.5];b=node_center(idx+1)+[0,-37.5]
            mid=(a+b)/2
            # Bent cable channels sit outside the chip's silhouette.
            path=np.array([a,a+[70,0],mid+[0,27],b+[-70,0],b])
            paths.append((idx,idx+1,project(path,t)))
    for col in range(6):
        for row in range(3):
            idx=row*6+col;a=node_center(idx)+[0,-37.5];b=node_center(idx+6)+[0,-37.5]
            path=np.array([a,a+[90,0],a+[106,60],b+[106,-60],b+[90,0],b])
            paths.append((idx,idx+6,project(path,t)))
    return paths


def _polyline(path,dist):
    delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1);ends=np.r_[0,np.cumsum(length)]
    idx=np.clip(np.searchsorted(ends,dist,side='right')-1,0,len(length)-1)
    u=(dist-ends[idx])/np.maximum(length[idx],1e-9)
    return path[idx]+delta[idx]*u[:,None]


def draw_network(im,t,batch):
    if not 8.56<t<13.45:return
    release=float(smoother((t-12.18)/1.12))
    for j,(a,b,path) in enumerate(_network_paths(t)):
        # The open socket has no active link or arriving packets. Its absence
        # is the cause of the later collective barrier, visible from this shot.
        if a==MISSING_RANK or b==MISSING_RANK:continue
        delay=activation_time(b)
        if t<delay:continue
        length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
        n=max(2,math.ceil(length/17));dist=np.linspace(0,length,n)
        xy=_polyline(path,dist)
        # Stable cable characters are carried downwards as guides retract.
        xy[:,1]+=release*(1200+60*(j%5))
        batch(im,xy,15,np.array(list('..--'))[np.arange(n)%4],.28,bold=True)
        # Packets start/end under opaque dies; recirculation never teleports a
        # visible mark through open space. The payload flows in compact groups.
        for pack in range(2):
            phase=((t-delay)*.77+pack*.48)%1
            head=phase*length
            trail=head-np.arange(6)*19
            visible=(trail>=0)&(trail<=length)
            if not visible.any():continue
            p=_polyline(path,trail[visible]);p[:,1]+=release*(1200+60*(j%5))
            glyph=np.array(list('10>01>'))[visible]
            batch(im,p,17,glyph,.95,bold=True)


@lru_cache(maxsize=1)
def _background_material():
    p=[];g=[];lanes=[]
    for col in range(-2,13):
        phrase=['COMPUTE','PARALLEL','AI_INFRA'][col%3]
        for row in range(-12,60):
            for j,ch in enumerate(phrase):
                p.append((col*180+j*12, row*27));g.append(ch);lanes.append(col)
    return np.asarray(p,dtype=float),np.asarray(g),np.asarray(lanes)


def draw_background(im,t,batch,text):
    if not START<=t<14:return
    p,g,lane=_background_material();p=p.copy()
    # A continuing word current passes the stage. It is not a fresh full-frame
    # texture at each scene boundary. At 7.3 all new columns are beyond right.
    arrival=float(smoother((t-START)/1.60))
    p[:,0]+=2600*(1-arrival)-75*(t-9.)+18*np.sin(p[:,1]*.005+t)
    p[:,1]+=12*np.sin(p[:,0]*.003-t*.8)
    depart=float(smoother((t-12.35)/1.55))
    p[:,1]-=depart*2150
    batch(im,p,18,g,np.where(lane%3==0,.31,.23),bold=True)


def draw_cluster(im,t,solid,batch,label,text):
    if not START<=t<HANDOFF:return
    for points,kind,key in structural_faces(t):_face_print(im,points,kind,t,batch,size=16)
    draw_network(im,t,batch)
    draw_trays(im,t,batch)
    states=[node_state(index,t) for index in range(24)]
    # Lead is rendered last so docking is legible against its moving guides.
    states.sort(key=lambda s:s['id']==PRIMARY)
    for s in states:
        if not s['visible']:continue
        if s['center'][0]+s['canvas'][0]/2<0 or s['center'][0]-s['canvas'][0]/2>1920:continue
        solid(im,s['state'],s['center'],s['canvas'],cell=s['cell'],alpha=s['contrast'])
    # The worker input tape starts while the same trays are still in this
    # shot, so the 14 s handoff adds no new row of marks all at once.
    if t>13.35:
        alpha=float(smooth((t-13.35)/.20))
        for idx in range(24):
            if idx==MISSING_RANK:continue
            c=node_center(idx);baseline=c[1]+NODE_HEIGHT*.34
            for k in range(11):
                dx=((k*24-(t-11.05-idx%6*.07)*130)%264)-170
                batch(im,np.array([[c[0]+dx*.46,baseline+11.04]]),10,'01'[k%2],alpha*.75)
            text(im,'________',(c[0]-95*.46,baseline+22*.46),11,round(alpha*240))
    draw_missing_rank(im,t,text)
    if 8.56<t<9.18:
        center=project(PRIMARY_CENTER,t)
        # A small mechanical collar closes after the processor reaches it.
        zoom,_=camera(t);d=ImageDraw.Draw(im)
        flash=float(smooth((t-8.56)/.10)*(1-smooth((t-9.04)/.14)))
        for side in (-1,1):
            x=center[0]+side*83*zoom
            text(im,'[=' if side<0 else '=]',(x,center[1]+78*zoom),24,fill=255*flash,anchor='mm')


def draw_missing_rank(im,t,text):
    """The same unoccupied socket remains identifiable through rack removal."""
    if not 9.9<=t<LATE_DOCK:return
    c=node_center(MISSING_RANK)+[0,-37.5]
    c=project(c,t)
    alpha=float(smooth((t-9.9)/.35))
    value=round(230*alpha)
    # The marker belongs to the retained socket, in front of withdrawing rails.
    # A close blue knockout prevents white structural ink erasing its meaning.
    d=ImageDraw.Draw(im)
    for dx,y0,y1 in [(66,-17,16),(39,22,44)]:
        d.rectangle((round(c[0]-dx),round(c[1]+y0),round(c[0]+dx),round(c[1]+y1)),fill=0)
    text(im,'[  23  ]',tuple(c),25,fill=value,anchor='mm')
    text(im,'OFFLINE',tuple(c+[0,33]),15,fill=round(value*.72),anchor='mm')


def draw_late_rank(im,t,solid):
    if not LATE_ENTRY<=t<LATE_HANDOFF:return
    s=node_state(MISSING_RANK,t)
    solid(im,s['state'],s['center'],s['canvas'],cell=s['cell'])


def cluster_state(t):
    return dict(time=t,camera=camera(t),nodes=[node_state(i,t,False) for i in range(24)])


def colliders(t):
    solids=[];polygons=[]
    if not START<=t<17.2:return dict(solids=solids,polygons=polygons)
    dt=1/240
    if t<HANDOFF:
        for index in range(24):
            state=node_state(index,t);a=node_state(index,t-dt,False);b=node_state(index,t+dt,False)
            if not state['visible']:continue
            v=(b['center']-a['center'])/(2*dt)
            rate=(b['height']-a['height'])/(2*dt*state['height'])
            solids.append(dict(id=index,state=state['state'],center=state['center'],canvas=state['canvas'],velocity=v,scale_rate=rate))
        before={k+f:p for p,f,k in structural_faces(t-dt)};after={k+f:p for p,f,k in structural_faces(t+dt)}
        for points,kind,key in structural_faces(t):
            # Camera and rail velocities sampled from the identical geometry.
            v=(after[key+kind].mean(axis=0)-before[key+kind].mean(axis=0))/(2*dt)
            polygons.append(dict(id=key+kind,points=points,velocity=v))
    if LATE_ENTRY<=t<LATE_HANDOFF:
        state=node_state(MISSING_RANK,t)
        a=node_state(MISSING_RANK,t-dt,False);b=node_state(MISSING_RANK,t+dt,False)
        solids.append(dict(id=MISSING_RANK,state=state['state'],center=state['center'],canvas=state['canvas'],
                           velocity=(b['center']-a['center'])/(2*dt),
                           scale_rate=(b['height']-a['height'])/(2*dt*state['height'])))
    for index in range(24):
        faces,alpha=tray_faces(index,t)
        if alpha<.5:continue
        pa,_=tray_faces(index,t-dt);pb,_=tray_faces(index,t+dt)
        for j,(points,kind) in enumerate(faces):
            v=(pb[j][0].mean(axis=0)-pa[j][0].mean(axis=0))/(2*dt)
            polygons.append(dict(id=f'tray-{index}-{kind}',points=points,velocity=v))
    return dict(solids=solids,polygons=polygons)
