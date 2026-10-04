"""Bake actual opaque 3D silhouettes and solid velocities for the GPU fluid."""
from pathlib import Path
import numpy as np
from PIL import Image
from render_director import curve,smooth
from ascii_solids import sample_worker,sample_chip
from chip_choreography import chip_state,CHIP_START,CHIP_END
from opening_impact import impact_mask
from stair_choreography import staircase_state,sample_climber

ROOT=Path(__file__).resolve().parents[1]

def bodies(t):
    result=[]
    if 4.65<=t<6.173:
        x=curve(t,[(4.65,2210),(4.76,1620),(4.93,1320),(5.10,1080),(5.28,930),(5.65,890),(6.173,845)])
        h=curve(t,[(4.65,640),(4.95,890),(5.2,1050),(5.7,1030),(6.173,1070)])
        result.append((0,x,545,h*110/120,h,t,(0,.62+.14*np.sin(t*3),.015*np.sin(t*8)),'worker'))
    elif CHIP_START<=t<CHIP_END:
        path=chip_state(t);h=path['height']
        result.append((1,path['x'],path['y'],h,h,t,path['rotation'],'chip'))
    elif 11<=t<17.05:
        zoom=curve(t,[(11,1),(11.65,1),(12.3,.70),(13.1,.51),(14,.46),(15.8,.46),(16.7,.46),(17.4,.46)])
        opening=float(smooth((t-11.15)/.7));reveal=float(smooth((t-12.25)/.85))
        for row in range(4):
            for col in range(6):
                idx=row*6+col
                if idx!=18 and t>16.61+idx*.009+.12:continue
                if row>0 and reveal<=0:continue
                birth=11.05+col*.07 if row==0 else 12.25+(row-1)*.13+col*.025
                if smooth((t-birth)/.28)<.5:continue
                x=960+(col-2.5)*580*zoom*opening
                y=535+(row-1.5)*505*zoom*(.5+.5*reveal)
                settle=16.12 if idx==23 else 14.65+(idx%6)*.10+(idx//6)*.055
                run_t=t+.063*idx
                if t>settle:run_t=settle+.063*idx+np.sin((t-settle)*9)*.026
                step=curve(t,[(birth,-110),(birth+.4,0),(14.6,0),(settle,32),(17.2,32)])
                x+=step*zoom
                if idx==23 and t>14.1:x+=curve(t,[(14.1,530),(14.8,505),(15.2,400),(15.65,175),(16.12,0),(17.2,0)])
                h=620*zoom
                result.append((100+idx,x,y,h*64/74,h,run_t,(0,.6,0),'worker'))
    elif 17.05<=t<19.75:
        path=staircase_state(t);x,y=path['center'];h=path['height']
        result.append((200,x,y,h*110/120,h,t,(0,0,0),'climber'))
    return result

def main():
    times=np.arange(3,19.75+1e-6,1/24)
    masks=[];velocities=[]
    xx,yy=np.meshgrid((np.arange(512)+.5)*3.75,(np.arange(288)+.5)*3.75)
    for fi,t in enumerate(times):
        layer,velocity=impact_mask(t,(512,288))
        if 17.18<=t<19.75:
            for x,y,w in ((230.,970.,250.),)+staircase_state(t)['platforms']:
                active=(xx>=x)&(xx<=x+w)&(yy>=y)&(yy<y+5)
                layer|=active
        before={b[0]:b for b in bodies(t-1/240)};after={b[0]:b for b in bodies(t+1/240)}
        for key,x,y,w,h,pose,rotation,kind in bodies(t):
            fn=sample_worker if kind=='worker' else sample_chip
            resolution=(48,max(30,round(48*h/w)))
            state=sample_climber(pose,resolution=resolution) if kind=='climber' else fn(pose,rotation=rotation,resolution=resolution)
            shape=Image.fromarray(np.uint8(state['mask'])*255)
            shape=shape.resize((max(1,round(w/3.75)),max(1,round(h/3.75))),Image.Resampling.BILINEAR)
            canvas=Image.new('L',(512,288));canvas.paste(shape,(round((x-w/2)/3.75),round((y-h/2)/3.75)))
            active=np.asarray(canvas)>120
            vx=vy=rate=0.
            if key in before and key in after:
                b,a=before[key],after[key]
                vx=(a[1]-b[1])*120;vy=(a[2]-b[2])*120;rate=(a[4]-b[4])*120/h
            velocity[active,0]=np.clip(vx+(xx[active]-x)*rate,-2600,2600)
            velocity[active,1]=np.clip(vy+(yy[active]-y)*rate,-2600,2600)
            layer|=active
        masks.append(layer);velocities.append(velocity.astype(np.float16))
        if fi%24==0:print(f'{t:.2f}s / {int(layer.sum())} obstacle cells',flush=True)
    path=ROOT/'work/physics-v5/solid-masks.npz'
    np.savez_compressed(path,times=times.astype('f4'),solid_mask=np.asarray(masks),solid_velocity=np.asarray(velocities))
    print(path,flush=True)

if __name__=='__main__':main()
