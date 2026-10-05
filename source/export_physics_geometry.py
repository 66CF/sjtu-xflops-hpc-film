"""Bake actual opaque 3D silhouettes and solid velocities for the GPU fluid."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw
from render_director import curve,smooth
from ascii_solids import sample_worker,sample_chip
from chip_choreography import CHIP_END
from hero_choreography import hero_state,sample_hero,HERO_START
from stair_platforms import platform_mask
from opening_impact import impact_mask
from worker_choreography import worker_state
from cluster_scene import colliders
from stair_choreography import staircase_state,sample_climber

ROOT=Path(__file__).resolve().parents[1]

def bodies(t):
    result=[]
    if HERO_START<=t<7.30:
        path=hero_state(t);h=path['height']
        result.append((0,path['x'],path['y'],path['width'],h,t,path['rotation'],'hero'))
    elif 14.0<=t<17.05:
        for idx in range(24):
            path=worker_state(idx,t)
            if path['opacity']<.5:continue
            # A dissolving surface transfers ownership to its printed bodies.
            # Its former solid must not squeeze the new glyphs back out of it.
            if idx!=18 and t>=16.665+idx*.009-1e-5:continue
            result.append((100+idx,path['x'],path['y'],path['width'],path['height'],
                           path['pose'],path['rotation'],'worker'))
    elif 17.05<=t<19.75:
        path=staircase_state(t);x,y=path['center'];h=path['height']
        result.append((200,x,y,h*110/120,h,t,(0,0,0),'climber'))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fps',type=int,default=24)
    parser.add_argument('--start',type=float,default=3.)
    parser.add_argument('--end',type=float,default=19.75)
    parser.add_argument('--output',default=str(ROOT/'work/physics-v5/solid-masks.npz'))
    args=parser.parse_args()
    times=np.arange(args.start,args.end+1e-6,1/args.fps)
    masks=[];velocities=[]
    xx,yy=np.meshgrid((np.arange(512)+.5)*3.75,(np.arange(288)+.5)*3.75)
    for fi,t in enumerate(times):
        layer,velocity=impact_mask(t,(512,288))
        if 17.05<=t<19.75:
            layer|=platform_mask(t,xx,yy)
        if 7.30<=t<17.05:
            cluster=colliders(float(t))
            for item in cluster['polygons']:
                canvas=Image.new('L',(512,288))
                points=[(float(x)/3.75,float(y)/3.75) for x,y in item['points']]
                ImageDraw.Draw(canvas).polygon(points,fill=255)
                active=np.asarray(canvas)>120
                velocity[active]=np.clip(item['velocity'],-2600,2600)
                layer|=active
            for item in cluster['solids']:
                state=item['state'];x,y=item['center'];w,h=item['canvas']
                shape=Image.fromarray(np.uint8(state['mask'])*255)
                shape=shape.resize((max(1,round(w/3.75)),max(1,round(h/3.75))),Image.Resampling.BILINEAR)
                canvas=Image.new('L',(512,288))
                canvas.paste(shape,(round((x-w/2)/3.75),round((y-h/2)/3.75)))
                active=np.asarray(canvas)>120
                vx,vy=item['velocity'];rate=item.get('scale_rate',0.)
                velocity[active,0]=np.clip(vx+(xx[active]-x)*rate,-2600,2600)
                velocity[active,1]=np.clip(vy+(yy[active]-y)*rate,-2600,2600)
                layer|=active
        before={b[0]:b for b in bodies(t-1/240)};after={b[0]:b for b in bodies(t+1/240)}
        for key,x,y,w,h,pose,rotation,kind in bodies(t):
            fn=sample_worker if kind=='worker' else sample_chip
            resolution=(110,120) if kind=='hero' else (48,max(30,round(48*h/w)))
            if kind=='hero':state=sample_hero(pose,resolution=resolution)
            elif kind=='climber':state=sample_climber(pose,resolution=resolution)
            else:state=fn(pose,rotation=rotation,resolution=resolution)
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
        if fi%args.fps==0:print(f'{t:.2f}s / {int(layer.sum())} obstacle cells',flush=True)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,times=times.astype('f4'),solid_mask=np.asarray(masks),solid_velocity=np.asarray(velocities))
    print(path,flush=True)

if __name__=='__main__':main()
