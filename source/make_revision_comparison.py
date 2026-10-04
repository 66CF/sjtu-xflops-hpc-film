#!/usr/bin/env python3
"""Compare the three revised motion passages at their original speed."""
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw,ImageFont

root=Path(__file__).resolve().parents[1]
work=root/'work/revision-comparison'
work.mkdir(parents=True,exist_ok=True)
font=ImageFont.truetype('/System/Library/Fonts/Menlo.ttc',21)
passages=[('COLLISION',65,115),('RIGHT EXIT',165,204),('FOOT CONTACT',406,480)]
parts=[]
for n,(label,start,end) in enumerate(passages):
    header=Image.new('RGB',(1920,48),(16,18,27));draw=ImageDraw.Draw(header)
    draw.text((26,11),'PREVIOUS / '+label,font=font,fill='white')
    draw.text((986,11),'REVISED / '+label,font=font,fill='white')
    png=work/f'header-{n}.png';header.save(png)
    part=work/f'part-{n}.mp4';parts.append(part)
    trim=f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,scale=960:540,setsar=1'
    subprocess.run(['ffmpeg','-y','-v','error',
        '-i',str(root/'output/SJTU_Xflops_Physics.mp4'),
        '-i',str(root/'output/SJTU_Xflops_Physics_v2.mp4'),
        '-loop','1','-i',str(png),
        '-filter_complex',f'[0:v]{trim}[a];[1:v]{trim}[b];[a][b]hstack[films];[2:v][films]vstack,format=yuv420p[out]',
        '-map','[out]','-an','-t',str((end-start)/24),'-r','24',
        '-c:v','libx264','-preset','fast','-crf','18','-movflags','+faststart',str(part)],check=True)
manifest=work/'parts.txt'
manifest.write_text(''.join(f"file '{p.name}'\n" for p in parts))
dest=root/'output/SJTU_Xflops_Motion_Changes.mp4'
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),
                '-c','copy','-movflags','+faststart',str(dest)],check=True)
print(dest)
