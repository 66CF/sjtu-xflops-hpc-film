#!/usr/bin/env python3
"""Build a finite, initially nonoverlapping physical inlet reservoir.

Retains all source IDs, original terminal and worker surface glyphs, and every
font/size. Unused inlet IDs have birth/release=1e6, so they neither render nor
exert invisible contact mass. Visible eligible IDs are selected once, then
Poisson packed upstream with measured ink rectangles and spacing. There is no
per-frame culling, respawn, or position correction in the rendered movie.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from glyph_material import visible_glyphs


def prepare(input_path,geometry_path,output_path,medium=600,large=50):
    with np.load(input_path) as original:d={k:original[k].copy() for k in original.files}
    measured=np.load(geometry_path);footprint=measured['footprint'];skin=float(measured['skin']);group=d['group'];eligible=visible_glyphs(group)
    selected=[]
    for g,count in [(1,medium),(2,large)]:
        ids=np.flatnonzero((group==g)&eligible)
        selected.extend(ids[np.linspace(0,len(ids)-1,min(count,len(ids))).round().astype(int)])
    selected=np.asarray(selected,np.int32)
    # Place the largest circles first. This conservative broad phase packs
    # any possible initial rotation, not just zero-angle glyph widths.
    radii=np.linalg.norm(footprint[:,:2]+skin,axis=1)+2.0
    selected=selected[np.argsort(radii[selected])[::-1]]
    rng=np.random.default_rng(552024);placed=[];pr=[];kept=[];cell=80.;bins={}
    for i in selected:
        r=float(radii[i]);found=None
        for trial in range(4000):
            p=np.array([rng.uniform(2070+r,4700-r),rng.uniform(-170+r,1240-r)])
            bx,by=np.floor(p/cell).astype(int);near=[]
            for y in range(by-1,by+2):
                for x in range(bx-1,bx+2):near.extend(bins.get((x,y),[]))
            if near:
                pp=np.asarray(placed)[near];rr=np.asarray(pr)[near]
                if np.any(np.sum((pp-p)**2,axis=1)<(rr+r)**2):continue
            found=p;break
        if found is None:continue
        bins.setdefault((bx,by),[]).append(len(placed));placed.append(found);pr.append(r);kept.append(i)
    disabled=(group==1)|(group==2)
    d['birth'][disabled]=1e6;d['release'][disabled]=1e6
    for i,p in zip(kept,placed):
        # Centers measured above are ink centers, not tile anchors.
        d['xy'][i]=p-footprint[i,2:]
        d['birth'][i]=0.;d['release'][i]=4.4 if group[i]==1 else 4.55
    meta={'version':5,'policy':'finite Poisson-packed upstream reservoir; no invisible contact mass; stable source IDs','active_inlet_counts':{str(g):int(np.sum((group[np.asarray(kept)]==g))) for g in [1,2]},'source_ids_retained':len(group),'minimum_initial_ink_skin_px':skin,'extra_radial_spacing_px':2.,'terminal_and_worker_sources_unchanged':True,'particle_font_sizes_unchanged':True}
    d['preparation_metadata']=np.array(json.dumps(meta))
    np.savez_compressed(output_path,**d);Path(output_path).with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--geometry',required=True);p.add_argument('--output',required=True);p.add_argument('--medium',type=int,default=600);p.add_argument('--large',type=int,default=50)
    a=p.parse_args();prepare(a.input,a.geometry,a.output,a.medium,a.large)
