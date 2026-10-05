#!/usr/bin/env python3
"""Measure the renderer's actual Menlo ink, and audit baked oriented contacts.

The NPZ sidecar stores each unrotated raster's ink bounding rectangle and its
center relative to the tile center. A 2.5 px skin accounts for bicubic rotated
edges and integer raster placement; the font and displayed size never change.
"""
from pathlib import Path
import argparse,json,math
import numpy as np


def export(input_path, output_path):
    from render_director import tile
    data=np.load(input_path)
    shapes=[]
    for glyph,size in zip(data['glyph'],data['size']):
        im=tile(str(glyph),int(np.clip(np.rint(size),7,84)),0,False,True)
        box=im.getbbox()
        if box is None:box=(im.width/2,im.height/2,im.width/2,im.height/2)
        x0,y0,x1,y1=box
        shapes.append(((x1-x0)/2,(y1-y0)/2,(x1+x0-im.width)/2,(y1+y0-im.height)/2))
    arr=np.asarray(shapes,np.float32)
    identity={};raster_ids=[];offsets=[];widths=[];heights=[];bit_rows=[];cursor=0
    for glyph,size in zip(data['glyph'],data['size']):
        key=(str(glyph),int(np.clip(np.rint(size),7,84)))
        if key not in identity:identity[key]=len(identity)
        raster_ids.append(identity[key])
    for glyph,size in identity:
        for angle in range(0,360,4):
            sprite=np.asarray(tile(glyph,size,angle,False,True))>0
            h,w=sprite.shape
            if w>128:raise ValueError('Raster atlas supports up to128px glyph width')
            padded=np.zeros((h,128),np.uint8);padded[:,:w]=sprite
            words=np.packbits(padded,axis=1,bitorder='little').view('<u4').reshape(-1)
            offsets.append(cursor);widths.append(w);heights.append(h);bit_rows.append(words);cursor+=len(words)
    np.savez_compressed(output_path,footprint=arr,glyph=data['glyph'],size=data['size'],skin=np.float32(2.5),
                        raster_ids=np.asarray(raster_ids,np.int32),raster_offsets=np.asarray(offsets,np.int32),
                        raster_width=np.asarray(widths,np.int32),raster_height=np.asarray(heights,np.int32),
                        raster_bits=np.concatenate(bit_rows).astype(np.uint32))
    print(json.dumps({'path':str(output_path),'particles':len(arr),'half_extent_range':np.stack([arr[:,:2].min(0),arr[:,:2].max(0)]).tolist()}))


def overlaps(xy,angle,footprint,skin=0.,ids=None):
    """Broad-phase spatial hash + vectorized 4-axis OBB SAT; no dependencies."""
    if ids is None:ids=np.arange(len(xy))
    xy=np.asarray(xy)[ids];angle=np.asarray(angle)[ids];f=np.asarray(footprint)[ids]
    a=np.deg2rad(np.rint(angle/4)*4)
    u=np.stack([np.cos(a),-np.sin(a)],1);v=np.stack([np.sin(a),np.cos(a)],1)
    center=xy+u*f[:,2,None]+v*f[:,3,None];half=f[:,:2]+skin
    rad=np.linalg.norm(half,axis=1)
    cell=max(16,float(np.max(rad))*2)
    bins={}
    for i,key in enumerate(map(tuple,np.floor(center/cell).astype(np.int32))):bins.setdefault(key,[]).append(i)
    pairs=[]
    for (x,y),value in bins.items():
        aa=np.asarray(value,np.int32)
        if len(aa)>1:
            q,r=np.triu_indices(len(aa),1);pairs.append(np.stack([aa[q],aa[r]],1))
        for ox,oy in [(1,-1),(1,0),(1,1),(0,1)]:
            other=bins.get((x+ox,y+oy))
            if other:
                bb=np.asarray(other,np.int32);pairs.append(np.stack([np.repeat(aa,len(bb)),np.tile(bb,len(aa))],1))
    if not pairs:return np.zeros((0,2),np.int32),np.zeros(0)
    pairs=np.concatenate(pairs);i,j=pairs.T;d=center[i]-center[j]
    near=np.sum(d*d,axis=1)<(rad[i]+rad[j])**2
    pairs=pairs[near];d=d[near];i,j=pairs.T
    minimum=np.full(len(pairs),np.inf)
    for axis in (u[i],v[i],u[j],v[j]):
        ai=half[i,0]*abs(np.sum(u[i]*axis,1))+half[i,1]*abs(np.sum(v[i]*axis,1))
        aj=half[j,0]*abs(np.sum(u[j]*axis,1))+half[j,1]*abs(np.sum(v[j]*axis,1))
        pen=ai+aj-abs(np.sum(d*axis,1));minimum=np.minimum(minimum,pen)
    hit=minimum>0
    return ids[pairs[hit]],minimum[hit]


def raster_overlap(xy,angle,size,glyph,ids):
    from render_director import tile
    coverage=np.zeros((1080,1920),np.uint16)
    owner=np.full((1080,1920),-1,np.int32);pairs=[]
    for i in ids:
        sp=tile(str(glyph[i]),int(np.clip(np.rint(size[i]),7,84)),int(np.rint(angle[i]/4)*4)%360,False,True)
        x=round(float(xy[i,0])-sp.width/2);y=round(float(xy[i,1])-sp.height/2)
        x0=max(0,x);x1=min(1920,x+sp.width);y0=max(0,y);y1=min(1080,y+sp.height)
        if x0>=x1 or y0>=y1:continue
        ink=np.asarray(sp)[y0-y:y1-y,x0-x:x1-x]>0
        patch=owner[y0:y1,x0:x1]
        touched=patch[ink&(patch>=0)]
        if touched.size:
            previous,counts=np.unique(touched,return_counts=True)
            pairs.extend({'ids':[int(j),int(i)],'pixels':int(count)} for j,count in zip(previous,counts))
        patch[ink]=int(i);coverage[y0:y1,x0:x1]+=ink
    return {'overlap_pixels':int(np.sum(coverage>1)),'maximum_ink_layers':int(coverage.max()),'overlap_pairs':pairs}


def audit(flow_path,shape_path,output_path,step=1,raster=False):
    from glyph_material import visible_glyphs
    with np.load(flow_path) as source:d={k:source[k] for k in source.files}
    f=np.load(shape_path)['footprint'];rows=[]
    shown=visible_glyphs(d['group']);birth=d['birth'];release=d['release']
    for frame in range(0,len(d['times']),step):
        t=float(d['times'][frame]);xy=d['xy'][frame]
        # Visible footprint crossing the aperture is included. Pinned terminal
        # letters count; unborn worker surface letters do not.
        active=t>=birth
        on=(xy[:,0]>-60)&(xy[:,0]<1980)&(xy[:,1]>-60)&(xy[:,1]<1140)
        row={'time':t}
        for name,choose in [('all',active&on),('visible',active&on&shown)]:
            ids=np.flatnonzero(choose);pairs,pen=overlaps(xy,d['angle'][frame],f,0,ids)
            row[name]={'glyphs':len(ids),'ink_bbox_pairs':len(pen),'max_penetration_px':float(pen.max()) if len(pen) else 0.,'pairs_over_0_25px':int((pen>.25).sum())}
        if raster:
            row['raster_visible']=raster_overlap(xy,d['angle'][frame],d['size'],d['glyph'],np.flatnonzero(active&on&shown))
        rows.append(row)
        if frame%48==0:print('audit',frame,row,flush=True)
    result={'flow':str(flow_path),'geometry':str(shape_path),'footprint':'actual Menlo Bold ink bounding rectangles at renderer 4-degree orientation; no skin in audit','frames':rows}
    result['summary']={name:{'maximum_penetration_px':max(r[name]['max_penetration_px'] for r in rows),'total_frame_pairs':sum(r[name]['ink_bbox_pairs'] for r in rows),'frames_with_penetration_over_0_25px':sum(r[name]['pairs_over_0_25px']>0 for r in rows)} for name in ['all','visible']}
    if raster:result['summary']['raster_visible']={'total_overlap_pixels':sum(r['raster_visible']['overlap_pixels'] for r in rows),'frames_with_overlap':sum(r['raster_visible']['overlap_pixels']>0 for r in rows),'maximum_ink_layers':max(r['raster_visible']['maximum_ink_layers'] for r in rows)}
    Path(output_path).write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    e=s.add_parser('export');e.add_argument('--input',required=True);e.add_argument('--output',required=True)
    a=s.add_parser('audit');a.add_argument('--flow',required=True);a.add_argument('--geometry',required=True);a.add_argument('--output',required=True);a.add_argument('--step',type=int,default=1);a.add_argument('--raster',action='store_true')
    args=p.parse_args()
    if args.command=='export':export(args.input,args.output)
    else:audit(args.flow,args.geometry,args.output,args.step,args.raster)
