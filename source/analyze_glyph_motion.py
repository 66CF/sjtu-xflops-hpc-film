#!/usr/bin/env python3
"""Report actual recovery moves and visible motion residuals from a baked cache.

Kinematic terminal scrolling, births, and aperture entrances/exits are excluded
from the velocity-residual comparison. A residual can include real collisions
and acceleration; it is a locator for visual review, not an artifact verdict.
"""
import argparse, json
from pathlib import Path
import numpy as np
from glyph_material import visible_glyphs

INTERVALS = [('opening',3.,7.3),('rack',7.3,13.65),
             ('late_rank',13.65,16.665),('ending',16.665,20.)]


def analyze(flow, output, threshold=24., limit=1000):
    with np.load(flow) as source:
        f={k:source[k] for k in ['xy','velocity','times','group','birth','release',
                                 'glyph','size','recovery_diagnostics']}
    xy=f['xy'];v=f['velocity'];tt=f['times'];g=f['group'];rel=f['release']
    delta=np.diff(xy,axis=0);dt=np.diff(tt)
    predicted=(v[1:]+v[:-1])*.5*dt[:,None,None]
    residual=np.linalg.norm(delta-predicted,axis=2)
    distance=np.linalg.norm(delta,axis=2);speed=np.linalg.norm(v,axis=2)
    screen=((xy[:,:,0]>-30)&(xy[:,:,0]<1950)&(xy[:,:,1]>-30)&(xy[:,:,1]<1110))
    valid=((tt[:-1,None]>=f['birth'])&(tt[:-1,None]>=rel)&
           visible_glyphs(g)&screen[:-1]&screen[1:])
    recovery=f['recovery_diagnostics'];intervals={}
    for name,lo,hi in INTERVALS:
        sel=valid&(tt[:-1,None]>=lo)&(tt[:-1,None]<hi)
        z=residual[sel];d=distance[sel]
        r=recovery[(recovery[:,0]>=lo)&(recovery[:,0]<hi)]
        vs=speed[:-1][sel]
        intervals[name]={
            'visible_transitions':len(z),
            'residual_p95_px':float(np.percentile(z,95)),
            'residual_max_px':float(z.max()),
            'residual_over_threshold':int(np.sum(z>=threshold)),
            'frame_distance_p95_px':float(np.percentile(d,95)),
            'frame_distance_max_px':float(d.max()),
            'visible_speed_p95_px_s':float(np.percentile(vs,95)),
            'visible_speed_max_px_s':float(vs.max()),
            'recovery_visible_corrections':int(r[:,5].sum()),
            'recovery_visible_ge8':int(r[:,8].sum()),
            'recovery_max_step_visible_p95_px':float(r[:,6].max()),
            'recovery_visible_max_px':float(r[:,7].max()),
            'recovery_unresolved_attempts':int(r[:,4].sum()),
        }
    frame,ids=np.where(valid&(residual>=threshold))
    order=np.argsort(residual[frame,ids])[::-1][:limit];rows=[]
    for q in order:
        k=int(frame[q]);i=int(ids[q])
        rows.append({'time_from':float(tt[k]),'time_to':float(tt[k+1]),
                     'id':i,'group':int(g[i]),'glyph':str(f['glyph'][i]),
                     'size':float(f['size'][i]),'release':float(rel[i]),
                     'distance_px':float(distance[k,i]),
                     'velocity_residual_px':float(residual[k,i]),
                     'from':xy[k,i].tolist(),'to':xy[k+1,i].tolist(),
                     'v_from':v[k,i].tolist(),'v_to':v[k+1,i].tolist()})
    summary={'flow':str(flow),'recorded_frames':len(tt),
             'integration_steps':len(recovery),'threshold_px':threshold,
             'residual_count':len(frame),'outliers_retained':len(rows),
             'definition':'Already-free glyphs visible at both endpoints; displacement minus trapezoidal velocity. Contacts and acceleration can contribute.',
             'intervals':intervals}
    Path(output).write_text(json.dumps({'summary':summary,'outliers':rows,
        'recovery_records':recovery.tolist()},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--flow',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,default=24.)
    p.add_argument('--limit',type=int,default=1000);a=p.parse_args()
    analyze(a.flow,a.output,a.threshold,a.limit)
