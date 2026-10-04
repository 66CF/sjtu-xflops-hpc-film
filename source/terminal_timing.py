"""Shared, frame-exact terminal output for the picture and original soundtrack.

A visible output event is emitted only when a new non-space character or a new
nonempty log line first appears. Existing lines moving up the screen never
produce a typing event. The simulation's continuous scrolling curve is retained.
"""
from functools import lru_cache
import math
import numpy as np

FPS=24
H=1080
def curve(t,keys):
    """Shape-preserving cubic Hermite curves, with per-channel key times."""
    arr=np.asarray(keys,dtype=float);tt=arr[:,0];vv=arr[:,1:]
    if t<=tt[0]:v=vv[0]
    elif t>=tt[-1]:v=vv[-1]
    else:
        slopes=np.diff(vv,axis=0)/np.diff(tt)[:,None]
        tangent=np.zeros_like(vv);tangent[0]=slopes[0];tangent[-1]=slopes[-1]
        for j in range(1,len(tt)-1):
            same=slopes[j-1]*slopes[j]>0
            tangent[j]=np.where(same,2*slopes[j-1]*slopes[j]/(slopes[j-1]+slopes[j]+1e-12),0)
        k=np.searchsorted(tt,t)-1;dt=tt[k+1]-tt[k];u=(t-tt[k])/dt
        v=(2*u**3-3*u*u+1)*vv[k]+(u**3-2*u*u+u)*dt*tangent[k]+(-2*u**3+3*u*u)*vv[k+1]+(u**3-u*u)*dt*tangent[k+1]
    return float(v[0]) if len(v)==1 else v


INTRO=[
    '/*','A program can run.', '',
    '    ...error... One core can only go so far.',
    'A bigger question has been detected.',
    'More than one mind required.', '',
    'An upgrade is in progress...', '',
    'Ready >>',
    'Set System Mode to: PARALLEL',
    '{ Initialize a new generation of computing }',
    '    ...from one core to collective intelligence...',
    '    ...breaking through the next bottleneck...',
    'The next breakthrough is a team effort.',
    '    >> Accelerate?', 'Sure.',
    '    // Set mode: XFLOPS.',
    'Open possibilities. Build the infrastructure.', '',
    'HPC system updating...', 'LOG(DEBUG): Executing <<', '',
    '#include <mpi.h>',
    'int main(int argc, char **argv) {',
    '    MPI_Init(&argc, &argv);',
    '    // Initialize the compute fabric',
    '    launch_parallel_kernels();',
    '    synchronize_workers();',
    '    return discover_what_is_next();',
    '}', '',
    '// Start the main loop',
    'while (residual > tolerance) {',
    '    exchange_boundaries();',
    '    iterate();',
    '}', '',
    '>>', 'Initializing system upgrade protocol...',
    'Activating: SJTU Xflops / Shanghai Jiao Tong University',
]
TASKS=['Mapping compute kernels to available execution units',
       'Scheduling independent tasks across the cluster',
       'Preparing high bandwidth interconnect routes',
       'Synchronizing collective communication across all ranks',
       'Retrieving memory pages for the next working set',
       'Building a new pipeline for distributed training',
       'Activating all-reduce gradient synchronization',
       'Removing unnecessary serialization from the hot path',
       'Overlapping memory transfers with active computation',
       'Restoring numerical stability across boundary conditions',
       'Creating the next simulation from a better question',
       'Checking shared state and solver convergence',
       'Committing a new generation of parallel potential',
       'Releasing the next wave of discovery']
LOGS=INTRO.copy()
for k in range(150):
    msg=TASKS[k%len(TASKS)]+'...'
    detail=f' rank={k%64:02d} / block={1024*(k%8+1):04d} / phase={k%9+1:02d} / active'
    LOGS.append((msg+detail)[:89].ljust(90)+['[OK]','[OK]','[>>]'][k%3])

SCROLL_KEYS=[(0,0),(2.25,0),(2.34,.8),(2.42,1.57),(2.51,1.97),(2.59,3.55),(2.67,2.77),(2.84,2.77),(2.92,.8),(3.01,.4),(3.17,.8),(3.26,.4),(3.43,.4),(3.51,0),(4.8,0)]
ST=np.arange(0,5.01,.001)
SV=np.array([curve(t,SCROLL_KEYS) for t in ST])*H
SI=np.cumsum(SV)*.001
def scroll(t):return math.floor(float(np.interp(t,ST,SI))/36)*36
ROW_KEYS=[(0,0),(.2,1),(.45,2),(1.45,2),(1.70,5),(1.95,8),(2.12,13),(2.25,19),(2.34,29),(3.8,29)]

SECOND_LOGS=['Ready >>','Set System Mode to: DISTRIBUTED',
               '{ Synchronizing the next generation of intelligence }',
               'GLOBAL COMPUTE ZONES:','',
               '>>> SYSTEM UPDATE: XFLOPS / AI INFRASTRUCTURE',
               'ModuleLoaded: TrainingEngine.cpp       RoutingOptimizer: ENABLED',
               'Autolayer: DATA_PARALLEL               GradientSync: ALL_REDUCE',
               'Interconnect: ACTIVE                  WorkerState: READY','',
               '// SYSTEM MESSAGE - Running background loop',
               'TensorShardingPlan: Resolved',
               'Checkpoint: Consistent',
               'CollectiveGroup: Synchronized',
               'Optimizer.step() -> NEXT_ITERATION','',
               '#include <xflops/collective.h>',
               'int main() {',
               '    launch_training_pipeline();',
               '    synchronize_gradients();',
               '    return next_breakthrough();','}', '',
               '>>> COMPUTE ONLINE - OPEN - COLLABORATIVE - PARALLEL']
for k in range(50):SECOND_LOGS.append(LOGS[40+k])


def frame_number(t):
    return max(0, math.floor(t*FPS+1e-7))


def opening_rows(t):
    """Rows exactly as drawn at this output frame, including the typed opener."""
    frame=frame_number(t)
    stamp=frame/FPS
    shift=scroll(stamp)
    rowmax=int(curve(stamp,ROW_KEYS)+shift/36)
    result=[]
    for row in range(min(rowmax,len(LOGS))):
        y=40+row*36-shift
        if -36<y<H:
            line=LOGS[row]
            if row==0:line=line[:max(0,frame-4)]
            elif row==1:line=line[:max(0,frame-10)]
            result.append((row,line,y))
    return result


def second_rows(t):
    local=t-9.45
    y0=curve(t,[(9.45,1060),(9.64,510),(9.90,70),(10.25,40),(11.5,-470),(12,-970)])
    limit=int(max(0,local)*85)+8
    return [(row,line,y0+row*36) for row,line in enumerate(SECOND_LOGS[:limit])
            if -40<y0+row*36<H]


def _visible_events(rows_at, first, last, scene):
    seen={}
    events=[]
    for frame in range(first,last+1):
        additions=[]
        changed_rows=[]
        for row,line,y in rows_at(frame/FPS):
            # A line wholly clipped below the picture is not audible yet.
            if y>=H-5:continue
            previous=seen.get(row,0)
            if len(line)>previous:
                new=line[previous:]
                if new.strip():
                    additions.append(new)
                    changed_rows.append(row)
                seen[row]=len(line)
        if additions:
            events.append(dict(frame=frame,time=frame/FPS,scene=scene,
                               kind='key' if scene=='opening' and max(changed_rows)<2 else 'line',
                               text=' | '.join(additions),rows=changed_rows,
                               characters=sum(len(part.strip()) for part in additions)))
    return events


@lru_cache(maxsize=1)
def opening_events():
    return tuple(_visible_events(opening_rows,0,72,'opening'))


@lru_cache(maxsize=1)
def second_events():
    return tuple(_visible_events(second_rows,math.ceil(9.45*FPS),math.floor(10.2*FPS),'second_terminal'))


def typed_count(t,born,cps,length):
    return min(length,max(0,int((t-born)*cps)))


def typed_events(text,born,cps,scene='closing',gain=.02):
    previous=0
    events=[]
    for frame in range(math.floor(born*FPS),math.ceil((born+len(text)/cps)*FPS)+2):
        count=typed_count(frame/FPS,born,cps,len(text))
        new=text[previous:count]
        if new.strip():
            events.append(dict(frame=frame,time=frame/FPS,scene=scene,kind='key',
                               text=new,rows=[],characters=len(new.strip()),gain=gain))
        previous=count
    return events


def closing_events():
    events=[dict(frame=481,time=481/FPS,scene='closing',kind='line',
                 text='System > Upgraded',rows=[],characters=17,gain=.024)]
    for text,born in [('Compute is;',21.0),('Parallel',21.7),('Connected',22.25),
                      ('Open',22.8),('HPC / AI INFRA',23.35)]:
        events.extend(typed_events(text,born,35,gain=.019))
    events.extend(typed_events('The next breakthrough starts here.',24.8,40,gain=.013))
    events.extend(typed_events('SJTU Xflops',27.65,24,gain=.018))
    return tuple(sorted(events,key=lambda event:event['frame']))
