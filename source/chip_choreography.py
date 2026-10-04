"""Shared image/physics trajectory for the entering and exiting chip."""
import math

CHIP_START=6.173
CHIP_END=7.80

def _curve(t,keys):
    if t<=keys[0][0]:return keys[0][1]
    for (a,x),(b,y) in zip(keys,keys[1:]):
        if t<=b:
            u=(t-a)/(b-a);return x+(y-x)*u*u*(3-2*u)
    return keys[-1][1]

def chip_state(t):
    # The chip retains its entrance/face-on beat, then accelerates across the
    # right boundary. Visibility ends only after the whole physical die exits.
    x=_curve(t,[(CHIP_START,48),(6.27,192),(6.35,451.2),(6.52,672),(6.68,806.4),(7.03,806.4)])
    if t>7.03:x+=1780*((t-7.03)/.72)**2
    h=_curve(t,[(CHIP_START,660),(6.27,710),(6.45,380),(6.61,240),
                (6.79,320),(7.03,310),(7.27,430),(7.63,550),(CHIP_END,580)])
    pitch=_curve(t,[(CHIP_START,-.4),(6.27,0),(6.52,1.5),(6.77,math.pi),
                    (7.02,math.pi*1.5),(7.27,math.tau)])
    if t>7.27:pitch+=(t-7.27)*5.8
    exit_progress=max(0,min(1,(t-7.03)/.72))
    return dict(x=x,y=535-55*exit_progress**2,width=h,height=h,
                rotation=(pitch,.13*math.sin(t*5),-.12-.16*exit_progress),
                visible=CHIP_START<=t<CHIP_END)
