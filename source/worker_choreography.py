"""Stable cluster workstations after deployment; shared by image and fluid."""
import math
from render_kinetic import curve,smooth


def worker_state(index,t):
    row,col=divmod(index,6)
    zoom=.46
    settle=16.12 if index==23 else 14.65+col*.10+row*.055
    pose=t+.063*index
    if t>settle:pose=settle+.063*index+math.sin((t-settle)*9)*.026
    step=curve(t,[(14.,0),(14.6,0),(settle,32),(17.2,32)])
    x=960+(col-2.5)*580*zoom+step*zoom
    y=535+(row-1.5)*505*zoom
    opacity=1. if index==18 else 1-float(smooth((t-(16.61+index*.009))/.24))
    if index==23 and t<16.12:opacity=0.
    if index==18 and t>=17.05:opacity=0.
    return dict(index=index,x=x,y=y,height=620*zoom,width=620*zoom*64/74,
                pose=pose,rotation=(0.,.60,0.),cell=9,opacity=opacity,
                settle=settle,zoom=zoom,visible=opacity>.001)
