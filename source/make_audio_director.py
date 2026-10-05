#!/usr/bin/env python3
"""Original 30-second, 120 BPM director-cut electronic score for SJTU Xflops.

Synthesized entirely from oscillators and seeded noise; no sampled or reference
audio is used. Run with Python 3 + NumPy. Output: ../work/soundtrack-director.wav.
Early accents follow the visible RUN contacts and the worker's folding body;
cluster cues follow docking, visible link traffic and mechanical rail release.
Later cues follow synchronization, grounded jumps and the quiet closing scenes.
Every sound is original.
"""

from pathlib import Path
import wave
import json
import numpy as np
from terminal_timing import opening_events, closing_events

SR = 48_000
DURATION = 30.0
TEMPO = 120
N = int(SR * DURATION)
RNG = np.random.default_rng(20261007)
master = np.zeros((N, 2), dtype=np.float64)
music = np.zeros_like(master)
terminal_bus = np.zeros_like(master)
terminal_audit = []
cluster_bus = np.zeros_like(master)
cluster_audit = []


def timeline(seconds):
    return np.arange(int(round(seconds * SR)), dtype=np.float64) / SR


def midi(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def add(signal, start, gain=1.0, pan=0.0, bus=master):
    i = int(round(start * SR))
    if i >= N:
        return
    if i < 0:
        signal = signal[-i:]
        i = 0
    length = min(len(signal), N - i)
    p = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    bus[i:i + length, 0] += signal[:length] * gain * np.cos(p)
    bus[i:i + length, 1] += signal[:length] * gain * np.sin(p)


def filtered_noise(seconds, lo, hi):
    t = timeline(seconds)
    spectrum = np.fft.rfft(RNG.normal(size=len(t)))
    freq = np.fft.rfftfreq(len(t), 1 / SR)
    filt = np.exp(-(freq / hi) ** 4)
    if lo:
        filt *= 1 - np.exp(-(freq / lo) ** 4)
    out = np.fft.irfft(spectrum * filt, n=len(t))
    return out / (np.std(out) + 1e-12)


def pluck(note, seconds=0.65, brightness=0.2):
    t = timeline(seconds)
    f = midi(note)
    # A rounded FM/triangle tone; short attack prevents digital clicks.
    phase = 2 * np.pi * f * t
    tone = np.sin(phase + brightness * np.sin(2 * phase) * np.exp(-t * 9))
    tone += 0.12 * np.sin(3 * phase) * np.exp(-t * 13)
    env = (1 - np.exp(-t / 0.008)) * np.exp(-t / 0.16)
    env *= np.clip((seconds - t) / 0.05, 0, 1)
    return tone * env


def kick():
    t = timeline(0.44)
    freq = 46 + 102 * np.exp(-t / 0.022)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-t / 0.115)
    body += 0.17 * np.sin(phase * 2) * np.exp(-t / 0.04)
    return body * (1 - np.exp(-t / 0.0015))


def snare():
    t = timeline(0.22)
    noise = filtered_noise(0.22, 700, 5000)
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.041)
    body = np.sin(2 * np.pi * 178 * t) * np.exp(-t / 0.025)
    return (0.72 * noise + 0.46 * body) * env


def hat(seconds=0.07):
    t = timeline(seconds)
    return filtered_noise(seconds, 3800, 8300) * np.exp(-t / 0.018) * (1 - np.exp(-t / 0.001))


def bass(note, seconds):
    t = timeline(seconds)
    phase = 2 * np.pi * midi(note) * t
    sig = np.sin(phase) + 0.17 * np.sin(2 * phase) + 0.07 * np.sin(3 * phase)
    env = (1 - np.exp(-t / 0.015)) * np.exp(-t / 0.7)
    env *= np.minimum(1, np.maximum(0, (seconds - t) / 0.055))
    return sig * env


def pad(notes, seconds=4.6):
    t = timeline(seconds)
    result = np.zeros(len(t))
    for i, note in enumerate(notes):
        f = midi(note)
        result += np.sin(2 * np.pi * f * t + 0.35 * i)
        result += 0.25 * np.sin(2 * np.pi * f * 1.0024 * t + i)
        result += 0.08 * np.sin(2 * np.pi * f * 2 * t + i)
    env = np.minimum(1, t / 0.6) * np.clip((seconds - t) / 1.0, 0, 1)
    return result * env / len(notes)


def terminal_events(events, gain=.035):
    """One dry key/line transient per actual visible output frame.

    Timing comes only from the renderer's shared event table. Seeded noise
    changes each key's timbre, never its onset or the gaps between output.
    At 24 fps / 48 kHz each video frame is exactly 2,000 audio samples.
    """
    for event in events:
        when=event['frame']/24
        is_key=event['kind']=='key'
        seconds=.027 if is_key else .033
        u=timeline(seconds)
        frequency=620+((event['frame']*17)%130)
        sig=filtered_noise(seconds,520,3400)
        sig+=.30*np.sin(2*np.pi*frequency*u)
        sig*=(1-np.exp(-u/.00065))*np.exp(-u/(.0044 if is_key else .0055))
        level=event.get('gain',gain)
        if event['scene']=='opening':
            level*=.65 if is_key else min(1.13,.80+.08*len(event['rows']))
        pan=-.10 if is_key else -.035
        add(sig,when,level,pan,terminal_bus)
        terminal_audit.append(dict(event,audio_sample=round(when*SR),
                                   peak_offset_samples=int(np.argmax(np.abs(sig))),
                                   duration_samples=len(sig)))


def whoosh(start, seconds, gain=.04, lo=220, hi=3200, direction=1):
    """A short unpitched rush with a broad stereo path, never a siren."""
    t = timeline(seconds)
    env = np.sin(np.pi * t / seconds) ** 1.45
    # The rise is slightly faster than the decay; the noisy timbre stays rounded.
    env *= .8 + .2 * np.exp(-t / max(.1, seconds * .4))
    sig = filtered_noise(seconds, lo, hi) * env
    add(sig, start, gain, -.62 * direction)
    add(sig, start + .031, gain * .71, .66 * direction)


def soft_impact(start, gain=.15, note=38):
    t = timeline(.55)
    f = midi(note)
    phase = 2 * np.pi * np.cumsum(f + 64 * np.exp(-t / .018)) / SR
    sig = np.sin(phase) * np.exp(-t / .105) * (1 - np.exp(-t / .002))
    add(sig, start, gain)


# A typed opener gives way to real output rows. Both image and sound use the
# same 24 fps event schedule, including the quiet hold and accelerating scroll.
add(pluck(74, .45, .05), .20, .085, -.15, music)
terminal_events(opening_events(),.052)
for when, note in [(1.50, 62), (2.27, 69), (2.87, 74)]:
    add(pluck(note, .36, .04), when, .049, .12, music)
add(pad([50, 57, 62], 3.9), .7, .08, -.15, music)

# The visible RUN cursor makes three light contacts with the terminal rows.
# Short dry taps and trailing friction replace the previous unseen fan-like rush.
for when, gain, pan in [(3.30,.047,-.48),(3.60,.044,-.10),(4.00,.042,.44)]:
    u=timeline(.18)
    tap=np.sin(2*np.pi*640*u)*np.exp(-u/.009)
    friction=filtered_noise(.18,430,2800)*.29*np.exp(-u/.042)
    envelope=(1-np.exp(-u/.0012))*np.clip((.18-u)/.035,0,1)
    add((tap*.63+friction)*envelope,when,gain,pan)

# First ten seconds: short motion accents at the actual reference action beats.
whoosh(4.47, .42, .072, 170, 3600, 1)
soft_impact(4.65, .17)
add(pluck(69, .45, .13), 4.67, .095, .12, music)
# The same worker folds its limbs into the die: a restrained mechanical
# closure belongs to this body, with no second entrance impact or passing rush.
u=timeline(.15)
closure=(.48*np.sin(2*np.pi*330*u)+filtered_noise(.15,500,2450)*.17)
closure*=np.exp(-u/.025)*(1-np.exp(-u/.002))*np.clip((.15-u)/.035,0,1)
add(closure,6.30,.036,-.06)
add(pluck(77,.40,.05),6.33,.035,-.08,music)
# Detail sounds are deliberately short and sparse; no continuous sci-fi siren.
def data_tick(when, gain=.042, note=76, pan=0):
    seconds=.075
    u=timeline(seconds)
    tone=.64*np.sin(2*np.pi*midi(note)*u)+.16*np.sin(2*np.pi*midi(note)*2.03*u)
    grit=filtered_noise(seconds,850,4100)*.13
    env=(1-np.exp(-u/.0009))*np.exp(-u/.012)
    add((tone+grit)*env,when,gain,pan)


def light_step(when, gain=.04, pan=0):
    seconds=.075
    u=timeline(seconds)
    freq=210+95*np.exp(-u/.007)
    body=np.sin(2*np.pi*np.cumsum(freq)/SR)
    wood=filtered_noise(seconds,360,2500)
    env=(1-np.exp(-u/.001))*np.exp(-u/.015)
    add((body*.60+wood*.23)*env,when,gain,pan)


def landing_contact(when, note, pan):
    # A dry planted contact with a small tonal answer after the sole touches.
    light_step(when,.047,pan)
    add(pluck(note,.25,.035),when+.018,.042,pan,music)
    add(pluck(note+7,.18,.02),when+.073,.020,-pan*.4,music)


def cluster_cue(signal, when, name, gain, pan=0.):
    # A continuous animation cue first becomes visible on this video frame.
    frame=int(np.ceil(when*24-1e-7));stamp=frame/24
    add(signal,stamp,gain,pan,cluster_bus)
    cluster_audit.append(dict(name=name,source_time=when,frame=frame,time=stamp,
                              audio_sample=frame*2000,duration_samples=len(signal),
                              pan=pan,gain=gain))


def docking_latch(when,pan):
    u=timeline(.14)
    # A small sprung latch: rounded contact followed by a short metal answer.
    body=.66*np.sin(2*np.pi*355*u)+.17*np.sin(2*np.pi*815*u)
    body+=filtered_noise(.14,680,2200)*.12
    env=(1-np.exp(-u/.0014))*np.exp(-u/.019)
    cluster_cue(body*env,when,'slot_lock',.053,pan)


def network_ping(when,pan,note):
    u=timeline(.16)
    f=midi(note)
    # Clean pitched packets distinguish network traffic from keyboard clicks.
    body=np.sin(2*np.pi*f*u+.11*np.sin(2*np.pi*2*f*u)*np.exp(-u*40))
    body+=.12*np.sin(2*np.pi*3*f*u)*np.exp(-u*32)
    env=(1-np.exp(-u/.0035))*np.exp(-u/.031)*np.clip((.16-u)/.04,0,1)
    cluster_cue(body*env,when,'network_packet',.043,pan)


def rail_slide(when,seconds,pan):
    u=timeline(seconds)
    # Close dry friction with shallow rail teeth, not a passing wind effect.
    body=filtered_noise(seconds,410,1680)*(.44+.11*np.sin(2*np.pi*22*u))
    body+=.20*np.sin(2*np.pi*175*u)
    env=np.minimum(1,u/.045)*np.clip((seconds-u)/.09,0,1)
    cluster_cue(body*env,when,'support_rail',.019,pan)


def tray_foot_contact(when,pan):
    u=timeline(.09)
    body=.57*np.sin(2*np.pi*270*u)+filtered_noise(.09,470,2400)*.19
    env=(1-np.exp(-u/.0013))*np.exp(-u/.017)
    cluster_cue(body*env,when,'tray_foot',.037,pan)


# Character stride: a 0.6s left/right cycle, so contacts occur about every 0.3s.
for j,when in enumerate([4.80,5.10,5.40,5.70,6.00]):
    light_step(when,.034,-.14 if j%2==0 else .14)

# Docking, visible interconnect traffic and the rack-to-worker handoff share
# an event timeline with cluster_scene. These are material contacts and tonal
# packets, never the deleted terminal keystrokes or annular burst sounds.
import cluster_scene as cluster

# The latch settles into the same on-screen socket as the visible processor.
lock_position=cluster.node_state(cluster.PRIMARY,cluster.DOCK_TIME,False)['center']
lock_pan=float(np.clip((lock_position[0]-960)/1100,-.72,.72))
docking_latch(cluster.DOCK_TIME,lock_pan)

# Each candidate is a visible packet crossing the midpoint of a real cable.
# Match draw_network's .77 cycles/s and two packet phases (0 and .48). Choose
# sparse crossings from that flow; never sound a node that is still offscreen.
packet_candidates=[]
for source,target,_ in cluster._network_paths(cluster.NETWORK_TIME):
    if cluster.MISSING_RANK in (source,target):continue
    delay=cluster.activation_time(target)
    for packet in range(2):
        for cycle in range(3):
            when=delay+(cycle+.5-packet*.48)/.77
            if not cluster.DOCK_TIME+.10<when<cluster.DEPLOY_TIME-.06:continue
            stamp=np.ceil(when*24-1e-7)/24
            path=next(path for a,b,path in cluster._network_paths(stamp)
                      if (a,b)==(source,target))
            length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
            phase=((stamp-delay)*.77+packet*.48)%1
            point=cluster._polyline(path,np.array([phase*length]))[0]
            if not (150<point[0]<1770 and 90<point[1]<990):continue
            packet_candidates.append(dict(time=when,source=source,target=target,packet=packet,
                                          screen_xy=point.tolist(),pan=float(np.clip((point[0]-960)/1100,-.72,.72))))
used=set()
for j,wanted in enumerate([8.72,8.90,9.15,9.42,9.70,10.0,10.36,10.78,11.20,11.64,12.0]):
    options=[(k,event) for k,event in enumerate(packet_candidates)
             if k not in used and wanted<=event['time']<=wanted+.16]
    if not options:continue
    desired_pan=(-.42 if j%2 else .42)
    k,event=min(options,key=lambda pair:abs(pair[1]['pan']-desired_pan)+.8*(pair[1]['time']-wanted))
    used.add(k)
    network_ping(event['time'],event['pan'],[74,77,69,72,65,74,77,69,72,74,69][j])
    cluster_audit[-1].update(source_node=event['source'],target_node=event['target'],
                             packet=event['packet'],screen_xy=event['screen_xy'])

# The established cluster resolves into a quiet chord instead of a scene cut.
cluster_cue(pad([53,57,62,65],1.60),cluster.NETWORK_TIME,'network_online',.086,0.)

# The guides visibly withdraw while the tray remains. Their dry friction stays
# close to the assembly, then shoes contact the retained platform and step off.
rail_slide(cluster.DEPLOY_TIME,.84,-.48)
rail_slide(cluster.DEPLOY_TIME+.085,.84,.48)
tray_foot_contact(cluster.FOOT_CONTACT_TIME,-.21)
tray_foot_contact(cluster.FOOT_CONTACT_TIME+1/24,.21)
tray_foot_contact(cluster.RUN_TIME+.06,-.17)
tray_foot_contact(cluster.RUN_TIME+.26,.17)


# Rank 23's empty socket persists. Its arrival and unfolding now cause the
# collective to complete; the sounds follow those contacts in the picture.
rail_slide(cluster.LATE_ENTRY+.12,.66,.65)
docking_latch(cluster.LATE_DOCK,.61)
tray_foot_contact(cluster.LATE_UNFOLD,.61)
for j,when in enumerate([14.76,15.08,15.68,15.90]):
    data_tick(when,.041,[69,72,74,76][j],.42)
data_tick(16.12,.059,72,0)
for j,note in enumerate([65,69,72]):
    add(pluck(note,.36,.045),16.12,.063,-.24+j*.24,music)

# Streams depart, then grounded push-offs and planted landings on underscores.
whoosh(17.16,.30,.021,310,2300,1)
for when,pan in [(17.28,-.24),(17.91,0),(18.54,.24)]:
    light_step(when,.021,pan)
landing_contact(17.70,69,-.20)
landing_contact(18.33,72,0)
landing_contact(18.96,74,.20)
for when,pan in [(19.28,.35),(19.48,.52),(19.68,.68)]:
    light_step(when,.020,pan)
# Clear the frame without a booming scene-change impact.
add(pluck(62,.28,.015),19.50,.030,-.10,music)

# Harmonic bed retains the initial two sections, then makes room for the action.
# D minor 9 -> Bb major 7 -> F major 9 -> C add 9 -> D minor 9.
sections = [
    (4.65, 4.00, [50,57,60,65], [62,69,72,77], 38),
    (8.65, 4.00, [46,53,57,62], [62,65,69,74], 34),
    (12.65,2.30, [53,60,64,67], [65,69,72,79], 41),
    (16.12,3.05, [48,55,62,64], [62,67,72,76], 36),
    (20.00,4.20, [50,57,60,65], [62,69,72,77], 38),
]
pattern=[0,2,1,3,2,1,3,1]
for start,length,notes,arp,root in sections:
    quiet=start>=20
    pad_gain=.115 if quiet else .235
    add(pad(notes,length+1.1),start,pad_gain,-.26,music)
    add(pad([n+12 for n in notes],length+1.1),start+.027,.033 if quiet else .070,.40,music)
    interval=1.0 if quiet else (.50 if start>=16 else .25)
    for n in range(int(length/interval)):
        when=start+n*interval
        velocity=(.036 if quiet else (.16 if n%4==0 else .089))
        if 10.7<=when<12.2:velocity*=.60
        if start>=16 and not quiet:velocity*=.67
        add(pluck(arp[pattern[n%8]],.7,.15),when,velocity,np.sin(n*1.15)*.48,music)
    if not quiet:
        for n in range(int(length*2)):
            when=start+n*.5+(.125 if n%4==3 else .055)
            note=root+(12 if n%8==7 else 0)
            gain=.185 if n%2==0 else .14
            if when>=16:gain*=.62
            add(bass(note,.36),when,gain)

# Minimal pulse. It pauses at the barrier and gives way to small physical taps.
k=kick();s=snare()
for beat in range(29):
    when=4.65+beat*.5
    if 15.0<=when<16.30:continue
    if when>=17.0 and beat%2:continue
    strength=.37 if when<11 else .35
    if 7.5<=when<12.2:strength=.15
    elif 12.2<=when<14:strength=.21
    if when>=16:strength=.20
    add(k,when,strength)
    if beat%2 and when<15:
        gain=.036 if 7.5<=when<14 else (.08 if when<11 else .067)
        add(s,when,gain,.06)
    if when<15:
        gain=.014 if 7.5<=when<14 else (.030 if when<10.7 else .019)
        add(hat(),when+.25,gain,-.2 if beat%2 else .2)

# Closing text is keyed on its visible reveal frames, including the holds.
terminal_events(closing_events())

# The closing line draws once. A spacious signature resolves at the quiet brand.
whoosh(24.69,.24,.012,460,2100,1)
add(pad([50,57,62,65,69],5.10),24.80,.215,-.17,music)
for j,note in enumerate([62,69,74]):
    add(pluck(note,1.1,.035),24.80+j*.28,.090-j*.012,-.24+j*.24,music)
add(bass(38,1.5),24.81,.090)
add(pluck(74,1.6,.025),27.65,.061,.08,music)
add(pluck(62,1.6,.020),27.65,.037,-.16,music)

# Duck only the tonal bus during the wait, so each small staggered arrival reads.
t=timeline(DURATION)
duck=np.interp(t,[0,14.96,15.20,16.055,16.18,30],[1,1,.40,.40,1,1])
music*=duck[:,None]
# Make the physical contacts and clean link pings legible without louder effects.
# Full scoring returns before the original rank barrier sequence at 15 seconds.
cluster_space=np.interp(t,[0,7.5,8.5,12.0,12.4,14.0,15.0,30],
                         [1,1,.60,.60,.72,1,1,1])
music*=cluster_space[:,None]
master+=music+terminal_bus+cluster_bus
# Restrained stereo echoes belong to tonal notes, not clicks or footsteps.
for delay,gain in [(.1875,.12),(.375,.085),(.5625,.045),(.75,.030),(1.125,.018)]:
    shift=int(round(delay*SR))
    master[shift:]+=music[:-shift,::-1]*gain

# No continuous atmospheric-noise bed: transients get genuinely quiet gaps.
master-=master.mean(axis=0,keepdims=True)
master=np.tanh(master*1.18)
master*=min(1.0,10**(-1/20)/np.max(np.abs(master)))
fade_in=np.minimum(1,t/.025)
fade_out=np.minimum(1,np.maximum(0,(DURATION-t)/1.85))**1.5
master*=(fade_in*fade_out)[:,None]

out_path=Path(__file__).resolve().parent.parent/'work'/'soundtrack-director.wav'
out_path.parent.mkdir(parents=True,exist_ok=True)
# Preserve an isolated event stem plus its exact sample/frame correspondence
# so that synchronization can be checked independently of the musical bed.
audit_dir=out_path.parent/'audio-sync-v5'
audit_dir.mkdir(parents=True,exist_ok=True)
with wave.open(str(audit_dir/'terminal-events.wav'),'wb') as stem:
    stem.setnchannels(2);stem.setsampwidth(2);stem.setframerate(SR)
    stem.writeframes(np.rint(np.clip(terminal_bus,-1,1)*32767).astype('<i2').tobytes())
(audit_dir/'events.json').write_text(json.dumps(dict(
    fps=24,sample_rate=SR,samples_per_frame=SR//24,
    event_count=len(terminal_audit),events=terminal_audit),indent=2)+'\n')
with wave.open(str(audit_dir/'cluster-events.wav'),'wb') as stem:
    stem.setnchannels(2);stem.setsampwidth(2);stem.setframerate(SR)
    stem.writeframes(np.rint(np.clip(cluster_bus,-1,1)*32767).astype('<i2').tobytes())
(audit_dir/'cluster-events.json').write_text(json.dumps(dict(
    fps=24,sample_rate=SR,event_count=len(cluster_audit),events=cluster_audit),indent=2)+'\n')
pcm=np.rint(np.clip(master,-1,1)*32767).astype('<i2')
with wave.open(str(out_path),'wb') as output:
    output.setnchannels(2)
    output.setsampwidth(2)
    output.setframerate(SR)
    output.writeframes(pcm.tobytes())
peak_db=20*np.log10(np.max(np.abs(master))+1e-12)
rms_db=20*np.log10(np.sqrt(np.mean(master*master))+1e-12)
print(f'{out_path}\n{DURATION:.3f}s | {SR} Hz | stereo PCM16 | {TEMPO} BPM')
print(f'Peak {peak_db:.2f} dBFS | RMS {rms_db:.2f} dBFS | frames {len(pcm)}')
