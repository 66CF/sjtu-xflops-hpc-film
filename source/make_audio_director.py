#!/usr/bin/env python3
"""Original 30-second, 120 BPM director-cut electronic score for SJTU Xflops.

Synthesized entirely from oscillators and seeded noise; no sampled or reference
audio is used. Run with Python 3 + NumPy. Output: ../work/soundtrack-director.wav.
Early accents follow the visible RUN contacts and the chip's rightward exit;
later cues follow synchronization, grounded jumps and the quiet closing scenes.
Every sound is original.
"""

from pathlib import Path
import wave
import numpy as np

SR = 48_000
DURATION = 30.0
TEMPO = 120
N = int(SR * DURATION)
RNG = np.random.default_rng(20261007)
master = np.zeros((N, 2), dtype=np.float64)
music = np.zeros_like(master)


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


def terminal(start, end, gain=.05, acceleration=True):
    """Bursty rounded key/line sounds with deterministic, uneven timing."""
    when = start
    while when < end:
        progress = (when - start) / (end - start)
        seconds = .035
        t = timeline(seconds)
        sig = filtered_noise(seconds, 520, 3400)
        sig += .28 * np.sin(2 * np.pi * (590 + 180 * RNG.random()) * t)
        sig *= (1 - np.exp(-t / .001)) * np.exp(-t / .006)
        add(sig, when, gain * (.6 + .4 * progress), float(RNG.uniform(-.22, .22)))
        interval = RNG.uniform(.045, .13)
        if acceleration:
            interval *= 1.1 - .56 * progress
        if RNG.random() < .12:
            interval += .07
        when += interval


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


# Deliberate boot, then increasingly urgent terminal output at 1.5–2.9 seconds.
add(pluck(74, .45, .05), .20, .085, -.15, music)
terminal(.50, 1.30, .029, False)
terminal(1.50, 2.90, .071, True)
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
soft_impact(6.17, .16, 41)
whoosh(6.14, .31, .043, 290, 3200, -1)
add(pluck(77, .65, .10), 6.19, .08, -.22, music)
# The delayed right channel ends at 7.70 s as the die clears the right edge.
whoosh(7.38, .289, .044, 320, 3200, 1)
add(pluck(74, .30, .08), 7.633, .049, .28, music)
whoosh(8.22, .50, .056, 180, 3100, -1)
soft_impact(8.47, .18)
for j, note in enumerate([62, 69, 77]):
    add(pluck(note, .68, .17), 8.47 + j * .065, .079 - j * .01, -.4 + j * .4, music)
whoosh(9.24, .32, .049, 450, 3550, 1)
terminal(9.60, 10.16, .050, True)


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


# Character stride: a 0.6s left/right cycle, so contacts occur about every 0.3s.
for j,when in enumerate([4.80,5.10,5.40,5.70,6.00]):
    light_step(when,.034,-.14 if j%2==0 else .14)

# Selection, source edit, and the six-way fork form one short readable gesture.
data_tick(10.90,.048,72,-.18)
data_tick(11.30,.055,79,.14)
for j,note in enumerate([62,65,69,74,77,81]):
    data_tick(11.40+j*.046,.040-j*.002,note,-.65+j*.26)
# Pull back to reveal the 24-worker quilt: a single low-energy wide expansion.
whoosh(12.52,.36,.020,300,2450,-1)
add(pluck(69,.48,.055),12.70,.061,-.27,music)
add(pluck(74,.48,.055),12.725,.045,.27,music)

# Six asynchronous arrivals become one brief synchronized response.
# No rolls between these taps: the gaps are part of the waiting sensation.
for j,when in enumerate([15.20,15.37,15.55,15.73,15.90,16.05]):
    data_tick(when,.041,[69,72,74,76,77,79][j],-.58+j*.232)
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
    if when>=16:strength=.20
    add(k,when,strength)
    if beat%2 and when<15:
        add(s,when,.08 if when<11 else .067,.06)
    if when<15:
        add(hat(),when+.25,.030 if when<10.7 else .019,-.2 if beat%2 else .2)

# Tiny summary terminal in a quiet field, not a constant typing/noise layer.
terminal(20.05,20.57,.023,False)
terminal(21.10,21.44,.020,False)
terminal(22.16,22.48,.020,False)
terminal(23.06,23.32,.018,False)

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
master+=music
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
