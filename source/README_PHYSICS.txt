SJTU XFLOPS — CLUSTER CUT V5

DELIVERY
output/SJTU_Xflops_Cluster_v5.mp4
30.000 seconds, 1920 x 1080, 24 fps / 720 frames.
H.264 video, AAC 48 kHz stereo; English text and original synthesized audio.

V3 CHECKPOINT
Commit 81e1f568f9c66f18906916e50c3d279ac4693ddd, tag v1.1.0.
https://github.com/66CF/sjtu-xflops-hpc-film/releases/tag/v1.1.0
That release contains the exact v3 film, physics-assets-v3.zip and checksums.
The local duplicate is work/checkpoints/v3/. V3's knot is intentionally not
part of the v5 story. Use the tagged source with the tagged assets to restore it.

NORMAL V5 REBUILD
    python3 source/build_director.py --check-only
    python3 source/build_director.py

Requires Python 3, NumPy, Pillow, FFmpeg/FFprobe and macOS Menlo at
/System/Library/Fonts/Menlo.ttc. No GPU is needed to render the baked cache.
The required cache is work/physics-v5/flow.npz. Preflight verifies dimensions,
finite values, the 120 Hz / 1536 x 432 fluid settings, v5 cluster metadata and
absence of the removed annular gather. A v3 cache is rejected. The final movie
is replaced only after resolution, frame count, duration and audio checks pass.

CONTINUOUS MIDDLE PASSAGE
7.30 s: the same processor is followed into a rack; position, scale and pose
match hero_choreography.py at the junction.
8.56 s: the processor reaches its socket; a collar closes and data links start.
8.64-10.20 s: a continuous camera pullback reveals 23 existing processors and an empty Rank 23 socket.
10.05-12.20 s: the AI INFRA tile remains legible while data traffic runs.
12.20-13.62 s: structural guides withdraw. The same processor faces stay in
place while arms and legs extend from behind them. Supporting trays remain.
14.00 s: the same geometry joins worker_choreography.py without an object swap.
Rank 23 is visibly absent from the rack and worker grid. Its tray remains.
14.42-15.32 s: the late processor arrives from the right and docks.
15.32-16.10 s: that same processor unfolds into the missing worker.
16.12 s: the collective can complete. The existing jumps and ending remain.
White emphasis labels assemble from inverse character cells and decode before
a pixel-stable reading hold. label_scramble.py was implemented by the same
agent that viewed 240 consecutive reference frames and measured the transitions.

cluster_scene.py owns the projected camera, solid processors, rack faces,
retained trays, data paths and shared collider geometry. Network packets follow
authored cable paths; they are not a claim of simulated network hardware or
rigid-body dynamics. Processor articulation and camera movement are directed
curves. Their silhouettes and boundary velocities drive the actual fluid.

ACTUAL GPU PHYSICS
The cache was recomputed on an NVIDIA GeForce RTX 4070 Laptop GPU, using
CuPy 14.2.0 / CUDA. It retains 18,479 source identity slots and 403 samples
at 24 fps from 3.0 through 19.75 s. The v5 inlet enables 600 medium and
50 large glyphs; unused inlet IDs are never born and exert no contact mass.
The 2,899 terminal and 6,980 worker-source glyph identities are preserved. A 1536 x 432 incompressible velocity field
is integrated at 120 Hz with 3.75 px cells in a padded 5760 x 1620 domain.

The solver uses RK2 semi-Lagrangian advection, spectral pressure projection,
viscosity, vorticity confinement, moving-solid Brinkman coupling, signed-distance
contact normals and iterative oriented glyph-footprint constraints. The same visible RUN, processor,
rack, worker and step geometry is used for its collision masks, sampled at
120 Hz to match each physics step. When worker surface glyphs are born, the
former parent solid stops colliding with them. Terminal glyphs
remain anchored until contact, neighboring impacts or accumulated fluid drag
release them. V4 removes the obsolete diagonal jet and annular knot forces.
A leftward current carries the old field away during the camera's rack reveal.
Visible gaps below each rack support provide lateral outflow. The collision
geometry has those same openings; no invisible walls trap glyphs in bays.

Physical glyph identities, seed sizes and Menlo Bold face remain stable across
shots. Camera distance in later shots is expressed through contrast, not a
font substitution. glyph_material.py provides a single stable visibility sample.
Free glyphs collide across source groups using measured ink bounds and the
same four-degree angle rounding as the renderer. Pair constraints add no
padding to the printed ink; a 2.5 px broad-phase guard catches antialiased
rotated edges for the exact-raster contact check without adding physical bulk.
Iterative position constraints resolve contact after fluid advection. In
nonlinear rack corners, a bounded nearest-feasible-position projection checks
the exact rotated glyph bitmaps and updates the persistent position, velocity
and spatial bins at each 120 Hz step. Its displacement statistics are retained
in recovery_diagnostics. The renderer uses the audited 24 fps states exactly.
Glyph
boundaries are open: an outgoing ID does not wrap back into the picture.
There is one printed copy per simulated body, without ghost-exposure copies.

The blue substrate is exactly RGB (20,31,244) for zero-ink pixels; no film grain,
vertical streaks, bloom or color separation is applied.

The closing remnants inherit positions, velocities, size, glyph and angle
from selected late fluid IDs. They follow the exact contact-solved cache until
19.75 s, including the interval when neighbouring physical letters remain
visible. Damped motion and a growing outward force then carry
individual characters through the frame edges. They are not re-randomized.

RECOMPUTE THE COLLISION CACHE AND FLUID
Preserve the original input.npz and obstacles.npz. Prepare the v5 reservoir
and exact font footprints on the render host:
    python3 source/glyph_contact_geometry.py export \
      --input work/physics-v5/input.npz --output work/physics-v5/glyph-footprints.npz
    python3 source/prepare_physics_input.py \
      --input work/physics-v5/input.npz \
      --geometry work/physics-v5/glyph-footprints.npz \
      --output work/physics-v5/input-v5.npz
Then export the same geometry used in the movie:
    python3 source/export_physics_geometry.py --fps 120 \
      --output work/physics-v5/solid-masks-120hz.npz

On a CUDA/CuPy host, copy simulate_glyph_fluid.py, glyph_material.py,
opening_impact.py and the NPZ inputs used below, then run:
    python3 source/simulate_glyph_fluid.py \
      --input work/physics-v5/input-v5.npz \
      --obstacles work/physics-v5/obstacles.npz \
      --solid-masks work/physics-v5/solid-masks-120hz.npz \
      --glyph-footprints work/physics-v5/glyph-footprints.npz \
      --output work/physics-v5/flow.npz \
      --grid 1536 432 --dx 3.75 --hz 120 --fps 24 \
      --start 3.0 --end 19.75 --viscosity 11

Audit every recorded frame on the macOS render host with the actual font:
    python3 source/glyph_contact_geometry.py audit \
      --flow work/physics-v5/flow.npz \
      --geometry work/physics-v5/glyph-footprints.npz \
      --output work/physics-v5/contact-audit.json --raster

The raster test counts all nonzero antialiased ink pixels, with exactly the
same four-degree rotation and integer placement as the film. Oriented ink
rectangles remain a conservative broad phase; two empty rectangle corners
touching do not imply that the printed characters intersect.

Do not use --gather-reference or simulate_type_knot.py for v5. Those paths are
preserved for historical reconstruction only. CUDA versions/hardware may
produce small floating-point differences; baked caches are authoritative.

SOUND
make_audio_director.py synthesizes a 30 s, 48 kHz stereo PCM16 WAV. Opening
and closing keystrokes use terminal_timing.py's actual visible output frames.
There are no second-terminal, knot absorption, burst or mpirun-edit sounds in
v5. The new latch, network traffic, guides and foot contacts use cluster_scene.py
constants. Network pings are aligned with visible packet positions and stereo
pan follows the packet's screen position. Sound-event audits are written to
work/audio-sync-v5/. The reference soundtrack is never sampled.

SOURCE MAP
build_director.py           Strict preflight, audio, render and final mux
render_director.py          Overall shot composition and ASCII printing
cluster_scene.py            Continuous rack/network/worker deployment stage
worker_choreography.py       Shared worker positions, poses and barrier timing
ascii_solids.py              Opaque 3D ray intersections and surface illumination
hero_choreography.py         Earlier runner-to-chip fold
export_physics_geometry.py   Shared foreground and rack collision masks
simulate_glyph_fluid.py      Actual CUDA fluid and glyph dynamics
glyph_contact_geometry.py    Menlo ink footprint export and collision audit
analyze_glyph_motion.py      Motion residuals and actual recovery displacement
prepare_physics_input.py     Finite spaced inlet, preserving source identities
label_scramble.py            Inverse-cell assembly and fixed-position decode
terminal_timing.py           Shared visible text/audio event timeline
make_audio_director.py       Original synchronized audio synthesis
film_material.py            Flat blue/white color mapping
glyph_material.py           Stable physical glyph visibility
stair_choreography.py        Ballistic jumps, planted feet and exit
stair_platforms.py           Thick ASCII step faces and collider geometry
ambient_glyphs.py            Sparse persistent closing remnants

The supplied Coinbase.mp4 was examined directly for art direction. Its footage
and audio are not embedded, sampled or needed for a normal rebuild.
