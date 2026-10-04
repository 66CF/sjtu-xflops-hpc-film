SJTU XFLOPS — PHYSICS ASCII FILM

DELIVERY
Final movie: output/SJTU_Xflops_Physics_v2.mp4
30.000 seconds; 1920 x 1080; 24 fps / 720 video frames
H.264 video; AAC stereo audio, 48 kHz
All on-screen copy is English. The soundtrack is original synthesis.
The supplied Coinbase.mp4 informed art direction and timing; its footage and
audio are not used in the output and are not needed for a normal rebuild.

NORMAL REBUILD — USES THE BAKED PHYSICS CACHES
From the project root:

    python3 source/build_director.py --check-only
    python3 source/build_director.py

The entry point also works from another working directory. It locates the
project relative to its own file. Requirements: Python 3, NumPy, Pillow,
FFmpeg/FFprobe on PATH, and /System/Library/Fonts/Menlo.ttc (macOS Menlo).
No CUDA GPU is needed to render the already simulated caches.

Required final caches:
    work/physics-v5/flow.npz
    work/physics-v5/knot.npz

Missing, incomplete, incompatible or non-finite caches cause a clear error.
The build does not substitute physics-qa.npz, older caches, or analytic paths.
It checks both caches, synthesizes work/soundtrack-director.wav, renders
work/director-silent.mp4 at full resolution, and muxes the final movie.
The final file is replaced only after the new movie passes format/duration
checks. --check-only validates dependencies and caches without changing media.
The older source/build.py and source/README.txt describe the previous version.

WHAT WAS ACTUALLY SIMULATED

1. GPU fluid and moving glyphs
The final fluid cache was computed on an NVIDIA GeForce RTX 4070 Laptop GPU,
using CUDA through CuPy 14.2.0. This is a two-dimensional incompressible velocity
field on a 1536 x 432 grid, integrated at 120 Hz. Grid spacing is 3.75 screen
pixels, giving a padded 5760 x 1620 pixel physical domain. Its periodic boundary
lies outside the visible 1920 x 1080 aperture.

The solver uses RK2 semi-Lagrangian velocity advection, viscosity, vorticity
confinement and a spectral pressure projection. Moving articulated silhouettes
couple to the field with Brinkman boundary forcing. Actual 3D worker/chip poses
are projected into collision masks; particle contacts use signed distances,
surface normals, moving-boundary velocity and sliding friction.

18,479 glyph tracers have their own velocity, drag response, release time and
angular response to fluid curl. Finite-radius repulsion (4–8 px radii) and
damped contacts apply between particles in the same material group. The cache
contains 403 samples at 24 fps, spanning 3.0–19.75 s. Rendering selects material
samples and opacity for legibility: 18,479 is the simulated count, not a claim
that every particle is visible in every frame.

A CUDA gather continuation starts at 8.05 s and applies damped annular
attraction to the selected visible glyphs through 8.50 s. It preserves their
identities, velocity and roll and writes gather-handoff.npz for the spring
knot. The approved later fluid checkpoint resumes at 10 s under the terminal
shot.

2. Three-dimensional elastic data knot
The knot uses CPU NumPy, not the CUDA fluid solver: unequal-mass glyph nodes
connected by near-neighbor Hooke/dashpot bonds, integrated at 240 Hz. Its node
and bond counts follow the visible glyphs in the final GPU handoff and can
change when the preceding motion is revised; the cache records those counts.
At 8.50 s it inherits the selected GPU glyphs' identities, positions and
velocities. The source_ids array records this connection to flow.npz.
An inertial, torque-driven anchor frame gathers an irregular two-lobed shape.
Spatially phased rest strain, mass differences and elastic lag keep the knot
deforming as it rotates. Particle positions are advanced from forces and
velocities; they are not lerped onto a rotating target shape.

At 9.30 s the bonds/anchors release, and an upward/radial impulse is added to
the existing velocities. Drag and a weak updraft then carry the glyphs out.
The cache contains 29 samples at 24 fps covering 8.50–9.6667 s. It includes source
particle IDs, position, velocity, glyph, size and glyph roll in degrees.
Coordinates are world pixels with Y down; projection uses a 1200 px camera
distance.

WHAT REMAINS CHOREOGRAPHED
The worker/chip geometry is articulated 3D geometry, but its poses, travel,
camera scale and platform choreography are directed curves, not a general
rigid-body simulation. These moving bodies supply the fluid collision masks.
Terminal output, selection/edit timing, independently readable white/blue
labels, scene composition and the final typography are separately animated.
The film does not claim that every word or every body is physically simulated.

STORY AND SOUND
A visible RUN cursor crosses the terminal and pushes its characters into flow;
a worker and accelerator pass through the field, and the accelerator speeds
off the right edge before FASTER. A deforming data knot bursts upward. A command changes from
one process to six, and a camera pullback reveals 24 workers. Most reach the
barrier while rank 23 arrives last at 16.12 s; the shared X flash follows.
Surface letters dissolve into flow. The leader lands on three ASCII platforms
at 17.70, 18.33 and 18.96 s, then runs right along the upper platform. A quiet
terminal summary leads to the final line
and the SJTU Xflops / HPC / AI INFRA identity.

make_audio_director.py synthesizes a 30 s, 48 kHz stereo PCM16 soundtrack from
oscillators and seeded noise. It includes sparse typing, soft flow accents,
staggered synchronization taps, three landing contacts and a quiet brand cue.
The reference soundtrack is not sampled. The final mux encodes this WAV to AAC.

V2 MOTION REVISION
The opening RUN cursor is now the visible cause of the terminal displacement;
its drawing and collision geometry share opening_impact.py. The previous four
invisible fan sources are removed. Terminal group 0 releases dynamically from
cursor contact, its wake and neighboring glyph impulses; still-attached
terminal glyphs participate in collision. Three quiet contact/friction sounds
at 3.30, 3.60 and 4.00 s replace the previous broad rush.
The accelerator has a continuous accelerated exit through the right boundary,
shared with its collision masks through chip_choreography.py. Its existing
rightward audio accent is softer and occupies 7.38–7.70 s.
The stair hero faces screen-right, anticipates each jump, follows a ballistic
flight, compresses with planted soles on landing and exits along the top
platform. Drawing and masks share stair_choreography.py; landing audio is
synchronized to 17.70, 18.33 and 18.96 s. Film duration, output format and the
403-frame fluid sampling range remain unchanged. The previous movie is kept
as output/SJTU_Xflops_Physics.mp4; the build writes the v2 filename above.

RECOMPUTE THE KNOT
This works on the normal NumPy render host:

    python3 source/simulate_type_knot.py

The final work/physics-v5/gather-handoff.npz is required for its initial state;
preserve it together with flow.npz. The script replaces knot.npz and runs
finite-value, scale, ongoing-spin, non-rigid-deformation, near-plane and
departure checks.

RECOMPUTE THE GPU FLUID
This is separate from the normal build. Use a compatible NVIDIA CUDA host
with CuPy and its CUDA runtime / NVRTC / cuFFT dependencies installed. The
production cache records CuPy 14.2.0. This command cannot run on the macOS
render host without an NVIDIA CUDA execution environment.

The following authored simulation inputs must be preserved:
    work/physics-v5/input.npz
    work/physics-v5/obstacles.npz
    work/physics-v5/solid-masks.npz

input.npz contains seed glyph positions, identities, sizes, groups, births and
release times. Its preset group-0 release times are ignored by the causal
opening solver; the output flow.npz release array records the measured release
times. obstacles.npz contains the obstacle/scroll control track. These are
required input assets, not regenerated by build_director.py.

To update articulated collision masks after changing body choreography, run
on the render host and copy the resulting inputs to the CUDA host:

    python3 source/export_physics_geometry.py

From the project root on the CUDA host, first compute the base simulation:

    python3 source/simulate_glyph_fluid.py \
      --input work/physics-v5/input.npz \
      --obstacles work/physics-v5/obstacles.npz \
      --solid-masks work/physics-v5/solid-masks.npz \
      --output work/physics-v5/flow-before-gather.npz \
      --grid 1536 432 --dx 3.75 --hz 120 --fps 24 \
      --start 3.0 --end 19.75 --viscosity 11

Then continue the selected existing glyphs through the GPU gather:

    python3 source/simulate_glyph_fluid.py \
      --input work/physics-v5/input.npz \
      --obstacles work/physics-v5/obstacles.npz \
      --solid-masks work/physics-v5/solid-masks.npz \
      --gather-reference work/physics-v5/flow-before-gather.npz \
      --handoff-output work/physics-v5/gather-handoff.npz \
      --output work/physics-v5/flow.npz \
      --grid 1536 432 --dx 3.75 --hz 120 --fps 24 \
      --start 3.0 --end 19.75 --viscosity 11

Preserve the newly computed flow-before-gather.npz as the continuation's input.
It must contain the same RUN, chip-exit and stair choreography as this version;
an older baseline is incompatible even if its dimensions match. Copy flow.npz
and gather-handoff.npz back to the render project. Recompute the knot with
simulate_type_knot.py, then run the normal build. CUDA versions/hardware can
produce small floating-point differences;
the baked caches are the authoritative inputs for reproducing this delivery.
If body choreography or seed layout changes, update the corresponding mask
and seed inputs before recomputing. GPU recomputation is never implicit.

SOURCE MAP
build_director.py           Strict cache preflight, render, audio and final mux
render_director.py          Film composition, typography and cached physics playback
ascii_solids.py             Articulated worker/chip geometry and surface sampling
film_material.py            Subtle vertical grain, narrow bloom and display material
simulate_glyph_fluid.py     CUDA fluid, inertial tracers and finite-radius contacts
export_physics_geometry.py  Projected articulated masks and boundary velocities
simulate_type_knot.py       240 Hz 3D elastic-knot simulation
make_audio_director.py      Original synchronized soundtrack synthesis
render_kinetic.py           Shared terminal/timing utilities used by the director cut
kinetic_geometry.py         Shared geometry utilities
opening_impact.py           Shared visible RUN cursor and collision geometry
chip_choreography.py        Shared accelerator entrance and right-edge exit
stair_choreography.py       Right-facing jumps, planted soles and grounded exit

Keep the complete source directory and required work/physics-v5 caches and
inputs together. No external reference-media download is required.
