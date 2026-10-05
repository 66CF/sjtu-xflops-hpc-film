# SJTU Xflops HPC — ASCII Motion Film

30-second, 1920 × 1080, 24 fps promotional film for SJTU Xflops. All on-screen copy is English, with an original synthesized stereo soundtrack.

The current working cut is **v5**, rendered to `output/SJTU_Xflops_Cluster_v5.mp4`. The middle passage follows one chip into a rack, reveals a connected cluster with Rank 23 missing, and unfolds the same processors onto their retained worker trays. The absent rank arrives later to complete the collective. The v3 checkpoint is preserved at tag `v1.1.0`, with its film and exact physics assets in the release.


![Film preview](assets/preview.jpg)

**[Watch / download the film](https://github.com/66CF/sjtu-xflops-hpc-film/releases/download/v1.1.0/SJTU_Xflops_Physics_v3.mp4)** · **[Motion revision comparison (silent)](https://github.com/66CF/sjtu-xflops-hpc-film/releases/download/v1.0.0/SJTU_Xflops_Motion_Changes.mp4)** · **[All release assets](https://github.com/66CF/sjtu-xflops-hpc-film/releases/tag/v1.1.0)**

## Motion and physics

- A visible `RUN` block collides with terminal characters. Letters release through contact, neighbouring impacts and accumulated fluid drag.
- The runner retracts its limbs into the same chip, flips back to its X face, and accelerates right into the rack while `FASTER` decodes.
- One chip docks into a rack. The camera reveals 23 processors and a visibly empty Rank 23 socket, linked by flowing data packets; the rack guides withdraw while the same chips unfold into running workers.
- The empty Rank 23 tray persists into the worker grid. A late chip enters from the right, docks and unfolds before the collective becomes ready. Parallel chip workers synchronize; the leader turns right, crouches, jumps onto three platforms, lands, and runs along the top platform.
- White emphasis cells assemble from scrambled symbols and decode into blue lettering at stable reading positions.

The GPU cache was computed on an RTX 4070 Laptop GPU using CuPy/CUDA: 18,479 retained source IDs, a finite inlet of 600 medium and 50 large glyphs, a 1536 × 432 fluid grid and 120 Hz integration. Contact constraints use actual Menlo ink footprints across source groups; unused reservoir IDs exert no invisible contact mass. Character choreography and independent captions are authored animation, not dynamic rigid-body simulation.

## Rebuild on macOS

Install Python 3, NumPy, Pillow and FFmpeg (including FFprobe). The renderer uses the macOS Menlo font at `/System/Library/Fonts/Menlo.ttc`.

```bash
git clone --branch v1.1.0 https://github.com/66CF/sjtu-xflops-hpc-film.git
cd sjtu-xflops-hpc-film
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gh release download v1.1.0 --repo 66CF/sjtu-xflops-hpc-film --pattern physics-assets-v3.zip
unzip physics-assets-v3.zip
python3 source/build_director.py --check-only
python3 source/build_director.py
```

The v1.1.0 bundle reproduces the tagged v3 source. The current v5 source needs its freshly computed cluster collision cache in `work/physics-v5/`; the earlier annular-gather cache is rejected by preflight. **No GPU is needed to render baked caches.** The current output is `output/SJTU_Xflops_Cluster_v5.mp4`.

For CUDA recomputation and detailed production notes, see [source/README_PHYSICS.txt](source/README_PHYSICS.txt). The v5 sequence does not use the historical gather pass.

## Source map

| File | Purpose |
| --- | --- |
| `source/build_director.py` | Validate assets, render, synthesize audio and mux the final film |
| `source/render_director.py` | Final film composition and ASCII material |
| `source/ascii_solids.py` | Analytic 3D surfaces and camera sampling |
| `source/opening_impact.py` | Shared visible cursor and collision geometry |
| `source/chip_choreography.py` | Shared chip entrance and rightward exit |
| `source/stair_choreography.py` | Grounded jump poses, foot contacts and exit |
| `source/export_physics_geometry.py` | Bake collision masks from the same visible geometry |
| `source/simulate_glyph_fluid.py` | CUDA fluid, contact and particle solver |
| `source/simulate_type_knot.py` | Preserved v3 elastic-knot simulation (not used by v5) |
| `source/make_audio_director.py` | Original synthesized soundtrack |
| `source/film_material.py` | Flat blue/white color mapping |
| `source/terminal_timing.py` | Shared frame events for visible text and synchronized sound |
| `source/glyph_material.py` | Stable particle visibility and printed size |
| `source/glyph_contact_geometry.py` | Exact glyph ink footprints and contact validation |
| `source/prepare_physics_input.py` | Finite nonoverlapping inlet with preserved glyph identities |
| `source/label_scramble.py` | Reference-measured inverse-cell assembly and symbol decoding |
| `source/hero_choreography.py` | Continuous runner-to-chip fold and rightward exit |
| `source/stair_platforms.py` | Thick ASCII top/front/side faces and collision masks |
| `source/ambient_glyphs.py` | Persistent closing remnants |
| `source/cluster_scene.py` | Continuous docking, camera pullback, network and workstation deployment |
| `source/worker_choreography.py` | Shared worker layout and barrier timing |

The `render_kinetic.py`, `kinetic_geometry.py` and `kinetic_later.py` modules retain shared helpers and preview fallbacks. `build_director.py` is the current build entry point and requires the final caches.

## Reference

Art direction was informed by a supplied Coinbase reference film. That reference footage and audio are not included in this repository or the main film. The comparison release compares two original SJTU Xflops revisions, not the Coinbase reference.

Large generated media and caches are distributed through Releases rather than Git history. SHA-256 checksums are included in each release.
