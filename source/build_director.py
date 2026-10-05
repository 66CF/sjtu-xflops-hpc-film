#!/usr/bin/env python3
"""Rebuild the physics director cut from its required baked simulation caches.

From the project root: python3 source/build_director.py
Preflight only:       python3 source/build_director.py --check-only
No GPU is needed for this normal build. Missing caches are fatal; this entry
point never substitutes an older cache or asks the renderer to invent paths.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import wave


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
WORK = ROOT / "work"
FINAL = ROOT / "output" / "SJTU_Xflops_Cluster_v5.mp4"
FONT = Path("/System/Library/Fonts/Menlo.ttc")


class BuildError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise BuildError(message)


def check_cache(path, position_key, frames, nodes, dimensions, time_start, time_end):
    import numpy as np

    require(path.is_file(),
            f"Required final physics cache is missing: {path}\n"
            "Restore it or follow source/README_PHYSICS.txt to recompute it. "
            "No older/analytic fallback will be used.")
    try:
        with np.load(path, allow_pickle=False) as data:
            expected = {"times", position_key, "velocity", "angle", "glyph", "size", "metadata"}
            if position_key == "xy":
                expected |= {"group", "release", "birth", "attached", "curl"}
            else:
                expected.add("source_ids")
            missing = expected - set(data.files)
            require(not missing, f"{path.name} is missing keys: {', '.join(sorted(missing))}")
            times = data["times"]
            if frames is None:
                frames = len(times)
                require(frames >= 2, f"{path.name}: insufficient physics time samples")
            if nodes is None:
                nodes = len(data["glyph"])
                require(0 < nodes <= 18479, f"{path.name}: invalid knot particle count")
            require(times.shape == (frames,), f"{path.name}: expected {frames} time samples, got {times.shape}")
            require(np.isfinite(times).all() and np.all(np.diff(times) > 0),
                    f"{path.name}: time samples must be finite and strictly increasing")
            if position_key == "xy":
                covers = abs(float(times[0])-time_start) < 2e-5 and abs(float(times[-1])-time_end) < 2e-5
            else:
                covers = float(times[0]) <= time_start+2e-5 and float(times[-1]) >= time_end-2e-5
            require(covers, f"{path.name}: must cover the full time range {time_start:g}–{time_end:g} seconds")
            require(np.allclose(np.diff(times), 1/24, atol=3e-6, rtol=0),
                    f"{path.name}: cache must contain 24 fps samples")
            for key in (position_key, "velocity"):
                array = data[key]
                require(array.shape == (frames, nodes, dimensions),
                        f"{path.name}/{key}: expected {(frames,nodes,dimensions)}, got {array.shape}")
                require(np.isfinite(array).all(), f"{path.name}/{key}: non-finite simulation values")
                del array
            angle = data["angle"]
            require(angle.shape == (frames,nodes) and np.isfinite(angle).all(),
                    f"{path.name}/angle: invalid shape or non-finite values")
            require(data["glyph"].shape == (nodes,), f"{path.name}: glyph count does not match {nodes}")
            sizes = data["size"]
            require(sizes.shape == (nodes,) and np.isfinite(sizes).all() and (sizes>0).all(),
                    f"{path.name}: invalid glyph sizes")
            if position_key == "xy":
                for key in ("group", "release", "birth"):
                    require(data[key].shape == (nodes,), f"{path.name}/{key}: invalid particle count")
                for key in ("attached", "curl"):
                    require(data[key].shape == (frames,nodes), f"{path.name}/{key}: invalid frame count")
            metadata = json.loads(str(data["metadata"]))
            if position_key == "xy":
                require(metadata.get("grid") == [1536,432] and
                        abs(metadata.get("dt",0)-1/120) < 1e-8,
                        "flow.npz is not the final 1536×432 / 120 Hz simulation")
                require(metadata.get("actual_silhouette_obstacles") is True and
                        "finite_size_contacts" in metadata,
                        "flow.npz is missing final articulated-collider/contact physics")
                require(metadata["finite_size_contacts"].get("same_group_only") is False and
                        "oriented" in metadata["finite_size_contacts"].get("model", ""),
                        "flow.npz needs actual glyph footprints and cross-group contacts")
                require(metadata.get("cluster_sequence",{}).get("version") == 5 and
                        metadata.get("annular_gather",{}).get("enabled") is False,
                        "flow.npz needs the v5 cluster colliders without the removed annular gather")
            else:
                source_ids = data["source_ids"]
                require(source_ids.shape == (nodes,) and source_ids.dtype.kind in "iu" and
                        len(np.unique(source_ids)) == nodes and
                        (source_ids >= 0).all() and (source_ids < 18479).all(),
                        "knot.npz must preserve unique, valid source_ids from the final GPU flow")
                require(metadata.get("simulation_hz") == 240 and metadata.get("no_position_lerp") is True,
                        "knot.npz is not the final 240 Hz mass-spring simulation")
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise BuildError(f"Cannot read required cache {path}: {error}") from error
    print(f"Verified {path.name}: {frames} frames / {nodes:,} particles", flush=True)
    return metadata


def preflight():
    for package in ("numpy", "PIL"):
        require(importlib.util.find_spec(package) is not None,
                "Python dependencies are missing. Install NumPy and Pillow in this interpreter.")
    for command in ("ffmpeg", "ffprobe"):
        require(shutil.which(command) is not None, f"Required executable is not on PATH: {command}")
    require(FONT.is_file(), f"Required Menlo font is missing: {FONT}\n"
            "This render configuration uses the macOS Menlo font collection.")
    for name in ("make_audio_director.py", "render_director.py", "render_kinetic.py",
                 "ascii_solids.py", "kinetic_geometry.py", "film_material.py",
                 "opening_impact.py", "chip_choreography.py", "stair_choreography.py",
                 "terminal_timing.py", "glyph_material.py", "hero_choreography.py",
                 "stair_platforms.py", "ambient_glyphs.py", "cluster_scene.py",
                 "worker_choreography.py", "label_scramble.py", "glyph_contact_geometry.py"):
        require((SOURCE/name).is_file(), f"Required source file is missing: {SOURCE/name}")
    flow = check_cache(WORK/"physics-v5/flow.npz", "xy", 403, 18479, 2, 3.0, 19.75)
    print(f"Baked fluid GPU: {flow.get('gpu','not recorded')}", flush=True)


def run(command):
    print("Running: " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def check_audio(path):
    with wave.open(str(path), "rb") as audio:
        require(audio.getnframes() == 1_440_000 and audio.getframerate() == 48_000 and
                audio.getnchannels() == 2 and audio.getsampwidth() == 2,
                "Director audio must be exactly 30 s, 48 kHz, stereo PCM16")


def check_movie(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames,sample_rate,channels",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    require(video.get("codec_name") == "h264" and video.get("width") == 1920 and
            video.get("height") == 1080 and video.get("r_frame_rate") == "24/1" and
            video.get("nb_frames") == "720", "Final video is not H.264 1080p / 24 fps / 720 frames")
    require(audio.get("codec_name") == "aac" and audio.get("sample_rate") == "48000" and
            audio.get("channels") == 2, "Final audio is not 48 kHz stereo AAC")
    require(abs(float(info["format"]["duration"])-30) <= .002, "Final movie duration is not 30.000 s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="validate dependencies and final caches without rendering or writing media")
    args = parser.parse_args()
    preflight()
    if args.check_only:
        print("Preflight passed. No media was changed.")
        return
    WORK.mkdir(exist_ok=True)
    FINAL.parent.mkdir(exist_ok=True)
    audio = WORK/"soundtrack-director.wav"
    silent = WORK/"director-silent.mp4"
    temporary = FINAL.with_name(FINAL.stem+".building.mp4")
    run([sys.executable, SOURCE/"make_audio_director.py"])
    check_audio(audio)
    # Required final caches were checked before the renderer can use any of
    # its interactive-preview fallbacks. Render settings are explicit.
    run([sys.executable, SOURCE/"render_director.py", "--start", "0", "--end", "30",
         "--width", "1920", "--output", silent])
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", silent, "-i", audio,
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
         "-ar", "48000", "-t", "30", "-movflags", "+faststart",
         "-metadata", "title=SJTU Xflops - Physics ASCII Film", temporary])
    check_movie(temporary)
    temporary.replace(FINAL)
    print(f"Verified final film: {FINAL}\n30.000 s / 1920×1080 / 24 fps / H.264 / AAC stereo 48 kHz")


if __name__ == "__main__":
    try:
        main()
    except (BuildError, subprocess.CalledProcessError, OSError, wave.Error) as error:
        print(f"Physics build failed: {error}", file=sys.stderr)
        sys.exit(1)
