#!/usr/bin/env python3
"""Continue the GPU gather as an 8.5–9.667 s elastic ASCII knot.

Every node is an existing visible fluid glyph. The 8.5 s handoff supplies
identity, printed size, position, velocity and clockwise roll. Initial z=vz=0
makes projected position AND screen velocity exactly match the 2D fluid.
Nothing is respawned, decimated, position-interpolated or faded in.

At 240 Hz, local Hooke bonds and force anchors gather an organic shape around
an inertial, torque-driven rotor. Springs start at incoming distances; both
their preferred lengths and motor forces engage gradually. Unequal masses,
elastic lag and breathing forces cause actual non-rigid dynamics.
At 9.30 s bonds and anchors break. An upward/radial impulse is ADDED to the
existing velocity; subsequent departure uses only drag and a weak updraft.

Output: work/physics-v5/knot.npz. World units are pixels relative to (960,540),
y downward; p=1200/(1200+z), screen=(960+x*p,540+y*p). Global times are at
24 fps, xyz/velocity are [F,N,3], angle [F,N] is clockwise degrees. Glyph,
size, group and original source_ids are [N]; size is already printed size.
"""

from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "physics-v5" / "knot.npz"
HANDOFF = OUT.with_name("gather-handoff.npz")
HZ = 240
FPS = 24
DT = 1.0 / HZ
START = 204 / FPS
END = 232 / FPS
BREAK = 9.30


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def rotate_increment(omega, dt):
    """Rodrigues increment for angular velocity in world coordinates."""
    a = float(np.linalg.norm(omega)) * dt
    if a < 1e-10:
        return np.eye(3)
    x, y, z = omega / np.linalg.norm(omega)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(a)*k + (1-math.cos(a))*(k @ k)


def angular_order(points, bins=32):
    """Neighbour-preserving correspondence: angular sectors, then radius."""
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2*np.pi)
    radius = np.linalg.norm(points[:, :2], axis=1)
    sector = np.minimum((theta*bins/(2*np.pi)).astype(int), bins-1)
    return np.lexsort((theta, radius, sector))


def rest_constellation(rng, incoming):
    count = len(incoming)
    # A spatial figure-eight: crossing strands pass at different depths.
    u = (np.arange(count)+rng.uniform(-.28, .28, count))*(2*np.pi/count)
    phi = np.arange(count)*2.39996323+rng.normal(0, .12, count)
    c = np.column_stack((126*np.sin(2*u), 160*np.sin(u), 82*np.cos(u)))
    tangent = unit(np.column_stack((252*np.cos(2*u), 160*np.cos(u), -82*np.sin(u))))
    normal = unit(np.cross(tangent, np.broadcast_to([0., 0., 1.], c.shape)))
    binormal = unit(np.cross(tangent, normal))
    radius = (58+15*np.cos(3*u+.6)+10*np.sin(5*u))*rng.uniform(.55, 1., count)**.35
    tube = normal*(radius*np.cos(phi))[:, None]
    tube += binormal*(radius*.80*np.sin(phi))[:, None]
    rest = c+tube+rng.normal(0, 2.5, (count, 3))
    rest -= rest.mean(axis=0)
    # A larger volume keeps all inherited letters without white front clumps.
    rest *= 305/np.percentile(np.linalg.norm(rest, axis=1), 97)
    rest[:, 2] *= 1.20
    # Match nearby projected sectors rather than random IDs. These are only
    # force anchors: the incoming position/velocity are never overwritten.
    assignment = np.empty(count, dtype=int)
    assignment[angular_order(incoming)] = angular_order(rest)
    return rest[assignment], u[assignment], phi[assignment]


def local_bonds(points):
    """Eight local neighbours; bounded memory for the ~2305 real glyphs."""
    count = len(points)
    neighbours = min(8, count-1)
    near = np.empty((count, neighbours), dtype=np.int32)
    for start in range(0, count, 192):
        stop = min(start+192, count)
        d = points[start:stop, None, :]-points[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", d, d)
        d2[np.arange(stop-start), np.arange(start, stop)] = np.inf
        near[start:stop] = np.argpartition(d2, neighbours-1, axis=1)[:, :neighbours]
    i = np.repeat(np.arange(count), neighbours)
    pairs = np.unique(np.sort(np.column_stack((i, near.ravel())), axis=1), axis=0)
    lengths = np.linalg.norm(points[pairs[:, 1]]-points[pairs[:, 0]], axis=1)
    return pairs.astype(np.int32), lengths


def simulate():
    rng = np.random.default_rng(510824)
    with np.load(HANDOFF, allow_pickle=False) as data:
        incoming = {key: data[key] for key in data.files}
    source_ids = incoming["source_ids"].astype(np.int64)
    count = len(source_ids)
    assert count > 8 and len(np.unique(source_ids)) == count
    incoming_xy = np.asarray(incoming["xy"], dtype=np.float64)
    incoming_velocity = np.asarray(incoming["velocity"], dtype=np.float64)
    incoming_angle = np.asarray(incoming["angle"], dtype=np.float64)
    size = np.asarray(incoming["size"], dtype=np.float64)
    glyph = incoming["glyph"].astype("<U1")
    group = incoming["group"]
    assert incoming_xy.shape == incoming_velocity.shape == (count, 2)
    assert incoming_angle.shape == size.shape == glyph.shape == (count,)
    for key in ("time", "t", "handoff_time"):
        if key in incoming:
            assert abs(float(np.asarray(incoming[key]).ravel()[0])-START) < 1e-6

    # This exact planar initial state preserves perspective position and its
    # derivative. Incoming printed size and selection are already final.
    x = np.column_stack((incoming_xy-[960., 540.], np.zeros(count)))
    v = np.column_stack((incoming_velocity, np.zeros(count)))
    rest, u, phi = rest_constellation(rng, x)
    bonds, incoming_lengths = local_bonds(x)
    bi, bj = bonds[:, 0], bonds[:, 1]
    target_lengths = np.linalg.norm(rest[bj]-rest[bi], axis=1)
    mass = np.maximum(.28, (size/22)**1.15)
    inv_mass = 1/mass

    rotation = np.eye(3)
    relative_x = x-np.average(x, axis=0, weights=mass)
    relative_v = v-np.average(v, axis=0, weights=mass)
    inertia_z = np.sum(mass*np.sum(relative_x[:, :2]**2, axis=1))
    omega_z = np.sum(mass*np.cross(relative_x, relative_v)[:, 2])/inertia_z
    omega = np.array([0., 0., omega_z])
    angle = np.radians(incoming_angle)
    # Actual roll derivative from the same particles, never random angles.
    if "angular_velocity" in incoming:
        roll_velocity = np.radians(incoming["angular_velocity"].astype(np.float64))
    else:
        with np.load(HANDOFF.with_name("flow.npz"), allow_pickle=False) as flow:
            j = int(np.argmin(abs(flow["times"]-START)))
            if j:
                delta = flow["angle"][j, source_ids]-flow["angle"][j-1, source_ids]
                roll_velocity = np.radians((delta+180)%360-180)/(flow["times"][j]-flow["times"][j-1])
            else:
                roll_velocity = np.zeros(count)
    broken = False
    positions, velocities, angles, roll_rates, times, diagnostic = [], [], [], [], [], []
    steps = round((END-START)*HZ)
    release_before = release_after = None

    for step in range(steps+1):
        t = START+step*DT
        if step % (HZ//FPS) == 0:
            times.append(t)
            positions.append(x.copy())
            velocities.append(v.copy())
            angles.append(np.degrees(angle.copy()))
            roll_rates.append(np.degrees(roll_velocity.copy()))
            radius = np.linalg.norm(x-x.mean(axis=0), axis=1)
            diagnostic.append([t, np.percentile(radius, 97), np.sqrt(np.mean(v*v)),
                               float(np.linalg.norm(omega))])
        if step == steps:
            break
        if t >= BREAK-1e-9 and not broken:
            release_before = v.copy()
            radial = unit(x-np.average(x, axis=0, weights=mass))
            impulse = radial*np.array([1180., 880., 880.])
            impulse[:, 0] += 200*np.sin(u*5)
            impulse[:, 1] -= 4650+250*np.sin(phi)
            impulse[:, 2] += 150
            v += impulse
            release_after = v.copy()
            roll_velocity += 2.2*np.sin(phi)+.6*np.cos(u*3)
            broken = True

        if not broken:
            elapsed = t-START
            drive = np.array([.85+.28*math.sin(elapsed*5.2),
                              2.7+.80*math.sin(elapsed*5.0+.6),
                              -.72+.28*math.sin(elapsed*4.5)])
            omega += (4.8*(drive-omega)-.10*omega)*DT
            rotation = rotate_increment(omega, DT) @ rotation
            phase = elapsed*8.7
            strain = 1+.055*np.sin(phase+u*2.2)+.025*np.cos(phase*.65-phi)
            local = rest*strain[:, None]
            local[:, 2] += 13*np.sin(phase*.8+u*3)
            target = np.einsum("ij,kj->ik", local, rotation)
            target_velocity = np.cross(np.broadcast_to(omega, target.shape), target)
            capture = math.exp(-elapsed/.20)
            engage = float(smooth(elapsed/.10))
            anchor_k = (118+160*capture)*engage
            anchor_damp = (10+20*capture)*engage
            force = anchor_k*(target-x)+anchor_damp*np.sqrt(mass)[:, None]*(target_velocity-v)

            delta = x[bj]-x[bi]
            dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
            direction = delta/dist[:, None]
            relative_speed = np.einsum("ij,ij->i", v[bj]-v[bi], direction)
            lengths = incoming_lengths+(target_lengths-incoming_lengths)*smooth(elapsed/.32)
            bond_k = 10+62*float(smooth(elapsed/.24))
            tension = (bond_k*(dist-lengths)+2.7*relative_speed)*engage
            edge_force = direction*tension[:, None]
            for axis in range(3):
                force[:, axis] += np.bincount(bi, edge_force[:, axis], minlength=count)
                force[:, axis] -= np.bincount(bj, edge_force[:, axis], minlength=count)
            acceleration = force*inv_mass[:, None]
            desired_roll = omega[2]+.30*np.sin(u*3+phase*.6)
            roll_velocity += (desired_roll-roll_velocity)*5.2*inv_mass*engage*DT
        else:
            acceleration = -v*.48
            acceleration[:, 1] -= 420
            roll_velocity *= math.exp(-.42*DT)
        v += acceleration*DT
        x += v*DT
        angle += roll_velocity*DT
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"non-finite physics state at t={t:.6f}")

    xyz = np.asarray(positions, dtype=np.float32)
    velocity = np.asarray(velocities, dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    angle_frames = np.asarray(angles, dtype=np.float32)
    angular_velocity_frames = np.asarray(roll_rates, dtype=np.float32)
    reference = xyz[0].astype(np.float64)
    reference -= np.average(reference, axis=0, weights=mass)
    physical_motion = []
    for at in [8.5, 8.75, 9.0, 9.25]:
        j = int(np.argmin(abs(times-at)))
        r = xyz[j].astype(np.float64)
        r -= np.average(r, axis=0, weights=mass)
        vrel = velocity[j]-np.average(velocity[j], axis=0, weights=mass)
        inertia = np.eye(3)*np.sum(mass*np.sum(r*r, axis=1))
        inertia -= np.einsum("n,ni,nj->ij", mass, r, r)
        momentum = np.sum(mass[:, None]*np.cross(r, vrel), axis=0)
        physical_omega = np.linalg.solve(inertia, momentum)
        aa, ss, bb = np.linalg.svd(np.einsum("ni,nj->ij", reference, r))
        # einsum avoids the platform BLAS warning path for this tiny SVD fit.
        fitted = np.einsum("ni,ij->nj", reference, np.einsum("ij,jk->ik", aa, bb))
        residual = np.sqrt(np.mean(np.sum((fitted-r)**2, axis=1)))
        physical_motion.append({"time": float(times[j]),
                                "angular_speed_rad_s": float(np.linalg.norm(physical_omega)),
                                "nonrigid_rms_from_8_5_px": float(residual)})
    p = 1200/(1200+xyz[-1, :, 2])
    sx, sy = 960+xyz[-1, :, 0]*p, 540+xyz[-1, :, 1]*p
    visible = (sx>-70)&(sx<1990)&(sy>-70)&(sy<1150)&(1200+xyz[-1, :, 2]>80)
    report = {
        "method": "exact GPU glyph continuation; 240Hz damped mass-spring; inertial rest rotor",
        "nodes": count, "bonds": len(bonds), "simulation_hz": HZ, "cache_fps": FPS,
        "frames": len(times), "time_range": [float(times[0]), float(times[-1])],
        "break_time": BREAK, "coordinates": "world pixels, y down, center 0; camera distance 1200",
        "last_frame_visible_nodes": int(visible.sum()),
        "finite": bool(np.isfinite(xyz).all() and np.isfinite(velocity).all()),
        "max_speed": float(np.linalg.norm(velocity, axis=2).max()),
        "rest_nominal_radius_px": 305, "rest_depth_multiplier": 1.20,
        "release_retains_velocity_error": float(np.max(abs(release_after-release_before-impulse))),
        "diagnostic_columns": ["time", "97pct_radius_about_com", "velocity_component_rms", "rotor_speed_rad_s"],
        "diagnostics": diagnostic, "no_position_lerp": True,
        "measured_particle_motion": physical_motion,
        "handoff": {
            "path": str(HANDOFF.relative_to(ROOT)), "time": START,
            "position_max_error_px": float(np.max(abs(xyz[0, :, :2]+[960, 540]-incoming_xy))),
            "velocity_max_error_px_s": float(np.max(abs(velocity[0, :, :2]-incoming_velocity))),
            "angle_max_error_deg": float(np.max(abs(angle_frames[0]-incoming_angle))),
            "size_max_error_px": float(np.max(abs(size-incoming["size"]))),
            "identical_glyphs": bool(np.array_equal(glyph, incoming["glyph"].astype("<U1"))),
            "source_ids_unique": bool(len(np.unique(source_ids)) == count),
            "initial_depth_zero": bool(np.all(xyz[0, :, 2] == 0) and np.all(velocity[0, :, 2] == 0)),
            "selection_reapplied": False,
        },
    }
    # Validate the handoff against the exact same frame in the published GPU
    # cache as well. This catches a stale intermediate file, ID mismatch or
    # accidental second application of the print-material scale.
    flow_path = HANDOFF.with_name("flow.npz")
    if flow_path.exists():
        with np.load(flow_path, allow_pickle=False) as flow:
            source_times = flow["times"]
            j = int(np.argmin(abs(source_times-START)))
            assert abs(float(source_times[j])-START) < 1e-6
            match = {
                "time": float(source_times[j]),
                "position_max_error_px": float(np.max(abs(incoming_xy-flow["xy"][j, source_ids]))),
                "velocity_max_error_px_s": float(np.max(abs(incoming_velocity-flow["velocity"][j, source_ids]))),
                "angle_max_error_deg": float(np.max(abs(incoming_angle-flow["angle"][j, source_ids]))),
                "print_size_max_error_px": float(np.max(abs(size-flow["size"][source_ids]))),
                "identical_glyphs": bool(np.array_equal(glyph, flow["glyph"][source_ids])),
                "identical_groups": bool(np.array_equal(group, flow["group"][source_ids])),
            }
            report["gpu_source_match"] = match
            assert match["position_max_error_px"] < .001
            assert match["velocity_max_error_px_s"] < .001
            assert match["angle_max_error_deg"] < .001
            assert match["print_size_max_error_px"] < .001
            assert match["identical_glyphs"] and match["identical_groups"]
    expected_frames = round((END-START)*FPS)+1
    assert report["finite"] and xyz.shape == (expected_frames, count, 3)
    assert visible.sum() == 0, "burst has not cleared the screen by the last frame"
    assert np.min(1200+xyz[:, :, 2]) > 80, "near-plane singularity"
    for at in [8.75, 9.0, 9.25]:
        radius97 = diagnostic[int(np.argmin(abs(times-at)))][1]
        assert 210 < radius97 < 440, "elastic knot volume collapsed or exploded"
    assert min(p["angular_speed_rad_s"] for p in physical_motion[2:]) > .5, "knot spin stopped"
    assert max(p["nonrigid_rms_from_8_5_px"] for p in physical_motion) > 10, "shape is only rigid rotation"
    assert report["handoff"]["position_max_error_px"] < .001
    assert report["handoff"]["velocity_max_error_px_s"] < .001
    assert report["handoff"]["angle_max_error_deg"] < .001
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, times=times, xyz=xyz, velocity=velocity,
                        source_ids=source_ids, group=group, glyph=glyph, size=size.astype(np.float32),
                        angle=angle_frames, angular_velocity=angular_velocity_frames,
                        bonds=bonds, rest=rest.astype(np.float32),
                        metadata=np.array(json.dumps(report)))
    print(json.dumps({k: v for k, v in report.items() if k not in ("diagnostics", "diagnostic_columns")}, indent=2))
    for at in [8.5, 8.75, 9.0, 9.25, 9.3333, 9.625, 9.6667]:
        j = int(np.argmin(abs(times-at)))
        print("t %.4f | radius97 %.1f px | velocity RMS %.1f px/s | rotor %.2f rad/s" % tuple(diagnostic[j]))
    print(OUT)


if __name__ == "__main__":
    simulate()
