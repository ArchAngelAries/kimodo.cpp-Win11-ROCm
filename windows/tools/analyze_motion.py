#!/usr/bin/env python3
"""Checks kimodo.cpp raw motion output and compares two runs.

root_positions.f32       float32 [T, 3]     (metres, Y up)
local_rotations_xyzw.f32 float32 [T, J, 4]  (local joint quaternions, xyzw)
"""
import array
import json
import math
import sys
from pathlib import Path


def load(run: Path):
    roots = array.array('f')
    roots.frombytes((run / 'root_positions.f32').read_bytes())
    rots = array.array('f')
    rots.frombytes((run / 'local_rotations_xyzw.f32').read_bytes())
    frames = len(roots) // 3
    if frames == 0 or len(rots) % (frames * 4):
        raise SystemExit(f'{run}: inconsistent sizes')
    joints = len(rots) // (frames * 4)
    return frames, joints, roots, rots


def quat_angle_deg(a, b):
    dot = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def summarise(run: Path):
    frames, joints, roots, rots = load(run)
    finite = all(math.isfinite(v) for v in roots) and all(math.isfinite(v) for v in rots)
    norms = []
    for i in range(0, len(rots), 4):
        norms.append(math.sqrt(sum(rots[i + k] ** 2 for k in range(4))))
    path_m = 0.0
    for f in range(1, frames):
        dx = roots[3 * f] - roots[3 * (f - 1)]
        dz = roots[3 * f + 2] - roots[3 * (f - 1) + 2]
        path_m += math.hypot(dx, dz)
    ys = [roots[3 * f + 1] for f in range(frames)]
    return {
        'frames': frames, 'joints': joints, 'seconds_at_30fps': round(frames / 30.0, 3),
        'all_finite': finite,
        'quat_norm_min': round(min(norms), 6), 'quat_norm_max': round(max(norms), 6),
        'root_path_m': round(path_m, 3),
        'root_height_m_min_max': [round(min(ys), 3), round(max(ys), 3)],
    }


def compare(a: Path, b: Path):
    fa, ja, ra, qa = load(a)
    fb, jb, rb, qb = load(b)
    if (fa, ja) != (fb, jb):
        return {'shape_mismatch': [[fa, ja], [fb, jb]]}
    root_max = max(abs(x - y) for x, y in zip(ra, rb))
    angles = []
    worst = 0.0
    for f in range(fa):
        for j in range(ja):
            base = (f * ja + j) * 4
            ang = quat_angle_deg(qa[base:base + 4], qb[base:base + 4])
            angles.append(ang)
            worst = max(worst, ang)
    angles.sort()
    mean = sum(angles) / len(angles)
    p99 = angles[int(0.99 * (len(angles) - 1))]
    return {'root_max_abs_diff_m': float(f'{root_max:.6g}'),
            'rotation_mean_deg': round(mean, 4), 'rotation_p99_deg': round(p99, 4),
            'rotation_max_deg': round(worst, 4), 'byte_identical': a.joinpath('root_positions.f32').read_bytes() ==
            b.joinpath('root_positions.f32').read_bytes() and a.joinpath('local_rotations_xyzw.f32').read_bytes() ==
            b.joinpath('local_rotations_xyzw.f32').read_bytes()}


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print(json.dumps(summarise(Path(sys.argv[1])), indent=2))
    elif len(sys.argv) == 3:
        print(json.dumps(compare(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
    else:
        raise SystemExit('usage: analyze_motion.py RUN_DIR [OTHER_RUN_DIR]')
