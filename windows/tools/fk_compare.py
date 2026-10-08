#!/usr/bin/env python3
"""World-space joint error between two kimodo.cpp runs (stdlib only).

Forward kinematics uses the SOMA-30 rest offsets and parent list from the
repo's export_glb.py (loaded by path, not executed as __main__). Reports
world MPJPE (cm), root-relative MPJPE (cm) and root difference (cm), the same
kind of metric the project uses in benchmarks/quantization/RESULTS.md.
"""
import array
import importlib.util
import math
import statistics
import sys
from pathlib import Path

# windows/tools/ sits two levels below the repository root.
REPO = Path(__file__).resolve().parents[2]


def skeleton():
    spec = importlib.util.spec_from_file_location('export_glb', REPO / 'scripts' / 'export_glb.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    soma = module.SKELETONS['soma30']
    return soma['parents'], soma['offsets']


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, w = q
    px, py, pz = v
    # v' = q * (v, 0) * conj(q), expanded
    tx = 2 * (y * pz - z * py)
    ty = 2 * (z * px - x * pz)
    tz = 2 * (x * py - y * px)
    return (px + w * tx + (y * tz - z * ty),
            py + w * ty + (z * tx - x * tz),
            pz + w * tz + (x * ty - y * tx))


def load(run: Path):
    roots = array.array('f')
    roots.frombytes((run / 'root_positions.f32').read_bytes())
    rots = array.array('f')
    rots.frombytes((run / 'local_rotations_xyzw.f32').read_bytes())
    frames = len(roots) // 3
    joints = len(rots) // (frames * 4)
    return frames, joints, roots, rots


def world_positions(frames, joints, roots, rots, parents, offsets):
    out = []
    for f in range(frames):
        rot_world = [None] * joints
        pos_world = [None] * joints
        for j in range(joints):
            base = (f * joints + j) * 4
            local = (rots[base], rots[base + 1], rots[base + 2], rots[base + 3])
            p = parents[j]
            if p < 0:
                pos_world[j] = (roots[3 * f], roots[3 * f + 1], roots[3 * f + 2])
                rot_world[j] = local
            else:
                off = offsets[j]
                step = qrot(rot_world[p], off)
                pos_world[j] = (pos_world[p][0] + step[0], pos_world[p][1] + step[1], pos_world[p][2] + step[2])
                rot_world[j] = qmul(rot_world[p], local)
        out.append(pos_world)
    return out


def main(a_dir: str, b_dir: str) -> None:
    parents, offsets = skeleton()
    fa, ja, ra, qa = load(Path(a_dir))
    fb, jb, rb, qb = load(Path(b_dir))
    assert (fa, ja) == (fb, jb), 'shape mismatch'
    pa = world_positions(fa, ja, ra, qa, parents, offsets)
    pb = world_positions(fb, jb, rb, qb, parents, offsets)
    world, rel, root = [], [], []
    for f in range(fa):
        dists = [math.dist(pa[f][j], pb[f][j]) for j in range(ja)]
        world.append(statistics.mean(dists) * 100)
        ca, cb = pa[f][0], pb[f][0]
        rel_d = [math.dist(tuple(pa[f][j][k] - ca[k] for k in range(3)),
                           tuple(pb[f][j][k] - cb[k] for k in range(3))) for j in range(ja)]
        rel.append(statistics.mean(rel_d) * 100)
        root.append(math.dist(ca, cb) * 100)
    def stats(values):
        s = sorted(values)
        return {'mean': round(statistics.mean(s), 3), 'p95': round(s[int(0.95 * (len(s) - 1))], 3),
                'max': round(s[-1], 3)}
    print({'frames': fa, 'joints': ja,
           'world_mpjpe_cm': stats(world), 'root_relative_mpjpe_cm': stats(rel),
           'root_cm': stats(root)})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
