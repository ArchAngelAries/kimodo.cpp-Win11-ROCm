#!/usr/bin/env python3
"""Structural checks for the exported GLB and BVH files (stdlib only, no engine import)."""
import json
import math
import struct
import sys
from pathlib import Path

COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COUNT = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def check_glb(path: Path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<4sII', data, 0)
    assert magic == b'glTF' and version == 2 and length == len(data), 'bad GLB header'
    json_len, json_type = struct.unpack_from('<I4s', data, 12)
    assert json_type == b'JSON'
    gltf = json.loads(data[20:20 + json_len].decode('utf-8'))
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from('<I4s', data, bin_off)
    assert bin_type.startswith(b'BIN')
    blob = data[bin_off + 8: bin_off + 8 + bin_len]
    for view in gltf['bufferViews']:
        assert view['byteOffset'] + view['byteLength'] <= len(blob), 'bufferView outside BIN'
    nodes = gltf['nodes']
    report = {'asset': gltf['asset']['generator'], 'nodes': len(nodes),
              'node_names_first_5': [n.get('name') for n in nodes[:5]],
              'skins': [{'joints': len(s['joints'])} for s in gltf.get('skins', [])],
              'animations': [], 'accessors': len(gltf['accessors'])}
    for anim in gltf.get('animations', []):
        channels = anim['channels']
        targets = {}
        for ch in channels:
            assert 0 <= ch['target']['node'] < len(nodes), 'channel targets missing node'
            sampler = anim['samplers'][ch['sampler']]
            inp = gltf['accessors'][sampler['input']]
            out = gltf['accessors'][sampler['output']]
            assert inp['count'] == out['count'], 'sampler input/output count mismatch'
            targets.setdefault(ch['target']['path'], 0)
            targets[ch['target']['path']] += 1
            if ch['target']['path'] == 'rotation':
                # every rotation key must be a unit quaternion
                view = gltf['bufferViews'][out['bufferView']]
                floats = struct.unpack_from('<%df' % (out['count'] * 4), blob, view['byteOffset'])
                worst = max(abs(math.sqrt(sum(floats[i + k] ** 2 for k in range(4))) - 1.0)
                            for i in range(0, len(floats), 4))
                assert worst < 1e-3, f'non-unit quaternion, worst error {worst}'
        times = gltf['accessors'][anim['samplers'][0]['input']]
        report['animations'].append({'name': anim.get('name'), 'channels': len(channels),
                                     'paths': targets, 'keys': times['count'],
                                     'duration_s': round(times['max'][0] - times['min'][0], 4)})
    return report


def check_bvh(path: Path):
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0].strip() == 'HIERARCHY', 'missing HIERARCHY'
    joints, channels_total, motion_at = [], 0, None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(('ROOT ', 'JOINT ')):
            joints.append(s.split()[1])
        if s.startswith('CHANNELS '):
            channels_total += int(s.split()[1])
        if s == 'MOTION':
            motion_at = i
            break
    assert motion_at is not None, 'missing MOTION'
    frames = int(lines[motion_at + 1].split()[1])
    frame_time = float(lines[motion_at + 2].split()[2])
    rows = [r for r in lines[motion_at + 3:] if r.strip()]
    assert len(rows) == frames, f'{len(rows)} data rows for {frames} frames'
    values = [len(r.split()) for r in rows]
    assert all(v == channels_total for v in values), 'value count != channel count'
    return {'joints': len(joints), 'channels': channels_total, 'frames': frames,
            'frame_time_s': frame_time, 'fps': round(1 / frame_time, 3), 'duration_s': round(frames * frame_time, 4)}


if __name__ == '__main__':
    target = Path(sys.argv[1])
    result = check_glb(target) if target.suffix.lower() == '.glb' else check_bvh(target)
    print(json.dumps(result, indent=2))
