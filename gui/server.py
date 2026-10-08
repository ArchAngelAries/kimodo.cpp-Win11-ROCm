#!/usr/bin/env python3
"""Local text-to-motion GUI for kimodo.cpp (ROCm build on Windows).

Serves the demo page in gui/web (adapted from demo/: ROCm labels, and "not installed"
for models that are missing) with the same JSON API as demo/main.go, and runs the native
worker (kmd-generate --server) for each request. The worker stays resident between
requests, so the weights are loaded once. Standard library only. Start it with
Kimodo-GUI.bat in the repository root.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import secrets
import struct
import subprocess
import sys
import threading
import webbrowser
from array import array
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = HERE / 'web'
SKELETONS = json.loads((HERE / 'skeletons.json').read_text(encoding='utf-8'))

FPS = 30
DEFAULT_FRAMES = 150
DEFAULT_STEPS = 100
DEFAULT_TRANSITION = 5
MAX_SEGMENTS = 16
MAX_BODY_BYTES = 32 << 10

NVIDIA_INTERNAL = ('https://www.nvidia.com/en-us/agreements/enterprise-software/'
                   'nvidia-internal-scientific-research-and-development-model-license/')
NVIDIA_OPEN = 'https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/'

# id, label, skeleton label, skeleton key, upstream, licence, licence URL, commercial, GGUF under weights/
MOTION_MODELS = [
    ('smplx-rp-v1', 'SMPL-X RP v1', 'SMPL-X 22 joints', 'smplx22', 'nvidia/Kimodo-SMPLX-RP-v1',
     'NVIDIA Internal Scientific R&D (non-commercial)', NVIDIA_INTERNAL, False,
     'models/kimodo-smplx-rp-v1-f32.gguf'),
    ('soma-rp-v1.1', 'SOMA RP v1.1', 'SOMA compact 30-joint control skeleton', 'soma30',
     'nvidia/Kimodo-SOMA-RP-v1.1', 'NVIDIA Open Model License', NVIDIA_OPEN, True,
     'models/kimodo-soma-rp-v1.1-f32.gguf'),
    ('soma-seed-v1.1', 'SOMA SEED v1.1', 'SOMA compact 30-joint control skeleton', 'soma30',
     'nvidia/Kimodo-SOMA-SEED-v1.1', 'NVIDIA Open Model License', NVIDIA_OPEN, True,
     'models/kimodo-soma-seed-v1.1-f32.gguf'),
    ('g1-rp-v1', 'G1 RP v1', 'Unitree G1 34 joints', 'g1skel34', 'nvidia/Kimodo-G1-RP-v1',
     'NVIDIA Open Model License', NVIDIA_OPEN, True, 'models/kimodo-g1-rp-v1-f32.gguf'),
    ('g1-seed-v1', 'G1 SEED v1', 'Unitree G1 34 joints', 'g1skel34', 'nvidia/Kimodo-G1-SEED-v1',
     'NVIDIA Open Model License', NVIDIA_OPEN, True, 'models/kimodo-g1-seed-v1-f32.gguf'),
]
DEFAULT_MODEL = 'soma-rp-v1.1'

# id, label, description, GGUF file under weights/ (tokenizer.gguf must sit beside it)
TEXT_BUNDLES = [
    ('bf16', 'BF16 reference', 'Highest-fidelity reference encoder.', 'Llama-3-Kimodo-BF16.gguf'),
    ('q8_0', 'Q8_0', 'Recommended quantized encoder; closest measured agreement with BF16.',
     'Llama-3-Kimodo-Q8_0.gguf'),
    ('q6_k', 'Q6_K', 'Smaller experimental encoder with increased motion divergence.',
     'Llama-3-Kimodo-Q6_K.gguf'),
    ('q5_k', 'Q5_K', 'Experimental; some prompt/noise pairs enter a different trajectory basin.',
     'Llama-3-Kimodo-Q5_K.gguf'),
    ('q4_k', 'Q4_K', 'Experimental uniform four-bit encoder with substantial measured divergence.',
     'Llama-3-Kimodo-Q4_K.gguf'),
    ('q4_k_m', 'Q4_K mixed', 'Experimental mixed-bit profile that improves on uniform Q4_K.',
     'Llama-3-Kimodo-Q4_K_M.gguf'),
]
DEFAULT_TEXT = 'q8_0'


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def read_f32(path: Path) -> array:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f'invalid F32 file: {path.name}')
    values = array('f')
    values.frombytes(data)
    return values


def write_skeleton_glb(path: Path, roots: array, rotations: array, skeleton: dict) -> None:
    """Node-only GLB with the same layout as the Go demo, plus glTF-required min/max."""
    parents, names, offsets = skeleton['parents'], skeleton['names'], skeleton['offsets']
    joints = len(parents)
    frames = len(roots) // 3
    if (frames < 1 or joints < 1 or len(names) != joints or len(offsets) != joints
            or len(roots) != frames * 3 or len(rotations) != frames * joints * 4):
        raise ValueError(f"invalid {skeleton['key']} motion for GLB export")
    times = array('f', [i / FPS for i in range(frames)])
    blob = bytearray()
    views: list[dict] = []

    def add_view(values: array) -> int:
        offset = len(blob)
        blob.extend(values.tobytes())
        views.append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(blob) - offset})
        return len(views) - 1

    time_view = add_view(times)
    root_view = add_view(roots)
    rotation_views = []
    for joint in range(joints):
        track = array('f', (rotations[(frame * joints + joint) * 4 + k]
                            for frame in range(frames) for k in range(4)))
        rotation_views.append(add_view(track))

    accessors = [
        {'bufferView': time_view, 'componentType': 5126, 'count': frames, 'type': 'SCALAR',
         'min': [times[0]], 'max': [times[-1]]},
        {'bufferView': root_view, 'componentType': 5126, 'count': frames, 'type': 'VEC3'},
    ]
    accessors += [{'bufferView': view, 'componentType': 5126, 'count': frames, 'type': 'VEC4'}
                  for view in rotation_views]

    nodes = []
    for joint in range(joints):
        node: dict = {'name': names[joint]}
        if joint != 0:
            node['translation'] = list(offsets[joint])
        children = [child for child, parent in enumerate(parents) if parent == joint]
        if children:
            node['children'] = children
        nodes.append(node)

    samplers, channels = [], []

    def add_channel(node: int, output: int, target: str) -> None:
        samplers.append({'input': 0, 'output': output, 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': node, 'path': target}})

    add_channel(0, 1, 'translation')
    for joint in range(joints):
        add_channel(joint, joint + 2, 'rotation')

    document = {
        'asset': {'version': '2.0', 'generator': 'kimodo.cpp skeleton exporter'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': nodes,
        'buffers': [{'byteLength': len(blob)}],
        'bufferViews': views,
        'accessors': accessors,
        'animations': [{'name': 'KimodoMotion', 'samplers': samplers, 'channels': channels}],
        'extras': {'skeleton': skeleton['key'], 'fps': FPS, 'rotation_order': 'xyzw'},
    }
    json_chunk = json.dumps(document, separators=(',', ':')).encode('utf-8')
    json_chunk += b' ' * (-len(json_chunk) % 4)
    blob += b'\x00' * (-len(blob) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(blob)
    out = bytearray(struct.pack('<4sII', b'glTF', 2, total))
    out += struct.pack('<I4s', len(json_chunk), b'JSON') + json_chunk
    out += struct.pack('<I4s', len(blob), b'BIN\x00') + blob
    path.write_bytes(bytes(out))


def export_skeleton_glb(directory: Path, skeleton: dict) -> None:
    roots = read_f32(directory / 'root_positions.f32')
    rotations = read_f32(directory / 'local_rotations_xyzw.f32')
    write_skeleton_glb(directory / 'animation.glb', roots, rotations, skeleton)


class Catalog:
    """Which motion models and text encoders exist under the weights folder."""

    def __init__(self, weights: Path):
        self.models: dict[str, dict] = {}
        self.model_paths: dict[str, Path] = {}
        for (model_id, label, skeleton_label, key, upstream, licence, licence_url,
             commercial, relative) in MOTION_MODELS:
            path = weights / relative
            skeleton = SKELETONS[key]
            entry = {'id': model_id, 'label': label, 'skeleton': skeleton_label,
                     'skeleton_key': key, 'upstream': upstream, 'license': licence,
                     'license_url': licence_url, 'commercial': commercial,
                     'available': path.is_file(), 'parents': skeleton['parents'],
                     'offsets': skeleton['offsets']}
            if not entry['available']:
                entry['reason'] = f'GGUF not found at {path}'
            self.models[model_id] = entry
            self.model_paths[model_id] = path
        self.texts: dict[str, dict] = {}
        self.text_paths: dict[str, Path] = {}
        for text_id, label, description, filename in TEXT_BUNDLES:
            path = weights / filename
            tokenizer = weights / 'tokenizer.gguf'
            entry = {'id': text_id, 'label': label, 'description': description,
                     'available': False}
            if not path.is_file():
                entry['reason'] = f'bundle not found at {path}'
            elif not tokenizer.is_file():
                entry['reason'] = f'tokenizer.gguf not found beside {path}'
            else:
                entry['available'] = True
                entry['bytes'] = path.stat().st_size + tokenizer.stat().st_size
            self.texts[text_id] = entry
            self.text_paths[text_id] = path

    def model_list(self) -> list[dict]:
        return [self.models[key] for key in sorted(self.models)]

    def text_list(self) -> list[dict]:
        return [self.texts[text[0]] for text in TEXT_BUNDLES]


class Session:
    """One native worker: kmd-generate --server MOTION TEXT, with the demo's settings."""

    def __init__(self, generator: Path, motion: Path, text: Path, key: str):
        env = os.environ.copy()
        env['KIMODO_TEXT_LAYER_CHUNK'] = '32'
        env['KIMODO_TEXT_RESIDENT_LIMIT_MIB'] = '10000'
        self.key = key
        self.process = subprocess.Popen([str(generator), '--server', str(motion), str(text)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def generate(self, item: dict, directory: Path, segments: list[dict], prompt_paths: list[Path]) -> None:
        fields = [str(item['transition_frames']), str(item['diffusion_steps']), str(item['seed']),
                  str(directory)]
        for segment, path in zip(segments, prompt_paths):
            fields += [str(segment['frames']), str(path)]
        try:
            self.process.stdin.write(('\t'.join(fields) + '\n').encode('utf-8'))
            self.process.stdin.flush()
            response = self.process.stdout.readline().decode('utf-8', 'replace').strip()
        except OSError as error:
            raise RuntimeError(f'native worker stopped: {error}') from error
        if not response:
            raise RuntimeError('native worker stopped')
        if response.startswith('ERR\t'):
            raise RuntimeError('native worker: ' + response[4:])
        if not response.startswith('OK\t'):
            raise RuntimeError(f'invalid native worker response {response!r}')

    def close(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()


class Gallery:
    """Saved animations (one JSON file each) plus a single generation worker thread."""

    def __init__(self, output: Path, catalog: Catalog, generator: Path):
        self.output = output
        self.catalog = catalog
        self.generator = generator
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.jobs: queue.Queue[str] = queue.Queue()
        self.session: Session | None = None
        output.mkdir(parents=True, exist_ok=True)
        self._load()
        threading.Thread(target=self._worker, name='generation-worker', daemon=True).start()

    def _load(self) -> None:
        for path in sorted(self.output.glob('*.json')):
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            item.setdefault('text_quantization', 'bf16')
            if item.get('status') in ('queued', 'running'):
                item['status'] = 'failed'
                item['error'] = 'interrupted before the animation finished'
                self.save(item)
            if item.get('status') == 'ready':
                model = self.catalog.models.get(item.get('model'))
                if model:
                    try:
                        export_skeleton_glb(self.output / item['id'], SKELETONS[model['skeleton_key']])
                    except (OSError, ValueError) as error:
                        print(f'[gui] could not rebuild GLB for {item["id"]}: {error}', flush=True)
            self.items[item['id']] = item

    def save(self, item: dict) -> None:
        text = json.dumps(item, indent=2, ensure_ascii=False)
        (self.output / f"{item['id']}.json").write_text(text, encoding='utf-8')

    def list(self) -> list[dict]:
        with self.lock:
            copies = [dict(item) for item in self.items.values()]
        return sorted(copies, key=lambda item: item['created_at'], reverse=True)

    def get(self, item_id: str) -> dict | None:
        with self.lock:
            item = self.items.get(item_id)
            return dict(item) if item else None

    def submit(self, item: dict) -> None:
        with self.lock:
            self.items[item['id']] = item
            self.save(item)
        self.jobs.put(item['id'])

    def _update(self, item_id: str, **changes) -> None:
        with self.lock:
            item = self.items[item_id]
            for key, value in changes.items():
                if value is None:
                    item.pop(key, None)
                else:
                    item[key] = value
            self.save(item)

    def _worker(self) -> None:
        while True:
            item_id = self.jobs.get()
            try:
                self._run(item_id)
                self._update(item_id, status='ready', progress=None)
                print(f'[gui] animation {item_id} ready', flush=True)
            except Exception as error:  # keep the worker alive whatever the failure
                self._update(item_id, status='failed', error=str(error), progress=None)
                print(f'[gui] animation {item_id} failed: {error}', flush=True)

    def _run(self, item_id: str) -> None:
        self._update(item_id, status='running')
        item = self.get(item_id)
        directory = self.output / item_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'prompt.txt').write_text(item['prompt'], encoding='utf-8')
        model = self.catalog.models.get(item['model'])
        if not model or not model['available']:
            raise RuntimeError(f"model {item['model']!r} is not available")
        text_id = item.get('text_quantization') or DEFAULT_TEXT
        text = self.catalog.texts.get(text_id)
        if not text or not text['available']:
            raise RuntimeError(f"text quantization {text_id!r} is not available")
        segments = item.get('segments') or [{'prompt': item['prompt'], 'frames': item['frames']}]
        prompt_paths = []
        for index, segment in enumerate(segments, start=1):
            path = directory / f'segment-{index:02d}.txt'
            path.write_text(segment['prompt'], encoding='utf-8')
            prompt_paths.append(path)
        self._update(item_id, progress=f'Generating {len(segments)} conditioned segments')
        motion_path = self.catalog.model_paths[item['model']]
        text_path = self.catalog.text_paths[text_id]
        key = f'{motion_path}\x00{text_path}'
        if self.session is None or self.session.key != key:
            if self.session is not None:
                self.session.close()
            self.session = None
            self.session = Session(self.generator, motion_path, text_path, key)
        try:
            self.session.generate(item, directory, segments, prompt_paths)
        except RuntimeError:
            self.session.close()
            self.session = None
            raise
        export_skeleton_glb(directory, SKELETONS[model['skeleton_key']])

    def shutdown(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None


def validate_request(body: dict, catalog: Catalog) -> tuple[dict | None, tuple[int, str] | None]:
    """Same rules and messages as the Go demo's /api/generate handler."""
    prompt = str(body.get('prompt') or '').strip()
    raw_segments = body.get('segments') or []
    frames = int(body.get('frames') or 0)
    steps = int(body.get('steps') or 0)
    transition = int(body.get('transition_frames') or 0)
    seed = body.get('seed') or 0
    if not isinstance(seed, int) or seed < 0:
        return None, (400, 'invalid JSON')
    if not raw_segments:
        raw_segments = [{'prompt': prompt, 'frames': frames}]
    if len(raw_segments) > MAX_SEGMENTS:
        return None, (400, f'at most {MAX_SEGMENTS} prompt segments')
    segments = []
    for raw in raw_segments:
        segment_prompt = str(raw.get('prompt') or '').strip()
        segment_frames = int(raw.get('frames') or 0) or DEFAULT_FRAMES
        if (not segment_prompt or len(segment_prompt.encode('utf-8')) > 4096
                or not 60 <= segment_frames <= 150):
            return None, (400, 'each prompt segment must be 60..150 frames and 1..4096 bytes')
        segments.append({'prompt': segment_prompt, 'frames': segment_frames})
    steps = steps or DEFAULT_STEPS
    transition = transition or DEFAULT_TRANSITION
    if not 1 <= transition <= 60 or not 1 <= steps <= 1000:
        return None, (400, 'transition frames must be 1..60 and steps 1..1000')
    model_id = str(body.get('model') or DEFAULT_MODEL)
    model = catalog.models.get(model_id)
    if not model or not model['available']:
        reason = model.get('reason', '') if model else 'unknown model'
        return None, (409, f'selected motion model is not available: {reason}')
    text_id = str(body.get('text_quantization') or DEFAULT_TEXT)
    text = catalog.texts.get(text_id)
    if not text:
        return None, (400, 'unknown text quantization')
    if not text['available']:
        return None, (409, f"selected text quantization is not available: {text.get('reason', '')}")
    item = {
        'id': secrets.token_hex(8),
        'prompt': segments[0]['prompt'],
        'frames': sum(segment['frames'] for segment in segments),
        'diffusion_steps': steps,
        'seed': seed,
        'created_at': utc_now(),
        'status': 'queued',
        'kind': 'generated',
        'model': model_id,
        'text_quantization': text_id,
        'segments': segments,
        'transition_frames': transition,
    }
    return item, None


class Handler(BaseHTTPRequestHandler):
    server_version = 'KimodoGUI/1.0'

    def log_message(self, format, *args):  # keep the console for start-up and job messages
        return

    @property
    def gallery(self) -> Gallery:
        return self.server.gallery

    @property
    def catalog(self) -> Catalog:
        return self.server.catalog

    def _bytes(self, status: int, data: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def _json(self, value, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self._bytes(status, data, 'application/json')

    def _error(self, status: int, message: str) -> None:
        self._bytes(status, (message + '\n').encode('utf-8'), 'text/plain; charset=utf-8')

    def _file(self, path: Path, content_type: str, headers: dict | None = None) -> None:
        if not path.is_file():
            self._error(404, 'not found')
            return
        self._bytes(200, path.read_bytes(), content_type, headers)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        no_store = {'Cache-Control': 'no-store'}
        if path == '/':
            self._file(WEB / 'index.html', 'text/html; charset=utf-8')
        elif path == '/compare':
            self._file(WEB / 'comparison.html', 'text/html; charset=utf-8', no_store)
        elif path == '/models.js':
            self._file(WEB / 'models.js', 'application/javascript; charset=utf-8', no_store)
        elif path == '/localai.png':
            self._file(ROOT / 'demo' / 'assets' / 'localai.png', 'image/png',
                       {'Cache-Control': 'public, max-age=86400'})
        elif path == '/api/models':
            self._json(self.catalog.model_list())
        elif path == '/api/text-quantizations':
            self._json(self.catalog.text_list())
        elif path == '/api/animations':
            self._json(self.gallery.list())
        elif path == '/api/comparisons':
            self._json(self._comparison_list())
        elif path.startswith('/api/comparisons/'):
            self._comparison_file(path[len('/api/comparisons/'):])
        elif path == '/api/generate':
            self._bytes(405, b'POST required\n', 'text/plain; charset=utf-8', {'Allow': 'POST'})
        elif path.startswith('/api/animations/'):
            self._animation_file(path[len('/api/animations/'):])
        else:
            self._error(404, 'not found')

    def do_POST(self) -> None:
        if urlsplit(self.path).path != '/api/generate':
            self._error(404, 'not found')
            return
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY_BYTES:
            self._error(400, 'invalid JSON')
            return
        try:
            body = json.loads(self.rfile.read(length).decode('utf-8'))
            if not isinstance(body, dict):
                raise ValueError('not an object')
            item, problem = validate_request(body, self.catalog)
        except (ValueError, TypeError):
            self._error(400, 'invalid JSON')
            return
        if problem:
            self._error(*problem)
            return
        self.gallery.submit(item)
        print(f"[gui] queued animation {item['id']} ({item['model']}, {item['text_quantization']}, "
              f"{item['frames']} frames, {item['diffusion_steps']} steps)", flush=True)
        self._json(item, 202)

    def _animation_file(self, rest: str) -> None:
        parts = rest.split('/')
        if len(parts) != 2 or parts[1] not in ('root.f32', 'rotations.f32', 'animation.glb'):
            self._error(404, 'not found')
            return
        item = self.gallery.get(parts[0])
        if not item or item.get('status') != 'ready':
            self._error(404, 'not found')
            return
        directory = self.gallery.output / item['id']
        if parts[1] == 'animation.glb':
            self._file(directory / 'animation.glb', 'model/gltf-binary',
                       {'Content-Disposition': f"attachment; filename=kimodo-{item['id']}.glb",
                        'Cache-Control': 'no-store'})
        else:
            name = 'root_positions.f32' if parts[1] == 'root.f32' else 'local_rotations_xyzw.f32'
            self._file(directory / name, 'application/octet-stream', {'Cache-Control': 'no-store'})

    def _comparison_root(self) -> Path:
        return self.server.comparisons

    def _comparison_list(self) -> list:
        root = self._comparison_root()
        result = []
        if root.is_dir():
            for entry in sorted(root.iterdir()):
                manifest = entry / 'comparison.json'
                if entry.is_dir() and _safe(entry.name) and manifest.is_file():
                    try:
                        result.append(json.loads(manifest.read_text(encoding='utf-8')))
                    except (OSError, json.JSONDecodeError):
                        continue
        return result

    def _comparison_file(self, rest: str) -> None:
        parts = rest.split('/')
        root = self._comparison_root()
        if len(parts) == 2 and parts[1] == 'comparison.json' and _safe(parts[0]):
            self._file(root / parts[0] / 'comparison.json', 'application/json', {'Cache-Control': 'no-store'})
        elif (len(parts) == 3 and _safe(parts[0]) and _safe(parts[1])
              and parts[2] in ('root_positions.f32', 'local_rotations_xyzw.f32')):
            self._file(root / parts[0] / parts[1] / parts[2], 'application/octet-stream',
                       {'Cache-Control': 'no-store'})
        else:
            self._error(404, 'not found')


def _safe(value: str) -> bool:
    return bool(value) and value not in ('.', '..') and all(
        ch.isascii() and (ch.isalnum() or ch in '-_.') for ch in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--addr', default='127.0.0.1:8094', help='listen address (default 127.0.0.1:8094)')
    parser.add_argument('--generator', default=str(ROOT / 'build' / 'rocm' / 'kmd-generate.exe'),
                        help='native text-to-motion program')
    parser.add_argument('--weights', default=str(ROOT / 'weights'), help='folder holding the GGUF files')
    parser.add_argument('--output', default=str(ROOT / 'gui-output'), help='saved animations folder')
    parser.add_argument('--comparisons', default=str(ROOT / 'quantization-comparisons'),
                        help='optional quantisation comparison folders')
    parser.add_argument('--no-browser', action='store_true', help='do not open the page automatically')
    args = parser.parse_args()

    host, _, port_text = args.addr.rpartition(':')
    generator = Path(args.generator)
    if not generator.is_file():
        print(f'[gui] native program not found: {generator}', flush=True)
        print('[gui] build it first (windows\\build-rocm.ps1) or pass --generator', flush=True)
        return 1
    catalog = Catalog(Path(args.weights))
    try:
        httpd = ThreadingHTTPServer((host or '127.0.0.1', int(port_text)), Handler)
    except OSError as error:
        print(f'[gui] cannot listen on {args.addr}: {error}', flush=True)
        print('[gui] is another Kimodo GUI already open?', flush=True)
        return 1
    httpd.gallery = Gallery(Path(args.output), catalog, generator)
    httpd.catalog = catalog
    httpd.comparisons = Path(args.comparisons)
    url = f'http://{host or "127.0.0.1"}:{port_text}/'
    print(f'[gui] Kimodo text to motion at {url}', flush=True)
    for model in catalog.model_list():
        state = 'ready' if model['available'] else 'not installed'
        print(f"[gui]   motion  {model['id']:<15} {state}", flush=True)
    for text in catalog.text_list():
        state = 'ready' if text['available'] else 'not installed'
        print(f"[gui]   text    {text['id']:<15} {state}", flush=True)
    print('[gui] close this window (or press Ctrl+C) to stop', flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.gallery.shutdown()
        httpd.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
