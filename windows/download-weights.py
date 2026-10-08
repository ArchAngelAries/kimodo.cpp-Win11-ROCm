#!/usr/bin/env python3
"""Download the Kimodo GGUF weights this fork uses, and check every file.

Run from anywhere. Paths are relative to the repository root:

    python windows/download-weights.py                        # SOMA RP v1.1 + Q8_0 text encoder
    python windows/download-weights.py --model soma-seed-v1.1
    python windows/download-weights.py --text-quantization bf16 --no-motion
    python windows/download-weights.py --verify-only          # check files already in weights/

Each file comes from a LocalAI-io Hugging Face repository. Its size and SHA-256 are compared
with the repository's MANIFEST.json, matched by exact path. Files are saved under weights/
with the layout the GUI and the command-line tools expect. A file that fails the check is
deleted.

Needs the huggingface_hub package:  python -m pip install huggingface_hub

Licences: the motion models are under the NVIDIA Open Model License. The text-encoder bundle
includes Meta Llama 3 material under the Meta Llama 3 Community License. Read both before use.
SMPL-X is not offered here: its licence allows internal research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ORG = 'LocalAI-io'
REPO_ROOT = Path(__file__).resolve().parent.parent
MOTION_MODELS = {
    'soma-rp-v1.1': ('Kimodo-SOMA-RP-v1.1-GGML', 'models/kimodo-soma-rp-v1.1-f32.gguf'),
    'soma-seed-v1.1': ('Kimodo-SOMA-SEED-v1.1-GGML', 'models/kimodo-soma-seed-v1.1-f32.gguf'),
    'g1-rp-v1': ('Kimodo-G1-RP-v1-GGML', 'models/kimodo-g1-rp-v1-f32.gguf'),
    'g1-seed-v1': ('Kimodo-G1-SEED-v1-GGML', 'models/kimodo-g1-seed-v1-f32.gguf'),
}
TEXT_REPO = 'Llama-3-Kimodo-GGML'
TEXT_QUANTIZATIONS = ('bf16', 'q8_0', 'q6_k', 'q5_k', 'q4_k', 'q4_k_m')
MANIFEST_FORMAT = 'kimodo-gguf-manifest-v1'


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_manifest(download, repo: str, output: Path) -> dict:
    folder = output / '.manifests' / repo
    path = Path(download(repo_id=f'{ORG}/{repo}', filename='MANIFEST.json', revision='main',
                         local_dir=folder))
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format') != MANIFEST_FORMAT:
        raise SystemExit(f'{repo}: unsupported manifest format {manifest.get("format")!r}')
    return manifest


def find_entry(manifest: dict, relative: str) -> dict:
    for entry in manifest.get('files', []):
        if entry.get('path') == relative:
            return entry
    raise SystemExit(f'{relative} is not listed in the manifest')


def is_good(target: Path, entry: dict) -> bool:
    return (target.is_file() and target.stat().st_size == entry['bytes']
            and sha256_of(target) == entry['sha256'])


def ensure_file(download, repo: str, relative: str, entry: dict, output: Path, verify_only: bool) -> None:
    target = output / relative
    if is_good(target, entry):
        print(f'ok        {relative}')
        return
    if verify_only:
        raise SystemExit(f'missing or does not match the manifest: {relative}')
    print(f'download  {relative} ({entry["bytes"] / 1e9:.2f} GB)')
    # Remove a damaged copy first: the hub may treat the cached record as current and skip the download.
    target.unlink(missing_ok=True)
    download(repo_id=f'{ORG}/{repo}', filename=relative, revision='main', local_dir=output)
    if not is_good(target, entry):
        target.unlink(missing_ok=True)
        raise SystemExit(f'checksum or size failed for {relative}; the file was removed. Try again.')
    print(f'verified  {relative}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Download and verify Kimodo GGUF weights.')
    parser.add_argument('--model', action='append', choices=sorted(MOTION_MODELS),
                        help='motion model to fetch (repeat for more); default soma-rp-v1.1')
    parser.add_argument('--no-motion', action='store_true', help='skip the motion models')
    parser.add_argument('--text-quantization', choices=TEXT_QUANTIZATIONS, default='q8_0',
                        help='text-encoder variant (default q8_0, the recommended one)')
    parser.add_argument('--no-text', action='store_true', help='skip the text encoder')
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'weights',
                        help='folder for the weights (default: weights/ in the repository)')
    parser.add_argument('--verify-only', action='store_true',
                        help='check the files already present; download nothing')
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('huggingface_hub is not installed. Run: python -m pip install huggingface_hub', file=sys.stderr)
        return 1

    output = args.output.resolve()
    if not args.no_motion:
        for model in args.model or ['soma-rp-v1.1']:
            repo, relative = MOTION_MODELS[model]
            manifest = fetch_manifest(hf_hub_download, repo, output)
            ensure_file(hf_hub_download, repo, relative, find_entry(manifest, relative), output, args.verify_only)
    if not args.no_text:
        manifest = fetch_manifest(hf_hub_download, TEXT_REPO, output)
        variant = manifest['variants'][args.text_quantization]
        for relative in (variant['weights'], variant['tokenizer']):
            ensure_file(hf_hub_download, TEXT_REPO, relative, find_entry(manifest, relative), output, args.verify_only)
    print(f'Done. Weights are in {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
