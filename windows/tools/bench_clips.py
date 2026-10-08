#!/usr/bin/env python3
"""Times kimodo.cpp clips through the same --server protocol the demo uses.

Clip 1 includes model load (cold). Later clips reuse the resident weights (warm).
Every clip uses the same prompt, seed, frame count and step count, so each clip
should reproduce clip 1 byte for byte on the same device.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', required=True, help='kmd-generate.exe for this build')
    ap.add_argument('--dll-dir', action='append', default=[], help='directories holding the build DLLs')
    ap.add_argument('--motion', required=True)
    ap.add_argument('--text', required=True, help='monolithic text GGUF (tokenizer.gguf beside it)')
    ap.add_argument('--prompt', required=True, help='prompt text file')
    ap.add_argument('--frames', type=int, default=150)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--clips', type=int, default=4)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--gpu-sample', action='store_true', help='record GPU counters for the process')
    ap.add_argument('--env', action='append', default=[], help='extra KEY=VALUE for the child')
    args = ap.parse_args()

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(args.dll_dir + [env.get('PATH', '')])
    env['KIMODO_PROFILE'] = '1'
    for pair in args.env:
        key, value = pair.split('=', 1)
        env[key] = value

    log = open(run / 'stderr.log', 'w', encoding='utf-8')
    wall_start = time.time()
    proc = subprocess.Popen([args.exe, '--server', args.motion, args.text],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env)
    sampler = None
    if args.gpu_sample:
        # typeperf writes one row per second for every GPU engine and adapter counter;
        # the post-run analysis sums the rows that belong to this process ID.
        sampler = subprocess.Popen(['typeperf', r'\GPU Engine(*)\Utilization Percentage',
                                    r'\GPU Adapter Memory(*)\Dedicated Usage',
                                    '-si', '1', '-f', 'CSV', '-o', str(run / 'gpu_typeperf.csv')],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

    clips = []
    for index in range(1, args.clips + 1):
        out = run / f'clip{index}'
        request = f"5\t{args.steps}\t{args.seed}\t{out}\t{args.frames}\t{args.prompt}\n"
        started = time.perf_counter()
        try:
            proc.stdin.write(request.encode('utf-8'))
            proc.stdin.flush()
            response = proc.stdout.readline().decode('utf-8', 'replace').strip()
        except OSError as error:
            response = f'SERVER_GONE ({error})'
        seconds = time.perf_counter() - started
        ok = response.startswith('OK')
        sha = None
        if ok:
            digest = hashlib.sha256()
            for name in ('root_positions.f32', 'local_rotations_xyzw.f32'):
                digest.update((out / name).read_bytes())
            sha = digest.hexdigest()
        clips.append({'clip': index, 'cold': index == 1, 'seconds': round(seconds, 3),
                      'response': response, 'sha256': sha})
        print(f"clip {index}: {seconds:.2f} s  {response}", flush=True)
        if not ok:
            break

    try:
        proc.stdin.close()
    except OSError:
        pass
    proc.wait(timeout=120)
    log.close()
    if proc.returncode != 0 or any(not c['response'].startswith('OK') for c in clips):
        tail = (run / 'stderr.log').read_text(encoding='utf-8', errors='replace').splitlines()[-6:]
        print('server failed; stderr tail:')
        print('\n'.join(tail))
    if sampler is not None:
        time.sleep(2)
        sampler.terminate()
        sampler.wait(timeout=30)

    identical = all(c['sha256'] == clips[0]['sha256'] for c in clips if c['sha256'])
    summary = {'exe': args.exe, 'pid': proc.pid, 'frames': args.frames, 'steps': args.steps,
               'seed': args.seed, 'prompt': args.prompt, 'exit_code': proc.returncode, 'clips': clips,
               'warm_outputs_match_cold': identical, 'env': args.env,
               'wall_start_epoch': wall_start, 'wall_end_epoch': time.time()}
    (run / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'warm_outputs_match_cold': identical, 'exit_code': proc.returncode}))
    return 0 if proc.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
