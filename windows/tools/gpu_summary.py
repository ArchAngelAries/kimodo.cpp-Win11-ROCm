#!/usr/bin/env python3
"""Summarises typeperf GPU counters for one kimodo.cpp run (stdlib only).

Engine utilisation is summed over this process's GPU engines (columns named
pid_<pid>_..._engtype_<X>) for every 1 s row inside the run window. Adapter
memory is the sum of all 'GPU Adapter Memory ... Dedicated Usage' columns.
"""
import csv
import datetime
import json
import statistics
import sys
from pathlib import Path

MIB = 1024 * 1024


def parse_time(text):
    return datetime.datetime.strptime(text.strip().strip('"'), '%m/%d/%Y %H:%M:%S.%f').timestamp()


def main(run_dir: str) -> None:
    run = Path(run_dir)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    pid_tag = f"pid_{summary['pid']}_"
    t0, t1 = summary['wall_start_epoch'], summary['wall_end_epoch']
    with open(run / 'gpu_typeperf.csv', encoding='utf-8', errors='replace', newline='') as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    util_cols, vram_cols = {}, []
    for index, name in enumerate(header):
        if index == 0:
            continue
        if 'Utilization Percentage' in name and pid_tag in name:
            util_cols[index] = name.split('engtype_')[-1].split(')')[0] if 'engtype_' in name else 'other'
        if 'GPU Adapter Memory' in name and 'Dedicated Usage' in name:
            vram_cols.append(index)
    totals, per_engine, vram = [], {}, []
    for row in rows[1:]:
        if len(row) < len(header):
            continue
        try:
            stamp = parse_time(row[0])
        except ValueError:
            continue
        if not (t0 <= stamp <= t1):
            continue
        total = 0.0
        for index, engine in util_cols.items():
            value = row[index].strip().strip('"')
            if value:
                number = float(value)
                total += number
                per_engine.setdefault(engine, []).append(number)
        totals.append(total)
        used = sum(float(row[i].strip().strip('"') or 0) for i in vram_cols)
        vram.append(used / MIB)
    report = {
        'run': run.name, 'pid': summary['pid'], 'window_s': round(t1 - t0, 2), 'samples': len(totals),
        'engine_util_sum_pct': {'mean': round(statistics.mean(totals), 1), 'max': round(max(totals), 1)} if totals else None,
        'per_engine_pct': {k: {'mean': round(statistics.mean(v), 1), 'max': round(max(v), 1)}
                           for k, v in sorted(per_engine.items())},
        'adapter_dedicated_mib': {'max': round(max(vram), 0), 'mean': round(statistics.mean(vram), 0)} if vram else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
