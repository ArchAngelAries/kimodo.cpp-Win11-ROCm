#!/usr/bin/env python3
"""Compares two kmd-encode outputs (float32 embedding files) and reports the usual metrics."""
import array
import math
import sys


def load(path):
    values = array.array('f')
    with open(path, 'rb') as handle:
        values.frombytes(handle.read())
    return values


def main(a_path: str, b_path: str) -> None:
    a, b = load(a_path), load(b_path)
    if len(a) != len(b):
        raise SystemExit(f'length mismatch: {len(a)} vs {len(b)}')
    diff = [x - y for x, y in zip(a, b)]
    rel_l2 = math.sqrt(sum(d * d for d in diff)) / math.sqrt(sum(x * x for x in a))
    dot = sum(x * y for x, y in zip(a, b))
    cosine = dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    print(f'values={len(a)} max_abs_diff={max(abs(d) for d in diff):.6g} '
          f'rel_L2={rel_l2:.6g} cosine={cosine:.8f} identical={a == b}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
