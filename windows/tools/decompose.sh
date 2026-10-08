#!/usr/bin/env bash
# Splits the difference between CPU and ROCm runs into text-encoder and denoiser parts.
#   E1  text embedding: CPU build vs ROCm build (kmd-encode, same prompt)
#   D1  denoiser:  CPU build, CPU embedding, fixed noise
#   D2  denoiser:  ROCm build, CPU embedding, same noise         -> GPU denoiser effect
#   D3  denoiser:  CPU build, ROCm embedding, same noise         -> GPU text-encoder effect
#   D4  denoiser:  ROCm build with KIMODO_BACKEND=cpu            -> compiler effect (clang vs MSVC)
#
# Needs the builds from windows/build-cpu.ps1 and windows/build-rocm.ps1, the SOMA RP v1.1
# motion model and the Q8_0 text encoder under weights/ (windows/download-weights.py), and
# Python with the standard library (set PYTHON to choose the interpreter).
#
# Usage: windows/tools/decompose.sh [FRAMES] [STEPS]      (defaults: 60 frames, 10 steps)
# Overrides: CPU_BUILD, ROCM_BUILD, HIP_BIN, DIAG_DIR, PYTHON.
set -uo pipefail

R="$(cd "$(dirname "$0")/../.." && pwd)"
FRAMES=${1:-60}
STEPS=${2:-10}
M=$R/weights/models/kimodo-soma-rp-v1.1-f32.gguf
T=$R/weights/Llama-3-Kimodo-Q8_0.gguf
P=$R/benchmarks/quantization/prompts/official-demo-run-jump.txt
CPU=${CPU_BUILD:-$R/build/cpu}
ROCM=${ROCM_BUILD:-$R/build/rocm}
HIP_BIN=${HIP_BIN:-/c/Program Files/AMD/ROCm/7.1/bin}
D=${DIAG_DIR:-$R/runs/diag}
PY=${PYTHON:-python}
mkdir -p "$D"

cpu_env() { PATH="$CPU/bin:$PATH" "$@"; }
rocm_env() { PATH="$HIP_BIN:$ROCM/bin:$PATH" "$@"; }

# Fixed initial noise: 369 floats per frame (9 + 12 x 30 joints), seeded so every run reuses it.
NOISE="$D/noise_${FRAMES}f.f32"
if [ ! -f "$NOISE" ]; then
    "$PY" -I -c "import array, random, sys
random.seed(1234)
values = array.array('f', (random.gauss(0.0, 1.0) for _ in range(int(sys.argv[1]) * 369)))
with open(sys.argv[2], 'wb') as handle:
    handle.write(values.tobytes())" "$FRAMES" "$NOISE"
fi

echo "== E1 text embeddings (CPU build, then ROCm build)"
cpu_env "$CPU/kmd-encode.exe" "$T" "$P" "$D/emb_cpu.f32" > "$D/enc_cpu.log" 2>&1; echo "cpu encode rc=$?"
rocm_env "$ROCM/kmd-encode.exe" "$T" "$P" "$D/emb_rocm.f32" > "$D/enc_rocm.log" 2>&1; echo "rocm encode rc=$?"

echo "== D1 CPU denoiser, CPU embedding"
cpu_env "$CPU/kmd-sample-embedding.exe" "$M" "$D/emb_cpu.f32" "$NOISE" "$FRAMES" "$STEPS" "$D/d1_cpu_${FRAMES}f" > "$D/d1.log" 2>&1; echo "rc=$?"

echo "== D2 ROCm denoiser, CPU embedding"
rocm_env "$ROCM/kmd-sample-embedding.exe" "$M" "$D/emb_cpu.f32" "$NOISE" "$FRAMES" "$STEPS" "$D/d2_rocm_${FRAMES}f" > "$D/d2.log" 2>&1; echo "rc=$?"

echo "== D3 CPU denoiser, ROCm embedding"
cpu_env "$CPU/kmd-sample-embedding.exe" "$M" "$D/emb_rocm.f32" "$NOISE" "$FRAMES" "$STEPS" "$D/d3_cpu_rocmemb_${FRAMES}f" > "$D/d3.log" 2>&1; echo "rc=$?"

echo "== D4 ROCm build forced to the CPU backend, CPU embedding"
KIMODO_BACKEND=cpu rocm_env "$ROCM/kmd-sample-embedding.exe" "$M" "$D/emb_cpu.f32" "$NOISE" "$FRAMES" "$STEPS" "$D/d4_hipcpu_${FRAMES}f" > "$D/d4.log" 2>&1; echo "rc=$?"

echo "Outputs are in $D. Compare with windows/tools/analyze_motion.py and fk_compare.py,"
echo "and the embeddings with windows/tools/compare_embeddings.py."
echo DECOMPOSE_DONE
