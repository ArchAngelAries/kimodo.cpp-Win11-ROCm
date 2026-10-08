# kimodo.cpp on Windows 11 with AMD ROCm

This fork makes AMD GPU use available on Windows 11. The motion model and the text encoder run on an AMD Radeon GPU through ROCm (the HIP backend of ggml). It also adds a local browser GUI with a double-click launcher, Windows build and download scripts, and tools for benchmarks and checks.

Everything here is relative to the repository root. Commands use Windows PowerShell or Command Prompt.

## Status and support

- **Tested:** Windows 11 Pro with an AMD Radeon RX 7900 XT (gfx1100, 20 GB VRAM), ROCm 7.1 (the AMD HIP SDK for Windows). Motion models: SOMA RP v1.1 and SOMA SEED v1.1. Text encoder: Q8_0.
- **Not tested:** other AMD GPUs and architectures, other ROCm versions, other Windows builds, the Vulkan path in this fork, and the upstream C++ test suite (its parity tests need SMPL-X fixtures that are not in the repository).
- **Not supported:** the original Go demo in `demo/`. This fork does not build, configure or test it. If you absolutely want it, install Go yourself and set it up yourself, following the upstream instructions. The GUI in `gui/` replaces it.
- **Future support is not guaranteed.** This is a community port of the upstream project. It may fall behind upstream, and parts of it may stop working when ROCm, Visual Studio or upstream change.

## Environment used to build and test this fork

| Part | Used |
| --- | --- |
| CPU | AMD Ryzen 9 3900X (12 cores, 24 threads) |
| RAM | 32 GB |
| GPU | AMD Radeon RX 7900 XT, gfx1100, 20 GB VRAM |
| OS | Windows 11 Pro, build 10.0.26220 |
| GPU runtime | AMD HIP SDK for Windows, ROCm 7.1 (clang 21 from the SDK, HIP runtime, rocBLAS, hipBLAS) |
| C++ toolchain | Visual Studio 2026 Insiders with MSVC 14.50, Windows SDK 10.0.26100 |
| Build tools | CMake 4.1.2, Ninja (from the Visual Studio CMake component) |
| Python | 3.12. The GUI and the tools use the standard library. The weight downloader needs `huggingface_hub`. |
| Other | Blender 5.2.2, used only to check that the exported BVH and GLB files import |

Other combinations are likely to work, but they have not been tested.

## Requirements

1. **AMD HIP SDK for Windows** (ROCm 7.1 was used). It provides `clang.exe` and the HIP runtime. The installer normally sets `HIP_PATH`.
2. **Visual Studio** with the "Desktop development with C++" workload. The build scripts load its x64 developer environment, which provides the MSVC headers and libraries that clang links against.
3. **CMake 3.25 or newer** and **Ninja**. Both come with the Visual Studio CMake component, or you can install them separately.
4. **Python 3.12** (the version used here). Needed for the GUI, the downloader and the tools.
5. **Git**, with submodules.

## Build

Clone with the ggml submodule:

```
git clone --recursive <url-of-this-fork> kimodo.cpp
cd kimodo.cpp
```

Build the ROCm (GPU) tools:

```
powershell -ExecutionPolicy Bypass -File windows\build-rocm.ps1
```

The default GPU target is `gfx1100`. To target another architecture, pass `-GpuTarget` (not tested here). The build writes to `build\rocm`. Use `-ConfigureOnly` to check the environment without compiling.

Build the CPU reference tools (optional, but needed for the comparison tools and for CPU-only machines):

```
powershell -ExecutionPolicy Bypass -File windows\build-cpu.ps1
```

This writes to `build\cpu`.

## Weights

The model weights are not in this repository. Their licences apply (see Licences below).

```
python -m pip install huggingface_hub
python windows\download-weights.py
```

The default downloads the SOMA RP v1.1 motion model and the Q8_0 text encoder. This is about 9.3 GB. Use `--model soma-seed-v1.1` to add the SEED model, and `--text-quantization` to choose another text encoder. Use `--verify-only` to check files you already have.

Each file is checked for size and SHA-256 against the repository's manifest. Files go to `weights\models\` and `weights\`, which is the layout the GUI and the tools expect.

We also found that the shell downloader `scripts/download_gguf_weights.sh` did not finish a text-encoder request on the current manifest. Its multi-file request fetched only one file, and its check matched a legacy path. Use `windows\download-weights.py` on Windows.

## Run the GUI

Double-click `Kimodo-GUI.bat` in the repository root, or run it from a terminal. The console shows which models and encoders are installed. The page opens at `http://127.0.0.1:8094/`. Close the console window to stop the GUI.

- Pick a motion model and a text encoder. Models that are not installed are shown as "not installed".
- Write a prompt. Each segment is 60 to 150 frames. Add more segments for a sequence, up to 16. Each generation is 150 frames at 100 steps by default.
- Click **Generate motion**. The first clip is slower while the models load.
- Select an animation in the gallery to play it, and use **Download GLB** to save it.

To use the CPU build instead of the GPU build:

```
python gui\server.py --generator build\cpu\kmd-generate.exe
```

Saved animations go to `gui-output\`. The GUI page always sends seed 0, which is the upstream behaviour. Change the prompt to get a different motion.

## Command line

The native program takes a motion model, a text encoder, a prompt file, a frame count, a step count, a seed and an output folder:

```
set "PATH=C:\Program Files\AMD\ROCm\7.1\bin;%CD%\build\rocm\bin;%PATH%"
build\rocm\kmd-generate.exe weights\models\kimodo-soma-rp-v1.1-f32.gguf weights\Llama-3-Kimodo-Q8_0.gguf benchmarks\quantization\prompts\official-demo-run-jump.txt 150 100 42 runs\example
```

This writes `runs\example\root_positions.f32` and `runs\example\local_rotations_xyzw.f32`: root translation in metres (T x 3) and local joint quaternions in xyzw order (T x 30 x 4 for SOMA).

Environment variables:

| Variable | Effect |
| --- | --- |
| `KIMODO_BACKEND=cpu` | Forces the CPU path in any build (the parity reference). |
| `KIMODO_TEXT_LAYER_CHUNK=32` | Keeps all text-encoder layers resident on the GPU. Default 8 streams them for every request. The GUI sets 32. |
| `KIMODO_TEXT_RESIDENT_LIMIT_MIB` | Caps the resident size of the text encoder. The GUI sets 10000. |
| `KIMODO_PROFILE=1` | Prints the device and phase timings to stderr. |
| `KIMODO_THREADS=n` | Sets the CPU thread count. |

## Export animations

```
python scripts\export_glb.py --motion-dir runs\example --output runs\example\animation.glb
python scripts\export_bvh.py --motion-dir runs\example --output runs\example\animation.bvh
```

- **GLB** (glTF 2.0, metres, 30 fps): `export_glb.py` writes 30 joints, a skin, a simple bone-cube mesh, and one animation, `KimodoMotion`. The GUI's download is the same animation as a node hierarchy with no mesh. Godot 4 imports GLB directly. Blender imports it through File > Import > glTF 2.0.
- **BVH** (centimetres, 30 fps, Euler channels): Blender reads it through File > Import > Motion Capture. Set the import scale to **0.01** to get metres, and set the scene frame rate to **30 fps**. Blender's default is 24 fps, which would play a 150-frame clip for 6.25 s instead of 5 s. Godot has no built-in BVH importer.

We imported the BVH and the `export_glb.py` GLB of a 150-frame clip in Blender 5.2.2, in background mode: the bones, frame ranges and keyframe counts were as expected. Godot was not tested.

## Benchmarks and checks

Scripts in `windows\tools\`:

- `bench_clips.py`: runs a server session with several clips, records timings and checks that warm clips match the cold clip. With `--gpu-sample` it records GPU engine use and VRAM with `typeperf`. Then `gpu_summary.py <run-dir>` summarises it.
- `analyze_motion.py <run>` checks a clip: finite values, unit quaternions, root path and height. With two runs it compares them.
- `fk_compare.py <run-a> <run-b>` gives the world-space joint error between two runs (MPJPE, in cm).
- `compare_embeddings.py <a.f32> <b.f32>` compares two text embeddings.
- `decompose.sh [FRAMES] [STEPS]` splits a CPU-versus-GPU difference into text-encoder and denoiser parts.
- `check_exports.py <file.glb|file.bvh>` checks the structure of an export.
- `blender_check.py` is a headless Blender import check. Run it with Blender's `--background --factory-startup --python` options.

## Measurements

All numbers are from one machine (see Environment). Clip = 150 frames (5 s at 30 fps) at 100 diffusion steps, seed 42, the official demo prompt. Warm = after the first clip in the same session.

Wall time per clip:

| Configuration | Cold clip | Warm clips (mean) | Text encode per request | Diffusion per step |
| --- | ---: | ---: | ---: | ---: |
| CPU, MSVC build, 24 threads (library defaults) | 160.9 s | 157.3 s | 8 to 9 s (weights streamed) | about 1.5 s |
| ROCm, library defaults (`KIMODO_TEXT_LAYER_CHUNK=8`) | 14.85 s | 13.78 s | 7.0 to 7.5 s (weights streamed) | about 70 ms |
| ROCm, resident encoder (`KIMODO_TEXT_LAYER_CHUNK=32`, `KIMODO_TEXT_RESIDENT_LIMIT_MIB=10000`) | 13.74 s | 7.49 s | about 0.1 s | about 70 ms |

Warm GPU clips are about 11 times faster than the CPU build with the library defaults, and about 21 times faster with the resident encoder. The GUI uses the resident setting.

GPU use during a run (Windows GPU engine counters summed over the process, sampled every second, so the window includes model loading):

| Configuration | Compute engine, mean (peak) | Peak adapter VRAM |
| --- | ---: | ---: |
| ROCm, library defaults | 37% (80%) | 6.5 GB |
| ROCm, resident encoder | 61.5% (82%) | 11.9 GB |

Output checks: every run gave finite values and unit-length quaternions. Warm clips reproduced the cold clip exactly, on the CPU and on the GPU. The default and resident GPU configurations gave byte-identical output.

## Accuracy: CPU and GPU motion

The GPU and CPU builds are not bit-identical, and the motion can diverge over 100 steps. What we measured:

- **Motion model alone** (same text embedding and same noise, 60 frames, 10 steps): the GPU and CPU agree to 0.001 cm (world-space joint error).
- **Text encoder alone:** the GPU and CPU embeddings differ by 1.4% (relative L2; cosine 0.99990). A compiler change alone (MSVC to clang, both on the CPU) gives 1.1% (cosine 0.99994).
- **Full 150-frame clip, 100 steps:** the GPU against the CPU gives a mean world-space joint error of 14.7 cm (p95 30 cm). The clang CPU build against the MSVC CPU build gives 11.1 cm. Two different trajectories can come from very small differences in the embedding, because the sampler amplifies them.

So the GPU result is in the same range as an ordinary rebuild. Small embedding differences can lead to different trajectories, because the 100-step sampler amplifies them. A single prompt and seed can't establish quality on its own. Compare several prompts and seeds before relying on the GPU output for anything that needs to match the CPU.

## Changes from upstream

- **Backend selection** (`src/gpu_backend.hpp`, new): `KIMODO_BACKEND=cpu` keeps the CPU path. Otherwise the first GPU device of the compiled-in backend is used, through ggml's backend registry.
- **Motion model and text encoder** (`src/ggml_weights.cpp`, `src/llm_text_encoder.cpp`): use the shared selection. The Vulkan path is unchanged. The Vulkan header is included only in Vulkan builds, so CPU-only builds no longer reference Vulkan symbols.
- **BF16 norm cast** (`src/llm_text_encoder.cpp`): the per-layer norm weights are BF16 in the bundle. The HIP element-wise kernels take only F32 or F16 operands and aborted on them. The weight is now cast to F32 before the repeat. The cast is exact, and the CPU output stayed byte-identical.
- **CMake** (`CMakeLists.txt`): `KIMODO_ENABLE_HIP` (default OFF) builds ggml's HIP backend, which replaces Vulkan in that build. The GPU link variables were renamed from `KIMODO_VK_*` to `KIMODO_GPU_*`.
- **Windows tools** (`windows\`, `Kimodo-GUI.bat`, `gui\`, `docs\`): new, as described above.

The modified source files carry a notice at the top saying so.

## Known issues and limits

- Default text-encoder settings stream its weights to the GPU for every request. Use `KIMODO_TEXT_LAYER_CHUNK=32` (the GUI does this).
- CPU and GPU motion differ over 100 steps (see Accuracy).
- Only gfx1100 has been built and run. Other architectures need `-GpuTarget` and are untested.
- The Vulkan backend (upstream) is not built or tested in this fork.
- The Go demo is not supported (see Status and support).
- The GLB download from the GUI contains only the joints, not a mesh.
- Godot import is not tested.
- ROCm prints about 90,000 GNU-extension warnings from ggml's common headers while building. They are harmless on this toolchain.

## Licences

- The changes in this fork are under the same Apache-2.0 licence as upstream (see `LICENSE`). The upstream NOTICE file still applies.
- ggml is MIT-licensed.
- The motion models (SOMA RP v1.1 and SEED v1.1) are under the NVIDIA Open Model License. It allows commercial use and derivative models, claims no ownership of outputs, and requires a notice when the weights are redistributed. It ends if you circumvent a model guardrail.
- The text-encoder bundle includes Meta Llama 3 material under the Meta Llama 3 Community License. Redistribution requires the licence and a "Built with Meta Llama 3" notice. The licence also has a threshold for very large services, and it prohibits using the material or its output to improve other large language models.
- SMPL-X (`smplx-rp-v1`) is under an NVIDIA internal R&D licence, which excludes commercial use. This fork's downloader does not offer it.

This is a summary of the licence texts, not legal advice. Read the licences yourself before using or distributing the weights.

## Trademarks

AMD, Radeon and ROCm are trademarks of Advanced Micro Devices, Inc. NVIDIA, Meta and LocalAI are trademarks of their owners. This fork is not affiliated with or endorsed by any of them.

## Verification performed

- The ROCm and CPU builds were compiled from this source. Warm and cold clips, and the default and resident GPU configurations, matched byte for byte.
- The GUI was tested end to end: model and encoder lists, validation errors, generation from the API and from the page, gallery selection, a two-segment sequence, the downloaded GLB, and the launcher starting and stopping the server.
- The GLB and BVH exports were checked for structure, and the BVH and GLB files were imported in Blender 5.2.2 in background mode.
- The downloader was checked against the local files and their manifests.
