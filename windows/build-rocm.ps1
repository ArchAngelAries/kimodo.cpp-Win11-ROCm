<#
.SYNOPSIS
    Builds kimodo.cpp with the AMD ROCm (HIP) backend on Windows 11.

.DESCRIPTION
    Needs the AMD HIP SDK for Windows (ROCm 7.x, which provides clang, HIP and rocBLAS)
    and Visual Studio with the "Desktop development with C++" workload. The script loads the
    Visual Studio x64 developer environment so that clang finds the MSVC headers and libraries.
    CMake (3.25 or newer) and Ninja must be installed, or come from Visual Studio.

    The build writes to <repository>\build\rocm by default. Run the GUI with Kimodo-GUI.bat
    in the repository root once the build has finished.

.PARAMETER GpuTarget
    AMD GPU architecture for the kernels. gfx1100 covers the Radeon RX 7900 series.
    Only gfx1100 has been tested.

.PARAMETER RocmPath
    HIP SDK root. Defaults to the HIP_PATH environment variable, then C:\Program Files\AMD\ROCm\7.1.

.PARAMETER BuildDir
    Build folder. Defaults to <repository>\build\rocm.

.PARAMETER ConfigureOnly
    Run CMake configuration only. Useful for checking the environment.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File windows\build-rocm.ps1
#>
[CmdletBinding()]
param(
    [string]$GpuTarget = 'gfx1100',
    [string]$RocmPath,
    [string]$BuildDir,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot

if (-not $RocmPath) {
    $RocmPath = 'C:\Program Files\AMD\ROCm\7.1'
    if ($env:HIP_PATH) { $RocmPath = $env:HIP_PATH }
}
$RocmPath = $RocmPath.TrimEnd('\')
$clang = Join-Path $RocmPath 'bin\clang.exe'
$clangxx = Join-Path $RocmPath 'bin\clang++.exe'
if (-not (Test-Path $clangxx)) {
    throw "The AMD HIP SDK was not found at $RocmPath (no $clangxx). Install the AMD HIP SDK for Windows, or pass -RocmPath."
}
if (-not $BuildDir) { $BuildDir = Join-Path $repo 'build\rocm' }

Enter-VisualStudioEnvironment
# ggml's HIP build reads ROCM_PATH on Windows.
$env:ROCM_PATH = $RocmPath
$env:HIP_PATH = $RocmPath

$cmakeArgs = @(
    '-S', $repo,
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DKIMODO_ENABLE_HIP=ON',
    '-DKIMODO_ENABLE_VULKAN=OFF',
    ('-DCMAKE_C_COMPILER=' + (ConvertTo-CMakePath $clang)),
    ('-DCMAKE_CXX_COMPILER=' + (ConvertTo-CMakePath $clangxx)),
    ('-DCMAKE_PREFIX_PATH=' + (ConvertTo-CMakePath $RocmPath)),
    ('-DGPU_TARGETS=' + $GpuTarget)
)
Write-Host "Configuring $BuildDir for $GpuTarget with the HIP SDK at $RocmPath"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
if ($ConfigureOnly) {
    Write-Host 'Configuration finished (-ConfigureOnly).'
    exit 0
}

$targets = @('kmd-generate', 'kmd-encode', 'kmd-inspect', 'kmd-sample-embedding', 'kimodo-generate-smoke')
& cmake --build $BuildDir --target $targets
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
Write-Host "Built: $(Join-Path $BuildDir 'kmd-generate.exe')"
Write-Host 'Next: download weights (windows\download-weights.py), then run Kimodo-GUI.bat.'
