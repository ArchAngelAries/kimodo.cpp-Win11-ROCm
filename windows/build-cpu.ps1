<#
.SYNOPSIS
    Builds kimodo.cpp for the CPU only on Windows 11 (no GPU needed).

.DESCRIPTION
    Uses the MSVC compiler from Visual Studio ("Desktop development with C++"), CMake 3.25 or
    newer, and Ninja. The CPU build is the reference path: KIMODO_BACKEND=cpu forces the CPU in
    any build, and this build is what the ROCm results are compared against.

    The build writes to <repository>\build\cpu by default.

.PARAMETER BuildDir
    Build folder. Defaults to <repository>\build\cpu.

.PARAMETER ConfigureOnly
    Run CMake configuration only.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File windows\build-cpu.ps1
#>
[CmdletBinding()]
param(
    [string]$BuildDir,
    [switch]$ConfigureOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$repo = Get-RepoRoot
if (-not $BuildDir) { $BuildDir = Join-Path $repo 'build\cpu' }

Enter-VisualStudioEnvironment

$cmakeArgs = @(
    '-S', $repo,
    '-B', $BuildDir,
    '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DKIMODO_ENABLE_HIP=OFF',
    '-DKIMODO_ENABLE_VULKAN=OFF'
)
Write-Host "Configuring $BuildDir (CPU only)"
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
