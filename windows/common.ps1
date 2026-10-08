# Shared helpers for the Windows build scripts. Dot-source it; do not run it directly.

function Get-RepoRoot {
    # This folder sits directly under the repository root.
    return (Split-Path -Parent $PSScriptRoot)
}

function Enter-VisualStudioEnvironment {
    # Loads the x64 developer environment of the newest Visual Studio installation that
    # has the C++ tools, so MSVC and clang find the Windows SDK headers and libraries.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'Visual Studio was not found (vswhere.exe is missing). Install Visual Studio with the "Desktop development with C++" workload.'
    }
    # -prerelease is needed for Visual Studio Insiders and preview channels.
    $install = & $vswhere -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) {
        throw 'No Visual Studio installation with the C++ build tools was found.'
    }
    $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    # Ninja ships with the Visual Studio CMake component; use it if Ninja is not on PATH.
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        $ninja = Join-Path $install 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
        if (Test-Path $ninja) { $env:Path = (Split-Path $ninja) + ';' + $env:Path }
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        throw 'CMake was not found. Install CMake 3.25 or newer, or the Visual Studio CMake component.'
    }
    if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
        throw 'Ninja was not found. Install Ninja, or the Visual Studio CMake component.'
    }
}

function ConvertTo-CMakePath([string]$Path) {
    return ($Path -replace '\\', '/')
}
