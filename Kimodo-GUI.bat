@echo off
rem Kimodo text to motion, Windows ROCm build. Double-click to start; close this window to stop.
title Kimodo GUI
cd /d "%~dp0"

rem The AMD HIP SDK: HIP_PATH (set by the installer), otherwise the default install folder.
set "ROCM_BIN=C:\Program Files\AMD\ROCm\7.1\bin"
if defined HIP_PATH set "ROCM_BIN=%HIP_PATH%\bin"

if not exist "build\rocm\kmd-generate.exe" (
    echo The ROCm build is missing: build\rocm\kmd-generate.exe
    echo Build it first, from this folder:
    echo     powershell -ExecutionPolicy Bypass -File windows\build-rocm.ps1
    pause
    exit /b 1
)

set "PATH=%ROCM_BIN%;%~dp0build\rocm\bin;%PATH%"

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    set "PY=python"
)
"%PY%" "gui\server.py"
echo.
echo The Kimodo GUI has stopped.
pause
