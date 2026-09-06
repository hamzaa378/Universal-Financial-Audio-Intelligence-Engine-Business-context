@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Devsoc AI v4.2 Diagnostics

echo ============================================================
echo   Devsoc AI v4.2 - Installation Diagnostics
echo ============================================================
echo.
echo Folder: %CD%
echo.

echo --- Python launcher ---
where py 2>nul
py --version 2>nul
py -0p 2>nul

echo.
echo --- python on PATH ---
where python 2>nul
python --version 2>nul

echo.
echo --- Virtual environment ---
if exist ".venv\Scripts\python.exe" (
    echo .venv exists.
    ".venv\Scripts\python.exe" --version
    ".venv\Scripts\python.exe" -m pip --version
) else (
    echo .venv does NOT exist.
)

echo.
echo --- Core import check ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; print('Python:',sys.version); import numpy; print('numpy',numpy.__version__); import librosa; print('librosa',librosa.__version__); import soundfile; print('soundfile',soundfile.__version__); import streamlit; print('streamlit',streamlit.__version__); import faster_whisper; print('faster-whisper',getattr(faster_whisper,'__version__','installed'))"
)

echo.
echo --- Optional semantic AI ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import fastembed; print('fastembed',getattr(fastembed,'__version__','installed'))" 2>nul
    if errorlevel 1 echo FastEmbed is not installed or cannot import. This is optional.
)

echo.
echo --- Recent installer log ---
if exist install_v4.log type install_v4.log

echo.
echo ============================================================
echo Copy the error text above if you need help diagnosing it.
echo ============================================================
pause
