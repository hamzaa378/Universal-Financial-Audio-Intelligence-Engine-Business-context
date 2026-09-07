@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if "%~1"=="" (
  echo Usage: profile_pre_asr.bat "C:\path\to\audio.wav"
  pause
  exit /b 1
)
%PYTHON% tools\profile_pre_asr.py "%~1" --json reports\pre_asr_profile_v412.json
pause
