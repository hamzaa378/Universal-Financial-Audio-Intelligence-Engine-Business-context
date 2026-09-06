@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install_v4.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage:
  echo   run_audio_manifest_benchmark.bat "C:\path\to\manifest.jsonl" [small^|medium^|large-v3]
  echo.
  echo See benchmarks\audio_manifest_example.jsonl for the expected format.
  pause
  exit /b 2
)
set "MODEL=%~2"
if "%MODEL%"=="" set "MODEL=small"
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="
"%PY%" tools\evaluate_audio_manifest.py --manifest "%~1" --model "%MODEL%" --speed fast --semantic --out reports\audio_manifest_report.json
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Report: reports\audio_manifest_report.json
pause
exit /b %RC%
