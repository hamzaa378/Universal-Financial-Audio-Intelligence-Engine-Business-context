@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found.
  pause
  exit /b 1
)
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="

echo Running a real tiny Faster-Whisper CUDA inference...
"%PY%" -c "from asr.fintech_asr import warmup_model; import pprint; pprint.pp(warmup_model('small',smoke_test=True)); print('GPU SMOKE TEST: PASS')"
if errorlevel 1 (
  echo.
  echo [FAIL] Real GPU inference did not complete.
  pause
  exit /b 1
)
echo.
pause
