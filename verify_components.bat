@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install_v4.bat first.
  pause
  exit /b 1
)
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="
echo Running full v5.1 component verification...
"%PY%" tools\startup_verify.py --model small --semantic --ner
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
