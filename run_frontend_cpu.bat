@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "FINAI_DEVICE=cpu"
set "FINAI_COMPUTE_TYPE=int8"
set "FINAI_STRICT_GPU=0"
set "FINAI_FORCE_CPU=1"
set "FINAI_SEMANTIC_DEVICE=cpu"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install_v4.bat first.
  pause
  exit /b 1
)
echo Starting explicit CPU fallback frontend...
"%PY%" -m streamlit run frontend.py --server.address localhost --server.port 8501
pause
