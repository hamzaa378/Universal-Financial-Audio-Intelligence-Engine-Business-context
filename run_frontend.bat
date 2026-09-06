@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Devsoc AI v4.4 - STRICT GPU Frontend

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv was not found.
  echo Run install_v4.bat first.
  echo.
  pause
  exit /b 1
)

rem ============================================================
rem STRICT GPU POLICY
rem Normal v4.4 frontend requests MUST run Faster-Whisper on CUDA.
rem CPU mode is available only through run_frontend_cpu.bat.
rem ============================================================
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="
set "FINAI_CUDA_INDEX=0"
set "FINAI_BEAM_SIZE=2"
set "FINAI_BEST_OF=1"
set "FINAI_SEMANTIC_DEVICE=auto"

cls
echo ============================================================
echo   Devsoc AI v4.4 - STRICT GPU Frontend
echo ============================================================
echo.
"%PY%" --version
echo.
echo Requested ASR backend : CUDA / float16
echo Strict GPU mode       : ON
echo CUDA device index     : 0
echo.

rem --- Required Faster-Whisper runtime DLLs ---
echo [1/4] Checking CUDA/cuDNN DLL visibility...
set "DLLFAIL=0"
for %%D in (cublas64_12.dll cublasLt64_12.dll cudnn64_9.dll) do (
  where %%D >nul 2>&1
  if errorlevel 1 (
    echo   [MISSING] %%D
    set "DLLFAIL=1"
  ) else (
    for /f "delims=" %%P in ('where %%D') do echo   [FOUND] %%D  --^> %%P
  )
)
if "!DLLFAIL!"=="1" (
  echo.
  echo [ERROR] Required CUDA/cuDNN DLLs are not visible on PATH.
  echo v4.4 will NOT silently fall back to CPU.
  echo.
  echo Typical paths are:
  echo   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
  echo   C:\Program Files\NVIDIA\CUDNN\v9.x\bin\12.x\x64
  echo.
  pause
  exit /b 1
)
echo [OK] Required DLLs are visible.
echo.

rem --- CTranslate2 must see a CUDA device ---
echo [2/4] Checking CTranslate2 CUDA access...
"%PY%" -c "import ctranslate2,sys; n=ctranslate2.get_cuda_device_count(); print('CTranslate2:',ctranslate2.__version__); print('CUDA devices:',n); sys.exit(0 if n>0 else 2)"
if errorlevel 1 (
  echo.
  echo [ERROR] CTranslate2 cannot see an NVIDIA CUDA device.
  echo Run diagnose_gpu.bat.
  echo.
  pause
  exit /b 1
)
echo [OK] CUDA device is visible to CTranslate2.
echo.

rem --- Verify v4.4 resolves the backend exactly as intended ---
echo [3/4] Verifying application GPU policy...
"%PY%" -c "from asr.fintech_asr import backend_status; s=backend_status(); print(s); raise SystemExit(0 if s.get('planned_device')=='cuda' and s.get('strict_gpu') else 3)"
if errorlevel 1 (
  echo.
  echo [ERROR] The application did not resolve to strict CUDA mode.
  echo.
  pause
  exit /b 1
)
echo [OK] Application backend is locked to CUDA.
echo.

rem --- Start Streamlit ---
echo [4/4] Starting Streamlit...
echo.
echo Open: http://localhost:8501
echo.
echo IMPORTANT:
echo   - Click "Warm up AI models" once after the page opens.
echo   - If GPU inference itself fails, the request stops with an error.
echo   - It will not be re-run on CPU in this launcher.
echo.
echo Press Ctrl+C here to stop the server.
echo ============================================================
echo.

"%PY%" -m streamlit run frontend.py --server.address localhost --server.port 8501
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] Streamlit exited with code %RC%.
pause
exit /b %RC%
