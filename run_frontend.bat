@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Devsoc AI v4.5 - STRICT GPU Frontend

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv was not found.
  echo Run install_v4.bat first.
  pause
  exit /b 1
)

rem ============================================================
rem STRICT GPU POLICY
rem Faster-Whisper must execute on CUDA/FP16 in this launcher.
rem No CPU fallback is permitted. Use run_frontend_cpu.bat only intentionally.
rem ============================================================
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="
set "FINAI_CUDA_INDEX=0"
set "FINAI_BEAM_SIZE=2"
set "FINAI_BEST_OF=1"
set "FINAI_SEMANTIC_DEVICE=auto"
set "FINAI_NER_DEVICE=auto"

cls
echo ============================================================
echo   Devsoc AI v4.5 - STRICT GPU Frontend
echo ============================================================
echo.
"%PY%" --version
echo Requested ASR backend : CUDA / float16
echo Strict GPU mode       : ON
echo.

rem --- Required Faster-Whisper runtime DLLs ---
echo [1/5] Checking CUDA/cuDNN DLL visibility...
set "DLLFAIL=0"
for %%D in (cublas64_12.dll cublasLt64_12.dll cudnn64_9.dll) do (
  where %%D >nul 2>&1
  if errorlevel 1 (
    echo   [MISSING] %%D
    set "DLLFAIL=1"
  ) else (
    for /f "delims=" %%P in ('where %%D') do echo   [FOUND] %%D --^> %%P
  )
)
if "!DLLFAIL!"=="1" (
  echo.
  echo [ERROR] Required CUDA/cuDNN DLLs are not visible on PATH.
  echo v4.5 will NOT silently fall back to CPU.
  pause
  exit /b 1
)
echo [OK] Required DLLs are visible.
echo.

rem --- CTranslate2 CUDA visibility ---
echo [2/5] Checking CTranslate2 CUDA access...
"%PY%" -c "import ctranslate2,sys; n=ctranslate2.get_cuda_device_count(); print('CTranslate2:',ctranslate2.__version__); print('CUDA devices:',n); sys.exit(0 if n>0 else 2)"
if errorlevel 1 (
  echo [ERROR] CTranslate2 cannot see an NVIDIA CUDA device.
  pause
  exit /b 1
)
echo [OK] CUDA device is visible to CTranslate2.
echo.

rem --- Actual CUDA inference, not just device visibility ---
echo [3/5] Running a real Faster-Whisper CUDA smoke test...
"%PY%" -c "from asr.fintech_asr import warmup_model; r=warmup_model('small',smoke_test=True); print('ASR GPU:',r['device'],r['compute_type']); raise SystemExit(0 if r['device']=='cuda' and str(r['compute_type']).lower()=='float16' else 3)"
if errorlevel 1 (
  echo.
  echo [ERROR] Faster-Whisper could not complete CUDA inference.
  echo Run diagnose_gpu.bat for details. CPU fallback is disabled.
  pause
  exit /b 1
)
echo [OK] Faster-Whisper CUDA FP16 inference works.
echo.

rem --- Optional AI components are reported separately ---
echo [4/5] Checking optional AI components...
"%PY%" -c "import importlib.util,os; print('Semantic AI package:', 'installed' if importlib.util.find_spec('fastembed') else 'not installed'); print('ONNX NER runtime:', 'installed' if importlib.util.find_spec('onnxruntime') and importlib.util.find_spec('tokenizers') else 'not installed'); print('Diarization:', 'installed' if importlib.util.find_spec('pyannote.audio') else 'not installed', '/ HF_TOKEN=' + ('set' if os.getenv('HF_TOKEN') else 'not set'))"
echo Note: click 'Warm up ^& verify AI' in the frontend for model-level readiness.
echo.

rem --- Refuse to attach to an old/stale Streamlit server ---
netstat -ano | findstr LISTENING | findstr ":8501" >nul 2>&1
if not errorlevel 1 (
  echo [ERROR] Port 8501 is already in use by another process.
  echo Stop the old Streamlit server first, then run this launcher again.
  echo Use: netstat -ano ^| findstr LISTENING ^| findstr :8501
  pause
  exit /b 1
)

rem --- Start Streamlit ---
echo [5/5] Starting Streamlit...
echo Open: http://localhost:8501
echo Press Ctrl+C here to stop the server.
echo ============================================================
echo.
"%PY%" -m streamlit run frontend.py --server.address localhost --server.port 8501
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo [ERROR] Streamlit exited with code %RC%.
pause
exit /b %RC%
