@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
echo ============================================================
echo Devsoc AI v4.4 GPU diagnostic
echo ============================================================
echo.
nvidia-smi
echo.
echo DLL visibility:
where cublas64_12.dll
where cublasLt64_12.dll
where cudnn64_9.dll
echo.
if exist "%PY%" (
  "%PY%" -c "import ctranslate2; print('CTranslate2:',ctranslate2.__version__); print('CUDA devices:',ctranslate2.get_cuda_device_count())"
  echo.
  set "FINAI_DEVICE=cuda"
  set "FINAI_COMPUTE_TYPE=float16"
  set "FINAI_STRICT_GPU=1"
  set "FINAI_FORCE_CPU="
  "%PY%" -c "from asr.fintech_asr import backend_status; import pprint; pprint.pp(backend_status())"
) else (
  echo [ERROR] .venv not found.
)
echo.
pause
