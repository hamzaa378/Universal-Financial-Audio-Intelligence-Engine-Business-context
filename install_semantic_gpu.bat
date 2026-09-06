@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install_v4.bat first.
  pause
  exit /b 1
)

echo ============================================================
echo Optional FastEmbed semantic-AI GPU installer
echo ============================================================
echo.
echo This changes ONLY the lightweight semantic decision layer.
echo Faster-Whisper GPU support is independent and is already handled by run_frontend.bat.
echo.
echo Removing CPU FastEmbed/ONNX Runtime packages to avoid provider conflicts...
"%PY%" -m pip uninstall -y fastembed fastembed-gpu onnxruntime onnxruntime-gpu >nul 2>&1

echo Installing fastembed-gpu...
"%PY%" -m pip install -r requirements-ai-gpu.txt
if errorlevel 1 (
  echo.
  echo [ERROR] fastembed-gpu installation failed.
  echo Restoring CPU semantic AI...
  "%PY%" -m pip install -r requirements-ai-lite.txt
  echo.
  pause
  exit /b 1
)

echo.
"%PY%" -c "import onnxruntime as ort; print('ONNX providers:',ort.get_available_providers()); raise SystemExit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 2)"
if errorlevel 1 (
  echo.
  echo [WARNING] fastembed-gpu installed, but CUDAExecutionProvider is not available.
  echo Semantic AI will not use the GPU yet. Check ONNX Runtime CUDA dependencies.
  echo.
  pause
  exit /b 2
)

echo.
echo [OK] Semantic AI GPU provider is available.
echo run_frontend.bat will use it automatically because FINAI_SEMANTIC_DEVICE=auto.
echo.
pause
