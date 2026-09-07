@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv not found. Run install_v4.bat first.
  pause
  exit /b 1
)

rem Make local project packages (decision_ai, nlp, etc.) importable by helper scripts.
set "PYTHONPATH=%CD%;%PYTHONPATH%"

echo Installing optional ONNX NER runtime...
"%PY%" -m pip install --upgrade onnxruntime tokenizers huggingface-hub
if errorlevel 1 goto :fail

echo.
echo Downloading the quantized multilingual privacy NER model.
echo This is optional and can be several hundred MB depending on available artifacts.
"%PY%" tools\install_ner_model.py
if errorlevel 1 goto :fail

echo.
echo [OK] Optional NER AI installed.
pause
exit /b 0

:fail
echo.
echo [ERROR] Optional NER AI installation failed. Core privacy rules remain usable.
pause
exit /b 1
