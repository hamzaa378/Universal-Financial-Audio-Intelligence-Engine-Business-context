@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" tools\runtime_fingerprint.py --output reports\runtime_fingerprint.json
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
