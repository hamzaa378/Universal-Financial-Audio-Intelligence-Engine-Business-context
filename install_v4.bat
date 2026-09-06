@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Devsoc AI v4.4 Installer

set "LOG=%CD%\install_v4.log"
>"%LOG%" echo Devsoc AI v4.4 installation log
>>"%LOG%" echo Started: %DATE% %TIME%
>>"%LOG%" echo Folder: %CD%

echo ============================================================
echo   Devsoc AI v4.4 - Safe Windows Installer
echo ============================================================
echo.
echo Project folder:
echo   %CD%
echo.

rem ------------------------------------------------------------
rem 1. Locate a usable Python interpreter.
rem ------------------------------------------------------------
set "PYBASE="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.14 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PYBASE=py -3.14"
    if not defined PYBASE py -3.13 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PYBASE=py -3.13"
    if not defined PYBASE py -3.12 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PYBASE=py -3.12"
    if not defined PYBASE py -3.11 -c "import sys; assert sys.version_info >= (3,11)" >nul 2>&1 && set "PYBASE=py -3.11"
)

if not defined PYBASE (
    where python >nul 2>&1
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1 && set "PYBASE=python"
    )
)

if not defined PYBASE (
    echo [ERROR] Python 3.11, 3.12, 3.13, or 3.14 was not found.
    echo.
    echo Install a 64-bit Python from https://www.python.org/downloads/
    echo During setup enable: "Add python.exe to PATH".
    echo.
    echo Diagnostics:
    where py 2>nul
    where python 2>nul
    >>"%LOG%" echo ERROR: No supported Python interpreter found.
    goto :failed
)

echo [1/6] Python interpreter selected: %PYBASE%
%PYBASE% --version
%PYBASE% --version >>"%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] The selected Python command could not be started.
    >>"%LOG%" echo ERROR: Selected Python failed to run: %PYBASE%
    goto :failed
)

echo.

rem ------------------------------------------------------------
rem 2. Verify venv support and create a clean project environment.
rem ------------------------------------------------------------
echo [2/6] Checking virtual-environment support...
%PYBASE% -c "import venv" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python's venv module is unavailable.
    echo Reinstall Python with the standard library and pip enabled.
    >>"%LOG%" echo ERROR: venv module unavailable.
    goto :failed
)

if exist ".venv\Scripts\python.exe" (
    echo Existing .venv found - reusing it.
) else (
    echo Creating .venv ...
    %PYBASE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create .venv.
        echo Try deleting any partially-created .venv folder and rerun this installer.
        >>"%LOG%" echo ERROR: venv creation failed.
        goto :failed
    )
)

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [ERROR] .venv was created but its Python executable is missing.
    >>"%LOG%" echo ERROR: .venv Python missing.
    goto :failed
)

echo.

rem ------------------------------------------------------------
rem 3. Bootstrap/upgrade pip inside the isolated environment.
rem ------------------------------------------------------------
echo [3/6] Preparing pip inside .venv ...
"%PY%" -m ensurepip --upgrade >nul 2>&1
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERROR] pip/setuptools/wheel upgrade failed.
    echo Check your internet connection, proxy/firewall, or certificate settings.
    >>"%LOG%" echo ERROR: pip bootstrap/upgrade failed.
    goto :failed
)
"%PY%" -m pip --version >>"%LOG%" 2>&1

echo.

rem ------------------------------------------------------------
rem 4. Install the packages required for ASR + privacy pipeline.
rem ------------------------------------------------------------
echo [4/6] Installing core audio and ASR dependencies...
"%PY%" -m pip install -r requirements-core.txt
if errorlevel 1 (
    echo [ERROR] Core dependency installation failed.
    echo The frontend cannot process audio until this succeeds.
    echo.
    echo Useful retry command:
    echo   "%PY%" -m pip install -r requirements-core.txt
    >>"%LOG%" echo ERROR: requirements-core.txt installation failed.
    goto :failed
)

echo.

rem ------------------------------------------------------------
rem 5. Install Streamlit frontend.
rem ------------------------------------------------------------
echo [5/6] Installing frontend dependencies...
"%PY%" -m pip install -r requirements-ui.txt
if errorlevel 1 (
    echo [ERROR] Streamlit/frontend installation failed.
    echo.
    echo Useful retry command:
    echo   "%PY%" -m pip install -r requirements-ui.txt
    >>"%LOG%" echo ERROR: requirements-ui.txt installation failed.
    goto :failed
)

echo.

rem ------------------------------------------------------------
rem 6. Optional lightweight semantic AI.
rem A failure here does NOT make the frontend unusable.
rem ------------------------------------------------------------
echo [6/6] Installing optional lightweight semantic AI...
"%PY%" -m pip show fastembed-gpu >nul 2>&1
if not errorlevel 1 (
    echo fastembed-gpu is already installed - keeping the GPU semantic runtime.
    >>"%LOG%" echo Existing fastembed-gpu preserved.
) else (
    "%PY%" -m pip install -r requirements-ai-lite.txt
    if errorlevel 1 (
        echo.
        echo [WARNING] Optional FastEmbed semantic AI could not be installed.
        echo The core ASR, PII detector, masking, protected audio, and frontend ARE installed.
        echo You can run the frontend now with Semantic AI switched OFF.
        echo.
        echo To retry later:
        echo   "%PY%" -m pip install -r requirements-ai-lite.txt
        >>"%LOG%" echo WARNING: Optional requirements-ai-lite.txt installation failed.
    ) else (
        echo Optional semantic AI installed successfully.
        >>"%LOG%" echo Optional semantic AI installed successfully.
    )
)

echo.
echo Running final import checks...
"%PY%" -c "import numpy, librosa, soundfile, streamlit; from faster_whisper import WhisperModel; print('Core imports: OK')"
if errorlevel 1 (
    echo [ERROR] Installation finished but one or more core imports failed.
    echo Run diagnose_install.bat for details.
    >>"%LOG%" echo ERROR: Final core import test failed.
    goto :failed
)

"%PY%" -c "import fastembed; print('FastEmbed import: OK')" >nul 2>&1
if errorlevel 1 (
    echo Semantic AI: unavailable ^(optional^)
) else (
    echo Semantic AI: available ^(CPU ONNX by default; optional GPU installer is included^)
)

>>"%LOG%" echo Completed: %DATE% %TIME%
>>"%LOG%" echo RESULT: SUCCESS

echo.
echo ============================================================
echo [OK] Installation completed.
echo ============================================================
echo.
echo Start the frontend with:
echo   run_frontend.bat
echo.
echo This launcher is STRICT GPU for Faster-Whisper.
echo Use run_frontend_cpu.bat only when you intentionally want CPU ASR.
echo.
echo Or manually:
echo   ".venv\Scripts\python.exe" -m streamlit run frontend.py
echo.
echo NOTE: The Whisper and semantic AI model files are downloaded/cached
 echo      when first used, so the first inference can require internet access.
echo.
pause
exit /b 0

:failed
>>"%LOG%" echo Completed: %DATE% %TIME%
>>"%LOG%" echo RESULT: FAILED

echo.
echo ============================================================
echo [INSTALLATION FAILED]
echo ============================================================
echo The error is shown above and this window will stay open.
echo.
echo A small diagnostic log was written to:
echo   %LOG%
echo.
echo You can also run:
echo   diagnose_install.bat
echo.
pause
exit /b 1
