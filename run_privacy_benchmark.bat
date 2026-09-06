@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found. Run install_v4.bat first.
  pause
  exit /b 1
)
if not exist reports mkdir reports

echo ============================================================
echo 1/5 Functional unit tests
echo ============================================================
%PYTHON% -m unittest discover -s tests -p "test_*.py" -v
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 2/5 PII + profanity accuracy, FPR and deterministic speed
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --iterations 200 --json reports\privacy_benchmark.json --csv reports\privacy_cases.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 3/5 Sensitive financial identifier regression
echo ============================================================
%PYTHON% tools\evaluate_sensitive_ids.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 4/5 Optional batched semantic-AI privacy stress test
echo ============================================================
%PYTHON% tools\evaluate_semantic_privacy.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 5/5 Optional full audio / GPU ASR benchmark
echo ============================================================
if "%~1"=="" (
  echo Skipped. To benchmark custom audio, run:
  echo   run_privacy_benchmark.bat "C:\path\to\audio_folder" small
  goto :success
)
set "MODEL=%~2"
if "%MODEL%"=="" set "MODEL=small"
set "FINAI_DEVICE=cuda"
set "FINAI_COMPUTE_TYPE=float16"
set "FINAI_STRICT_GPU=1"
set "FINAI_FORCE_CPU="
%PYTHON% tools\benchmark_audio.py --dir "%~1" --model "%MODEL%" --speed fast --protected-audio --json reports\audio_benchmark.json --csv reports\audio_benchmark.csv
if errorlevel 1 goto :fail

:success
echo.
echo [PASS] v4.5 benchmark completed. Open the reports folder for JSON/CSV results.
pause
exit /b 0

:fail
echo.
echo [FAIL] A regression or runtime error was detected.
echo Review the reports folder and the error above.
pause
exit /b 2
