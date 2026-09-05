@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

%PYTHON% --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found. Activate your venv or add Python to PATH.
  exit /b 1
)

if not exist reports mkdir reports

echo ============================================================
echo 1/3 Functional unit tests
echo ============================================================
%PYTHON% -m unittest discover -s tests -p "test_*.py" -v
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 2/3 PII + profanity accuracy, masking, false positives, speed
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --iterations 200 --json reports\privacy_benchmark.json --csv reports\privacy_cases.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 3/3 Optional full audio / ASR benchmark
echo ============================================================
if "%~1"=="" (
  echo Skipped. To benchmark audio, run:
  echo   run_privacy_benchmark.bat "C:\path\to\audio_folder" medium
  goto :success
)
set "MODEL=%~2"
if "%MODEL%"=="" set "MODEL=small"
%PYTHON% tools\benchmark_audio.py --dir "%~1" --model "%MODEL%" --json reports\audio_benchmark.json --csv reports\audio_benchmark.csv
if errorlevel 1 goto :fail

:success
echo.
echo [PASS] Benchmark completed. Open the reports folder for JSON and CSV results.
exit /b 0

:fail
echo.
echo [FAIL] A regression or runtime error was detected. Review the output above and reports\privacy_cases.csv.
exit /b 2
