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
echo 1/13 Functional unit tests
echo ============================================================
%PYTHON% -m unittest discover -s tests -p "test_*.py" -v
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 2/13 PII + profanity accuracy, FPR and deterministic speed
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --iterations 200 --json reports\privacy_benchmark.json --csv reports\privacy_cases.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 3/13 v4.6 context/false-positive stress benchmark
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\context_precision_v46.jsonl --iterations 200 --json reports\context_precision_v46.json --csv reports\context_precision_v46.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 4/13 v4.7 context/recall stress benchmark
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\context_recall_v47.jsonl --iterations 200 --json reports\context_recall_v47.json --csv reports\context_recall_v47.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 5/13 v4.8 precision-guard regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\precision_guard_v48.jsonl --iterations 200 --json reports\precision_guard_v48.json --csv reports\precision_guard_v48.csv
if errorlevel 1 goto :fail


echo.
echo ============================================================
echo 6/13 v4.9 ASR-robust privacy regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\asr_robust_v49.jsonl --iterations 200 --json reports\asr_robust_v49.json --csv reports\asr_robust_v49.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 7/13 v4.10 STT-guard precision regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\stt_guard_v410.jsonl --iterations 200 --json reports\stt_guard_v410.json --csv reports\stt_guard_v410.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 8/13 v4.11 reliability-guard regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\reliability_guard_v411.jsonl --iterations 200 --json reports\reliability_guard_v411.json --csv reports\reliability_guard_v411.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 9/13 v4.12 optimized STT-guard regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\optimized_guard_v412.jsonl --iterations 200 --json reports\optimized_guard_v412.json --csv reports\optimized_guard_v412.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 10/13 v5.0 correction-aware precision regression
echo ============================================================
%PYTHON% tools\evaluate_privacy.py --pii benchmarks\correction_guard_v50.jsonl --iterations 200 --json reports\correction_guard_v50.json --csv reports\correction_guard_v50.csv
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 11/13 Sensitive financial identifier regression
echo ============================================================
%PYTHON% tools\evaluate_sensitive_ids.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 12/13 Optional batched semantic-AI privacy stress test
echo ============================================================
%PYTHON% tools\evaluate_semantic_privacy.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo 13/13 Optional full audio / GPU ASR benchmark
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
echo [PASS] v5.0 benchmark completed. Open the reports folder for JSON/CSV results.
pause
exit /b 0

:fail
echo.
echo [FAIL] A regression or runtime error was detected.
echo Review the reports folder and the error above.
pause
exit /b 2
