@echo off
setlocal EnableExtensions

rem Usage:
rem   pull_remote_results.bat [remote_host] [remote_repo]
rem
rem Defaults:
rem   remote_host = ustc-lab
rem   remote_repo = ~/xhth

set "REMOTE_HOST=%~1"
if not defined REMOTE_HOST set "REMOTE_HOST=ustc-lab"

set "REMOTE_REPO=%~2"
if not defined REMOTE_REPO set "REMOTE_REPO=~/xhth"

set "STRATEGY_REL=examples/responder_assisted_lgb_catboost_strategy"
set "REMOTE_STRATEGY=%REMOTE_REPO%/%STRATEGY_REL%"
set "LOCAL_STRATEGY=%~dp0"

where scp >nul 2>nul
if errorlevel 1 (
    echo [error] scp was not found in PATH.
    echo [hint] Install or enable Windows OpenSSH Client first.
    exit /b 1
)

echo [pull] remote repository: %REMOTE_HOST%:%REMOTE_REPO%
echo [pull] local strategy:    %LOCAL_STRATEGY%

call :copy_required_dir model
if errorlevel 1 exit /b 1

call :copy_required_dir audit
if errorlevel 1 exit /b 1

call :copy_optional_dir analysis
call :copy_optional_dir model_single_responder
call :copy_optional_file submission.csv

rem OOF artifacts are useful for responder redundancy and combination analysis.
rem Cache shards remain excluded because they are much larger and reproducible.
call :copy_optional_artifact_dir "work/oof_models" "work\oof_models"
call :copy_optional_artifact_file "work/oof_responder_hat.dat" "work\oof_responder_hat.dat"
call :copy_optional_artifact_file "work/cache/cache.json" "work\cache\cache.json"
call :copy_optional_artifact_dir "work_single_responder/oof_models" "work_single_responder\oof_models"
call :copy_optional_artifact_file "work_single_responder/oof_responder_hat.dat" "work_single_responder\oof_responder_hat.dat"
call :copy_optional_artifact_file "work_single_responder/cache/cache.json" "work_single_responder\cache\cache.json"

call :require_file "model\metadata.json"
if errorlevel 1 exit /b 1
call :require_file "model\ablation_report.json"
if errorlevel 1 exit /b 1
call :require_file "model\target_feature_importance.csv"
if errorlevel 1 exit /b 1
call :require_file "model\target_lightgbm.txt"
if errorlevel 1 exit /b 1
call :require_file "audit\responder_audit_report.json"
if errorlevel 1 exit /b 1

echo [done] remote model, audit, analysis, and available OOF artifacts were copied
echo [note] cache metadata was copied, but large cache shard arrays were excluded
exit /b 0

:copy_required_dir
set "RESULT_DIR=%~1"
echo [pull] %REMOTE_HOST%:%REMOTE_STRATEGY%/%RESULT_DIR%/ -^> %LOCAL_STRATEGY%%RESULT_DIR%\
if not exist "%LOCAL_STRATEGY%%RESULT_DIR%" mkdir "%LOCAL_STRATEGY%%RESULT_DIR%"
scp -r "%REMOTE_HOST%:%REMOTE_STRATEGY%/%RESULT_DIR%/." "%LOCAL_STRATEGY%%RESULT_DIR%"
if errorlevel 1 (
    echo [error] failed to copy required directory: %RESULT_DIR%
    exit /b 1
)
exit /b 0

:copy_optional_dir
set "RESULT_DIR=%~1"
echo [pull] trying optional directory: %RESULT_DIR%\
if not exist "%LOCAL_STRATEGY%%RESULT_DIR%" mkdir "%LOCAL_STRATEGY%%RESULT_DIR%"
scp -r "%REMOTE_HOST%:%REMOTE_STRATEGY%/%RESULT_DIR%/." "%LOCAL_STRATEGY%%RESULT_DIR%"
if errorlevel 1 echo [warn] remote %RESULT_DIR%\ is unavailable; skipped
exit /b 0

:copy_optional_file
set "RESULT_FILE=%~1"
echo [pull] trying optional file: %RESULT_FILE%
scp "%REMOTE_HOST%:%REMOTE_STRATEGY%/%RESULT_FILE%" "%LOCAL_STRATEGY%%RESULT_FILE%"
if errorlevel 1 echo [warn] remote %RESULT_FILE% is unavailable; skipped
exit /b 0

:copy_optional_artifact_dir
set "REMOTE_ARTIFACT=%~1"
set "LOCAL_ARTIFACT=%~2"
echo [pull] trying OOF directory: %REMOTE_ARTIFACT%\
if not exist "%LOCAL_STRATEGY%%LOCAL_ARTIFACT%" mkdir "%LOCAL_STRATEGY%%LOCAL_ARTIFACT%"
scp -r "%REMOTE_HOST%:%REMOTE_STRATEGY%/%REMOTE_ARTIFACT%/." "%LOCAL_STRATEGY%%LOCAL_ARTIFACT%"
if errorlevel 1 echo [warn] remote %REMOTE_ARTIFACT%\ is unavailable; skipped
exit /b 0

:copy_optional_artifact_file
set "REMOTE_ARTIFACT=%~1"
set "LOCAL_ARTIFACT=%~2"
for %%D in ("%LOCAL_STRATEGY%%LOCAL_ARTIFACT%") do set "LOCAL_ARTIFACT_DIR=%%~dpD"
if not exist "%LOCAL_ARTIFACT_DIR%" mkdir "%LOCAL_ARTIFACT_DIR%"
echo [pull] trying OOF file: %REMOTE_ARTIFACT%
scp "%REMOTE_HOST%:%REMOTE_STRATEGY%/%REMOTE_ARTIFACT%" "%LOCAL_STRATEGY%%LOCAL_ARTIFACT%"
if errorlevel 1 echo [warn] remote %REMOTE_ARTIFACT% is unavailable; skipped
exit /b 0

:require_file
set "REQUIRED_FILE=%~1"
if not exist "%LOCAL_STRATEGY%%REQUIRED_FILE%" (
    echo [error] required result is missing: %LOCAL_STRATEGY%%REQUIRED_FILE%
    exit /b 1
)
for %%F in ("%LOCAL_STRATEGY%%REQUIRED_FILE%") do (
    if %%~zF EQU 0 (
        echo [error] required result is empty: %%~fF
        exit /b 1
    )
)
exit /b 0
