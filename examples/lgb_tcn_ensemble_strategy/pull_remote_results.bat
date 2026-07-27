@echo off
setlocal EnableExtensions

set "REMOTE_HOST=%~1"
if not defined REMOTE_HOST set "REMOTE_HOST=ustc-lab"

set "REMOTE_REPO=%~2"
if not defined REMOTE_REPO set "REMOTE_REPO=~/xhth"

set "STRATEGY_REL=examples/lgb_tcn_ensemble_strategy"
set "REMOTE_STRATEGY=%REMOTE_REPO%/%STRATEGY_REL%"
set "LOCAL_STRATEGY=%~dp0"

where scp >nul 2>nul
if errorlevel 1 (
    echo [error] scp was not found in PATH.
    exit /b 1
)

if not exist "%LOCAL_STRATEGY%model" mkdir "%LOCAL_STRATEGY%model"
echo [pull] TCN model and evaluation report
scp -r "%REMOTE_HOST%:%REMOTE_STRATEGY%/model/." "%LOCAL_STRATEGY%model"
if errorlevel 1 exit /b 1

if not exist "%LOCAL_STRATEGY%work" mkdir "%LOCAL_STRATEGY%work"
echo [pull] validation prediction artifact
scp "%REMOTE_HOST%:%REMOTE_STRATEGY%/work/validation_predictions.npz" "%LOCAL_STRATEGY%work\validation_predictions.npz"
if errorlevel 1 echo [warn] validation prediction artifact was unavailable

echo [done] TCN ensemble results copied
exit /b 0
