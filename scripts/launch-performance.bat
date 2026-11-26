@echo off
rem ---------------------------------------------------------------------------
rem Run Play-VIctory.cmd from the repository root with the performance set
rem (Series S profile, Pareto residency):
rem
rem   scripts\launch-performance.bat
rem
rem VICTORY_GAME_DIR must point at your own dump, as Play-VIctory.cmd requires.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions

for %%I in ("%~dp0..") do set "VICTORY_REPO=%%~fI"
set "VICTORY_LAUNCHER=%VICTORY_REPO%\Play-VIctory.cmd"

if not exist "%VICTORY_LAUNCHER%" (
    echo error: launcher script not found: %VICTORY_LAUNCHER%
    goto :fail
)

pushd "%VICTORY_REPO%" || goto :fail
call "%VICTORY_LAUNCHER%" pareto-residency
set "RC=%ERRORLEVEL%"
popd

if not "%RC%"=="0" (
    echo.
    echo Play-VIctory.cmd exited with status %RC%.
    goto :fail
)
exit /b 0

:fail
echo.
if not defined VICTORY_NO_PAUSE pause
exit /b 1
