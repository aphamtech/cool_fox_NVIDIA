@echo off
rem ---------------------------------------------------------------------------
rem Launch with UMA Bridge, Velocity I/O, and fast Series S preset settings
rem (precompile UI enabled by default; pass -NoPrecompile or --no-precompile
rem to skip).
rem
rem   scripts\launch-game-fast.bat
rem   scripts\launch-game-fast.bat -NoPrecompile
rem   scripts\launch-game-fast.bat --no-precompile
rem ---------------------------------------------------------------------------
setlocal EnableExtensions DisableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PRECOMPILE_FLAG=--precompile"
if defined VICTORY_NO_PRECOMPILE if not "%VICTORY_NO_PRECOMPILE%"=="0" set "PRECOMPILE_FLAG="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--no-precompile" (
    set "PRECOMPILE_FLAG="
    shift
    goto :parse_args
)
if /i "%~1"=="-NoPrecompile" (
    set "PRECOMPILE_FLAG="
    shift
    goto :parse_args
)
echo Usage: scripts\launch-game-fast.bat [--no-precompile ^| -NoPrecompile] 1>&2
exit /b 2

:args_done
call "%SCRIPT_DIR%config.bat" || goto :fail

if not exist "%VICTORY_TITLE_EXE%" (
    echo error: title executable not found:
    echo        %VICTORY_TITLE_EXE%
    echo Set VICTORY_GAME_DIR or edit scripts\config.bat.
    goto :fail
)

rem "runner=%VICTORY_REPO%\victory-3d-rtx-pareto-residency.exe"
if not exist "%runner%" set "runner=%VICTORY_DIAG%"

echo Launching %VICTORY_TITLE_EXE%
echo   Runner      : %runner%
echo   Cache Dir   : %VICTORY_CACHE_DIR%
echo   UMA Bridge  : %VICTORY_UMA_BRIDGE%
echo   Velocity IO : %VICTORY_VELOCITY_IO% (legacy BCPack: %VICTORY_VELOCITY_IO_LEGACY_BCPACK%)
echo.

"%runner%" run "%VICTORY_TITLE_EXE%" --console series-s --execute %PRECOMPILE_FLAG%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo victory exited with status %RC%.
    echo See crashes.log in the data folder ^(scripts\check-setup.bat shows where^).
    goto :fail
)
exit /b 0

:fail
echo.
if not defined VICTORY_NO_PAUSE pause
exit /b 1
.5