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
if not exist "%runner%" set "runner=%VICTORY_RELEASE%"

set "VICTORY_UMA_BRIDGE=0"
set "VICTORY_VELOCITY_IO=0"
set "VICTORY_VELOCITY_IO_LEGACY_BCPACK=0"
if not defined VICTORY_AMBIENT_AIR set "VICTORY_AMBIENT_AIR=1"
if not defined VICTORY_CACHE_DIR set "VICTORY_CACHE_DIR=%VICTORY_GAME_DIR%\cache"
set "VICTORY_TEXTURE_NO_WAIT=1"
set "VICTORY_RELAY=2"
set "VICTORY_WIDE_BACK=1"
set "VICTORY_CARD_REHASH_RUNS=1"
set "VICTORY_JOIN_COPIES=1"
set "VICTORY_INDIRECT_WINDOW=1"
set "VICTORY_RELAY_COPY=1"
set "VICTORY_CARD_STEPS=2"
set "VICTORY_INDIRECT_BATCH=2"
set "VICTORY_MAP_THREADS=1"
set "VICTORY_TITLE_PRIORITIES=0"
set "VICTORY_CARD_AWAKE=1"
set "VICTORY_SHOW_AFTER=1"
set "VICTORY_AMBIENT=0.5"
set "VICTORY_THIN=1"
set "VICTORY_JOIN_PARTS=8"
set "VICTORY_UPLOAD_AHEAD=1"
set "VICTORY_LEVELS_ON_DEMAND=1"
set "VICTORY_CARD_BUFFERS_MIN_KB=64"
set "VICTORY_RAYTRACING=1"
SET "VICTORY_LAND=3"
if not defined VICTORY_DLSS set "VICTORY_DLSS=OFF"
if not defined VICTORY_FRAMEGEN set "VICTORY_FRAMEGEN=OFF"
if not defined VICTORY_SHADOW_SCALE set "VICTORY_SHADOW_SCALE=1"

if not defined VICTORY_DYNAMIC_SLOTS set "VICTORY_DYNAMIC_SLOTS=524288"
if not defined VICTORY_LOG if not defined RUST_LOG set "VICTORY_LOG=INFO"

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
