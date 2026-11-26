@echo off
rem ---------------------------------------------------------------------------
rem Launch the title via scripts\play.ps1 (diag build) with UMA Bridge
rem (ReBAR + aliasing + barrier batching), Velocity I/O (IOCP async ring +
rem Async Compute decompression), and the launch shader precompile UI enabled.
rem
rem   scripts\launch-game.bat                     default speed set + precompile UI
rem   scripts\launch-game.bat -NoPrecompile       skip shader precompilation UI on launch
rem   scripts\launch-game.bat -Safe               safer fallback settings
rem   scripts\launch-game.bat -RayTracing         enable ray-traced indirect light
rem   scripts\launch-game.bat -Build release      run target\release\victory.exe
rem
rem   set VICTORY_REBUILD=1
rem   scripts\launch-game.bat                     rebuild target\diag first
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
call "%~dp0config.bat" || goto :fail

if not exist "%VICTORY_TITLE_EXE%" (
    echo error: title executable not found:
    echo        %VICTORY_TITLE_EXE%
    echo Set VICTORY_GAME_DIR or edit scripts\config.bat.
    goto :fail
)

if not exist "%VICTORY_PLAY%" (
    echo error: launcher script not found: %VICTORY_PLAY%
    goto :fail
)

call :ensure_diag || goto :fail

echo Launching %VICTORY_TITLE_EXE%
echo   UMA Bridge  : %VICTORY_UMA_BRIDGE%
echo   Velocity IO : %VICTORY_VELOCITY_IO% (legacy BCPack: %VICTORY_VELOCITY_IO_LEGACY_BCPACK%)
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "& $env:VICTORY_PLAY -Exe $env:VICTORY_TITLE_EXE -More @{VICTORY_UMA_BRIDGE=$env:VICTORY_UMA_BRIDGE;VICTORY_VELOCITY_IO=$env:VICTORY_VELOCITY_IO;VICTORY_VELOCITY_IO_LEGACY_BCPACK=$env:VICTORY_VELOCITY_IO_LEGACY_BCPACK} %*"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo victory exited with status %RC%.
    echo See crashes.log in the data folder ^(scripts\check-setup.bat shows where^).
    goto :fail
)
exit /b 0

:ensure_diag
if defined VICTORY_REBUILD goto :build_diag
if exist "%VICTORY_DIAG%" exit /b 0
:build_diag
echo Building victory ^(diag, diagnostics^), this takes a minute...
cargo build --locked --profile diag -p victory --features diagnostics || exit /b 1
exit /b 0

:fail
echo.
if not defined VICTORY_NO_PAUSE pause
exit /b 1
