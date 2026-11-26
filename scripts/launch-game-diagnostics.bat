@echo off
rem ---------------------------------------------------------------------------
rem Launch with the diagnostics build + verbose logging + F3 HUD panel
rem (VICTORY_PANEL=1) + UMA Bridge and Velocity I/O enabled.
rem
rem   scripts\launch-game-diagnostics.bat
rem   scripts\launch-game-diagnostics.bat -NoPrecompile
rem   scripts\launch-game-diagnostics.bat -Safe
rem
rem   set VICTORY_REBUILD=1
rem   scripts\launch-game-diagnostics.bat         rebuild target\diag first
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
call "%~dp0config.bat" || goto :fail

if not exist "%VICTORY_TITLE_EXE%" (
    echo error: title executable not found:
    echo        %VICTORY_TITLE_EXE%
    echo Set VICTORY_GAME_DIR or edit scripts\config.bat.
    goto :fail
)

if defined VICTORY_REBUILD goto :build
if exist "%VICTORY_DIAG%" goto :launch

:build
echo Building victory ^(diag, diagnostics^), this takes a minute...
cargo build --locked --profile diag -p victory --features diagnostics || goto :fail

:launch
if not defined VICTORY_LOG set "VICTORY_LOG=victory_loader=debug,victory_gpu::graphics=debug,victory_hle::modules::dstorage=info,info"

echo Launching %VICTORY_TITLE_EXE% ^(diagnostics + HUD panel^)
echo   UMA Bridge  : %VICTORY_UMA_BRIDGE%
echo   Velocity IO : %VICTORY_VELOCITY_IO% (legacy BCPack: %VICTORY_VELOCITY_IO_LEGACY_BCPACK%)
echo   VICTORY_LOG : %VICTORY_LOG%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "& $env:VICTORY_PLAY -Exe $env:VICTORY_TITLE_EXE -Build diag -More @{VICTORY_PANEL='1';VICTORY_LOG=$env:VICTORY_LOG;VICTORY_UMA_BRIDGE=$env:VICTORY_UMA_BRIDGE;VICTORY_VELOCITY_IO=$env:VICTORY_VELOCITY_IO;VICTORY_VELOCITY_IO_LEGACY_BCPACK=$env:VICTORY_VELOCITY_IO_LEGACY_BCPACK} %*"
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo victory exited with status %RC%.
    goto :fail
)
exit /b 0

:fail
echo.
if not defined VICTORY_NO_PAUSE pause
exit /b 1
