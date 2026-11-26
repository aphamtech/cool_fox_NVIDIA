@echo off
rem ---------------------------------------------------------------------------
rem Launch with UMA Bridge and Velocity I/O forced ON (shortcut for
rem scripts\launch-game.bat).
rem
rem   scripts\launch-game-fast.bat
rem   scripts\launch-game-fast.bat -NoPrecompile
rem   scripts\launch-game-fast.bat -RayTracing
rem ---------------------------------------------------------------------------
setlocal EnableExtensions

set "VICTORY_UMA_BRIDGE=1"
set "VICTORY_VELOCITY_IO=1"
set "VICTORY_VELOCITY_IO_LEGACY_BCPACK=1"

call "%~dp0launch-game.bat" %*
exit /b %ERRORLEVEL%
