@echo off
rem ---------------------------------------------------------------------------
rem Show the resolved paths, check them, and report what is missing.
rem Nothing is launched. Extra arguments go to `victory`:
rem
rem   scripts\check-setup.bat
rem   scripts\check-setup.bat census --missing
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
call "%~dp0config.bat" || goto :fail

call :ensure_binary || goto :fail

if "%~1"=="" (
    "%VICTORY_BIN%" config
) else (
    "%VICTORY_BIN%" %*
)
set "RC=%ERRORLEVEL%"

echo.
echo binary      : %VICTORY_BIN%
echo uma_bridge  : %VICTORY_UMA_BRIDGE%
echo velocity_io : %VICTORY_VELOCITY_IO% (legacy BCPack: %VICTORY_VELOCITY_IO_LEGACY_BCPACK%)
if exist "%VICTORY_TITLE_EXE%" (
    echo title       : %VICTORY_TITLE_EXE%
) else (
    echo title       : MISSING - %VICTORY_TITLE_EXE%
)

if not defined VICTORY_NO_PAUSE pause
exit /b %RC%

:ensure_binary
if not defined VICTORY_REBUILD (
    if exist "%VICTORY_DIAG%" (
        set "VICTORY_BIN=%VICTORY_DIAG%"
        exit /b 0
    )
    if exist "%VICTORY_RELEASE%" (
        set "VICTORY_BIN=%VICTORY_RELEASE%"
        exit /b 0
    )
)
echo Building victory ^(diag, diagnostics^), this takes a minute...
cargo build --locked --profile diag -p victory --features diagnostics || exit /b 1
set "VICTORY_BIN=%VICTORY_DIAG%"
exit /b 0

:fail
echo.
if not defined VICTORY_NO_PAUSE pause
exit /b 1
