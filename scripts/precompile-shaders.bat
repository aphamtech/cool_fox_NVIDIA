@echo off
rem ---------------------------------------------------------------------------
rem Translate every shader into the cache a run reads (`victory precompile`),
rem showing the precompile progress UI so the loading screen does not wait.
rem Extra arguments go to `precompile`:
rem
rem   scripts\precompile-shaders.bat
rem   scripts\precompile-shaders.bat --jobs 12 --validate
rem   scripts\precompile-shaders.bat --no-gui
rem
rem Run it again after a git pull that touches victory-shader.
rem ---------------------------------------------------------------------------
setlocal EnableExtensions
call "%~dp0config.bat" || goto :fail

if not exist "%VICTORY_GAME_DIR%\" (
    echo error: game folder not found: %VICTORY_GAME_DIR%
    echo Set VICTORY_GAME_DIR or edit scripts\config.bat.
    goto :fail
)

call :ensure_binary || goto :fail

echo Precompiling shaders in %VICTORY_GAME_DIR%
"%VICTORY_BIN%" precompile "%VICTORY_GAME_DIR%" %*
if not "%ERRORLEVEL%"=="0" goto :fail

echo.
echo Done. Optional: run "%VICTORY_BIN%" warm --used to warm driver PSO cache.
if not defined VICTORY_NO_PAUSE pause
exit /b 0

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
