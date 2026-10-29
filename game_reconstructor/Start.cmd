@echo off
setlocal
set "APP_DIR=%~dp0app"
if not exist "%APP_DIR%\reconstructor\__main__.py" (
    echo The installed reconstruction program is missing from "%APP_DIR%".
    pause
    exit /b 1
)
set "PYTHONPATH=%APP_DIR%;%PYTHONPATH%"
where python.exe >nul 2>nul
if errorlevel 1 (
    echo Python 3 was not found on PATH.
    echo Install Python 3.11 or newer, then run this launcher again.
    pause
    exit /b 1
)
cd /d "%APP_DIR%"
python.exe -m reconstructor gui
if errorlevel 1 pause