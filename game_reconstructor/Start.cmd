@echo off
cd /d "%~dp0"
where /q pythonw.exe
if not errorlevel 1 (
    start "" pythonw.exe "%~dp0Start.pyw"
    exit /b
)
python reconstruct.py gui
if errorlevel 1 pause
