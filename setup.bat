@echo off
REM One-click environment setup for the vehicle warranty cost system (Windows).
REM NOTE: This file is intentionally ASCII-only. cmd.exe parses .bat files using
REM the system ANSI code page (GBK on Chinese Windows), so non-ASCII text here
REM would be mis-decoded and break parsing.
setlocal

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment .venv ...
python -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment and upgrading pip ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/3] Installing dependencies from requirements.txt ...
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some dependencies failed to install (possibly optional pymc/arviz).
    echo        If you only need the MLE Weibull model, remove pymc/arviz from
    echo        requirements.txt and run setup.bat again.
)

echo.
echo Setup complete. Run start.bat to launch the system.
pause
