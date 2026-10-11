@echo off
REM One-click launcher for the vehicle warranty cost system (Windows).
REM ASCII-only on purpose (see the note in setup.bat about cmd.exe code pages).
setlocal

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
streamlit run app.py
pause
