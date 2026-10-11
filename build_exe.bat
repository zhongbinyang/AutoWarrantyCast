@echo off
setlocal
echo ========================================================
echo   Building Vehicle Warranty Cost Actuarial System (.exe)
echo ========================================================

set "PYINSTALLER_EXE=C:\Users\zhong\AppData\Local\Programs\Python\Python311\Scripts\pyinstaller.exe"

if not exist "%PYINSTALLER_EXE%" (
    echo [ERROR] PyInstaller not found at: %PYINSTALLER_EXE%
    exit /b 1
)

echo Cleaning old build artifacts...
if exist build rmdir /s /q build

echo Running PyInstaller build...
"%PYINSTALLER_EXE%" warranty_cost_system.spec --clean --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed with exit code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

echo ========================================================
echo   Build Successful! Executable is in the dist folder.
echo ========================================================
endlocal
