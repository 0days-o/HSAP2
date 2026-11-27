@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo HEART-SOUND AI SCIENCE FAIR PROJECT
echo Educational prototype only - NOT a medical diagnostic device
echo ================================================================
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher was not found. Install Python 3.11 from:
  echo https://www.python.org/downloads/
  echo During installation, enable the option to add Python to PATH.
  pause
  exit /b 1
)
py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 was not found. Please install Python 3.11 from:
  echo https://www.python.org/downloads/
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating the project environment...
  py -3.11 -m venv .venv
  if errorlevel 1 goto failed
)
call ".venv\Scripts\activate.bat"
echo Installing/checking software libraries. This is only needed the first time or after updates.
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements.txt
if errorlevel 1 goto failed
python main.py
goto finished
:failed
echo.
echo Setup failed. Check your internet connection and read docs\BUILD_GUIDE.md.
:finished
echo.
pause
endlocal
