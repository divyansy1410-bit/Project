@echo off
color 0B
echo ============================================
echo  IPL Score Predictor - Command Prompt Runner
echo ============================================
echo.

cd /d "%~dp0"

echo [Step 1] Starting Flask Server...
echo.
echo Please keep this window open to run the dashboard.
echo The dashboard will be available at: http://127.0.0.1:5000
echo Turn off by pressing CTRL + C
echo.

set PYTHONIOENCODING=utf-8
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" app.py

pause
