@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Follow the README setup instructions first.
  pause
  exit /b 1
)
echo Starting Expression + Pose Lab on camera 0.
echo Model loading may take 20-40 seconds. Press Q or Escape in the live window to quit.
echo No video or screenshots are saved by this launcher.
".venv\Scripts\python.exe" app.py --camera 0
if errorlevel 1 pause
