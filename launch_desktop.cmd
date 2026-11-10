@echo off
cd /d "%~dp0"
if exist "dist\Expression Lab\Expression Lab.exe" (
  start "" "dist\Expression Lab\Expression Lab.exe"
) else (
  ".venv\Scripts\python.exe" desktop.py
  if errorlevel 1 pause
)
