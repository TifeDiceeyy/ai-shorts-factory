@echo off
REM Review/approve queue at http://127.0.0.1:8420
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo no .venv found - run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt 1>&2
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m uvicorn shorts_factory.dashboard.app:app --host 127.0.0.1 --port 8420 --reload
exit /b %ERRORLEVEL%
