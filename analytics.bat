@echo off
REM YouTube analytics pull for already-published videos.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo no .venv found - run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt 1>&2
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m shorts_factory.analytics
exit /b %ERRORLEVEL%
