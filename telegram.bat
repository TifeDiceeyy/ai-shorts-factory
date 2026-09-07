@echo off
REM Start the Telegram bot. Runs in the foreground - close the window to stop.
REM /plan drives: topic -> keywords -> language -> length -> confirm.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo no .venv found - run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt 1>&2
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m shorts_factory.telegram_bot
exit /b %ERRORLEVEL%
