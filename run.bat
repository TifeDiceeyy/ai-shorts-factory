@echo off
REM One command, end to end:  run.bat soap
REM Windows counterpart of run.sh. Produces artifacts\<topic>\ with the mp4,
REM script json, captions, cost report and verification report. Exits non-zero
REM if any verification criterion fails or the topic is safety-blocked.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo no .venv found - run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt 1>&2
  exit /b 1
)

if "%~1"=="" (
  echo usage: run.bat ^<topic^> 1>&2
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m shorts_factory.pipeline %*
exit /b %ERRORLEVEL%
