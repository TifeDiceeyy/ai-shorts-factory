@echo off
REM Fetch and verify citations for a topic:  retrieve.bat "roman concrete"
REM Needed before generating a topic the brain does not already cover.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo no .venv found - run: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt 1>&2
  exit /b 1
)

if "%~1"=="" (
  echo usage: retrieve.bat ^<topic^> 1>&2
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" -m shorts_factory.retrieval %*
exit /b %ERRORLEVEL%
