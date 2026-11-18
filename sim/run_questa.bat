@echo off
rem QuestaSim launcher for sim/run_questa.ps1.
rem All arguments are forwarded unchanged.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_questa.ps1" %*
exit /b %ERRORLEVEL%
