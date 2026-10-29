@echo off
rem Double-click to set up First-Mover Flow on Windows (runs setup.ps1; Python does not need to be installed).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if "%~1"=="" pause
