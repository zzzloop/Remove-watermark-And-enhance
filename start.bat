@echo off
setlocal
cd /d "%~dp0"
title Watermark Remover

echo.
echo Exit tip: press Ctrl+C to stop the server.
echo If Windows shows "Terminate batch job (Y/N)?", type Y and press Enter.
echo.

where powershell >nul 2>nul
if errorlevel 1 (
  echo PowerShell was not found.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. See startup.log in this folder.
  pause
  exit /b 1
)
