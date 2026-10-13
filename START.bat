@echo off
title EarlyDrop
cd /d "%~dp0"
echo.
echo   EarlyDrop - starting up
echo   ------------------------
docker info >nul 2>&1
if errorlevel 1 (
  echo   Docker Desktop is not running.
  echo   Open Docker Desktop, wait until it says "Engine running", then double-click START.bat again.
  pause
  exit /b 1
)
echo   Building and starting. The first run downloads about 1 GB and takes a few minutes.
docker compose up -d --build
if errorlevel 1 (
  echo   Something went wrong. Take a screenshot of this window and send it to Shaurya.
  pause
  exit /b 1
)
echo   Waiting for the dashboard to come online...
:wait
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health; exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto wait
start "" http://localhost:8501
start "" http://localhost:8080
echo.
echo   Dashboard : http://localhost:8501
echo   Deck      : http://localhost:8080   (press F for fullscreen)
echo   To stop everything, double-click STOP.bat
echo.
pause
