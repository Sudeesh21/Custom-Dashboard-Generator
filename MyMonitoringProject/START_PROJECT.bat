@echo off
TITLE My Monitoring Project Starter
CLS

REM RECTIFIED: CRITICAL FIX for "file not found" errors.
REM Set current directory to the location of this script.
cd /d "%~dp0"

echo =====================================================
echo       MY MONITORING PROJECT - AUTOMATED START
echo =====================================================
echo.
echo NOTE: This script requires Administrator privileges to
echo       install the Windows Exporter service.
echo.
echo Checking for admin rights...

REM This command checks if the script is running as Administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Administrator privileges confirmed.
) else (
    color 0C
    echo.
    echo [ERROR] Current permissions inadequate.
    echo =====================================================
    echo PLEASE CLOSE THIS WINDOW.
    echo RIGHT-CLICK this file and select "Run as Administrator".
    echo =====================================================
    pause
    exit /b
)

echo.
echo =====================================================
echo STEP 1: Installing/Checking Windows Exporter Service
echo =====================================================
REM Run the PowerShell script we created in Step 1, bypassing execution policies
powershell -NoProfile -ExecutionPolicy Bypass -File ".\install_exporter.ps1"

IF %ERRORLEVEL% NEQ 0 (
   color 0C
   echo.
   echo [ERROR] Windows Exporter installation failed.
   pause
   exit /b
)


echo.
echo =====================================================
echo STEP 2: Starting Docker Containers (Compose)
echo =====================================================
echo Ensuring Docker Desktop is running...
docker info >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
   color 0C
   echo.
   echo [ERROR] Docker is not running! 
   echo Please start Docker Desktop and try again.
   pause
   exit /b
)

echo Building images (if necessary) and starting containers...
echo Using your existing docker-compose.yml configuration.
REM Start containers in detached mode using your specific compose file
docker-compose up -d --build

IF %ERRORLEVEL% NEQ 0 (
   color 0C
   echo.
   echo [ERROR] Docker Compose failed to start. Check the error messages above.
   pause
   exit /b
)


echo.
echo =====================================================
echo           SETUP COMPLETE - SYSTEMS ONLINE
echo =====================================================
echo.
echo [1] Grafana Dashboard:    http://localhost:3001
echo     (Login: admin / admin)
echo.
echo [2] Prometheus UI:        http://localhost:9090
echo.
echo [3] Pushgateway:          http://localhost:9091
echo.
echo [4] Predictor AI:         Running in background
echo.
echo =====================================================
echo You may close this window. The services will run in the background.
pause