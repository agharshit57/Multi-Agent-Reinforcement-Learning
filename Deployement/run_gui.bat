@echo off
REM Launch the deployment console. Example:
REM   Deployement\run_gui.bat --demo --mode mock
setlocal
cd /d "%~dp0.."
if exist deploy-venv\Scripts\activate.bat call deploy-venv\Scripts\activate.bat
python -m Deployement.app --gui %*
