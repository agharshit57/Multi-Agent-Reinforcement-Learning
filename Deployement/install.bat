@echo off
REM Installs the Cyber MARL deployment app (Windows).
REM Usage: Deployement\install.bat
setlocal
cd /d "%~dp0.."

if not exist deploy-venv (
  python -m venv deploy-venv
)
call deploy-venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
python -m pip install -r Deployement\requirements-deploy.txt

echo --- full test suite (no weights, no network) ---
python -m pytest Deployement\tests\ -q
if errorlevel 1 (
  echo pytest unavailable, falling back to unittest
  python -m unittest Deployement.tests.test_deployment
)
if errorlevel 1 exit /b 1
echo --- headless demo ---
python -m Deployement.app --demo --headless --cycles 5 --mode shadow
if errorlevel 1 exit /b 1
echo installed. Launch the console with:
echo   Deployement\run_gui.bat --demo --mode mock
echo Live enforcement stays disabled unless you pass --enable-live
echo with --mode live AND wire an EnforcementBackend. Secrets are
echo never stored in files: export DEPLOY_*-TOKEN-style variables.
