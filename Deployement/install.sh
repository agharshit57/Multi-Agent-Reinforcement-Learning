#!/usr/bin/env bash
# Installs the Cyber MARL deployment app (Linux/macOS/WSL).
# Usage: bash Deployement/install.sh [--no-venv]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "${1:-}" != "--no-venv" ]; then
  python3 -m venv deploy-venv
  # shellcheck disable=SC1091
  source deploy-venv/bin/activate
fi

python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
python -m pip install -r Deployement/requirements-deploy.txt

echo "--- full test suite (no weights, no network) ---"
python -m pytest Deployement/tests/ -q || python -m unittest Deployement.tests.test_deployment
echo "--- headless demo ---"
python -m Deployement.app --demo --headless --cycles 5 --mode shadow
echo "installed. Launch the console with:"
echo "  ./Deployement/run_gui.sh --demo --mode mock"
echo "Live enforcement stays disabled unless you pass --enable-live"
echo "with --mode live AND wire an EnforcementBackend. Secrets are"
echo "never stored in files: export DEPLOY_*_TOKEN-style variables."
