#!/usr/bin/env bash
# Launch the deployment console. Pass-through args, e.g.:
#   ./Deployement/run_gui.sh --demo --mode mock
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f deploy-venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source deploy-venv/bin/activate
fi
exec python -m Deployement.app --gui "$@"
