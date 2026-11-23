#!/usr/bin/env bash
# Start the web app (or run the environment check).
#   bash run.sh           start the server at http://127.0.0.1:5000
#   bash run.sh --check   run the smoke test instead

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${CONDA_PREFIX:-}" ] && [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
fi

# cuDNN on WSL2 can't find libcuda.so without this
if grep -qi microsoft /proc/version 2>/dev/null; then
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

cd "$ROOT/cvd_vit"

if [ "${1:-}" = "--check" ]; then
  exec python smoke_test.py
fi

echo "Starting on http://127.0.0.1:5000  (Ctrl+C to stop)"
exec python app.py
