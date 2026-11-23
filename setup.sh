#!/usr/bin/env bash
# One-shot setup for the CVD recolouring web app.
#
# Run from the root of this repo:   bash setup.sh
#
# What it does (safe to re-run; every step checks whether it is already done):
#   1. Clones the original paper code (tomgillooly/cvd_vit) at a pinned commit
#   2. Creates a Python 3.10 virtual environment and installs dependencies
#   3. Clones DINO at the pinned commit and applies the corrected patch
#   4. Adds the 'custom' option to --dataset_type in adjoint_multires_interp.py
#   5. Copies our own files (web app, CVD matrices, test scripts) into place
#   6. On WSL2, adds the library path that cuDNN needs
#
# Environment variables:
#   PYTHON_BIN   python executable to use (default: python3). Must be 3.9-3.11.
#   SKIP_PIP=1   skip the pip install step (used for testing the other steps)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python3}"

UPSTREAM_URL="https://github.com/tomgillooly/cvd_vit.git"
UPSTREAM_COMMIT="3147f3e1e5e863f013755ed95576d6da06689a36"
DINO_URL="https://github.com/facebookresearch/dino.git"
DINO_COMMIT="7c446df5b9f45747937fb0d72314eb9f7b66930a"

step() { printf '\n== %s ==\n' "$1"; }

# ---------------------------------------------------------------- 0. checks
step "0. Checking prerequisites"
command -v git >/dev/null || { echo "git is required."; exit 1; }
command -v "$PY" >/dev/null || { echo "Python not found: $PY"; exit 1; }

PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
case "$PY_VER" in
  3.9|3.10|3.11) echo "Python $PY_VER OK" ;;
  *)
    if [ "${SKIP_PIP:-0}" = "1" ]; then
      echo "Python $PY_VER (not 3.9-3.11, but SKIP_PIP=1 so it doesn't matter here)"
    else
      echo "Python $PY_VER is not supported (need 3.9-3.11, because the paper's code pins torch 2.0.1)."
      echo "Easiest fix: install Miniconda, then run:"
      echo "    conda create -n cvdvit python=3.10 -y && conda activate cvdvit"
      echo "and re-run this script."
      exit 1
    fi
    ;;
esac

if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU detected."
else
  echo "WARNING: no working NVIDIA GPU found (nvidia-smi failed)."
  echo "         The optimisation step needs a CUDA GPU. Continuing setup anyway."
fi

# ------------------------------------------------- 1. original paper code
step "1. Original paper code"
if [ ! -d "$ROOT/cvd_vit/.git" ]; then
  git clone --quiet "$UPSTREAM_URL" "$ROOT/cvd_vit"
fi
git -C "$ROOT/cvd_vit" checkout --quiet "$UPSTREAM_COMMIT" 2>/dev/null \
  || echo "(note: could not check out pinned commit; using whatever is cloned)"
echo "cvd_vit ready at $ROOT/cvd_vit"

# ------------------------------------------------------------ 2. python env
step "2. Python environment"
if [ "${SKIP_PIP:-0}" = "1" ]; then
  echo "SKIP_PIP=1 -> skipping pip install"
else
  if [ -z "${CONDA_PREFIX:-}" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    [ -d "$ROOT/venv" ] || "$PY" -m venv "$ROOT/venv"
    # shellcheck disable=SC1091
    source "$ROOT/venv/bin/activate"
    echo "Using virtual environment at $ROOT/venv"
  else
    echo "Using the already-active environment: ${CONDA_PREFIX:-$VIRTUAL_ENV}"
  fi
  python -m pip install --upgrade pip
  # torch alone is ~620 MB; on flaky connections (common on WSL2) pip needs
  # generous retries. Re-running this script resumes from pip's cache.
  python -m pip install -r "$ROOT/cvd_vit/requirements.txt" --retries 10 --timeout 120
  # Newer setuptools removed pkg_resources, which the pinned lightning-utilities needs.
  python -m pip install "setuptools<82" flask
fi

# ---------------------------------------------------------------- 3. DINO
step "3. DINO (with the attention hook patch)"
if [ ! -d "$ROOT/cvd_vit/dino/.git" ]; then
  git clone --quiet "$DINO_URL" "$ROOT/cvd_vit/dino"
fi
(
  cd "$ROOT/cvd_vit/dino"
  git checkout --quiet "$DINO_COMMIT"
  if grep -q "def get_first_n_attn" vision_transformer.py; then
    echo "Patch already applied."
  else
    # The dino.patch shipped upstream has no file headers and fails with
    # both `git apply` and `patch`; dino_fixed.patch is the corrected version.
    git apply "$ROOT/dino_fixed.patch"
    echo "Patch applied."
  fi
)

# ---------------------------------------- 4. add 'custom' dataset option
step "4. Enabling --dataset_type custom"
"$PY" - "$ROOT/cvd_vit/adjoint_multires_interp.py" <<'PYEOF'
import sys
path = sys.argv[1]
src = open(path).read()
old = "'wikiart_landscape', 'wikiart_still_life']"
new = "'wikiart_landscape', 'wikiart_still_life', 'custom']"
if "'custom']" in src:
    print("Already enabled.")
elif old in src:
    open(path, "w").write(src.replace(old, new, 1))
    print("Enabled.")
else:
    sys.exit("Could not find the dataset_type choices line to patch; "
             "the upstream file may have changed.")
PYEOF

# ------------------------------------------------- 5. our own files
step "5. Installing web app and helper scripts"
mkdir -p "$ROOT/cvd_vit/static" "$ROOT/cvd_vit/data/custom"
cp "$ROOT/overlay/app.py" \
   "$ROOT/overlay/cvd_matrices.py" \
   "$ROOT/overlay/smoke_test.py" \
   "$ROOT/overlay/visualize_results.py" "$ROOT/cvd_vit/"
cp "$ROOT/overlay/static/index.html" "$ROOT/cvd_vit/static/index.html"
echo "Copied into $ROOT/cvd_vit"

# --------------------------------------------------------- 6. WSL2 fix
if grep -qi microsoft /proc/version 2>/dev/null; then
  step "6. WSL2 detected"
  LINE='export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH'
  if grep -qF "/usr/lib/wsl/lib" "$HOME/.bashrc" 2>/dev/null; then
    echo "LD_LIBRARY_PATH fix already in ~/.bashrc"
  else
    echo "$LINE" >> "$HOME/.bashrc"
    echo "Added the cuDNN library path to ~/.bashrc."
    echo "Open a NEW terminal (or run: source ~/.bashrc) before launching the app."
  fi
fi

cat <<EOF

Setup finished.

Next:
  1. Activate your environment (conda activate cvdvit, or: source venv/bin/activate)
  2. bash run.sh --check     # quick environment test (downloads the ~80 MB DINO weights once)
  3. bash run.sh             # starts the web app at http://127.0.0.1:5000
EOF
