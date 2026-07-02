#!/usr/bin/env bash
set -euo pipefail

# Run this once from the repository root on ASA-X:
#   bash cluster/asax/setup_env.sh

cd "$(dirname "$0")/../.."

# If ASA-X requires modules, uncomment and adjust one of these after checking `module avail`.
# module purge
# module load python/3.11
# module load gcc

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv-tdc-hmh}"

echo "Using Python command: ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import sys
print("Python:", sys.version)
import torch
import rdkit
import tdc
import sklearn
import scipy
print("torch:", torch.__version__)
print("rdkit import OK")
print("tdc import OK")
print("sklearn:", sklearn.__version__)
print("scipy:", scipy.__version__)
PY

echo
echo "Environment ready:"
echo "  source ${ENV_DIR}/bin/activate"

