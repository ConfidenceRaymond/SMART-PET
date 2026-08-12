#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PROFILE="generic"
WITH_EXCEL=0
ASSET_PROFILE=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_environment.sh [options]

Create an isolated SMART-PET environment and install the repository.

Options:
  --alliance              Alliance Canada/Narval profile (tested PyTorch 2.6.0).
  --venv PATH             New environment path (default: REPO/.venv).
  --python PATH           Python interpreter used to create the venv.
  --with-excel            Install optional XLSX metadata support.
  --assets PROFILE        Install gdown and download verified public assets.
                          PROFILE: inference | finetune | all. Requires internet.
  -h, --help              Show this help.

Examples:
  bash scripts/setup_environment.sh
  bash scripts/setup_environment.sh --with-excel --assets inference
  bash scripts/setup_environment.sh --alliance --with-excel
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --alliance)
      PROFILE="alliance"
      shift
      ;;
    --venv)
      VENV="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --with-excel)
      WITH_EXCEL=1
      shift
      ;;
    --assets)
      ASSET_PROFILE="$2"
      case "$ASSET_PROFILE" in
        inference|finetune|all) ;;
        *) echo "[ERROR] --assets must be inference, finetune, or all" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

if [[ -e "$VENV" ]]; then
  echo "[ERROR] Environment path already exists: $VENV" >&2
  echo "Choose a new --venv path or remove the old environment first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("SMART-PET requires Python >=3.10")
print(f"[OK] Python {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Alliance EasyBuild modules can expose Python packages such as SciPy through
# EBPYTHONPREFIXES/PYTHONPATH. ANTs binaries do not require those Python paths.
unset PYTHONPATH || true
unset EBPYTHONPREFIXES || true
export PYTHONNOUSERSITE=1

if [[ "$PROFILE" == "alliance" ]]; then
  python -m pip install 'torch==2.6.0'
else
  python -m pip install 'torch>=2.2,<2.7'
fi

EXTRAS=""
if [[ $WITH_EXCEL -eq 1 && -n "$ASSET_PROFILE" ]]; then
  EXTRAS='[excel,assets]'
elif [[ $WITH_EXCEL -eq 1 ]]; then
  EXTRAS='[excel]'
elif [[ -n "$ASSET_PROFILE" ]]; then
  EXTRAS='[assets]'
fi

python -m pip install -c requirements/preprocessing-tested.txt ".${EXTRAS}"
python -m pip check

python - <<'PY'
import importlib.metadata
import sys
import torch

print(f"[OK] smartpet={importlib.metadata.version('smartpet')}")
print(f"[OK] python={sys.version.split()[0]}")
print(f"[OK] torch={torch.__version__}")
print(f"[OK] torch_cuda_runtime={torch.version.cuda}")
print(f"[OK] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] gpu={torch.cuda.get_device_name(0)}")
PY

for cli in \
  smartpet-prepare-external \
  smartpet-train \
  smartpet-infer \
  smartpet-infer-batch \
  smartpet-evaluate \
  smartpet-validate-manifest \
  smartpet-audit-checkpoint \
  smartpet-export-weights \
  smartpet-audit-weights \
  smartpet-convert-legacy-checkpoint \
  smartpet-audit-inference \
  smartpet-conformance \
  smartpet-download-assets; do
  command -v "$cli" >/dev/null
  "$cli" --help >/dev/null
  echo "[OK] $cli"
done

if [[ -n "$ASSET_PROFILE" ]]; then
  smartpet-download-assets \
    --profile "$ASSET_PROFILE" \
    --output-dir "$ROOT/resources"
fi

if [[ "$PROFILE" == "alliance" ]]; then
  echo "[INFO] For raw_activity sessions on Alliance, use:"
  echo "       source scripts/activate_alliance.sh"
else
  echo "[INFO] raw_activity additionally requires the official ANTs command-line tools."
fi

echo "[OK] SMART-PET environment ready: $VENV"
echo "Activate with: source '$VENV/bin/activate'"
