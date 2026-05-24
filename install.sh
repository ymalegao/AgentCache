#!/bin/bash
set -e

VENV_DIR="venv"
VLLM_VERSION="0.20.0"

echo "=== Creating virtual environment: $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Installing pip ==="
pip install --upgrade pip --quiet

echo "=== Installing vLLM $VLLM_VERSION ==="
pip install "vllm==$VLLM_VERSION"

echo "=== Installing requirements ==="
pip install -r requirements.txt

echo "=== Patching vLLM with centroid injection files ==="
SITE=$(python -c "import site; print(site.getsitepackages()[0])")

cp vllm/centroid_injector.py       "$SITE/vllm/"
cp vllm/centroid_integration.py    "$SITE/vllm/"
cp vllm/v1/core/sched/scheduler.py "$SITE/vllm/v1/core/sched/"
cp vllm/v1/worker/gpu_model_runner.py "$SITE/vllm/v1/worker/"
cp vllm/v1/worker/gpu/model_runner.py "$SITE/vllm/v1/worker/gpu/"

echo "=== Verifying patch ==="
python -c "from vllm.centroid_injector import CentroidInjector; print('Patch OK')"

echo ""
echo "Done. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
