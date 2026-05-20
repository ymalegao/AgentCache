#!/bin/bash
# Download a HuggingFace model into models/<name>/.
# Gated models (e.g. Llama) require: huggingface-cli login
set -e

MODEL=${1:-"meta-llama/Llama-3.2-1B-Instruct"}
DEST="models/$(basename "$MODEL")"

mkdir -p "$DEST"
source venv/bin/activate

echo "Downloading $MODEL -> $DEST"
huggingface-cli download "$MODEL" --local-dir "$DEST"

echo ""
echo "Model saved to: $(pwd)/$DEST"
echo "Use with: --model $(pwd)/$DEST"
