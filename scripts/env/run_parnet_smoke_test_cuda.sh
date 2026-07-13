#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.pixi/bin:$PATH"
export PIXI_CACHE_DIR="$HOME/.cache/pixi"
export UV_LOCK_TIMEOUT=1200

PARNET_REPO="$HOME/storage_ml4rg26-deconvgclip/twang/repos/parnet--demo--train-models"
IDEA1_REPO="$HOME/storage_ml4rg26-deconvgclip/twang/repos/idea1-globalclip-deconvolution"
CHECKPOINT="$HOME/storage_ml4rg26-deconvgclip/twang/models/parnet.7m-0.0.pt"

echo "Using Parnet repo: $PARNET_REPO"
echo "Using Idea1 repo:  $IDEA1_REPO"
echo "Using checkpoint:  $CHECKPOINT"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

cd "$PARNET_REPO"

pixi run -e parnet-dev-cu11 \
  python "$IDEA1_REPO/scripts/smoke_test_parnet.py" \
  --config "$IDEA1_REPO/configs/vm.yaml" \
  --checkpoint "$CHECKPOINT" \
  --num-windows 2 \
  --device cuda
