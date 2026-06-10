#!/usr/bin/env bash
# 将官方 DiT 工程与预训练权重下载到 GMC/DiT/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIT_DIR="$ROOT/DiT"
CKPT="$DIT_DIR/pretrained_models/DiT-XL-2-256x256.pt"

if [[ ! -d "$DIT_DIR/diffusion" ]]; then
  echo ">>> Cloning facebookresearch/DiT into $DIT_DIR"
  git clone --depth 1 https://github.com/facebookresearch/DiT.git "$DIT_DIR"
else
  echo ">>> DiT code already present at $DIT_DIR"
fi

if [[ ! -f "$CKPT" ]]; then
  echo ">>> Downloading DiT-XL-2-256x256.pt (~2.5GB)"
  (cd "$DIT_DIR" && python3 download.py)
else
  echo ">>> Checkpoint already present: $CKPT"
fi

echo ">>> Done. Run:"
echo "    cd $ROOT && python3 GMC-DiT/generate.py --preset default --class_id 207"
