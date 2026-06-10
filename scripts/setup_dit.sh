#!/usr/bin/env bash
# 将 DiT 工程、预训练权重与 VAE 下载到 GMC/ 目录下（自包含，无需外部路径）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIT_DIR="$ROOT/DiT"
CKPT="$DIT_DIR/pretrained_models/DiT-XL-2-256x256.pt"
VAE_DIR="$ROOT/pretrained_models/sd-vae-ft-mse"

if [[ ! -d "$DIT_DIR/diffusion" ]]; then
  echo ">>> Cloning facebookresearch/DiT into $DIT_DIR"
  git clone --depth 1 https://github.com/facebookresearch/DiT.git "$DIT_DIR"
else
  echo ">>> DiT code already present at $DIT_DIR"
fi

if [[ ! -f "$CKPT" ]]; then
  echo ">>> Downloading DiT-XL-2-256x256.pt (~2.5GB)"
  (cd "$DIT_DIR" && python3 -c "from download import download_model; download_model('DiT-XL-2-256x256.pt')")
else
  echo ">>> Checkpoint already present: $CKPT"
fi

if [[ ! -f "$VAE_DIR/config.json" ]]; then
  echo ">>> Downloading VAE (stabilityai/sd-vae-ft-mse)"
  mkdir -p "$ROOT/pretrained_models"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    'stabilityai/sd-vae-ft-mse',
    local_dir='${VAE_DIR}',
    local_dir_use_symlinks=False,
)
print('VAE saved to ${VAE_DIR}')
PY
else
  echo ">>> VAE already present: $VAE_DIR"
fi

echo ">>> Done. Run:"
echo "    cd $ROOT && python3 GMC-DiT/generate.py --preset default --class_id 207"
