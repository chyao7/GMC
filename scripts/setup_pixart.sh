#!/usr/bin/env bash
# PixArt-α 推理：源码在 GMC-PixArt/，权重与 T5 下载到 pretrained_models/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIXART_DIR="$ROOT/GMC-PixArt"
CKPT="$ROOT/pretrained_models/PixArt-XL-2-256x256.pth"
T5_DIR="$ROOT/pretrained_models/t5_ckpts/t5-v1_1-xxl"
VAE_DIR="$ROOT/pretrained_models/sd-vae-ft-mse"

if [[ ! -d "$PIXART_DIR/diffusion" ]]; then
  echo ">>> 缺少 PixArt 源码：$PIXART_DIR/diffusion"
  echo "    请确保仓库已包含 GMC-PixArt/diffusion 与 GMC-PixArt/tools/"
  exit 1
fi
echo ">>> PixArt 源码：$PIXART_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo ">>> 下载 PixArt-XL-2-256x256.pth"
  mkdir -p "$ROOT/pretrained_models"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id='PixArt-alpha/PixArt-alpha',
    filename='PixArt-XL-2-256x256.pth',
    local_dir='${ROOT}/pretrained_models',
)
dest = '${CKPT}'
if os.path.abspath(path) != os.path.abspath(dest):
    shutil.move(path, dest)
print('Checkpoint saved to', dest)
PY
else
  echo ">>> Checkpoint 已存在: $CKPT"
fi

if [[ ! -f "$T5_DIR/config.json" ]]; then
  echo ">>> 下载 T5-v1.1-xxl"
  mkdir -p "$ROOT/pretrained_models/t5_ckpts"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    'google/t5-v1_1-xxl',
    local_dir='${T5_DIR}',
    local_dir_use_symlinks=False,
)
print('T5 saved to ${T5_DIR}')
PY
else
  echo ">>> T5 已存在: $T5_DIR"
fi

if [[ ! -f "$VAE_DIR/config.json" ]]; then
  echo ">>> 下载 VAE (stabilityai/sd-vae-ft-mse)"
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
  echo ">>> VAE 已存在: $VAE_DIR"
fi

echo ">>> 完成。运行："
echo "    cd $ROOT && python3 GMC-PixArt/generate.py \\"
echo "      --model_path $CKPT \\"
echo "      --t5_path $ROOT/pretrained_models/t5_ckpts \\"
echo "      --vae_path $VAE_DIR \\"
echo "      --prompt \"A cat wearing sunglasses.\""
