#!/usr/bin/env python3
"""DiT-XL/2 + GMC 单样本生成示例。"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

GMC_DIT = os.path.dirname(__file__)
GMC_ROOT = os.path.abspath(os.path.join(GMC_DIT, '..'))
DIT_ROOT = os.path.normpath(os.path.join(GMC_DIT, '..', 'DiT'))
CKPT = os.path.join(DIT_ROOT, 'pretrained_models/DiT-XL-2-256x256.pt')
def _resolve_vae_path() -> str:
    if os.environ.get('VAE_PATH'):
        return os.environ['VAE_PATH']
    for name in ('sd-vae-ft-mse', 'sd-vae-ft-ema'):
        local = os.path.join(GMC_ROOT, 'pretrained_models', name)
        if os.path.isfile(os.path.join(local, 'config.json')):
            return local
    return 'stabilityai/sd-vae-ft-mse'


VAE_PATH = _resolve_vae_path()

if not os.path.isdir(os.path.join(DIT_ROOT, 'diffusion')):
    hint = ''
    if os.environ.get('DIT_ROOT'):
        hint = '\n提示：请执行 unset DIT_ROOT DIT_CKPT，避免旧环境变量干扰。'
    raise FileNotFoundError(
        f'未找到 DiT 工程：{DIT_ROOT}\n'
        f'请先运行：cd {GMC_ROOT} && bash scripts/setup_dit.sh{hint}'
    )

sys.path.insert(0, DIT_ROOT)
sys.path.insert(0, GMC_ROOT)
sys.path.insert(0, GMC_DIT)

from diffusion import create_diffusion  # noqa: E402
from config import ALL_PRESETS  # noqa: E402
from gmc_model import DiTWithGMC  # noqa: E402


def load_ckpt(model, path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'未找到 DiT 权重：{path}\n'
            '请先运行：bash scripts/setup_dit.sh'
        )
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)), strict=False)


def parse_args():
    p = argparse.ArgumentParser(description='DiT-XL/2 GMC generation')
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--preset', default='default', choices=list(ALL_PRESETS.keys()))
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--class_id', type=int, default=207)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='gmc_dit_sample.png')
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--vae_path', default=VAE_PATH)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    gmc_cfg = ALL_PRESETS[args.preset]['gmc']
    model = DiTWithGMC(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=32,
        gmc_config=gmc_cfg, total_sampling_steps=args.steps,
    ).to(device)
    load_ckpt(model, args.ckpt)
    model.eval()

    use_cache = not args.no_cache
    model.enable_cache(use_cache)
    model.reset_cache()

    diffusion = create_diffusion(str(args.steps))
    model.set_sampling_steps(args.steps)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    vae.eval()

    n = 2 if args.cfg > 1.0 else 1
    z = torch.randn(n, 4, 32, 32, device=device)
    y = torch.tensor([args.class_id] * n, device=device)
    model_kwargs = dict(y=y, cfg_scale=args.cfg)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device,
    )
    if use_cache:
        print('GMC stats:', model.get_cache_stats())

    img = vae.decode(samples[:1] / 0.18215).sample
    img = (img.clamp(-1, 1) + 1) / 2
    save_image(img, args.out)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
