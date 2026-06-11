#!/usr/bin/env python3
"""DiT-XL/2 + GMC v4 单样本生成（DDIM）。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

GMC_DIT = os.path.dirname(os.path.abspath(__file__))
GMC_ROOT = os.path.dirname(GMC_DIT)
DIT_ROOT = os.path.join(GMC_ROOT, 'DiT')
CKPT = os.path.join(DIT_ROOT, 'pretrained_models/DiT-XL-2-256x256.pt')


def _resolve_vae_path() -> str:
    if os.environ.get('VAE_PATH'):
        return os.environ['VAE_PATH']
    for name in ('sd-vae-ft-mse', 'sd-vae-ft-ema'):
        local = os.path.join(GMC_ROOT, 'pretrained_models', name)
        if os.path.isfile(os.path.join(local, 'config.json')):
            return local
    return 'stabilityai/sd-vae-ft-mse'


def _load_vae(path: str, device: str) -> AutoencoderKL:
    if os.path.isdir(path):
        print(f'[VAE] 本地: {path}')
        return AutoencoderKL.from_pretrained(path).to(device)
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    return AutoencoderKL.from_pretrained(path).to(device)


VAE_PATH = _resolve_vae_path()
if not os.path.isdir(os.path.join(DIT_ROOT, 'diffusion')):
    raise FileNotFoundError(f'未找到 DiT：{DIT_ROOT}，请运行 bash scripts/setup_dit.sh')

sys.path.insert(0, DIT_ROOT)
sys.path.insert(0, GMC_ROOT)
sys.path.insert(0, GMC_DIT)

from diffusion import create_diffusion  # noqa: E402
from config import ALL_PRESETS  # noqa: E402
from gmc_model import DiTWithGMC  # noqa: E402
from gmc_utils import GMCConfig, build_sa_refresh_mask  # noqa: E402


def v4_config(base, *, anchor=30, interval=3):
    return replace(base, sa_cycle_length=3, enable_mlp_cache=False,
                   mlp_anchor_step=anchor, mlp_post_anchor_interval=interval)


def load_ckpt(model, path: str) -> None:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)), strict=False)


def parse_args():
    p = argparse.ArgumentParser(description='DiT GMC v4 generation')
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--class_id', type=int, default=207)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='gmc_v4_sample.png')
    p.add_argument('--vae_path', default=VAE_PATH)
    p.add_argument('--mlp_anchor_step', type=int, default=30)
    p.add_argument('--mlp_post_anchor_interval', type=int, default=3)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--sampler', choices=['ddim', 'p'], default='ddim')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    gmc_cfg = v4_config(ALL_PRESETS['default']['gmc'],
                        anchor=args.mlp_anchor_step, interval=args.mlp_post_anchor_interval)
    model = DiTWithGMC(depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=32,
                       gmc_config=gmc_cfg, total_sampling_steps=args.steps).to(device)
    load_ckpt(model, args.ckpt)
    model.eval()
    use_cache = not args.baseline
    model.enable_cache(use_cache)
    model.reset_cache()
    model.set_sampling_steps(args.steps)
    if use_cache:
        model.set_gmc_config(gmc_cfg)
        model._sa_refresh = build_sa_refresh_mask(gmc_cfg, args.steps, model.depth)
    diffusion = create_diffusion(str(args.steps))
    vae = _load_vae(args.vae_path, device)
    vae.eval()
    n = 2 if args.cfg > 1.0 else 1
    z = torch.randn(n, 4, 32, 32, device=device)
    y = torch.tensor([args.class_id] * n, device=device)
    fn = diffusion.ddim_sample_loop if args.sampler == 'ddim' else diffusion.p_sample_loop
    t0 = time.perf_counter()
    samples = fn(model.forward_with_cfg, z.shape, z, clip_denoised=False,
                 model_kwargs=dict(y=y, cfg_scale=args.cfg), progress=True, device=device)
    print(f'采样耗时: {time.perf_counter()-t0:.3f}s')
    if use_cache:
        print('GMC stats:', model.get_cache_stats())
    img = vae.decode(samples[:1] / 0.18215).sample
    save_image((img.clamp(-1, 1) + 1) / 2, args.out)
    print(f'已保存 → {args.out}')


if __name__ == '__main__':
    main()
