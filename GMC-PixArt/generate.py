#!/usr/bin/env python3
"""PixArt-α + GMC 单卡生成（参考 DGC/pixart_generate.py）。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
PIXART_ROOT = ROOT / 'PixArt-alpha-ToCa'
GMC_ROOT = Path(__file__).resolve().parents[1]
GMC_PIXART = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIXART_ROOT))
sys.path.insert(0, str(GMC_ROOT))
sys.path.insert(0, str(GMC_PIXART))

_ar_spec = importlib.util.spec_from_file_location(
    'pixart_ar_utils', PIXART_ROOT / 'diffusion/data/datasets/utils.py',
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
_ar_spec.loader.exec_module(_ar_mod)
ASPECT_RATIO_256_TEST = _ar_mod.ASPECT_RATIO_256_TEST

from diffusion import DPMS  # noqa: E402
from diffusion.model.nets import PixArt_XL_2  # noqa: E402
from diffusion.model.t5 import T5Embedder  # noqa: E402
from diffusion.model.utils import prepare_prompt_ar  # noqa: E402
from tools.download import find_model  # noqa: E402

import diffusion.model.dpm_solver as dpm_solver_mod  # noqa: E402
import diffusion.model.cache_functions as cache_functions_mod  # noqa: E402

from config import DEFAULT_GMC_PIXART_CONFIG  # noqa: E402
from gmc_cache import gmc_cache_init  # noqa: E402
from gmc_pixart_block import apply_gmc_blocks  # noqa: E402


def _patch_cache_init(gmc_cfg):
    def _init(model_kwargs, num_steps):
        depth = model_kwargs.get('gmc_depth', 28)
        return gmc_cache_init(gmc_cfg, num_steps, depth=depth)

    cache_functions_mod.cache_init = _init
    dpm_solver_mod.cache_init = _init


def parse_args():
    p = argparse.ArgumentParser(description='PixArt-α GMC generation')
    p.add_argument('--model_path', required=True)
    p.add_argument('--t5_path', required=True)
    p.add_argument('--vae_path', required=True)
    p.add_argument('--prompt', default='A golden retriever playing in the snow.')
    p.add_argument('--out', default='gmc_pixart_sample.png')
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg_scale', type=float, default=4.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attn_interval', type=int, default=4)
    p.add_argument('--no_cache', action='store_true')
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    gmc_cfg = DEFAULT_GMC_PIXART_CONFIG
    gmc_cfg.attn_interval = args.attn_interval

    latent_size = args.image_size // 8
    model = PixArt_XL_2(input_size=latent_size, lewei_scale=1).to(device)
    state_dict = find_model(args.model_path)
    del state_dict['state_dict']['pos_embed']
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval().to(torch.float16)

    if not args.no_cache:
        apply_gmc_blocks(model)
        _patch_cache_init(gmc_cfg)

    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    t5 = T5Embedder(
        device='cuda', local_cache=True, cache_dir=args.t5_path,
        torch_dtype=torch.float16,
    )

    prompt_clean, _, hw, ar, _ = prepare_prompt_ar(
        args.prompt, ASPECT_RATIO_256_TEST, device=device, show=False,
    )
    hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device)
    ar = torch.tensor([[1.0]], device=device)
    caption_embs, emb_masks = t5.get_text_embeddings([prompt_clean.strip()])
    caption_embs = caption_embs.float()[:, None]
    null_y_b = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    model_kwargs = dict(
        data_info={'img_hw': hw, 'aspect_ratio': ar},
        mask=emb_masks,
        gmc_depth=len(model.blocks),
    )
    dpm = DPMS(
        model.forward_with_dpmsolver,
        condition=caption_embs,
        uncondition=null_y_b,
        cfg_scale=args.cfg_scale,
        model_kwargs=model_kwargs,
    )
    samples = dpm.sample(
        z, steps=args.steps, order=2, skip_type='time_uniform',
        method='multistep', model_kwargs=model_kwargs,
    )
    img = vae.decode(samples / 0.18215).sample
    save_image(img, args.out, normalize=True, value_range=(-1, 1))
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
