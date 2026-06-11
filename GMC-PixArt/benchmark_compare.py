#!/usr/bin/env python3
"""PixArt 文生图：基线 vs ToCa vs GMC 性能与 FID 对比。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.utils import save_image

GMC_PIXART = Path(__file__).resolve().parent
GMC_ROOT = GMC_PIXART.parent
sys.path.insert(0, str(GMC_PIXART))
sys.path.insert(0, str(GMC_ROOT))

from diffusion import DPMS  # noqa: E402
from diffusion.data.datasets.utils import ASPECT_RATIO_256_TEST  # noqa: E402
from diffusion.model.nets import PixArt_XL_2  # noqa: E402
from diffusion.model.t5 import T5Embedder  # noqa: E402
from diffusion.model.utils import prepare_prompt_ar  # noqa: E402
from tools.download import find_model  # noqa: E402

import diffusion.model.cache_functions as cache_functions_mod  # noqa: E402
import diffusion.model.dpm_solver as dpm_solver_mod  # noqa: E402

from config import DEFAULT_GMC_PIXART_CONFIG  # noqa: E402
from gmc_cache import gmc_cache_init  # noqa: E402
from gmc_pixart_block import (  # noqa: E402
    BASELINE_CACHE_KWARGS,
    TOCA_CACHE_KWARGS,
    apply_gmc_blocks,
)

ORIGINAL_CACHE_INIT = cache_functions_mod.cache_init


@dataclass
class RunResult:
    name: str
    seconds: float
    cache_stats: dict | None = None


@dataclass
class QualityMetrics:
    latent_mse: float
    latent_mae: float
    latent_rel_l2: float
    latent_cosine: float
    pixel_mse: float
    pixel_psnr: float


def _resolve_vae_path() -> str:
    for name in ('sd-vae-ft-ema', 'sd-vae-ft-mse'):
        local = GMC_ROOT / 'pretrained_models' / name
        if (local / 'config.json').is_file():
            return str(local)
    ext = Path('/home/chyao/projects/ToCa/pretrained_models/sd-vae-ft-ema')
    if (ext / 'config.json').is_file():
        return str(ext)
    return 'stabilityai/sd-vae-ft-mse'


def _resolve_model_path() -> str:
    for p in (
        GMC_ROOT / 'pretrained_models/PixArt-XL-2-256x256.pth',
        Path('/home/chyao/projects/ToCa/pretrained_models/PixArt-XL-2-256x256.pth'),
    ):
        if p.is_file():
            return str(p)
    raise FileNotFoundError('未找到 PixArt-XL-2-256x256.pth')


def _resolve_t5_path() -> str:
    for p in (
        GMC_ROOT / 'pretrained_models/t5_ckpts',
        Path('/home/chyao/projects/ToCa/pretrained_models/t5_ckpts'),
    ):
        if p.is_dir():
            return str(p)
    raise FileNotFoundError('未找到 T5 ckpts')


def _sync(device: str) -> None:
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def setup_cache(mode: str, gmc_cfg) -> None:
    if mode == 'gmc':
        def _init(model_kwargs, num_steps):
            depth = model_kwargs.get('gmc_depth', 28)
            return gmc_cache_init(gmc_cfg, num_steps, depth=depth)

        cache_functions_mod.cache_init = _init
        dpm_solver_mod.cache_init = _init
    else:
        cache_functions_mod.cache_init = ORIGINAL_CACHE_INIT
        dpm_solver_mod.cache_init = ORIGINAL_CACHE_INIT


def load_model(mode: str, latent_size: int, device: str, model_path: str | None = None) -> PixArt_XL_2:
    model = PixArt_XL_2(input_size=latent_size, lewei_scale=1).to(device)
    state_dict = find_model(model_path or _resolve_model_path())
    del state_dict['state_dict']['pos_embed']
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval().to(torch.float16)
    apply_gmc_blocks(model)
    model.to(device=device, dtype=torch.float16)
    return model


def build_model_kwargs(mode: str, hw, ar, emb_masks, model) -> dict:
    base = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
    if mode == 'gmc':
        base['gmc_depth'] = len(model.blocks)
        return base
    if mode == 'toca':
        return {**base, **TOCA_CACHE_KWARGS}
    return {**base, **BASELINE_CACHE_KWARGS}


@torch.inference_mode()
def encode_prompt(t5, prompt: str, image_size: int, device: str):
    prompt_clean, _, _, _, _ = prepare_prompt_ar(
        prompt, ASPECT_RATIO_256_TEST, device=device, show=False,
    )
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device)
    ar = torch.tensor([[1.0]], device=device)
    caption_embs, emb_masks = t5.get_text_embeddings([prompt_clean.strip()])
    caption_embs = caption_embs.float()[:, None]
    return caption_embs, emb_masks, hw, ar


@torch.inference_mode()
def sample_latents(
    model,
    mode: str,
    gmc_cfg,
    caption_embs,
    null_y_b,
    emb_masks,
    hw,
    ar,
    z: torch.Tensor,
    steps: int,
    cfg_scale: float,
    device: str,
) -> torch.Tensor:
    setup_cache(mode, gmc_cfg)
    model_kwargs = build_model_kwargs(mode, hw, ar, emb_masks, model)
    dpm = DPMS(
        model.forward_with_dpmsolver,
        condition=caption_embs,
        uncondition=null_y_b,
        cfg_scale=cfg_scale,
        model_kwargs=model_kwargs,
    )
    return dpm.sample(
        z, steps=steps, order=2, skip_type='time_uniform',
        method='multistep', model_kwargs=model_kwargs,
    )


def sample_timed(
    model,
    mode: str,
    gmc_cfg,
    caption_embs,
    null_y_b,
    emb_masks,
    hw,
    ar,
    z: torch.Tensor,
    steps: int,
    cfg_scale: float,
    device: str,
) -> RunResult:
    _sync(device)
    t0 = time.perf_counter()
    sample_latents(
        model, mode, gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
        z, steps, cfg_scale, device,
    )
    _sync(device)
    stats = None
    if mode == 'gmc':
        # stats live inside last cache_dic; approximate via re-sample not ideal
        stats = {'note': 'see gmc_pixart_block stats in full run'}
    return RunResult(name=mode, seconds=time.perf_counter() - t0, cache_stats=stats)


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def _decode_pixels(vae, latents: torch.Tensor) -> torch.Tensor:
    imgs = vae.decode(latents[:1] / 0.18215).sample
    return (imgs.clamp(-1, 1) + 1) / 2


def compute_quality(ref_latent, cand_latent, ref_pixel, cand_pixel) -> QualityMetrics:
    ref = ref_latent[:1].float()
    cand = cand_latent[:1].float()
    diff = cand - ref
    latent_mse = diff.pow(2).mean().item()
    latent_mae = diff.abs().mean().item()
    latent_rel_l2 = diff.norm().item() / ref.norm().clamp(min=1e-8).item()
    latent_cosine = F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0).item()
    px_diff = cand_pixel - ref_pixel
    pixel_mse = px_diff.pow(2).mean().item()
    pixel_psnr = 10 * math.log10(1.0 / max(pixel_mse, 1e-12))
    return QualityMetrics(
        latent_mse=latent_mse, latent_mae=latent_mae,
        latent_rel_l2=latent_rel_l2, latent_cosine=latent_cosine,
        pixel_mse=pixel_mse, pixel_psnr=pixel_psnr,
    )


def load_prompts(path: Path, limit: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    return lines[:limit]


def save_png(tensor_01: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(tensor_01, str(path))


def run_speed_benchmark(args, device, t5, gmc_cfg, latent_size) -> dict:
    model_b = load_model('baseline', latent_size, device, args.model_path)
    model_t = load_model('toca', latent_size, device, args.model_path)
    model_g = load_model('gmc', latent_size, device, args.model_path)

    caption_embs, emb_masks, hw, ar = encode_prompt(
        t5, args.speed_prompt, args.image_size, device,
    )
    null_y_b = model_b.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

    torch.manual_seed(args.seed)
    z = torch.randn(1, 4, latent_size, latent_size, device=device)

    results = {}
    for name, model in [('baseline', model_b), ('toca', model_t), ('gmc', model_g)]:
        times = []
        for i in range(args.warmup + args.runs):
            torch.manual_seed(args.seed)
            z_i = torch.randn(1, 4, latent_size, latent_size, device=device)
            r = sample_timed(
                model, name, gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
                z_i, args.steps, args.cfg_scale, device,
            )
            if i >= args.warmup:
                times.append(r.seconds)
        mean, std = _mean_std(times)
        results[name] = {'mean_s': mean, 'std_s': std, 'runs': args.runs}

    base = results['baseline']['mean_s']
    print('\n=== 性能对比 (DPM-Solver 采样耗时，不含 T5/VAE) ===')
    print(f'prompt: {args.speed_prompt[:60]}...' if len(args.speed_prompt) > 60 else f'prompt: {args.speed_prompt}')
    print(f'steps={args.steps}, cfg={args.cfg_scale}, seed={args.seed}, runs={args.runs}')
    print('┌──────────┬──────────────┬──────────┬──────────┐')
    print('│ 方法     │ 耗时 (mean)  │   std    │ speedup  │')
    print('├──────────┼──────────────┼──────────┼──────────┤')
    for name, label in [('baseline', '基线'), ('toca', 'ToCa'), ('gmc', 'GMC')]:
        m, s = results[name]['mean_s'], results[name]['std_s']
        sp = base / m if m > 0 else float('inf')
        print(f'│ {label:<8} │ {m:8.3f}s   │ {s:6.3f}s │ {sp:7.2f}x │')
    print('└──────────┴──────────────┴──────────┴──────────┘')
    if results['toca']['mean_s'] > 0:
        print(f"GMC 相对 ToCa: {results['toca']['mean_s'] / results['gmc']['mean_s']:.2f}x")
    return results


@torch.inference_mode()
def run_quality_benchmark(args, device, t5, vae, gmc_cfg, latent_size) -> dict:
    prompts = load_prompts(Path(args.prompts_file), args.quality_prompts)
    seeds = [int(s) for s in args.quality_seeds.split(',') if s.strip()]

    model_b = load_model('baseline', latent_size, device, args.model_path)
    model_t = load_model('toca', latent_size, device, args.model_path)
    model_g = load_model('gmc', latent_size, device, args.model_path)
    null_y_b = model_b.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

    toca_metrics: list[QualityMetrics] = []
    gmc_metrics: list[QualityMetrics] = []

    print('\n=== 质量对比（相对基线，同 prompt + seed）===')
    print(f'{len(prompts)} prompts × {len(seeds)} seeds, steps={args.steps}')

    for seed in seeds:
        for prompt in prompts:
            torch.manual_seed(seed)
            z = torch.randn(1, 4, latent_size, latent_size, device=device)
            caption_embs, emb_masks, hw, ar = encode_prompt(t5, prompt, args.image_size, device)

            ref = sample_latents(
                model_b, 'baseline', gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
                z, args.steps, args.cfg_scale, device,
            )
            toca = sample_latents(
                model_t, 'toca', gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
                z, args.steps, args.cfg_scale, device,
            )
            gmc = sample_latents(
                model_g, 'gmc', gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
                z, args.steps, args.cfg_scale, device,
            )

            ref_px = _decode_pixels(vae, ref)
            toca_metrics.append(compute_quality(ref, toca, ref_px, _decode_pixels(vae, toca)))
            gmc_metrics.append(compute_quality(ref, gmc, ref_px, _decode_pixels(vae, gmc)))

    def avg(ms: list[QualityMetrics]) -> QualityMetrics:
        n = len(ms)
        return QualityMetrics(
            latent_mse=sum(m.latent_mse for m in ms) / n,
            latent_mae=sum(m.latent_mae for m in ms) / n,
            latent_rel_l2=sum(m.latent_rel_l2 for m in ms) / n,
            latent_cosine=sum(m.latent_cosine for m in ms) / n,
            pixel_mse=sum(m.pixel_mse for m in ms) / n,
            pixel_psnr=sum(m.pixel_psnr for m in ms) / n,
        )

    t_avg = avg(toca_metrics)
    g_avg = avg(gmc_metrics)

    print('\n┌──────────┬────────────┬────────────┬──────────┬──────────┬────────────┐')
    print('│ 方法     │ latent MSE │ rel L2     │ cosine   │ PSNR(dB) │ pixel MSE  │')
    print('├──────────┼────────────┼────────────┼──────────┼──────────┼────────────┤')
    print(f'│ ToCa     │ {t_avg.latent_mse:10.2e} │ {t_avg.latent_rel_l2:10.4f} │ {t_avg.latent_cosine:10.6f} │ {t_avg.pixel_psnr:10.2f} │ {t_avg.pixel_mse:10.2e} │')
    print(f'│ GMC      │ {g_avg.latent_mse:10.2e} │ {g_avg.latent_rel_l2:10.4f} │ {g_avg.latent_cosine:10.6f} │ {g_avg.pixel_psnr:10.2f} │ {g_avg.pixel_mse:10.2e} │')
    print('└──────────┴────────────┴────────────┴──────────┴──────────┴────────────┘')

    return {'toca': asdict(t_avg), 'gmc': asdict(g_avg)}


@torch.inference_mode()
def run_fid_benchmark(args, device, t5, vae, gmc_cfg, latent_size) -> dict:
    from cleanfid import fid

    prompts = load_prompts(Path(args.prompts_file), args.num_fid)
    out_root = Path(args.fid_out_dir)
    dirs = {k: out_root / k for k in ('baseline', 'toca', 'gmc')}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    models = {k: load_model(k, latent_size, device, args.model_path) for k in dirs}
    null_y_b = models['baseline'].y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

    print(f'\n=== FID 评测：生成 {len(prompts)} 张/方法 → {out_root} ===')
    for idx, prompt in enumerate(prompts):
        seed = args.fid_seed + idx
        torch.manual_seed(seed)
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        caption_embs, emb_masks, hw, ar = encode_prompt(t5, prompt, args.image_size, device)
        for mode, model in models.items():
            lat = sample_latents(
                model, mode, gmc_cfg, caption_embs, null_y_b, emb_masks, hw, ar,
                z, args.steps, args.cfg_scale, device,
            )
            px = _decode_pixels(vae, lat)
            save_png(px, dirs[mode] / f'{idx:05d}.png')
        if (idx + 1) % 10 == 0 or idx == len(prompts) - 1:
            print(f'  已生成 {idx + 1}/{len(prompts)}')

    print('\n计算 FID (clean-fid, 相对基线生成集)...')
    fid_scores = {}
    for mode in ('toca', 'gmc'):
        score = fid.compute_fid(
            fdir1=str(dirs['baseline']),
            fdir2=str(dirs[mode]),
            mode='clean',
            num_workers=4,
            batch_size=32,
            verbose=False,
        )
        fid_scores[f'fid_vs_baseline_{mode}'] = float(score)
        print(f'  FID(基线 vs {mode.upper()}): {score:.4f}  (越低表示越接近基线分布)')

    if args.coco_ref and Path(args.coco_ref).is_dir():
        stats_name = args.coco_stats_name
        if not fid.test_stats_exists(stats_name, 'clean', 'custom'):
            print(f'  构建 COCO 参考统计: {args.coco_ref} → {stats_name}')
            fid.make_custom_stats(
                stats_name, args.coco_ref, mode='clean', num_workers=4,
            )
        for mode in ('baseline', 'toca', 'gmc'):
            score = fid.compute_fid(
                fdir1=str(dirs[mode]),
                dataset_name=stats_name,
                mode='clean',
                dataset_split='custom',
                dataset_res=256,
                num_workers=4,
                verbose=False,
            )
            fid_scores[f'fid_coco_{mode}'] = float(score)
            print(f'  FID(COCO vs {mode.upper()}): {score:.4f}')

    return fid_scores


def parse_args():
    p = argparse.ArgumentParser(description='PixArt 基线 vs ToCa vs GMC 对比')
    p.add_argument('--model_path', default=None)
    p.add_argument('--t5_path', default=None)
    p.add_argument('--vae_path', default=_resolve_vae_path())
    p.add_argument('--prompts_file', default=str(GMC_PIXART / 'benchmark_prompts.txt'))
    p.add_argument('--image_size', type=int, default=256)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg_scale', type=float, default=4.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--speed_prompt', default='A golden retriever playing in the snow.')
    p.add_argument('--skip_speed', action='store_true')
    p.add_argument('--skip_quality', action='store_true')
    p.add_argument('--skip_fid', action='store_true')
    p.add_argument('--quality_prompts', type=int, default=2)
    p.add_argument('--quality_seeds', default='0')
    p.add_argument('--num_fid', type=int, default=5)
    p.add_argument('--fid_seed', type=int, default=1000)
    p.add_argument('--fid_out_dir', default=str(GMC_PIXART / 'benchmark_fid'))
    p.add_argument('--coco_ref', default=None, help='COCO val 图像目录，用于真实 FID')
    p.add_argument('--coco_stats_name', default='mscoco_val2014_256')
    p.add_argument('--json_out', default=str(GMC_PIXART / 'benchmark_results.json'))
    p.add_argument('--casa_interval', type=int, default=4)
    p.add_argument('--mlp_anchor_step', type=int, default=30)
    p.add_argument('--mlp_interval', type=int, default=4)
    p.add_argument('--attn_interval', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.model_path:
        os.environ['PIXART_CKPT'] = args.model_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f'设备: {torch.cuda.get_device_name(0)}')
    else:
        print('警告: 未检测到 CUDA')

    gmc_cfg = DEFAULT_GMC_PIXART_CONFIG
    gmc_cfg.casa_interval = args.casa_interval if args.attn_interval is None else args.attn_interval
    gmc_cfg.mlp_anchor_step = args.mlp_anchor_step
    gmc_cfg.mlp_interval = args.mlp_interval
    latent_size = args.image_size // 8

    print('\n对比: 统一 Block | 基线 (每步全算) | ToCa (token 级) | GMC (步级 SA/CA + 分层 MLP)')
    print(f'steps={args.steps}, cfg={args.cfg_scale}, GMC casa_interval={gmc_cfg.casa_interval}, mlp_anchor={gmc_cfg.mlp_anchor_step}, mlp_interval={gmc_cfg.mlp_interval}')

    print('\n[加载 T5]...')
    t5 = T5Embedder(
        device='cuda' if device == 'cuda' else 'cpu',
        local_cache=True,
        cache_dir=args.t5_path or _resolve_t5_path(),
        torch_dtype=torch.float16,
    )

    report = {}
    if not args.skip_speed:
        report['speed'] = run_speed_benchmark(args, device, t5, gmc_cfg, latent_size)

    vae = None
    if not args.skip_quality or not args.skip_fid:
        print('\n[加载 VAE]...')
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        vae.eval()

    if not args.skip_quality and vae is not None:
        report['quality'] = run_quality_benchmark(args, device, t5, vae, gmc_cfg, latent_size)

    if not args.skip_fid and vae is not None:
        report['fid'] = run_fid_benchmark(args, device, t5, vae, gmc_cfg, latent_size)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f'\n结果已保存: {args.json_out}')


if __name__ == '__main__':
    main()
