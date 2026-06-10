#!/usr/bin/env python3
"""DiT 基线 vs GMC v1 vs GMC v2 性能与质量对比。"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

GMC_DIT = os.path.dirname(os.path.abspath(__file__))
GMC_ROOT = os.path.dirname(GMC_DIT)
DIT_ROOT = os.path.join(GMC_ROOT, 'DiT')
CKPT = os.path.join(DIT_ROOT, 'pretrained_models/DiT-XL-2-256x256.pt')

sys.path.insert(0, DIT_ROOT)
sys.path.insert(0, GMC_ROOT)
sys.path.insert(0, GMC_DIT)

from diffusion import create_diffusion  # noqa: E402
from config import ALL_PRESETS  # noqa: E402
from gmc_model import DiTWithGMC  # noqa: E402
from gmc_utils import GMCConfig, build_sa_refresh_mask  # noqa: E402
import gmc_utils_v1 as v1_utils  # noqa: E402


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
    if os.environ.get('VAE_PATH'):
        return os.environ['VAE_PATH']
    for name in ('sd-vae-ft-mse', 'sd-vae-ft-ema'):
        local = os.path.join(GMC_ROOT, 'pretrained_models', name)
        if os.path.isfile(os.path.join(local, 'config.json')):
            return local
    return 'stabilityai/sd-vae-ft-mse'


def _load_vae(path: str, device: str) -> AutoencoderKL:
    if os.path.isdir(path):
        return AutoencoderKL.from_pretrained(path).to(device)
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    return AutoencoderKL.from_pretrained(path).to(device)


def _sync(device: str) -> None:
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def load_ckpt(model, path: str) -> None:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)), strict=False)


def build_model(device: str, steps: int) -> DiTWithGMC:
    model = DiTWithGMC(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=32,
        gmc_config=GMCConfig(), total_sampling_steps=steps,
    ).to(device)
    load_ckpt(model, CKPT)
    model.eval()
    return model


def _v1_config(base: GMCConfig) -> GMCConfig:
    return replace(base, enable_mlp_cache=True, attn_interval=base.attn_interval)


def _v2_config(base: GMCConfig) -> GMCConfig:
    return replace(base, enable_mlp_cache=False, sa_cycle_length=3)


def _apply_v1_schedule(model: DiTWithGMC, cfg: GMCConfig, steps: int) -> None:
    model.set_gmc_config(cfg)
    mask_1d = v1_utils.build_sa_refresh_mask(cfg, steps)
    model._sa_refresh = [list(mask_1d) for _ in range(model.depth)]


def _apply_v2_schedule(model: DiTWithGMC, cfg: GMCConfig, steps: int) -> None:
    model.set_gmc_config(cfg)
    model._sa_refresh = build_sa_refresh_mask(cfg, steps, model.depth)


@torch.no_grad()
def sample_latents(
    model: DiTWithGMC,
    diffusion,
    z: torch.Tensor,
    y: torch.Tensor,
    cfg: float,
    device: str,
    use_cache: bool,
    steps: int,
    schedule_fn=None,
) -> torch.Tensor:
    model.enable_cache(use_cache)
    model.reset_cache()
    model.set_sampling_steps(steps)
    if schedule_fn is not None:
        schedule_fn(model)
    model_kwargs = dict(y=y, cfg_scale=cfg)
    return diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=False, device=device,
    )


def sample_once(
    model: DiTWithGMC,
    diffusion,
    z: torch.Tensor,
    y: torch.Tensor,
    cfg: float,
    device: str,
    use_cache: bool,
    steps: int,
    schedule_fn=None,
) -> RunResult:
    _sync(device)
    t0 = time.perf_counter()
    sample_latents(model, diffusion, z, y, cfg, device, use_cache, steps, schedule_fn)
    _sync(device)
    elapsed = time.perf_counter() - t0
    stats = model.get_cache_stats() if use_cache else None
    return RunResult(name='', seconds=elapsed, cache_stats=stats)


def run_trials(
    model: DiTWithGMC,
    diffusion,
    z: torch.Tensor,
    y: torch.Tensor,
    cfg: float,
    device: str,
    use_cache: bool,
    steps: int,
    runs: int,
    warmup: int,
    schedule_fn=None,
) -> list[RunResult]:
    for _ in range(warmup):
        sample_once(model, diffusion, z, y, cfg, device, use_cache, steps, schedule_fn)
    return [
        sample_once(model, diffusion, z, y, cfg, device, use_cache, steps, schedule_fn)
        for _ in range(runs)
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def _sa_compute_ratio(sa_mask: list[list[bool]]) -> float:
    total = sum(len(row) for row in sa_mask)
    compute = sum(sum(row) for row in sa_mask)
    return compute / total if total else 0.0


def _decode_to_pixels(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    imgs = vae.decode(latents[:1] / 0.18215).sample
    return (imgs.clamp(-1, 1) + 1) / 2


def compute_quality(ref_latent: torch.Tensor, cand_latent: torch.Tensor, ref_pixel, cand_pixel) -> QualityMetrics:
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
        latent_mse=latent_mse,
        latent_mae=latent_mae,
        latent_rel_l2=latent_rel_l2,
        latent_cosine=latent_cosine,
        pixel_mse=pixel_mse,
        pixel_psnr=pixel_psnr,
    )


def _avg_quality(metrics: list[QualityMetrics]) -> QualityMetrics:
    n = len(metrics)
    return QualityMetrics(
        latent_mse=sum(m.latent_mse for m in metrics) / n,
        latent_mae=sum(m.latent_mae for m in metrics) / n,
        latent_rel_l2=sum(m.latent_rel_l2 for m in metrics) / n,
        latent_cosine=sum(m.latent_cosine for m in metrics) / n,
        pixel_mse=sum(m.pixel_mse for m in metrics) / n,
        pixel_psnr=sum(m.pixel_psnr for m in metrics) / n,
    )


def run_speed_benchmark(args, device: str, model, diffusion, z, y, cfg_v1, cfg_v2) -> None:
    dit_results = run_trials(
        model, diffusion, z, y, args.cfg, device, False, args.steps, args.runs, args.warmup,
    )
    dit_mean, dit_std = _mean_std([r.seconds for r in dit_results])

    def apply_v1(m: DiTWithGMC) -> None:
        _apply_v1_schedule(m, cfg_v1, args.steps)

    apply_v1(model)
    v1_sa_ratio = _sa_compute_ratio(model._sa_refresh)
    v1_results = run_trials(
        model, diffusion, z, y, args.cfg, device, True, args.steps, args.runs, args.warmup,
        schedule_fn=apply_v1,
    )
    v1_mean, v1_std = _mean_std([r.seconds for r in v1_results])
    v1_stats = v1_results[-1].cache_stats

    def apply_v2(m: DiTWithGMC) -> None:
        _apply_v2_schedule(m, cfg_v2, args.steps)

    apply_v2(model)
    v2_sa_ratio = _sa_compute_ratio(model._sa_refresh)
    v2_results = run_trials(
        model, diffusion, z, y, args.cfg, device, True, args.steps, args.runs, args.warmup,
        schedule_fn=apply_v2,
    )
    v2_mean, v2_std = _mean_std([r.seconds for r in v2_results])
    v2_stats = v2_results[-1].cache_stats

    def speedup(base: float, cached: float) -> float:
        return base / cached if cached > 0 else float('inf')

    print('\n=== 性能对比 ===')
    print('┌─────────────┬──────────────┬──────────┬──────────┬────────────────────────────────────┐')
    print('│ 版本        │ 耗时 (mean)  │  std     │ speedup  │ 缓存统计                           │')
    print('├─────────────┼──────────────┼──────────┼──────────┼────────────────────────────────────┤')
    print(f'│ DiT 基线    │ {dit_mean:8.3f}s   │ {dit_std:6.3f}s │   1.00x  │ —                                  │')
    print(
        f'│ GMC v1      │ {v1_mean:8.3f}s   │ {v1_std:6.3f}s │ {speedup(dit_mean, v1_mean):7.2f}x  │ '
        f'SA计算={v1_sa_ratio:.0%}, attn_skip={v1_stats["attn_skipped"]}, '
        f'mlp_skip={v1_stats["mlp_skipped"]} │'
    )
    print(
        f'│ GMC v2 当前 │ {v2_mean:8.3f}s   │ {v2_std:6.3f}s │ {speedup(dit_mean, v2_mean):7.2f}x  │ '
        f'SA计算={v2_sa_ratio:.0%}, attn_skip={v2_stats["attn_skipped"]}, '
        f'mlp_skip={v2_stats["mlp_skipped"]} │'
    )
    print('└─────────────┴──────────────┴──────────┴──────────┴────────────────────────────────────┘')

    saved_v1 = (1.0 - v1_mean / dit_mean) * 100 if dit_mean > 0 else 0.0
    saved_v2 = (1.0 - v2_mean / dit_mean) * 100 if dit_mean > 0 else 0.0
    print(f'相对 DiT 基线节省时间: v1 {saved_v1:.1f}%, v2 {saved_v2:.1f}%')
    if v1_mean > 0:
        print(f'v2 相对 v1: {speedup(v1_mean, v2_mean):.2f}x ({(1 - v2_mean / v1_mean) * 100:+.1f}%)')


@torch.no_grad()
def run_quality_benchmark(args, device: str, model, diffusion, cfg_v1, cfg_v2, vae) -> None:
    seeds = [int(s) for s in args.quality_seeds.split(',') if s.strip()]
    class_ids = [int(c) for c in args.quality_classes.split(',') if c.strip()]

    def apply_v1(m: DiTWithGMC) -> None:
        _apply_v1_schedule(m, cfg_v1, args.steps)

    def apply_v2(m: DiTWithGMC) -> None:
        _apply_v2_schedule(m, cfg_v2, args.steps)

    v1_metrics: list[QualityMetrics] = []
    v2_metrics: list[QualityMetrics] = []
    n = 2 if args.cfg > 1.0 else 1

    print('\n=== 质量对比（相对 DiT 基线）===')
    print(f'seeds={seeds}, classes={class_ids}, steps={args.steps}, cfg={args.cfg}')
    print('指标: latent 空间 MSE/MAE/相对L2/余弦相似度; 像素空间 MSE/PSNR (VAE decode 后)')

    first_ref_pixel = None
    first_v1_pixel = None
    first_v2_pixel = None

    for seed in seeds:
        for class_id in class_ids:
            torch.manual_seed(seed)
            z = torch.randn(n, 4, 32, 32, device=device)
            y = torch.tensor([class_id] * n, device=device)

            ref_lat = sample_latents(model, diffusion, z, y, args.cfg, device, False, args.steps)
            v1_lat = sample_latents(model, diffusion, z, y, args.cfg, device, True, args.steps, apply_v1)
            v2_lat = sample_latents(model, diffusion, z, y, args.cfg, device, True, args.steps, apply_v2)

            ref_px = _decode_to_pixels(vae, ref_lat)
            v1_px = _decode_to_pixels(vae, v1_lat)
            v2_px = _decode_to_pixels(vae, v2_lat)

            v1_metrics.append(compute_quality(ref_lat, v1_lat, ref_px, v1_px))
            v2_metrics.append(compute_quality(ref_lat, v2_lat, ref_px, v2_px))

            if first_ref_pixel is None:
                first_ref_pixel, first_v1_pixel, first_v2_pixel = ref_px, v1_px, v2_px

            print(
                f'  seed={seed:3d} class={class_id:4d} | '
                f'v1 rel_l2={v1_metrics[-1].latent_rel_l2:.4f} psnr={v1_metrics[-1].pixel_psnr:.2f}dB | '
                f'v2 rel_l2={v2_metrics[-1].latent_rel_l2:.4f} psnr={v2_metrics[-1].pixel_psnr:.2f}dB'
            )

    v1_avg = _avg_quality(v1_metrics)
    v2_avg = _avg_quality(v2_metrics)

    print('\n┌─────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐')
    print('│ 版本        │ latent MSE │ latent MAE │ rel L2     │ cosine     │ pixel PSNR │ pixel MSE  │')
    print('├─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤')
    print(
        f'│ GMC v1      │ {v1_avg.latent_mse:10.2e} │ {v1_avg.latent_mae:10.2e} │ '
        f'{v1_avg.latent_rel_l2:10.4f} │ {v1_avg.latent_cosine:10.6f} │ '
        f'{v1_avg.pixel_psnr:10.2f} dB │ {v1_avg.pixel_mse:10.2e} │'
    )
    print(
        f'│ GMC v2 当前 │ {v2_avg.latent_mse:10.2e} │ {v2_avg.latent_mae:10.2e} │ '
        f'{v2_avg.latent_rel_l2:10.4f} │ {v2_avg.latent_cosine:10.6f} │ '
        f'{v2_avg.pixel_psnr:10.2f} dB │ {v2_avg.pixel_mse:10.2e} │'
    )
    print('└─────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘')
    print('说明: rel L2 / MSE 越小越好; cosine 越接近 1 越好; PSNR 越大越好 (通常 >30dB 视觉接近)')

    if args.quality_out and first_ref_pixel is not None:
        grid = torch.cat([first_ref_pixel, first_v1_pixel, first_v2_pixel], dim=0)
        save_image(grid, args.quality_out, nrow=3)
        print(f'对比图已保存: {args.quality_out}  (左→右: DiT基线 / v1 / v2)')


def parse_args():
    p = argparse.ArgumentParser(description='DiT vs GMC v1 vs v2 性能与质量对比')
    p.add_argument('--preset', default='default', choices=list(ALL_PRESETS.keys()))
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--class_id', type=int, default=207)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--ckpt', default=CKPT)
    p.add_argument('--skip_speed', action='store_true')
    p.add_argument('--skip_quality', action='store_true')
    p.add_argument('--quality_seeds', default='0,1,2,3,4', help='质量对比用的 seed 列表')
    p.add_argument('--quality_classes', default='207,279,923,980,417', help='质量对比用的 class 列表')
    p.add_argument('--quality_out', default='gmc_quality_compare.png', help='质量对比拼图输出路径')
    p.add_argument('--vae_path', default=_resolve_vae_path())
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('警告：未检测到 CUDA，CPU 计时仅供参考。')
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f'设备: {torch.cuda.get_device_name(0)}')

    base_cfg = ALL_PRESETS[args.preset]['gmc']
    cfg_v1 = _v1_config(base_cfg)
    cfg_v2 = _v2_config(base_cfg)

    print(f'\n对比: DiT 基线 | GMC v1 (SA 等间隔 + MLP 复用) | GMC v2 (n 步 SA 周期 + MLP 全算)')
    print(f'preset={args.preset}, steps={args.steps}, cfg={args.cfg}')

    torch.manual_seed(args.seed)
    model = build_model(device, args.steps)
    diffusion = create_diffusion(str(args.steps))

    n = 2 if args.cfg > 1.0 else 1
    z = torch.randn(n, 4, 32, 32, device=device)
    y = torch.tensor([args.class_id] * n, device=device)

    if not args.skip_speed:
        run_speed_benchmark(args, device, model, diffusion, z, y, cfg_v1, cfg_v2)

    if not args.skip_quality:
        print('\n[VAE] 加载解码器用于像素质量指标...')
        vae = _load_vae(args.vae_path, device)
        vae.eval()
        run_quality_benchmark(args, device, model, diffusion, cfg_v1, cfg_v2, vae)


if __name__ == '__main__':
    main()
