#!/usr/bin/env python3
"""DiT 基线 vs GMC 加速性能对比。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch

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


if not os.path.isdir(os.path.join(DIT_ROOT, 'diffusion')):
    raise FileNotFoundError(
        f'未找到 DiT 工程：{DIT_ROOT}\n'
        f'请先运行：cd {GMC_ROOT} && bash scripts/setup_dit.sh'
    )

sys.path.insert(0, DIT_ROOT)
sys.path.insert(0, GMC_ROOT)
sys.path.insert(0, GMC_DIT)

from diffusion import create_diffusion  # noqa: E402
from config import ALL_PRESETS, SPEED_PRESETS  # noqa: E402
from gmc_model import DiTWithGMC  # noqa: E402


@dataclass
class RunResult:
    name: str
    seconds: float
    cache_stats: dict | None = None


def _sync(device: str) -> None:
    if device.startswith('cuda'):
        torch.cuda.synchronize()


def load_ckpt(model, path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'未找到 DiT 权重：{path}\n'
            '请先运行：bash scripts/setup_dit.sh'
        )
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt.get('ema', ckpt.get('model', ckpt)), strict=False)


def build_model(device: str, preset: str, steps: int) -> DiTWithGMC:
    gmc_cfg = ALL_PRESETS[preset]['gmc']
    model = DiTWithGMC(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=32,
        gmc_config=gmc_cfg, total_sampling_steps=steps,
    ).to(device)
    load_ckpt(model, CKPT)
    model.eval()
    return model


def sample_once(
    model: DiTWithGMC,
    diffusion,
    z: torch.Tensor,
    y: torch.Tensor,
    cfg: float,
    device: str,
    use_cache: bool,
    steps: int,
) -> RunResult:
    model.enable_cache(use_cache)
    model.reset_cache()
    model.set_sampling_steps(steps)

    n = 2 if cfg > 1.0 else 1
    model_kwargs = dict(y=y, cfg_scale=cfg)

    _sync(device)
    t0 = time.perf_counter()
    diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=False, device=device,
    )
    _sync(device)
    elapsed = time.perf_counter() - t0

    stats = model.get_cache_stats() if use_cache else None
    label = 'GMC' if use_cache else 'DiT'
    return RunResult(name=label, seconds=elapsed, cache_stats=stats)


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
) -> list[RunResult]:
    for _ in range(warmup):
        sample_once(model, diffusion, z, y, cfg, device, use_cache, steps)
    return [
        sample_once(model, diffusion, z, y, cfg, device, use_cache, steps)
        for _ in range(runs)
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, var ** 0.5


def _print_stats_line(label: str, results: list[RunResult]) -> tuple[float, float]:
    times = [r.seconds for r in results]
    mean, std = _mean_std(times)
    print(f'  {label:<6}  {mean:7.3f}s ± {std:.3f}s  (n={len(times)})')
    return mean, std


def _print_cache_stats(stats: dict) -> None:
    attn = stats['attn_skipped']
    mlp = stats['mlp_skipped']
    fresh = stats['fresh_tokens']
    total = stats['total_tokens']
    reuse = (1.0 - fresh / total) if total else 0.0
    print(
        f'         cache: attn_skip={attn}, mlp_skip={mlp}, '
        f'token_reuse={reuse:.1%}'
    )


def benchmark_preset(
    preset: str,
    steps: int,
    cfg: float,
    class_id: int,
    seed: int,
    runs: int,
    warmup: int,
    device: str,
) -> None:
    print(f'\n=== preset={preset}, steps={steps}, cfg={cfg}, class={class_id} ===')

    torch.manual_seed(seed)
    model = build_model(device, preset, steps)
    diffusion = create_diffusion(str(steps))

    n = 2 if cfg > 1.0 else 1
    z = torch.randn(n, 4, 32, 32, device=device)
    y = torch.tensor([class_id] * n, device=device)

    dit_results = run_trials(model, diffusion, z, y, cfg, device, False, steps, runs, warmup)
    gmc_results = run_trials(model, diffusion, z, y, cfg, device, True, steps, runs, warmup)

    dit_mean, _ = _print_stats_line('DiT', dit_results)
    gmc_mean, _ = _print_stats_line('GMC', gmc_results)

    speedup = dit_mean / gmc_mean if gmc_mean > 0 else float('inf')
    saved = (1.0 - gmc_mean / dit_mean) * 100 if dit_mean > 0 else 0.0
    print(f'  speedup {speedup:.2f}x  (节省 {saved:.1f}% 时间)')

    if gmc_results[-1].cache_stats:
        _print_cache_stats(gmc_results[-1].cache_stats)


def parse_args():
    p = argparse.ArgumentParser(description='DiT vs GMC 性能对比')
    p.add_argument('--preset', default='default', choices=list(ALL_PRESETS.keys()))
    p.add_argument('--all_presets', action='store_true', help='扫 SPEED_PRESETS 全部配置')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--class_id', type=int, default=207)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--runs', type=int, default=3, help='计时重复次数')
    p.add_argument('--warmup', type=int, default=1, help='预热次数（不计入统计）')
    p.add_argument('--ckpt', default=CKPT, help='DiT 权重路径')
    return p.parse_args()


def main():
    args = parse_args()
    global CKPT
    CKPT = args.ckpt

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('警告：未检测到 CUDA，CPU 计时仅供参考。')
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f'设备: {torch.cuda.get_device_name(0)}')

    presets = SPEED_PRESETS if args.all_presets else [args.preset]
    print(f'对比项: DiT（无缓存） vs GMC（{", ".join(presets)}）')
    print(f'采样: steps={args.steps}, cfg={args.cfg}, runs={args.runs}, warmup={args.warmup}')

    for preset in presets:
        benchmark_preset(
            preset=preset,
            steps=args.steps,
            cfg=args.cfg,
            class_id=args.class_id,
            seed=args.seed,
            runs=args.runs,
            warmup=args.warmup,
            device=device,
        )


if __name__ == '__main__':
    main()
