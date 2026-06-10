"""Granularity-Matched Caching (GMC) 共享工具。

对应论文 §3.3–§3.4：
- Self-attention / Cross-attention：步级复用，默认间隔 n=4
- Cross-attention 尾段（后 T_tail 步 & 深层 L>=L_ca）：间隔 floor(n/2)
- MLP：分层 token 级 fresh 比例 + linear stale 外推
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GMCConfig:
    """GMC 默认超参与论文一致。"""

    attn_interval: int = 4
    ca_tail_steps: int = 10
    ca_tail_min_layer: int = 20
    mlp_full_reuse_layers: int = 6
    mlp_mid_reuse_max_layer: int = 18
    mlp_mid_fresh_ratio: float = 0.025
    mlp_deep_fresh_ratio: float = 0.07
    fresh_threshold: int = 5
    score_s1_weight: float = 1.0
    score_s3_weight: float = 0.25
    score_drift_weight: float = 0.5
    score_anchor_weight: float = 0.3
    velocity_damping: float = 0.35
    spatial_bonus: float = 0.4
    spatial_grid: int = 2
    unify_cfg_indices: bool = True
    force_full_first_last: bool = True
    stale_reuse_mode: str = 'linear'  # 'linear' | 'copy'


@dataclass
class LayerCacheState:
    attn_out: Optional[torch.Tensor] = None
    ca_out: Optional[torch.Tensor] = None
    attn_map: Optional[torch.Tensor] = None
    cross_attn_map: Optional[torch.Tensor] = None
    mlp_out: Optional[torch.Tensor] = None
    mlp_out_prev_step: Optional[torch.Tensor] = None
    mlp_last_written: Optional[torch.Tensor] = None
    mlp_prev_written: Optional[torch.Tensor] = None
    cache_index: Optional[torch.Tensor] = None
    score_buf: Optional[torch.Tensor] = None


def _boundary_refresh(cfg: GMCConfig, step: int, num_steps: int) -> bool:
    return cfg.force_full_first_last and (step == 0 or step == num_steps - 1)


def should_compute_self_attention(cfg: GMCConfig, step: int, num_steps: int) -> bool:
    """R_SA(t)：式 (eq:sa-schedule)。"""
    if _boundary_refresh(cfg, step, num_steps):
        return True
    interval = max(cfg.attn_interval, 1)
    return step % interval == 0


def cross_attention_interval(cfg: GMCConfig, step: int, num_steps: int, layer_idx: int) -> int:
    """n_ca(t,l)：尾段区域用 floor(n/2)，其余用 n。"""
    n = max(cfg.attn_interval, 1)
    in_tail = (
        step >= num_steps - cfg.ca_tail_steps
        and layer_idx >= cfg.ca_tail_min_layer
    )
    if in_tail:
        return max(1, n // 2)
    return n


def should_compute_cross_attention(
    cfg: GMCConfig, step: int, num_steps: int, layer_idx: int,
) -> bool:
    """R_CA(t,l)：式 (eq:ca-schedule)。"""
    if _boundary_refresh(cfg, step, num_steps):
        return True
    interval = cross_attention_interval(cfg, step, num_steps, layer_idx)
    return step % interval == 0


def is_full_refresh_step(cfg: GMCConfig, step: int, num_steps: int) -> bool:
    """DiT 仅 SA：与 should_compute_self_attention 相同。"""
    return should_compute_self_attention(cfg, step, num_steps)


def is_mlp_full_refresh(
    cfg: GMCConfig,
    step: int,
    num_steps: int,
    layer_idx: int,
    *,
    has_cross_attention: bool = False,
) -> bool:
    """Attention 刷新步强制 MLP 全量重算。"""
    if should_compute_self_attention(cfg, step, num_steps):
        return True
    if has_cross_attention and should_compute_cross_attention(cfg, step, num_steps, layer_idx):
        return True
    return False


def mlp_fresh_ratio_for_layer(cfg: GMCConfig, layer_idx: int) -> float:
    if layer_idx < cfg.mlp_full_reuse_layers:
        return 0.0
    if layer_idx < cfg.mlp_mid_reuse_max_layer:
        return cfg.mlp_mid_fresh_ratio
    return cfg.mlp_deep_fresh_ratio


def build_sa_refresh_mask(cfg: GMCConfig, num_steps: int) -> list[bool]:
    """预计算 SA 步级刷新表（与 ``should_compute_self_attention`` 等价）。"""
    return [should_compute_self_attention(cfg, s, num_steps) for s in range(num_steps)]


def build_ca_refresh_mask(cfg: GMCConfig, num_steps: int, depth: int) -> list[list[bool]]:
    """预计算 CA 步级刷新表 [layer][step]。"""
    return [
        [should_compute_cross_attention(cfg, s, num_steps, layer_idx) for s in range(num_steps)]
        for layer_idx in range(depth)
    ]


def build_layer_fresh_ratios(cfg: GMCConfig, depth: int) -> list[float]:
    """预计算每层 MLP fresh 比例。"""
    return [mlp_fresh_ratio_for_layer(cfg, layer_idx) for layer_idx in range(depth)]


def is_mlp_full_refresh_from_masks(
    sa_refresh: bool,
    ca_refresh: bool,
    *,
    has_cross_attention: bool = False,
) -> bool:
    """与 ``is_mlp_full_refresh`` 等价，使用预计算调度掩码。"""
    if sa_refresh:
        return True
    return has_cross_attention and ca_refresh


def spatial_bonus(score: torch.Tensor, grid: int = 2, bonus: float = 0.4) -> torch.Tensor:
    b, n = score.shape
    side = int(math.sqrt(n))
    if side * side != n or grid <= 1:
        return score
    blk = grid * grid
    s = score.view(b, side // grid, grid, side // grid, grid)
    s = s.permute(0, 1, 3, 2, 4).contiguous().view(b, -1, blk)
    max_val, max_idx = s.max(dim=-1, keepdim=True)
    mask = torch.zeros_like(s)
    mask.scatter_(-1, max_idx, 1.0)
    boosted = s + mask * max_val * bonus
    out = boosted.view(b, side // grid, side // grid, grid, grid)
    out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, n)
    return out


def _spatial_bonus_inplace(score: torch.Tensor, grid: int = 2, bonus: float = 0.4) -> torch.Tensor:
    """与 ``spatial_bonus`` 数值一致，结果写回 ``score``。"""
    out = spatial_bonus(score, grid, bonus)
    if out is not score:
        score.copy_(out)
    return score


def _normalize_per_batch(score: torch.Tensor) -> torch.Tensor:
    return score / score.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def compute_cache_score(
    cache_index: torch.Tensor,
    cfg: GMCConfig,
    attn_map: Optional[torch.Tensor] = None,
    mlp_out: Optional[torch.Tensor] = None,
    mlp_out_prev_step: Optional[torch.Tensor] = None,
    mlp_last_written: Optional[torch.Tensor] = None,
    score_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, n = cache_index.shape
    device = cache_index.device
    if attn_map is not None:
        s1 = _normalize_per_batch(attn_map.sum(dim=-1))
    else:
        s1 = torch.ones(b, n, device=device)
    s3 = cache_index.float() / max(cfg.fresh_threshold, 1)
    base = cfg.score_s1_weight * s1 + cfg.score_s3_weight * s3

    reuse_buf = (
        score_buf is not None
        and score_buf.shape == (b, n)
        and score_buf.device == device
    )
    if reuse_buf:
        score_buf.copy_(base)
        score = score_buf
    else:
        score = base

    if (
        cfg.score_drift_weight > 0
        and mlp_out is not None
        and mlp_out_prev_step is not None
        and (cache_index > 0).any()
    ):
        drift = (mlp_out - mlp_out_prev_step).norm(dim=-1)
        term = cfg.score_drift_weight * _normalize_per_batch(drift)
        if reuse_buf:
            score.add_(term)
        else:
            score = score + term

    if (
        cfg.score_anchor_weight > 0
        and mlp_out is not None
        and mlp_last_written is not None
        and (cache_index > 0).any()
    ):
        anchor = (mlp_out - mlp_last_written).norm(dim=-1)
        term = cfg.score_anchor_weight * _normalize_per_batch(anchor)
        if reuse_buf:
            score.add_(term)
        else:
            score = score + term

    if reuse_buf:
        return _spatial_bonus_inplace(score, cfg.spatial_grid, cfg.spatial_bonus)
    return spatial_bonus(score, cfg.spatial_grid, cfg.spatial_bonus)


def _pick_one_batch_indices(
    score: torch.Tensor,
    cache_index: Optional[torch.Tensor],
    k: int,
    force_stale_after: int,
) -> torch.Tensor:
    """单 batch fresh 索引选择（纯 GPU，语义与原 Python 版一致）。"""
    n = score.shape[0]
    device = score.device
    parts: list[torch.Tensor] = []
    n_stale = 0
    if force_stale_after > 0 and cache_index is not None:
        stale = (cache_index >= force_stale_after).nonzero(as_tuple=False).view(-1)
        if stale.numel() > 0:
            n_take = min(k, stale.numel())
            parts.append(stale[:n_take])
            n_stale = n_take
    remain = k - n_stale
    if remain > 0:
        mask = torch.ones(n, dtype=torch.bool, device=device)
        if n_stale > 0:
            mask[parts[0]] = False
        rest = score.masked_fill(~mask, float('-inf')).argsort(descending=True)[:remain]
        parts.append(rest)
    if not parts:
        return score.new_zeros(0, dtype=torch.long)
    chosen = parts[0] if len(parts) == 1 else torch.cat(parts)
    return chosen[:k]


def select_fresh_indices(
    score: torch.Tensor,
    fresh_ratio: float,
    unify_cfg: bool = False,
    cache_index: Optional[torch.Tensor] = None,
    force_stale_after: int = 0,
) -> torch.Tensor:
    b, n = score.shape
    k = int(fresh_ratio * n)
    k = min(max(k, 0), n)
    if k == 0:
        return score.new_zeros(b, 0, dtype=torch.long)

    if unify_cfg and b >= 2 and b % 2 == 0:
        ci0 = cache_index[0] if cache_index is not None else None
        idx = _pick_one_batch_indices(score[0], ci0, k, force_stale_after)
        half = b // 2
        out = idx.unsqueeze(0).expand(half, -1)
        return torch.cat([out, out], dim=0)

    if cache_index is not None:
        return torch.stack([
            _pick_one_batch_indices(score[i], cache_index[i], k, force_stale_after)
            for i in range(b)
        ], dim=0)

    return score.argsort(dim=-1, descending=True)[:, :k]


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.take_along_dim(tokens, indices.unsqueeze(-1), dim=1).squeeze(2)


def scatter_tokens_inplace(
    base: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if indices.shape[1] == 0:
        return base
    idx = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    base.scatter_(dim=1, index=idx, src=values)
    return base


def _stale_mask(
    shape: tuple[int, int],
    fresh_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    b, n = shape
    stale = torch.ones(b, n, dtype=torch.bool, device=device)
    if fresh_indices.shape[1] > 0:
        stale.scatter_(1, fresh_indices, False)
    return stale


def _store_mlp_out_prev_step(state: LayerCacheState, mlp_out: torch.Tensor) -> None:
    snapshot = mlp_out.detach()
    if state.mlp_out_prev_step is None:
        state.mlp_out_prev_step = snapshot.clone()
    else:
        state.mlp_out_prev_step.copy_(snapshot)


def update_written_history(
    state: LayerCacheState,
    indices: Optional[torch.Tensor],
    values: torch.Tensor,
) -> None:
    if state.mlp_last_written is None:
        state.mlp_last_written = values.detach().clone()
        state.mlp_prev_written = values.detach().clone()
        return
    if indices is None:
        if state.mlp_prev_written is None:
            state.mlp_prev_written = state.mlp_last_written.detach().clone()
        else:
            state.mlp_prev_written.copy_(state.mlp_last_written)
        state.mlp_last_written.copy_(values.detach())
        return
    if indices.shape[1] == 0:
        return
    idx = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    last = torch.gather(state.mlp_last_written, dim=1, index=idx)
    state.mlp_prev_written.scatter_(dim=1, index=idx, src=last.detach())
    state.mlp_last_written.scatter_(dim=1, index=idx, src=values.detach())


def apply_linear_extrapolation(
    mlp_out: torch.Tensor,
    state: LayerCacheState,
    cfg: GMCConfig,
    fresh_indices: Optional[torch.Tensor] = None,
    all_tokens: bool = False,
) -> torch.Tensor:
    if cfg.stale_reuse_mode == 'copy':
        return mlp_out
    if state.cache_index is None:
        return mlp_out

    interval = max(cfg.attn_interval, 1)
    gap = state.cache_index.clamp(min=1).float().unsqueeze(-1)

    if state.mlp_last_written is not None and state.mlp_prev_written is not None:
        delta_hist = state.mlp_last_written - state.mlp_prev_written
        if delta_hist.abs().max() > 1e-8:
            delta_step = delta_hist / interval
            lin_pred = state.mlp_last_written + gap * delta_step
        elif state.mlp_out_prev_step is not None:
            step_vel = mlp_out - state.mlp_out_prev_step
            lin_pred = mlp_out + cfg.velocity_damping * step_vel
        else:
            return mlp_out
    else:
        return mlp_out

    if all_tokens:
        mlp_out.copy_(lin_pred)
        return mlp_out

    stale = _stale_mask(mlp_out.shape[:2], fresh_indices, mlp_out.device)
    lin_mask = stale & (state.cache_index >= 1)
    if lin_mask.any():
        mlp_out.copy_(torch.where(lin_mask.unsqueeze(-1), lin_pred, mlp_out))
    return mlp_out


def merge_mlp_partial(
    mlp_out: torch.Tensor,
    fresh_values: torch.Tensor,
    fresh_indices: torch.Tensor,
    state: LayerCacheState,
    cfg: GMCConfig,
    layer_idx: int,
    fresh_ratio: Optional[float] = None,
) -> torch.Tensor:
    if fresh_ratio is None:
        fresh_ratio = mlp_fresh_ratio_for_layer(cfg, layer_idx)

    if fresh_ratio <= 0.0:
        state.cache_index += 1
        return apply_linear_extrapolation(
            mlp_out, state, cfg, fresh_indices=None, all_tokens=True,
        )

    if fresh_indices.shape[1] > 0:
        scatter_tokens_inplace(mlp_out, fresh_indices, fresh_values)
        update_written_history(state, fresh_indices, fresh_values)

    state.cache_index += 1
    return apply_linear_extrapolation(mlp_out, state, cfg, fresh_indices)
