"""统一 PixArt Block：结构一致，按 cache_dic 切换 ToCa / GMC 缓存策略。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

_GMC_ROOT = Path(__file__).resolve().parents[1]
if str(_GMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMC_ROOT))

from gmc_utils import (
    GMCConfig,
    LayerCacheState,
    compute_cache_score,
    resolve_mlp_anchor,
    should_compute_mlp,
    should_store_mlp_reuse_output,
    gather_tokens,
    is_mlp_full_refresh_from_masks,
    merge_mlp_partial,
    select_fresh_indices,
    update_written_history,
)


def _import_pixart_blocks():
    gmc_pixart = Path(__file__).resolve().parent
    if str(gmc_pixart) not in sys.path:
        sys.path.insert(0, str(gmc_pixart))
    from diffusion.model.nets.PixArt_blocks import (  # noqa: WPS433
        MultiHeadCrossAttention,
        WindowAttention,
        t2i_modulate,
    )
    return WindowAttention, MultiHeadCrossAttention, t2i_modulate


WindowAttention, MultiHeadCrossAttention, t2i_modulate = _import_pixart_blocks()

BASELINE_CACHE_KWARGS = dict(
    cache_type='attention',
    fresh_ratio=0.30,
    fresh_threshold=1,
    force_fresh='global',
    ratio_scheduler='ToCa',
    soft_fresh_weight=0.25,
)

TOCA_CACHE_KWARGS = dict(
    cache_type='attention',
    fresh_ratio=0.30,
    fresh_threshold=3,
    force_fresh='global',
    ratio_scheduler='ToCa',
    soft_fresh_weight=0.25,
)


def _import_toca_cache_fns():
    gmc_pixart = Path(__file__).resolve().parent
    if str(gmc_pixart) not in sys.path:
        sys.path.insert(0, str(gmc_pixart))
    from diffusion.model.cache_functions import (  # noqa: WPS433
        cache_cutfresh,
        force_init,
        global_force_fresh,
        update_cache,
    )
    return global_force_fresh, cache_cutfresh, update_cache, force_init


class PixArtBlockGMC(nn.Module):
    """统一 PixArt Block（权重与原版 PixArtBlock 相同，缓存策略由 cache_dic 决定）。"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        window_size=0,
        input_size=None,
        use_rel_pos=False,
        layer_idx: int = 0,
        **block_kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos, **block_kwargs,
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu, drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    @classmethod
    def from_pixart_block(cls, block, layer_idx: int) -> 'PixArtBlockGMC':
        gmc = cls(
            hidden_size=block.hidden_size,
            num_heads=block.attn.num_heads,
            mlp_ratio=block.mlp.fc1.out_features / block.hidden_size,
            drop_path=0.0 if isinstance(block.drop_path, nn.Identity) else block.drop_path.drop_prob,
            window_size=block.window_size,
            input_size=getattr(block.attn, 'input_size', None),
            use_rel_pos=getattr(block.attn, 'use_rel_pos', False),
            layer_idx=layer_idx,
        )
        gmc.load_state_dict(block.state_dict(), strict=False)
        gmc.scale_shift_table.data.copy_(block.scale_shift_table.data)
        return gmc

    def _layer_state(self, cache_dic: dict) -> LayerCacheState:
        return cache_dic['gmc_layers'][self.layer_idx]

    def _forward_mlp_gmc(
        self,
        x: torch.Tensor,
        mlp_input: torch.Tensor,
        gate_mlp: torch.Tensor,
        cfg: GMCConfig,
        state: LayerCacheState,
        step: int,
        attn_map,
        stats: dict,
        sa_refresh: bool,
        ca_refresh: bool,
        layer_fresh_ratio: float,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        full_refresh = is_mlp_full_refresh_from_masks(
            sa_refresh, ca_refresh, has_cross_attention=True,
        )

        if state.mlp_out is None:
            mlp_out = self.mlp(mlp_input)
            state.mlp_out = mlp_out
            update_written_history(state, None, mlp_out)
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)
            state.mlp_out_prev_step = mlp_out.detach()
            return x + self.drop_path(gate_mlp * mlp_out)

        if state.cache_index is None:
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)

        fresh_ratio = 1.0 if full_refresh else layer_fresh_ratio
        if fresh_ratio >= 1.0:
            mlp_out = self.mlp(mlp_input)
            state.mlp_out = mlp_out
            update_written_history(state, None, mlp_out)
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)
        elif fresh_ratio <= 0.0:
            stats['mlp_skipped'] += n * b
            merge_mlp_partial(
                state.mlp_out,
                state.mlp_out.new_zeros(b, 0, state.mlp_out.shape[-1]),
                state.mlp_out.new_zeros(b, 0, dtype=torch.long, device=x.device),
                state, cfg, self.layer_idx,
                fresh_ratio=layer_fresh_ratio,
            )
        else:
            score = compute_cache_score(
                state.cache_index, cfg, attn_map=attn_map,
                mlp_out=state.mlp_out,
                mlp_out_prev_step=state.mlp_out_prev_step,
                mlp_last_written=state.mlp_last_written,
            )
            fresh_idx = select_fresh_indices(
                score, fresh_ratio,
                unify_cfg=cfg.unify_cfg_indices,
                cache_index=state.cache_index,
                force_stale_after=cfg.fresh_threshold,
            )
            stats['fresh_tokens'] += fresh_idx.shape[1] * b
            stats['mlp_skipped'] += (n - fresh_idx.shape[1]) * b
            if fresh_idx.shape[1] > 0:
                state.cache_index.scatter_(1, fresh_idx, torch.zeros_like(fresh_idx))
                fresh_in = gather_tokens(mlp_input, fresh_idx)
                fresh_out = self.mlp(fresh_in)
            else:
                fresh_out = state.mlp_out.new_zeros(b, 0, state.mlp_out.shape[-1])
            merge_mlp_partial(
                state.mlp_out, fresh_out, fresh_idx, state, cfg, self.layer_idx,
                fresh_ratio=layer_fresh_ratio,
            )

        state.mlp_out_prev_step = state.mlp_out.detach()
        return x + self.drop_path(gate_mlp * state.mlp_out)

    def _forward_toca(self, x, y, t, current, cache_dic, mask=None, **kwargs):
        """ToCa 缓存策略（含 fresh_threshold=1 的基线全算模式）。"""
        global_force_fresh, cache_cutfresh, update_cache, force_init = _import_toca_cache_fns()

        b, n, _ = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(b, 6, -1)
        ).chunk(6, dim=1)
        is_force_fresh = global_force_fresh(cache_dic, current)
        current['is_force_fresh'] = is_force_fresh

        if is_force_fresh:
            current['module'] = 'attn'
            cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['attn_map'][-1][current['layer']] = (
                self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
            )
            force_init(cache_dic, current, x)
            x = x + self.drop_path(gate_msa * cache_dic['cache'][-1][current['layer']][current['module']])

            current['module'] = 'cross-attn'
            cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['cross_attn_map'][-1][current['layer']] = (
                self.cross_attn(x, y, mask)
            )
            force_init(cache_dic, current, x)
            x = x + cache_dic['cache'][-1][current['layer']][current['module']]

            current['module'] = 'mlp'
            cache_dic['cache'][-1][current['layer']][current['module']] = (
                self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
            )
            force_init(cache_dic, current, x)
            x = x + self.drop_path(gate_mlp * cache_dic['cache'][-1][current['layer']][current['module']])
        else:
            current['module'] = 'attn'
            x = x + self.drop_path(gate_msa * cache_dic['cache'][-1][current['layer']][current['module']])

            current['module'] = 'cross-attn'
            fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            fresh_tokens, fresh_cross_attn_map = self.cross_attn(fresh_tokens, y, mask)
            update_cache(
                fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic,
                current=current, fresh_attn_map=fresh_cross_attn_map,
            )
            x = x + cache_dic['cache'][-1][current['layer']][current['module']]

            current['module'] = 'mlp'
            fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, x, current)
            fresh_tokens = self.mlp(t2i_modulate(self.norm2(fresh_tokens), shift_mlp, scale_mlp))
            update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current)
            x = x + self.drop_path(gate_mlp * cache_dic['cache'][-1][current['layer']][current['module']])

        return x

    def _forward_gmc(self, x, y, t, current, cache_dic, mask=None, **kwargs):
        cfg: GMCConfig = cache_dic['gmc_cfg']
        stats = cache_dic['stats']
        state = self._layer_state(cache_dic)
        step = current['step']
        sa_refresh = cache_dic['sa_refresh'][self.layer_idx][step]
        ca_refresh = cache_dic['ca_refresh'][self.layer_idx][step]
        layer_fresh_ratio = cache_dic['layer_rho'][self.layer_idx]

        b, n, _ = x.shape
        t = t.to(device=x.device, dtype=x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        if sa_refresh or state.attn_out is None:
            attn_out, attn_map = self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa))
            state.attn_out = attn_out
            state.attn_map = attn_map
        else:
            stats['sa_skipped'] += n * b
            attn_out = state.attn_out
            attn_map = state.attn_map

        x = x + self.drop_path(gate_msa * attn_out)

        if ca_refresh or state.ca_out is None:
            ca_out, ca_map = self.cross_attn(x, y, mask)
            state.ca_out = ca_out
            state.cross_attn_map = ca_map
        else:
            stats['ca_skipped'] += n * b
            ca_out = state.ca_out

        x = x + ca_out

        mlp_input = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        num_steps = current['num_steps']
        if cfg.enable_mlp_cache:
            score_map = state.cross_attn_map if state.cross_attn_map is not None else attn_map
            return self._forward_mlp_gmc(
            x, mlp_input, gate_mlp, cfg, state, step, score_map, stats,
            sa_refresh, ca_refresh, layer_fresh_ratio,
        )

        if should_compute_mlp(cfg, step, num_steps):
            mlp_out = self.mlp(mlp_input)
            anchor = resolve_mlp_anchor(cfg)
            if anchor is not None and should_store_mlp_reuse_output(cfg, step, anchor):
                state.mlp_anchor_out = mlp_out
            return x + self.drop_path(gate_mlp * mlp_out)

        stats['mlp_skipped'] += n * b
        if state.mlp_anchor_out is not None:
            mlp_out = state.mlp_anchor_out
        else:
            mlp_out = self.mlp(mlp_input)
            state.mlp_anchor_out = mlp_out
        return x + self.drop_path(gate_mlp * mlp_out)

    def forward(self, x, y, t, current, cache_dic, mask=None, **kwargs):
        if cache_dic.get('gmc_mode', False):
            return self._forward_gmc(x, y, t, current, cache_dic, mask, **kwargs)
        return self._forward_toca(x, y, t, current, cache_dic, mask, **kwargs)


def apply_gmc_blocks(model, depth: int | None = None) -> int:
    """将 PixArt blocks 替换为统一 Block（三种缓存策略共用同一结构）。"""
    depth = depth or len(model.blocks)
    new_blocks = nn.ModuleList([
        PixArtBlockGMC.from_pixart_block(block, layer_idx=i)
        for i, block in enumerate(model.blocks)
    ])
    model.blocks = new_blocks
    return depth
