#!/usr/bin/env python3
"""DiT + GMC (Granularity-Matched Caching) 模型实现。"""

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

_GMC_ROOT = Path(__file__).resolve().parents[1]
if str(_GMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMC_ROOT))

from gmc_utils import (
    GMCConfig,
    LayerCacheState,
    build_layer_fresh_ratios,
    build_sa_refresh_mask,
    compute_cache_score,
    gather_tokens,
    is_full_refresh_step,
    merge_mlp_partial,
    resolve_mlp_anchor,
    select_fresh_indices,
    should_compute_mlp,
    should_store_mlp_reuse_output,
    update_written_history,
    _store_mlp_out_prev_step,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool = False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class AttentionWithMap(Attention):
    def forward(self, x: torch.Tensor, store_attn_map: bool = True) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if store_attn_map:
            self.last_attn_map = attn.mean(dim=1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x)


class DiTBlockWithGMC(nn.Module):
    """GMC DiT Block：SA 步级复用 + MLP 分层 token 级 linear 复用。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        layer_idx: int = 0,
        depth: int = 28,
        **block_kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.depth = depth
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithMap(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.use_cache = False
        self.cache_state = LayerCacheState()
        self.stats = {'fresh_tokens': 0, 'total_tokens': 0, 'attn_skipped': 0, 'mlp_skipped': 0}
        self._cached_fresh_ratio = 0.0

    def set_fresh_ratio(self, fresh_ratio: float) -> None:
        self._cached_fresh_ratio = fresh_ratio

    def _ensure_score_buf(self, state: LayerCacheState, b: int, n: int, device: torch.device) -> torch.Tensor:
        if (
            state.score_buf is None
            or state.score_buf.shape != (b, n)
            or state.score_buf.device != device
        ):
            state.score_buf = torch.empty(b, n, device=device)
        return state.score_buf

    def reset_cache(self):
        self.cache_state = LayerCacheState()
        self.stats = {'fresh_tokens': 0, 'total_tokens': 0, 'attn_skipped': 0, 'mlp_skipped': 0}

    def enable_cache(self, use_cache: bool = True):
        self.use_cache = use_cache
        if not use_cache:
            self.reset_cache()

    def _forward_mlp_cached(
        self,
        x: torch.Tensor,
        mlp_input_fn,
        gmc_cfg: GMCConfig,
        state: LayerCacheState,
        b: int,
        n: int,
        attn_map,
        full_refresh: bool,
        layer_fresh_ratio: float,
    ) -> torch.Tensor:
        if state.mlp_out is None:
            mlp_out = self.mlp(mlp_input_fn(x))
            state.mlp_out = mlp_out
            update_written_history(state, None, mlp_out)
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)
            _store_mlp_out_prev_step(state, mlp_out)
            return mlp_out

        if state.cache_index is None:
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)

        fresh_ratio = 1.0 if full_refresh else layer_fresh_ratio
        if fresh_ratio >= 1.0:
            mlp_out = self.mlp(mlp_input_fn(x))
            state.mlp_out = mlp_out
            update_written_history(state, None, mlp_out)
            state.cache_index = torch.zeros(b, n, dtype=torch.long, device=x.device)
        elif fresh_ratio <= 0.0:
            self.stats['mlp_skipped'] += n
            merge_mlp_partial(
                state.mlp_out,
                state.mlp_out.new_zeros(b, 0, state.mlp_out.shape[-1]),
                state.mlp_out.new_zeros(b, 0, dtype=torch.long, device=x.device),
                state,
                gmc_cfg,
                self.layer_idx,
                fresh_ratio=layer_fresh_ratio,
            )
        else:
            score = compute_cache_score(
                state.cache_index,
                gmc_cfg,
                attn_map=attn_map,
                mlp_out=state.mlp_out,
                mlp_out_prev_step=state.mlp_out_prev_step,
                mlp_last_written=state.mlp_last_written,
                score_buf=self._ensure_score_buf(state, b, n, x.device),
            )
            fresh_idx = select_fresh_indices(
                score, fresh_ratio,
                unify_cfg=gmc_cfg.unify_cfg_indices,
                cache_index=state.cache_index,
                force_stale_after=gmc_cfg.fresh_threshold,
            )
            self.stats['fresh_tokens'] += fresh_idx.shape[1]
            self.stats['mlp_skipped'] += n - fresh_idx.shape[1]

            if fresh_idx.shape[1] > 0:
                state.cache_index.scatter_(1, fresh_idx, torch.zeros_like(fresh_idx))
                fresh_out = self.mlp(mlp_input_fn(gather_tokens(x, fresh_idx)))
            else:
                fresh_out = state.mlp_out.new_zeros(b, 0, state.mlp_out.shape[-1])

            merge_mlp_partial(
                state.mlp_out, fresh_out, fresh_idx, state, gmc_cfg, self.layer_idx,
                fresh_ratio=layer_fresh_ratio,
            )

        _store_mlp_out_prev_step(state, state.mlp_out)
        return state.mlp_out

    def _forward_cached(
        self,
        x: torch.Tensor,
        attn_input: torch.Tensor,
        mlp_input_fn,
        gate_msa: torch.Tensor,
        gate_mlp: torch.Tensor,
        gmc_cfg: GMCConfig,
        compute_sa: bool,
        step: int,
        num_steps: int,
    ) -> torch.Tensor:
        state = self.cache_state
        b, n, _ = x.shape
        self.stats['total_tokens'] += n

        if compute_sa or state.attn_out is None:
            attn_out = self.attn(attn_input, store_attn_map=True)
            state.attn_out = attn_out
        else:
            self.stats['attn_skipped'] += 1
            attn_out = state.attn_out

        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_input = mlp_input_fn(x)
        if gmc_cfg.enable_mlp_cache:
            attn_map = getattr(self.attn, 'last_attn_map', None)
            full_refresh = compute_sa
            mlp_out = self._forward_mlp_cached(
                x, mlp_input_fn, gmc_cfg, state, b, n, attn_map, full_refresh, self._cached_fresh_ratio,
            )
        elif should_compute_mlp(gmc_cfg, step, num_steps):
            mlp_out = self.mlp(mlp_input)
            anchor = resolve_mlp_anchor(gmc_cfg)
            if anchor is not None and should_store_mlp_reuse_output(gmc_cfg, step, anchor):
                state.mlp_anchor_out = mlp_out
        else:
            self.stats['mlp_skipped'] += 1
            if state.mlp_anchor_out is not None:
                mlp_out = state.mlp_anchor_out
            else:
                mlp_out = self.mlp(mlp_input)
                state.mlp_anchor_out = mlp_out
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor, cache_ctx: Optional[dict] = None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)

        if self.use_cache and cache_ctx is not None:
            def mlp_input_fn(h):
                return modulate(self.norm2(h), shift_mlp, scale_mlp)
            step = cache_ctx['step']
            sa_refresh = cache_ctx['sa_refresh']
            compute_sa = (
                sa_refresh[self.layer_idx][step]
                if sa_refresh
                else is_full_refresh_step(
                    cache_ctx['gmc_cfg'], step, cache_ctx['num_steps'],
                    self.layer_idx, self.depth,
                )
            )
            return self._forward_cached(
                x, attn_input, mlp_input_fn,
                gate_msa, gate_mlp,
                cache_ctx['gmc_cfg'],
                compute_sa,
                step,
                cache_ctx['num_steps'],
            )

        attn_out = self.attn(attn_input, store_attn_map=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class DiTWithGMC(nn.Module):
    """集成 GMC 加速的 DiT 模型。"""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        gmc_config: Optional[GMCConfig] = None,
        total_sampling_steps: int = 50,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.gmc_config = gmc_config or GMCConfig()
        self.total_sampling_steps = total_sampling_steps
        self._diffusion_step = 0
        self._sa_refresh: list[list[bool]] = []
        self._layer_fresh_ratios: list[float] = []

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlockWithGMC(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                layer_idx=i, depth=depth,
            )
            for i in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()
        self.use_cache = False
        self._rebuild_schedules()

    def _rebuild_schedules(self) -> None:
        steps = self.total_sampling_steps
        cfg = self.gmc_config
        self._sa_refresh = build_sa_refresh_mask(cfg, steps, self.depth)
        self._layer_fresh_ratios = build_layer_fresh_ratios(cfg, self.depth)
        for i, block in enumerate(self.blocks):
            block.set_fresh_ratio(self._layer_fresh_ratios[i])

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def reset_cache(self):
        self._diffusion_step = 0
        for block in self.blocks:
            block.reset_cache()

    def set_sampling_steps(self, steps: int):
        self.total_sampling_steps = steps
        self._rebuild_schedules()

    def set_gmc_config(self, gmc_config: GMCConfig) -> None:
        self.gmc_config = gmc_config
        self._rebuild_schedules()

    def enable_cache(self, use_cache: bool = True):
        self.use_cache = use_cache
        for block in self.blocks:
            block.enable_cache(use_cache)
        if not use_cache:
            self.reset_cache()

    def get_cache_stats(self) -> dict:
        attn_skipped = sum(b.stats['attn_skipped'] for b in self.blocks)
        mlp_skipped = sum(b.stats['mlp_skipped'] for b in self.blocks)
        total = sum(b.stats['total_tokens'] for b in self.blocks)
        fresh = sum(b.stats['fresh_tokens'] for b in self.blocks)
        return {
            'attn_skipped': attn_skipped,
            'mlp_skipped': mlp_skipped,
            'fresh_tokens': fresh,
            'total_tokens': total,
        }

    def _cache_context(self) -> dict:
        step = self._diffusion_step
        num_steps = self.total_sampling_steps
        return {
            'gmc_cfg': self.gmc_config,
            'step': step,
            'num_steps': num_steps,
            'sa_refresh': self._sa_refresh,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb

        cache_ctx = self._cache_context() if self.use_cache else None
        for block in self.blocks:
            x = block(x, c, cache_ctx)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        if self.use_cache:
            self._diffusion_step += 1
        return x

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float):
        if len(x) == 1 or cfg_scale == 1.0:
            return self.forward(x, t, y)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def DiT_XL_2_GMC(**kwargs):
    return DiTWithGMC(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs,
    )
