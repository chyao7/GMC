#!/usr/bin/env python3
"""GMC 核心逻辑单元测试（无需 GPU）。"""

import sys
from pathlib import Path

import torch

GMC_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GMC_ROOT))

from gmc_utils import (
    GMCConfig,
    build_ca_refresh_mask,
    build_layer_fresh_ratios,
    build_sa_refresh_mask,
    cross_attention_interval,
    is_mlp_full_refresh,
    is_mlp_full_refresh_from_masks,
    mlp_fresh_ratio_for_layer,
    select_fresh_indices,
    should_compute_cross_attention,
    should_compute_self_attention,
)


def _pick_one_batch_ref(sc, ci, k, force_stale_after, n):
    chosen = []
    if force_stale_after > 0 and ci is not None:
        stale = (ci >= force_stale_after).nonzero(as_tuple=False).squeeze(-1)
        if stale.numel() > 0:
            chosen = stale[:k].tolist()
    remain = k - len(chosen)
    if remain > 0:
        mask = torch.ones(n, dtype=torch.bool, device=sc.device)
        if chosen:
            mask[torch.tensor(chosen, device=sc.device, dtype=torch.long)] = False
        rest = sc.masked_fill(~mask, float('-inf')).argsort(descending=True)
        chosen.extend(rest[:remain].tolist())
    return torch.tensor(chosen[:k], device=sc.device, dtype=torch.long)


def select_fresh_indices_ref(score, fresh_ratio, unify_cfg=False, cache_index=None, force_stale_after=0):
    b, n = score.shape
    k = int(fresh_ratio * n)
    k = min(max(k, 0), n)
    if k == 0:
        return score.new_zeros(b, 0, dtype=torch.long)
    if unify_cfg and b >= 2 and b % 2 == 0:
        idx = _pick_one_batch_ref(
            score[0], cache_index[0] if cache_index is not None else None, k, force_stale_after, n,
        )
        out = torch.stack([idx] * (b // 2), dim=0)
        return torch.cat([out, out], dim=0)
    if cache_index is not None:
        return torch.stack([
            _pick_one_batch_ref(score[i], cache_index[i], k, force_stale_after, n)
            for i in range(b)
        ], dim=0)
    return score.argsort(dim=-1, descending=True)[:, :k]


def test_sa_schedule():
    cfg = GMCConfig(sa_cycle_length=5, force_full_first_last=True)
    depth = 28
    # 首末步强制全量
    assert should_compute_self_attention(cfg, 0, 20, 0, depth)
    assert should_compute_self_attention(cfg, 19, 20, 0, depth)
    # n=5 循环：pos0 计算, pos1..4 复用
    assert should_compute_self_attention(cfg, 5, 20, 0, depth)   # pos0
    assert not should_compute_self_attention(cfg, 6, 20, 0, depth)  # pos1
    assert not should_compute_self_attention(cfg, 7, 20, 0, depth)  # pos2
    assert not should_compute_self_attention(cfg, 8, 20, 0, depth)  # pos3
    assert not should_compute_self_attention(cfg, 9, 20, 0, depth)  # pos4
    assert not should_compute_self_attention(cfg, 8, 20, 20, depth)  # 各层一致


def test_ca_tail_schedule():
    cfg = GMCConfig(attn_interval=4, ca_tail_steps=10, ca_tail_min_layer=20)
    # 尾段深层：interval=2
    assert cross_attention_interval(cfg, 15, 20, 25) == 2
    assert should_compute_cross_attention(cfg, 14, 20, 25)
    assert not should_compute_cross_attention(cfg, 13, 20, 25)
    # 浅层尾段：interval=4
    assert cross_attention_interval(cfg, 15, 20, 10) == 4


def test_mlp_rho():
    cfg = GMCConfig(enable_mlp_cache=True)
    assert mlp_fresh_ratio_for_layer(cfg, 3) == 0.0
    assert mlp_fresh_ratio_for_layer(cfg, 10) == 0.025
    assert mlp_fresh_ratio_for_layer(cfg, 22) == 0.07


def test_mlp_full_refresh():
    cfg = GMCConfig(sa_cycle_length=5)
    depth = 28
    assert is_mlp_full_refresh(cfg, 5, 20, 25, depth=depth, has_cross_attention=True)
    assert not is_mlp_full_refresh(cfg, 6, 20, 25, depth=depth, has_cross_attention=True)
    assert is_mlp_full_refresh(cfg, 10, 20, 25, depth=depth, has_cross_attention=True)


def test_precomputed_masks():
    cfg = GMCConfig(attn_interval=4, ca_tail_steps=10, ca_tail_min_layer=20, enable_mlp_cache=True)
    steps, depth = 20, 28
    sa = build_sa_refresh_mask(cfg, steps, depth)
    ca = build_ca_refresh_mask(cfg, steps, depth)
    rho = build_layer_fresh_ratios(cfg, depth)
    for l in range(depth):
        assert rho[l] == mlp_fresh_ratio_for_layer(cfg, l)
        for s in range(steps):
            assert sa[l][s] == should_compute_self_attention(cfg, s, steps, l, depth)
            assert ca[l][s] == should_compute_cross_attention(cfg, s, steps, l)
            assert is_mlp_full_refresh_from_masks(
                sa[l][s], ca[l][s], has_cross_attention=True,
            ) == is_mlp_full_refresh(cfg, s, steps, l, depth=depth, has_cross_attention=True)


def test_select_fresh_indices_gpu():
    torch.manual_seed(0)
    for n in (64, 256):
        for b in (1, 2, 4):
            score = torch.randn(b, n)
            ci = torch.randint(0, 8, (b, n))
            for ratio in (0.025, 0.07, 0.1):
                for unify in (False, True):
                    for thr in (0, 5):
                        ref = select_fresh_indices_ref(
                            score, ratio, unify_cfg=unify, cache_index=ci, force_stale_after=thr,
                        )
                        out = select_fresh_indices(
                            score, ratio, unify_cfg=unify, cache_index=ci, force_stale_after=thr,
                        )
                        assert torch.equal(ref, out), f'mismatch n={n} b={b} ratio={ratio} unify={unify} thr={thr}'




def apply_linear_extrapolation_ref(mlp_out, state, cfg, fresh_indices=None, all_tokens=False):
    """优化前参考实现：全量 lin_pred + torch.where。"""
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

    stale = torch.ones(*mlp_out.shape[:2], dtype=torch.bool, device=mlp_out.device)
    if fresh_indices is not None and fresh_indices.shape[1] > 0:
        stale.scatter_(1, fresh_indices, False)
    lin_mask = stale & (state.cache_index >= 1)
    if lin_mask.any():
        mlp_out.copy_(torch.where(lin_mask.unsqueeze(-1), lin_pred, mlp_out))
    return mlp_out


def test_compute_cache_score_buf():
    from gmc_utils import GMCConfig, LayerCacheState, compute_cache_score

    torch.manual_seed(0)
    cfg = GMCConfig()
    b, n, d = 2, 256, 64
    ci = torch.randint(0, 6, (b, n))
    mlp_out = torch.randn(b, n, d)
    prev = torch.randn(b, n, d)
    last = torch.randn(b, n, d)
    attn = torch.softmax(torch.randn(b, n, n), dim=-1)

    ref = compute_cache_score(ci, cfg, attn_map=attn, mlp_out=mlp_out,
                              mlp_out_prev_step=prev, mlp_last_written=last)
    buf = torch.empty(b, n)
    out = compute_cache_score(ci, cfg, attn_map=attn, mlp_out=mlp_out,
                              mlp_out_prev_step=prev, mlp_last_written=last, score_buf=buf)
    assert out is buf
    assert torch.allclose(ref, out)


def test_apply_linear_extrap_equivalence():
    from gmc_utils import GMCConfig, LayerCacheState, apply_linear_extrapolation

    torch.manual_seed(1)
    cfg = GMCConfig()
    for all_tokens in (False, True):
        for _ in range(20):
            b, n, d = 2, 256, 64
            mlp_out = torch.randn(b, n, d)
            state = LayerCacheState(
                cache_index=torch.randint(0, 5, (b, n)),
                mlp_last_written=torch.randn(b, n, d),
                mlp_prev_written=torch.randn(b, n, d),
                mlp_out_prev_step=torch.randn(b, n, d),
            )
            k = int(0.07 * n)
            fresh_idx = torch.stack([
                torch.randperm(n)[:k] for _ in range(b)
            ], dim=0)

            ref_out = mlp_out.clone()
            new_out = mlp_out.clone()
            apply_linear_extrapolation_ref(ref_out, state, cfg, fresh_idx, all_tokens=all_tokens)
            apply_linear_extrapolation(new_out, state, cfg, fresh_idx, all_tokens=all_tokens)
            assert torch.allclose(ref_out, new_out, atol=1e-6, rtol=1e-5), all_tokens

    cfg_copy = GMCConfig(stale_reuse_mode='copy')
    out = torch.randn(2, 64, 32)
    st = LayerCacheState(cache_index=torch.zeros(2, 64, dtype=torch.long))
    assert apply_linear_extrapolation(out.clone(), st, cfg_copy) is not None


def test_gather_tokens_equivalence():
    from gmc_utils import gather_tokens

    torch.manual_seed(2)
    tokens = torch.randn(2, 128, 32)
    idx = torch.stack([torch.randperm(128)[:10], torch.randperm(128)[:10]], dim=0)
    ref_idx = idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    ref = torch.gather(tokens, dim=1, index=ref_idx)
    out = gather_tokens(tokens, idx)
    assert torch.equal(ref, out)

if __name__ == '__main__':
    test_sa_schedule()
    test_ca_tail_schedule()
    test_mlp_rho()
    test_mlp_full_refresh()
    test_precomputed_masks()
    test_select_fresh_indices_gpu()
    test_compute_cache_score_buf()
    test_apply_linear_extrap_equivalence()
    test_gather_tokens_equivalence()
    print('All GMC unit tests passed.')
