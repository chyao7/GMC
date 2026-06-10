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
    cfg = GMCConfig(attn_interval=4)
    assert should_compute_self_attention(cfg, 0, 20)
    assert should_compute_self_attention(cfg, 19, 20)
    assert should_compute_self_attention(cfg, 4, 20)
    assert not should_compute_self_attention(cfg, 3, 20)


def test_ca_tail_schedule():
    cfg = GMCConfig(attn_interval=4, ca_tail_steps=10, ca_tail_min_layer=20)
    # 尾段深层：interval=2
    assert cross_attention_interval(cfg, 15, 20, 25) == 2
    assert should_compute_cross_attention(cfg, 14, 20, 25)
    assert not should_compute_cross_attention(cfg, 13, 20, 25)
    # 浅层尾段：interval=4
    assert cross_attention_interval(cfg, 15, 20, 10) == 4


def test_mlp_rho():
    cfg = GMCConfig()
    assert mlp_fresh_ratio_for_layer(cfg, 3) == 0.0
    assert mlp_fresh_ratio_for_layer(cfg, 10) == 0.025
    assert mlp_fresh_ratio_for_layer(cfg, 22) == 0.07


def test_mlp_full_refresh():
    cfg = GMCConfig(attn_interval=4)
    assert is_mlp_full_refresh(cfg, 4, 20, 25, has_cross_attention=True)
    assert not is_mlp_full_refresh(cfg, 3, 20, 25, has_cross_attention=True)
    assert is_mlp_full_refresh(cfg, 14, 20, 25, has_cross_attention=True)


def test_precomputed_masks():
    cfg = GMCConfig(attn_interval=4, ca_tail_steps=10, ca_tail_min_layer=20)
    steps, depth = 20, 28
    sa = build_sa_refresh_mask(cfg, steps)
    ca = build_ca_refresh_mask(cfg, steps, depth)
    rho = build_layer_fresh_ratios(cfg, depth)
    for s in range(steps):
        assert sa[s] == should_compute_self_attention(cfg, s, steps)
    for l in range(depth):
        assert rho[l] == mlp_fresh_ratio_for_layer(cfg, l)
        for s in range(steps):
            assert ca[l][s] == should_compute_cross_attention(cfg, s, steps, l)
            assert is_mlp_full_refresh_from_masks(
                sa[s], ca[l][s], has_cross_attention=True,
            ) == is_mlp_full_refresh(cfg, s, steps, l, has_cross_attention=True)


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


if __name__ == '__main__':
    test_sa_schedule()
    test_ca_tail_schedule()
    test_mlp_rho()
    test_mlp_full_refresh()
    test_precomputed_masks()
    test_select_fresh_indices_gpu()
    print('All GMC unit tests passed.')
