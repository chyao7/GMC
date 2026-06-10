"""GMC PixArt 缓存初始化。"""

from __future__ import annotations

import sys
from pathlib import Path

_GMC_ROOT = Path(__file__).resolve().parents[1]
if str(_GMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMC_ROOT))

from gmc_utils import GMCConfig, LayerCacheState


def gmc_cache_init(gmc_cfg: GMCConfig, num_steps: int, depth: int = 28):
    """GMC 专用 cache_dic，兼容 PixArt DPMS 采样循环。"""
    from gmc_utils import (
        build_ca_refresh_mask,
        build_layer_fresh_ratios,
        build_sa_refresh_mask,
    )

    cache_dic = {
        'gmc_mode': True,
        'gmc_cfg': gmc_cfg,
        'gmc_layers': {i: LayerCacheState() for i in range(depth)},
        'sa_refresh': build_sa_refresh_mask(gmc_cfg, num_steps),
        'ca_refresh': build_ca_refresh_mask(gmc_cfg, num_steps, depth),
        'layer_rho': build_layer_fresh_ratios(gmc_cfg, depth),
        'stats': {
            'sa_skipped': 0,
            'ca_skipped': 0,
            'mlp_skipped': 0,
            'fresh_tokens': 0,
        },
    }
    current = {'step': 0, 'num_steps': num_steps}
    cache_dic['_num_steps'] = num_steps
    return cache_dic, current


def reset_gmc_cache(cache_dic: dict, depth: int = 28) -> None:
    cache_dic['gmc_layers'] = {i: LayerCacheState() for i in range(depth)}
    gmc_cfg = cache_dic['gmc_cfg']
    num_steps = cache_dic.get('_num_steps')
    if num_steps is not None:
        from gmc_utils import (
            build_ca_refresh_mask,
            build_layer_fresh_ratios,
            build_sa_refresh_mask,
        )
        cache_dic['sa_refresh'] = build_sa_refresh_mask(gmc_cfg, num_steps)
        cache_dic['ca_refresh'] = build_ca_refresh_mask(gmc_cfg, num_steps, depth)
        cache_dic['layer_rho'] = build_layer_fresh_ratios(gmc_cfg, depth)
    cache_dic['stats'] = {
        'sa_skipped': 0,
        'ca_skipped': 0,
        'mlp_skipped': 0,
        'fresh_tokens': 0,
    }
