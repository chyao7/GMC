"""GMC DiT 默认与消融配置。"""

from __future__ import annotations

from dataclasses import replace

import sys
from pathlib import Path

_GMC_ROOT = Path(__file__).resolve().parents[1]
if str(_GMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMC_ROOT))

from gmc_utils import GMCConfig


def _preset(name: str, cfg: GMCConfig) -> dict:
    return {'cache_mode': f'gmc_{name}', 'gmc': cfg}


def _from_default(name: str, **kwargs) -> dict:
    return _preset(name, replace(DEFAULT_GMC_CONFIG['gmc'], **kwargs))


DEFAULT_GMC_CONFIG = _preset(
    'default',
    GMCConfig(
        attn_interval=3,
        ca_tail_steps=10,
        ca_tail_min_layer=20,
        mlp_full_reuse_layers=6,
        mlp_mid_reuse_max_layer=18,
        mlp_mid_fresh_ratio=0.025,
        mlp_deep_fresh_ratio=0.07,
        fresh_threshold=5,
        score_s1_weight=1.0,
        score_s3_weight=0.25,
        score_drift_weight=0.5,
        score_anchor_weight=0.3,
        spatial_bonus=0.4,
        unify_cfg_indices=True,
        force_full_first_last=True,
        enable_mlp_cache=False,
        stale_reuse_mode='linear',
    ),
)

ALL_PRESETS = {
    'default': DEFAULT_GMC_CONFIG,
    'turbo': _from_default(
        'turbo',
        attn_interval=3,
        mlp_full_reuse_layers=12,
        mlp_mid_fresh_ratio=0.02,
        mlp_deep_fresh_ratio=0.05,
    ),
    'aggressive': _from_default(
        'aggressive',
        attn_interval=3,
        mlp_full_reuse_layers=16,
        mlp_mid_fresh_ratio=0.02,
        mlp_deep_fresh_ratio=0.05,
    ),
    'ultra': _from_default(
        'ultra',
       attn_interval=3,
        mlp_full_reuse_layers=0,
        mlp_mid_fresh_ratio=1,
        mlp_deep_fresh_ratio=1,
    ),
    'extreme': _from_default(
        'extreme',
        attn_interval=2,
        mlp_full_reuse_layers=12,
        mlp_mid_fresh_ratio=0.02,
        mlp_deep_fresh_ratio=0.05,
    ),
    'hyper': _from_default(
        'hyper',
        attn_interval=2,
        mlp_full_reuse_layers=16,
        mlp_mid_fresh_ratio=0.02,
        mlp_deep_fresh_ratio=0.05,
    ),
    'ablation_no_attn_cache': _from_default('ablation_no_attn_cache', attn_interval=1),
    'ablation_copy': _from_default('ablation_copy', stale_reuse_mode='copy'),
}

# GMC vs ToCa 速度扫参（不含消融）
SPEED_PRESETS = ['default', 'turbo', 'aggressive', 'ultra', 'extreme', 'hyper']
