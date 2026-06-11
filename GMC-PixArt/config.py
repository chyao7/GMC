"""GMC-PixArt 默认配置。"""

import sys
from pathlib import Path

_GMC_ROOT = Path(__file__).resolve().parents[1]
if str(_GMC_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMC_ROOT))

from gmc_utils import GMCConfig

DEFAULT_GMC_PIXART_CONFIG = GMCConfig(
    casa_interval=4,
    mlp_anchor_step=30,
    mlp_interval=4,
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
    stale_reuse_mode='linear',
)
