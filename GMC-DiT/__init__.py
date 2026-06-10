"""GMC-DiT：类条件 DiT-XL/2 + Granularity-Matched Caching。"""

from .config import ALL_PRESETS, DEFAULT_GMC_CONFIG
from .gmc_model import DiTWithGMC, DiT_XL_2_GMC

__all__ = [
    'DiTWithGMC',
    'DiT_XL_2_GMC',
    'DEFAULT_GMC_CONFIG',
    'ALL_PRESETS',
]
