# GMC — Granularity-Matched Caching

论文 **Granularity-Matched Caching (GMC)** 的代码实现，与 `DGC/` 中已有 DiT 实现对应，并扩展 PixArt 支持。

## 目录结构

```
GMC/
├── gmc_utils.py          # 共享：GMCConfig、SA/CA 步级调度、MLP token 级缓存
├── test_gmc.py           # 无 GPU 单元测试
├── GMC-DiT/              # 类条件 DiT-XL/2
│   ├── gmc_model.py      # DiTWithGMC
│   ├── config.py         # 默认/消融 preset
│   └── generate.py       # 生成示例
└── GMC-PixArt/           # 文本到图像 PixArt-α
    ├── gmc_pixart_block.py  # SA+CA 步级 + MLP 分层
    ├── gmc_cache.py
    ├── config.py
    └── generate.py
```

## 论文对应

| 模块 | 策略 | 实现位置 |
|------|------|----------|
| Self-attention | 步级复用，间隔 n=4 | `should_compute_self_attention` |
| Cross-attention | 步级复用；尾段深层 interval ⌊n/2⌋ | `should_compute_cross_attention` |
| MLP | 分层 ρ_l + fresh score + linear 外推 | `merge_mlp_partial` |

默认超参：`n=4`, `T_tail=10`, `L_ca=20`, `L_s=6`, `L_m=18`, `ρ_mid=0.025`, `ρ_deep=0.07`, `τ_c=5`。

## 快速开始

### 单元测试

```bash
python GMC/test_gmc.py
```

### DiT-XL/2（GMC-DiT）

```bash
export DIT_ROOT=/path/to/DiT
export DIT_CKPT=/path/to/DiT-XL-2-256x256.pt

python GMC/GMC-DiT/generate.py \
  --preset default \
  --class_id 207 \
  --out sample.png
```

### PixArt-α（GMC-PixArt）

```bash
python GMC/GMC-PixArt/generate.py \
  --model_path /path/to/PixArt-XL-2-256x256.pth \
  --t5_path /path/to/t5 \
  --vae_path stabilityai/sd-vae-ft-mse \
  --prompt "A cat wearing sunglasses."
```

## 与 DGC 的关系

- **GMC-DiT** 逻辑与 `DGC/dgc_model.py` 一致，命名与论文对齐，并独立维护于 `GMC/`。
- **GMC-PixArt** 在 ToCa 版 PixArt 上新增：CA 步级复用（含尾段 ⌊n/2⌋）、MLP 分层 ρ + linear stale，替代原 ToCa token 级 CA 策略。

## API 示例

```python
from GMC.gmc_utils import GMCConfig
from GMC.GMC-DiT.gmc_model import DiTWithGMC

model = DiTWithGMC(gmc_config=GMCConfig(), total_sampling_steps=50)
model.enable_cache(True)
model.reset_cache()
# ... DDIM 采样 ...
stats = model.get_cache_stats()
```

```python
from GMC.GMC-PixArt.gmc_pixart_block import apply_gmc_blocks
from GMC.GMC-PixArt.gmc_cache import gmc_cache_init

apply_gmc_blocks(pixart_model)
cache_dic, current = gmc_cache_init(GMCConfig(), num_steps=20)
```
