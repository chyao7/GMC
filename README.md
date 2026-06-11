# GMC — Granularity-Matched Caching

论文 **Granularity-Matched Caching (GMC)** 的代码实现，DiT 与 PixArt 共用 `gmc_utils.py` 中的步级调度逻辑。

## 目录结构

```
GMC/
├── gmc_utils.py          # 共享：GMCConfig、SA/CA/MLP 步级调度
├── test_gmc.py           # 无 GPU 单元测试
├── DiT/                  # 官方 DiT 工程（facebookresearch/DiT）
├── GMC-DiT/              # 类条件 DiT-XL/2
│   ├── gmc_model.py      # DiTWithGMC
│   ├── config.py         # 默认/消融 preset
│   └── generate.py       # 生成示例
└── GMC-PixArt/           # 文本到图像 PixArt-α
    ├── diffusion/        # PixArt 推理代码（vendored）
    ├── tools/
    ├── gmc_pixart_block.py  # 统一 Block + 缓存策略分发
    ├── gmc_cache.py
    ├── config.py
    ├── benchmark_compare.py
    └── generate.py
```

## GMC 复用逻辑（DiT / PixArt 共用）

默认均为 **步级复用**，**无 token 级 linear 外推**（`enable_mlp_cache=False`）。stale 步直接复用缓存输出。

| 模块 | 策略 | DiT | PixArt |
|------|------|-----|--------|
| Self-attention | 间隔 `casa_interval`：step 0 全算 → 后 3 步复用 → 再全算…；**首末 step 强制全算** | ✓ | ✓ |
| Cross-attention | 与 SA 同频；**尾段（最后 1/5 步）频率减半** | — | ✓ |
| MLP | `step < mlp_anchor_step` 每步全算；之后每 `mlp_interval` 步刷新 | ✓ | ✓ |

核心配置参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `casa_interval` | 4 | SA（及 PixArt CA 前期）更新频率 |
| `mlp_anchor_step` | 30 | MLP 锚定步数，此前每步全算 |
| `mlp_interval` | 4 | MLP 锚定后的更新频率 |

实现位置：`should_compute_self_attention`、`should_compute_cross_attention`（PixArt）、`should_compute_mlp`（`gmc_utils.py`）。

PixArt 另含 ToCa 基线对比路径（`_forward_toca`）；GMC 路径与 DiT 调度一致，DiT 无 Cross-attention 模块。

## 快速开始

### 单元测试

```bash
python test_gmc.py
```

### DiT-XL/2（GMC-DiT）

```bash
bash scripts/setup_dit.sh

python GMC-DiT/generate.py \
  --preset default \
  --class_id 207 \
  --out sample.png
```

### PixArt-α（GMC-PixArt）

```bash
bash scripts/setup_pixart.sh

python GMC-PixArt/generate.py \
  --model_path /path/to/PixArt-XL-2-256x256.pth \
  --t5_path /path/to/t5_ckpts \
  --vae_path /path/to/sd-vae-ft-mse \
  --prompt "A cat wearing sunglasses."
```

性能对比（基线 / ToCa / GMC，PixArt）：

```bash
python GMC-PixArt/benchmark_compare.py \
  --model_path /path/to/PixArt-XL-2-256x256.pth \
  --t5_path /path/to/t5_ckpts \
  --num_fid 3
```

## 与 DGC 的关系

- **GMC-DiT** 与 **GMC-PixArt** 共享 `GMCConfig` 与步级调度；PixArt 额外支持 ToCa 基线对比（统一 Block）。
- 命名与论文对齐，独立维护于 `GMC/`。

## API 示例

```python
from gmc_utils import GMCConfig
from GMC-DiT.gmc_model import DiTWithGMC

model = DiTWithGMC(
    gmc_config=GMCConfig(casa_interval=4, mlp_anchor_step=30, mlp_interval=4),
    total_sampling_steps=50,
)
model.enable_cache(True)
model.reset_cache()
# ... DDIM 采样 ...
stats = model.get_cache_stats()
```

```python
from GMC-PixArt.gmc_pixart_block import apply_gmc_blocks
from GMC-PixArt.gmc_cache import gmc_cache_init
from GMC-PixArt.config import DEFAULT_GMC_PIXART_CONFIG

apply_gmc_blocks(pixart_model)
cache_dic, current = gmc_cache_init(DEFAULT_GMC_PIXART_CONFIG, num_steps=20)
```
