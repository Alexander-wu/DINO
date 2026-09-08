# DINO

**Differential-Integral Neural Operator**：使用局部卷积与全局注意力预测流体场的时空演化。模型采用编码、算子演化、解码结构，并通过特征 skip connection 融合浅层空间信息。

支持 McWilliams 和 Forced Li 两套任务，提供单步训练、自回归预测，以及默认关闭的零阶矩约束和网格缩放选项。

## 文档

- [模型结构与接口](docs/MODEL.md)
- [训练、推理与对比实验](docs/EXPERIMENTS.md)
- [验证范围与待处理事项](docs/VALIDATION.md)

## 目录

```text
DINO/
├── model/
│   ├── dino.py                    # 唯一核心模型实现
│   └── DINO.ipynb                 # 导入模型的随机输入示例
├── model_baselines/              # 对比模型
├── dataloader_McWilliams.py
├── dataloader_Forced_Li.py
├── train_McWilliams.py
├── train_Forced_Li.py
├── inference_McWilliams.py
├── inference_Forced_Li.py
├── config_McWilliams.yaml
├── config_Forced_Li.yaml
├── ablation_study/               # 结构消融入口（部分实现尚缺）
├── data/                        # 自备数据
└── docs/
```

运行后，日志、权重和预测结果分别写入 `runs/logs`、`runs/checkpoints` 和 `runs/results`。交付目录不包含历史实验图、日志、数据或预训练权重。

## 模型调用

```python
import torch
from model.dino import DINO

model = DINO(
    shape_in=(1, 1, 128, 128),
    spatial_hidden_dim=128,
    temporal_hidden_dim=256,
    output_channels=1,
    num_spatial_layers=4,
    num_temporal_layers=8,
    moment_constraint=False,
    grid_scaling=False,
).eval()

with torch.no_grad():
    prediction = model(torch.randn(1, 1, 1, 128, 128))
# prediction.shape: [1, 1, 1, 128, 128]
```

输入和输出维度均为 `[B, T, C, H, W]`。当前两套主任务使用 `T=1`、`C=1`。

## 开始实验

1. 准备 PyTorch、timm、NumPy、SciPy、PyYAML、tqdm、Matplotlib、pandas；当前尚未锁定完整依赖版本。
2. 在主配置的 `datas.DINO.data_path` 填写实际数据路径，或把数据放入 `data/`。默认训练使用 CUDA/NCCL。
3. 从项目根目录启动，例如：

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_McWilliams.py
```

推理前，按设备数量和实际 checkpoint 调整脚本设置，详见[实验指南](docs/EXPERIMENTS.md)。

两份主配置均默认选择 DINO，并关闭以下两个独立开关：

```yaml
moment_constraint: false
grid_scaling: false
```

矩约束仅保证指定局部卷积核的零阶矩为零；缩放按隐藏网格间距执行。二者均不自动保证完整模型的微分收敛或长期稳定性。
