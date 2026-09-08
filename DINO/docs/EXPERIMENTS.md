# DINO 实验指南

## 数据与任务

| 任务 | 数据格式 | 时间与空间 | 推理 |
| --- | --- | --- | --- |
| McWilliams | `.pt`，读取 `vorticity` | `[1280,100,128,128]` | 单帧输入，滚动 99 步 |
| Forced Li | `.mat`，读取 `u` | `[1200,64,64,20]` | 单帧输入，滚动 19 步 |

统一输入格式为 `[B,T,C,H,W]`，当前变量为单通道涡量。数据路径在对应 YAML 的 `datas.DINO.data_path` 中设置。

## 训练

两份主 YAML 均通过 `selected_model: DINO` 选择模型。`models`、`trainings`、`datas`、`loggings` 分别管理结构、训练、数据和输出参数。

默认训练设置：单步 MSE、Adam、初始学习率 0.001、余弦学习率调度、500 轮、随机种子 42、每进程 batch size 20。默认使用 DistributedDataParallel，依赖 CUDA/NCCL。

从项目根目录运行：

```bash
torchrun --nnodes=1 --nproc_per_node=8 train_McWilliams.py
# 或
torchrun --nnodes=1 --nproc_per_node=8 train_Forced_Li.py
```

可按实际 GPU 数量修改进程数，例如单 GPU 用 `--nproc_per_node=1`。训练与推理脚本目前读取固定名称的 YAML，尚未提供 `--config` 参数。

## 对比实验开关

在 `models.DINO.parameters` 下设置：

```yaml
moment_constraint: false
grid_scaling: false
grid_spacing: null
derivative_order: 1
domain_size: [1.0, 1.0]
```

| 实验 | moment_constraint | grid_scaling | 建议 backbone 后缀 |
| --- | --- | --- | --- |
| 基线 | false | false | `_base` |
| 仅零阶矩约束 | true | false | `_moment` |
| 仅网格缩放 | false | true | `_scale` |
| 两者都开启 | true | true | `_moment_scale` |

保持数据、随机种子、通道、训练轮数等其他设置一致。每组运行前修改 `loggings.DINO.backbone`，例如 `DINO_McWilliams_base`，并保留该组 YAML 副本。

**不同实验组必须使用不同 backbone。** 训练入口会尝试加载同名最佳 checkpoint，且并非完整恢复优化器与调度器状态。独立实验应使用全新的输出名称。

## 推理

```bash
python inference_McWilliams.py
# 或
python inference_Forced_Li.py
```

- McWilliams 入口目前在有 CUDA 时指定 `cuda:3`，加载 `epoch=250` 的权重；运行前按设备与权重修改这两项。
- Forced Li 入口优先加载最佳权重，否则尝试第 50 轮权重。
- 使用与训练一致的模型参数、约束开关和域配置。
- 输出包括预测数组、真值和可视化，位于 `runs/results`。

## 结构消融与基线

`ablation_study/` 提供结构消融配置与入口，但部分所引用模型实现尚未包含，因此不能直接运行全部结构消融。当前可直接配置的四组开关实验使用两套主训练入口。

`model_baselines/` 包含多个对比模型；并非每个文件都已接入所有训练注册表或拥有完整配置。切换基线前需检查四个配置节及模型注册表。
