# DINO 模型结构

## 数据流

```text
输入 [B,T,C,H,W]
  ↓
LiftingOperator：卷积编码与降采样 ─── 浅层特征 ───┐
  ↓                                            │
OperatorEvolution：局部与全局模块                │
  ↓                                            │
ProjectionOperator：上采样 + 特征拼接 ←─────────┘
  ↓
输出 [B,T,C_out,H,W]
```

四层空间编码采用步幅 `[1,2,1,2]`，空间尺寸约缩小四倍。解码时通过通道拼接融合浅层特征。这里的 skip connection 是特征连接。

算子演化将时间与通道维合并。`OperatorEvolutionBlock` 在输入输出通道相同且层号大于零时选择 `MHSA`，否则选择 `Conv`。默认通道为 128 → 256 → 128、演化深度为八层，因此排列为：

`Conv → MHSA × 6 → Conv`

## 模块

| 模块 | 实现 |
| --- | --- |
| `LiftingOperator` | 多层空间卷积、归一化、激活及降采样 |
| `LocalDifferentialOperator` | 深度卷积位置嵌入、1×1 投影、5×5 depthwise 卷积及残差 MLP |
| `GlobalIntegralOperator` | 空间 token 自注意力、LayerNorm、残差 MLP 和可学习层缩放 |
| `AttentionIntegralKernel` | 多头 Q/K/V 注意力和全局加权聚合 |
| `ProjectionOperator` | 转置卷积上采样、浅层特征拼接和输出通道投影 |
| `MomentConstrainedConv2d` | 可独立开启零阶矩约束与网格缩放的 Conv2d |

## 构造参数

| 参数 | 含义 |
| --- | --- |
| `shape_in` | `[T,C,H,W]` |
| `spatial_hidden_dim` | 编码、解码及潜在空间通道数 |
| `temporal_hidden_dim` | 算子演化隐藏通道数 |
| `output_channels` | 输出变量数 |
| `num_spatial_layers` | 空间编码与解码层数 |
| `num_temporal_layers` | 算子演化层数，至少为 2 |
| `moment_constraint` | 零阶矩约束，默认 `False` |
| `grid_scaling` | 网格缩放，默认 `False` |
| `grid_spacing` | 显式隐藏网格间距；默认 `None` 表示自动计算 |
| `derivative_order` | 缩放指数 p，默认 1，必须为正整数 |
| `domain_size` | `[Ly,Lx]`，默认单位域 `[1,1]` |

`in_time_seq_length` 和 `out_time_seq_length` 当前仅保存为属性，不会改变输出时间长度。输入时间长度应与构造时 `shape_in[0]` 一致。

## 零阶矩约束

仅作用于局部模块的 5×5 depthwise 卷积。开启后每次前向使用：

```python
weight = weight - weight.mean(dim=(-2, -1), keepdim=True)
```

每个空间核的权重和为零，并忽略该层偏置。投影保留梯度；保存的参数本身不必零和。位置嵌入、1×1 卷积、注意力及编码解码器不受约束。

保持零填充，因此常数场抵消只在不受边界填充影响的内部区域成立。该约束不固定导数方向或一阶矩，不保证整个残差模块对常数输入输出为零。

## 网格缩放

开启时把局部 5×5 卷积输出除以 `h**p`。仅开启缩放时偏置也随输出缩放；同时开启矩约束时偏置被忽略。

自动模式采用周期网格约定：`hy=Ly/H_latent`、`hx=Lx/W_latent`，根据实际传入局部算子的张量尺寸计算，要求两个间距相等。非等距网格会报错；当前不实现分方向的各向异性微分缩放。

可通过 `grid_spacing` 显式指定标量隐藏网格间距，此时它覆盖自动推断。非周期含端点网格需显式设置适当间距。默认域长度仅表示归一化坐标，实际物理长度应由实验配置指定。

设置 `p=2` 只改变缩放指数，不会自动施加二阶微分所需的矩条件。已知差分核的收敛性质不能直接推广到任意学习核或完整网络。

## 权重

注册模块的参数路径保持固定，便于已有同结构权重加载。两个开关不新增可训练参数，也不改变局部卷积的 `state_dict` 键名。开关、域长度及间距不保存在 `state_dict` 中，必须与实验配置一起管理。

局部/全局模块排列或通道数不同的 checkpoint 不能保证兼容。当前推理入口支持去除分布式保存的 `module.` 前缀。
