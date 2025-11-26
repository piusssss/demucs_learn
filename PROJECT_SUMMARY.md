# Demucs 学习与改进记录

## 项目背景

因音乐伴奏分离需求接触 Demucs，发现分离效果优秀后决定深入学习。

---

## 阶段一：源码理解

### 学习内容

- 完整分析项目源码：预处理、模型架构、训练流程
- 模型演进：Demucs → HDemucs → HTDemucs
- 涉及领域：深度学习、信号处理、音频处理
- 学习方式：AI 辅助分析和解释，针对困惑点进行讨论验证

### 成果

掌握音源分离系统完整工作原理。

---

## 阶段二：自定义训练

### 目标

使用自定义数据集训练模型（单源伴奏分离，因硬件限制和个人偏好）。

### 数据问题

网络数据源存在严重问题：
- 非官方数据不一致
- 重采样、标准化后数据不对齐
- 导致训练失败

### 解决方案

- 收集标准格式原曲
- 使用官方模型精细分离生成伴奏作为训练标签
- 类似知识蒸馏思路

### 结果

训练成功，在特定音乐风格上表现良好。

---

## 阶段三：架构改进

### 目标

尝试设计改进架构，超越原版性能。

### 实验设置

- 数据集：MUSDB18 标准数据集
- 分离目标：单源伴奏（instrumental）
- 基础框架：HTDemucs
- 方法：迭代式设计和验证

**注**：选择单源分离主要因硬件限制和个人偏好

### 实现变体

**官方 HTDemucs 基线性能**：（segment=10）
- FLOPs: 137.52 G
- 参数量: 20.93 M
- 处理时间: 6.98s / 180s 音频

共设计实现至少 11 个变体：

注：所有性能指标与通道数高度相关，仅作参考。
#### htdemucs_p 设计思路

**核心改进**：
- 深度从 4 层增加到 6 层
- 重新引入 HDemucs 的分支融合分离机制
- 拆解 Transformer 的交替自注意力和交叉注意力
- 将注意力机制分布到编解码器的融合分离前后

**性能指标**：
- FLOPs: 89.14 G
- 参数量: 53.57 M
- 处理时间: 13.15s / 180s 音频

**备注**：第一代设计，不够成熟

#### htdemucs_i 设计思路

**核心改进**：
- 延续 p 的思路，但更规范精细
- 引入 Transformer 部分的**可学习残差权重**（6 个独立权重参数）
- 更精确的 Transformer 位置：
  - 编码器：分支融合前交叉注意力（idx == depth-3）→ 融合后自注意力（idx == depth-2）
  - 解码器：自注意力（idx == 0）→ 分支分离时交叉注意力（idx == depth-len(tdecoder)）
- 4 个 Transformer 组件对称分布

**可学习残差权重**：
```python
# 6 个独立的可学习权重
residual_weight_cross_en      # 编码器交叉注意力（频域）
residual_weight_cross_en_t    # 编码器交叉注意力（时域）
residual_weight_2en           # 编码器自注意力
residual_weight_2de           # 解码器自注意力
residual_weight_cross_de      # 解码器交叉注意力（频域）
residual_weight_cross_de_t    # 解码器交叉注意力（时域）

# 使用方式：alpha = sigmoid(weight)
x = alpha * transformer_output + (1 - alpha) * residual
```
**性能指标**：
- FLOPs: 99.01 G
- 参数量: 77.92 M
- 处理时间: 7.08s / 180s 音频

#### htdemucs_c 设计思路

**核心改进**：
- 完全取消 Transformer 组件（t_layers=0）
- 将节省的计算资源倾斜给卷积层
- 更大的卷积核（kernel_size=16）
- 更大的 STFT 窗口（nfft=8192）
- 深度增加到 6 层

**设计理念**：
- 纯卷积架构，去除注意力机制
- 通过更大的感受野补偿 Transformer 的全局建模能力

- 在拟合测试中效果优异

**性能指标**：
- FLOPs: 217.38 G
- 参数量: 46.95 M
- 处理时间: 5.63s / 180s 音频

#### htdemucs_s 设计思路

**核心改进**：
- 类似原版架构，但深度从 4 层减少到 3 层
- 节省的计算资源平衡分配到其他参数：
  - 更大的通道数（channels=96 vs 原版 48）
  - 更大的 STFT 窗口（nfft=8192）
  - 更大的卷积核（kernel_size=16）
  - 更大的 Transformer 隐藏层（t_hidden_scale=10.0）
  - 更大的步长（stride=8, time_stride=8）

**设计理念**：
- 减少深度，增加宽度
- 用更大的参数补偿层数减少

**性能指标**：
- FLOPs: 314.81 G
- 参数量: 29.59 M
- 处理时间: 6.73s / 180s 音频

#### htdemucs_d 设计思路

**核心改进**：
- 实现分层的卷积核和步长（逐层 //2）
- 分层设计：
  - layers_1=1：初始层，保持原始 kernel 和 stride
  - layers_2=3：中间层，kernel 和 stride 减半
  - layers_3=5：深层，kernel 和 stride 再减半
- 快速下采样后进行精细卷积
- Transformer 设计演进：
  - 旧版：层有对应的 Transformer
  - 当前版：只保留中间层的 Transformer（t_layers=3）

**设计理念**：
- 前期快速下采样，后期精细处理
- 减少冗余的 Transformer 计算

**性能指标**：
- FLOPs: 104.32 G
- 参数量: 29.19 M
- 处理时间: 6.14s / 180s 音频

#### htdemucs_d2 设计思路

**核心改进**：
- 延续 d 的分层思路，调整部分参数：
  - 通道数：128 → 108
  - Transformer 隐藏层：t_hidden_scale=2.0 → 0.5
  - Transformer 注意力头：t_heads=4 → 8
  - DConv 初始化：dconv_init=1e-2 → 3e-3
  - 权重缩放：rescale=0.2 → 0.15
  - 段长度：segment=4 → 10
- 分层参数变化策略调整

**设计理念**：
- 微调 d 的参数配置
- 平衡计算量和性能

**性能指标**：
- FLOPs: 231.08 G
- 参数量: 59.14 M
- 处理时间: 5.68s / 180s 音频

#### htdemucs_d3 设计思路

**核心改进**：
- 发现之前下采样可能过激，采用更缓和的策略
- 更缓和的下采样：stride=4（vs d/d2 的 stride=8）
- 分层通道增长策略（应对早期信息丢失）：
  - 初始：growth=2.5（激进增长）
  - layers_1 后：growth=2.0
  - layers_2 后：growth=1.5（减缓增长）
  - layers_3 后：growth=1.5
- 取消 Transformer（t_layers=0）
- 更大的卷积核（kernel_size=32）
- 更小的初始通道数（channels=24）
- 深度减少到 5 层

**设计理念**：
- 浅层激进增长通道，保留更多信息
- 深层减缓增长，控制参数量
- 缓和下采样，减少信息丢失

**性能指标**：
- FLOPs: 284.49 G
- 参数量: 11.57 M
- 处理时间: 6.86s / 180s 音频

#### htdemucs_d4 设计思路

**核心改进**：
- 更加缓和的策略，主要增长卷积核而非下采样
- 分层策略调整：
  - layers_1=2, layers_2=4, layers_3=6（更均匀的分层）
- 卷积核增长策略（逐层 ×2）：
  - 初始：kernel_size=8
  - layers_2 后：ker × 2
  - layers_3 后：ker × 2
  - 最深层：ker × 2
- 保持缓和的下采样（stride=4）
- 更小的通道增长（growth=1.5（仅第0层））
- 中等初始通道数（channels=64）
- 恢复 Transformer（t_layers=3）

**设计理念**：
- 通过增大卷积核扩大感受野，而非激进下采样
- 更均匀的网络结构

**性能指标**：
- FLOPs: 280.50 G
- 参数量: 25.46 M
- 处理时间: 7.27s / 180s 音频

#### htdemucs_n 设计思路

**核心改进**（较大改进）：
- 改用**线性 Transformer**（降低计算复杂度）
- 引入**多频域分辨率处理**：[2048, 4096, 8192]
- 每个分辨率独立编解码处理

**多分辨率融合机制**：
1. 编解码器之间，Transformer 前：
   - 三个分辨率带权重融合
   - 融合后与时域进行 Transformer 交互
2. Transformer 后：
   - 带权分离回各分辨率
   - 加上融合前的残差
3. 最终输出：
   - 三个分辨率结果再次加权融合  （后改为nf的卷积融合）

**可学习融合权重**：
```python
fusion_weights = nn.Parameter(torch.ones(3) / 3)  # 3 个分辨率的权重
weight_ema = torch.ones(3) / 3  # EMA 平滑
weight_momentum = 0.9  # 动量平滑
```

**突破性成果**：
- **首个较明显超越原版的变体**

**性能指标**：
- FLOPs: 86.27  G
- 参数量: 9.61 M
- 处理时间: 5.33s / 180s 音频

#### htdemucs_nc 设计思路

**核心改进**：
- 完全抛弃 Transformer
- 完全抛弃频域处理（STFT）
- **纯时域卷积架构**
- 非常精简的实现

**架构特点**：
- 只有时域编解码器（tencoder/tdecoder）
- 深度增加到 7 层
- 更小的步长（stride=2）
- 更强的 DConv（dconv_mode=3）
- 极简设计

**实验结果**：
- 效果一般

**性能指标**：
- FLOPs: 233.23 G
- 参数量: 33.06 M
- 处理时间: 9.91s / 180s 音频

#### htdemucs_nf 设计思路

**核心改进**：
- **纯频域架构**
- 完全抛弃时域分支
- 完全抛弃 Transformer
- 扩展多分辨率：[1024, 2048, 4096, 8192, 16384]（5 个分辨率）

**多分辨率融合机制**：
- 不再使用可学习权重融合
- 改用**大卷积对 5 个频域进行维度融合**：
  ```python
    fusion_conv_wide=129
    self.fusion_conv = nn.Conv2d(
        in_channels=num_groups, 
        out_channels=num_groups,  
        kernel_size=[self.num_resolutions, fusion_conv_wide],  
        stride=1,
        padding=[0,(fusion_conv_wide-1)//2],
        groups=num_groups//self.audio_channels  
    )
  ```
- 每个分辨率独立编解码
- 最终通过融合卷积合并

**实验结果**：
- 效果不错

**性能指标**：
- FLOPs: 400.00 G
- 参数量: 27.39 M
- 处理时间: 7.41s / 180s 音频

#### htdemucs_nn 设计思路

**核心改进**：
- 基于 htdemucs_n，引入**源特异性融合权重**
- 每个源（drums, bass, other, vocals）有独立的分辨率权重
- 删除卷积融合，改用轻量级的可学习权重

**源特异性融合机制**：
```python
# 全局权重（瓶颈处）：所有源共享
fusion_weights = nn.Parameter(torch.ones(3) / 3)  # [3]

# 源特异性权重（最终融合）：每个源独立
final_fusion_weights = nn.Parameter(
    torch.ones(4, 3) / 3  # [4源, 3分辨率]
)

# 融合方式
for s in range(sources):
    for r in range(resolutions):
        output[:, s] += weights[s, r] * x_time_list[r][:, s]
```

**设计理念**：
- 不同源对不同分辨率的需求不同
- Drums 偏好低分辨率（时间精度）
- Vocals 偏好中分辨率（平衡时频）
- 让模型自动学习每个源的最优分辨率组合

**参数增加**：
- 全局权重：3 个参数
- 源特异性权重：4×3 = 12 个参数
- 总计仅增加 12 个参数

**实验结果**：
- 参数量几乎不变，但提供更灵活的融合策略

#### htdemucs_dnf 设计思路

**核心改进**：
- 基于 htdemucs_nf（5 分辨率）
- 将卷积融合替换为 **TimeUNet2D 融合模块**
- 引入编码器-解码器结构学习分辨率融合

**TimeUNet2D 融合模块**：
```python
class TimeUNet2D(nn.Module):
    # 输入: [B, C, num_res, T]
    # 编码器: 逐步压缩分辨率维度
    # 解码器: 恢复到单一输出
    # 输出: [B, C, 1, T]
```

**架构特点**：
- Encoder-Decoder 结构
- Skip connections 连接编解码器
- GLU 激活函数
- 自适应学习分辨率融合模式

**设计理念**：
- 用神经网络学习复杂的融合策略
- 而非简单的加权或卷积

**实验结果**：
- 计算量增加
- 效果不如简单加权融合（可能信息损失）

#### htdemucs_dn 设计思路

**核心改进**：
- dnf思路迁移到n

**实验结果**：
- 计算量增加
- 效果一般，推测第二个unet太简略。



### 实验结果

- 前期变体：与原版持平或略低
- **最终突破**：多频率分辨率窗口 超越原版

### 其他探索

- ONNX 导出（不兼容，主要由于stft的复数计算）
- TorchScript 转换 （反而在gpu下推理更慢）
- 模型量化
- 该架构的音乐生成，风格迁移
（未深入）

---

## 技术实现

### 训练配置

**数据**：
- 阶段二：自定义数据 + 官方模型生成标签（单源伴奏）
- 阶段三：MUSDB18 标准数据集（单源伴奏）

**超参数**：
- 学习率：手动根据情况调整
- 优化器：Adam
- 损失函数：L1 Loss
- 其余参数： 略


### 性能对比（musdb18_hq test单源）

htdemucs(官方预训练):
Tracks evaluated: 50
Overall SDR Mean: 18.411 dB
Overall SDR Median: 16.347 dB
Overall SDR Std: 9.786 dB

htdemucs(同一训练策略下):
Tracks evaluated: 50                  Tracks evaluated: 50   
Overall SDR Mean: 14.219 dB           Overall SDR Mean: 14.675 dB
Overall SDR Median: 11.140 dB         Overall SDR Median: 11.198 dB
Overall SDR Std: 9.826 dB             Overall SDR Std: 10.316 dB
### 10epoch退火收敛                  13epoch重启快速退火
htdemucs_n(同一训练策略下):
Tracks evaluated: 50                  Tracks evaluated: 50
Overall SDR Mean: 17.454 dB           Overall SDR Mean: 18.183 dB
Overall SDR Median: 12.470 dB         Overall SDR Median: 12.556 dB
Overall SDR Std: 12.708 dB            Overall SDR Std: 13.523 dB
htdemucs_nn(同一训练策略下):
Tracks evaluated: 50                  Tracks evaluated: 50
Overall SDR Mean: 18.323 dB           Overall SDR Mean: 18.474 dB
Overall SDR Median: 12.435 dB         Overall SDR Median: 12.561 dB
Overall SDR Std: 13.744 dB            Overall SDR Std: 13.898 dB



---

常用脚本：
calculate_complexity.py  # 计算模型复杂度
train_windows_fixed.py   # 训练脚本
evaluate_instrumental.py # 评估脚本
```

---

## 关键技术点

### 成功改进

1. **多频率分辨率窗口**
   - 不同窗口捕捉不同时频特性
   - 提供更全面的频谱信息

2. **线性 Transformer**
   - 降低计算复杂度
   - 保持性能同时提升效率
   - （原版transformer在该任务下对性能消耗很大，但收益似乎并没有这么显著。或许在大规模训练中才能体现？）


---

