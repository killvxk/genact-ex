# Phase 2: Training Loop - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

实现核心训练循环，包括初始化阶段（模型加载、分布式环境）、训练进度显示（epoch/step 进度条、loss 指标）和 GPU 状态网格（64+ GPU 的显存、利用率、温度显示）。用户将看到逼真的 PyTorch/HuggingFace 风格训练输出。

</domain>

<decisions>
## Implementation Decisions

### 初始化输出风格
- 详细日志风格，类似 PyTorch DDP 的输出
- 每行带完整时间戳，格式 `[2026-01-30 10:45:32]`
- 显示分布式进程信息（Rank/Node），如 `[Rank 0/64] [Node 0/8]`
- 使用彩色日志区分不同类型（INFO 绿色、WARNING 黄色等）
- 每个初始化步骤后显示耗时，如 "Model loaded in 12.3s"

### 初始化内容
- 模型加载：显示名称、参数量、架构细节（层数、hidden size、attention heads）
- 模型加载过程显示进度条，如 "Loading model weights... [=====>    ] 50%"
- Tokenizer 加载：显示名称、vocab 大小
- 数据集加载：显示名称、样本数、batch 配置
- 分布式初始化：NCCL 版本、backend、详细通信测试（AllReduce 测试结果）
- GPU 检测：逐个显示 64+ 个 GPU 的检测结果

### 训练进度显示
- 混合模式：进度条 + 定期输出详细日志
- 双层进度条：epoch 进度条 + step 进度条
- 进度条旁显示指标：Loss、Perplexity、学习率、Tokens/s
- 详细日志每 10 个 step 输出一次
- Loss 变化模式：波动下降（下降趋势中带有小幅波动，更真实）
- 显示已用时间和预估剩余时间 (ETA)
- 每个 epoch 结束时显示摘要（平均 loss、总时间等）
- 显示梯度信息（grad norm、梯度裁剪）

### GPU 状态网格
- 按节点分组显示，每个节点显示其 GPU
- 每个 GPU 显示：显存使用、利用率、温度、功耗
- 随详细日志一起更新（每 10 steps）
- 使用彩色状态标识健康状态（绿色正常、黄色警告、红色危险）

### 输出节奏与时机
- 全程保持稳定节奏
- 输出速度由 Claude 决定最自然的速度
- 模拟多个 epoch 完整循环
- 每个 epoch 有 500-2000 个 step（中等数量）
- 训练过程中显示分布式通信日志（NCCL AllReduce 等）
- 偶尔显示 WARNING 级别日志增加真实感（如显存临近上限）
- 显示学习率调度器信息（warmup、decay 等）
- 显示混合精度训练信息（AMP/FP16/BF16）

### Claude's Discretion
- 具体输出速度/延迟时间
- 进度条的具体样式
- 警告信息的具体内容和频率
- 通信日志的具体格式

</decisions>

<specifics>
## Specific Ideas

- 输出风格参考 PyTorch DDP 和 HuggingFace Trainer 的真实日志
- GPU 状态网格类似 nvidia-smi 的信息密度，但按节点分组
- Loss 下降曲线应该有真实训练的波动特征，不是完美平滑

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-training-loop*
*Context gathered: 2026-01-30*
