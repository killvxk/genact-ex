# Phase 3: Validation & Checkpoints - Context

**Gathered:** 2026-01-30
**Status:** Ready for planning

<domain>
## Phase Boundary

在训练过程中显示验证阶段和检查点保存。用户看到 epoch 结束后的验证进度、验证指标、以及基于 loss 改善的检查点保存。验证和检查点是训练循环的附加输出，不改变 Phase 2 的核心训练逻辑。

</domain>

<decisions>
## Implementation Decisions

### 验证触发时机
- 每个 epoch 结束后触发一次验证
- 验证开始时显示视觉分隔符 + 消息（如 `=== Validation ===`）
- 每次验证持续 10-20 秒（模拟时间）
- 验证集 step 数量随机在 80-150 步之间

### 验证输出格式
- 复用训练进度条样式，保持视觉一致
- 验证过程显示扩展指标：loss、perplexity、accuracy、tokens/sec
- 验证完成后显示详细多行报告（对比训练指标）
- 验证 loss 应该稍高于训练 loss（约 5-15% 更高），更真实

### 检查点保存行为
- 仅当验证 loss 改善时才保存检查点（best model 策略）
- 文件命名格式：`model-step-{N}.safetensors`
- 显示 GB 级别的文件大小（如 2.1GB、5.7GB）
- 保存过程带进度条，模拟写入过程

### 异常与警告日志
- 警告类型包括：梯度警告、Early stopping 提示、NaN 检测警告、Loss 上升警告
- 警告频率：30-40% 几率出现
- 复用 Phase 2 的 WARNING 日志格式，保持一致
- Early stopping 显示 patience 计数（如 "patience: 2/5"），但永不实际触发停止

### Claude's Discretion
- 具体的 loss/accuracy 数值范围
- 验证指标的具体格式细节
- 进度条的精确样式
- 检查点保存的速度模拟

</decisions>

<specifics>
## Specific Ideas

- 验证分隔符风格参考 PyTorch Lightning 的输出风格
- 检查点使用 safetensors 格式名称（现代 HuggingFace 风格）
- Early stopping patience 计数增加紧张感但不中断训练（无限循环需求）
- 警告消息应该看起来严肃但不会让观察者担心（模拟常见训练问题）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-validation-checkpoints*
*Context gathered: 2026-01-30*
