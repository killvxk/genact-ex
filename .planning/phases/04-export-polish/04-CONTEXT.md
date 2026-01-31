# Phase 4: Export & Polish - Context

**Gathered:** 2026-01-31
**Status:** Ready for planning

<domain>
## Phase Boundary

训练完成后的导出阶段和最终收尾。用户看到模型导出过程（格式转换、文件保存），训练完成总结（统计信息），然后无限循环开始新一轮训练。需通过 clippy 检查并确保 WASM 兼容。

</domain>

<decisions>
## Implementation Decisions

### 导出过程展示
- 使用 SafeTensors 格式导出（与 checkpoint 保持一致）
- 多步骤显示：合并权重 → 优化模型 → 序列化 → 写入文件
- 每个步骤都有进度条
- 合并权重步骤最慢，写文件较快
- 显示详细层级信息（每层参数量、shard 数量）
- 显示 shard 分片导出（如 model-00001-of-00004.safetensors）
- 显示配套文件保存：config.json、tokenizer.json

### 训练完成总结
- 导出完成后显示总结报告
- 使用结构化表格格式（类似 PyTorch Lightning）
- 包含内容：
  - 时间与进度统计（总训练时间、epochs、steps）
  - 损失与指标（最终 loss、最佳 loss、最终 perplexity）
  - GPU 使用统计（平均利用率、峰值内存）
  - 保存信息（模型保存路径、总大小）
- 时间显示使用可读格式（2h 34m 12s）

### 导出文件信息
- 路径格式：HuggingFace 风格（./outputs/llama-7b-chat-v2/checkpoint-final/）
- 文件大小与模型名称匹配（7B → ~14GB 等）
- 多 shard 显示汇总（4 shards, 14.2GB total）
- 显示配套文件保存成功：
  - config.json
  - tokenizer.json
  - generation_config.json
  - model.safetensors.index.json

### 收尾视觉效果
- 使用简洁勾号风格成功消息（✔ Training completed successfully!）
- 导出和总结之间有视觉分隔线
- 无限循环：完成后开始新一轮训练
- 循环过渡：显示过渡消息（"启动新训练任务..."）

### Claude's Discretion
- 具体的分隔线样式（等号、破折号等）
- 过渡消息的具体文案
- 表格的具体对齐和边框样式
- 各步骤的具体时间分配

</decisions>

<specifics>
## Specific Ideas

- 输出风格参考 HuggingFace Transformers 和 PyTorch Lightning
- shard 文件命名参考 HuggingFace Hub 格式（model-00001-of-00004.safetensors）
- 总结表格参考 PyTorch Lightning 的 trainer.fit() 完成后输出

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-export-polish*
*Context gathered: 2026-01-31*
