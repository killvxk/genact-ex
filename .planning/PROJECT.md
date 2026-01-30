# genact LLM Training Module

## What This Is

为 genact 添加一个 LLM 训练模拟模块，模拟大规模分布式训练过程。模块将显示逼真的 PyTorch/HuggingFace 风格训练输出，包括多 GPU 集群状态、NCCL 通信、训练指标变化和进度条。目标用户是想"假装在训练 AI 模型"的开发者。

## Core Value

**逼真地模拟大规模 LLM 训练过程，让观看者相信正在进行真正的模型训练。**

## Requirements

### Validated

- 现有 genact 模块系统运行正常 — existing
- 跨平台 I/O 抽象 (native/WASM) 可用 — existing
- Module trait 定义完善 — existing

### Active

- [ ] 初始化阶段：显示模型加载、分布式环境初始化
- [ ] 训练阶段：多 epoch 循环，带进度条和 step 计数
- [ ] 训练指标：loss/perplexity 随训练逐渐下降
- [ ] GPU 状态：64+ GPU 集群的显存、利用率、温度显示
- [ ] 分布式通信：NCCL AllReduce 等通信日志
- [ ] Checkpoint 保存：定期显示保存检查点
- [ ] 验证阶段：在验证集上评估模型
- [ ] 导出阶段：保存模型、转换格式
- [ ] 速度/时间：tokens/s、samples/s、ETA 估算
- [ ] 模型名称：随机混合真实模型名和恶搞名称

### Out of Scope

- 真实的训练功能 — genact 只做模拟，不执行实际计算
- 网络请求/外部 API — 保持离线运行
- 配置文件支持 — 使用随机生成的配置即可

## Context

**现有代码库状态：**
- genact 是一个成熟的 Rust 项目，使用 async trait 模块系统
- 已有 20 个模块作为参考（cargo、docker_build、terraform 等）
- 使用 `io::*` 函数进行跨平台输出，支持 WASM
- 使用 `yansi` crate 实现终端颜色
- 使用 `progress_string` crate 实现进度条
- 数据文件放在 `data/` 目录，通过 `include_str!()` 嵌入

**参考模块：**
- `cargo.rs` - 简单的进度输出模式
- `docker_build.rs` - 多阶段输出
- `kernel_compile.rs` - 长时间运行的复杂输出

## Constraints

- **平台兼容**: 必须同时支持 native 和 wasm32 目标
- **无 println!**: 使用 `io::print()`, `io::dprint()`, `io::newline()` 替代
- **Clippy 通过**: `cargo clippy -- -D warnings` 必须通过
- **现有架构**: 遵循 Module trait 和现有代码风格

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch/HuggingFace 输出风格 | 用户熟悉，最具代表性 | — Pending |
| 随机混合真实/恶搞模型名 | 增加趣味性和逼真度 | — Pending |
| 大规模集群 (64+ GPU) | 更壮观的视觉效果 | — Pending |

---
*Last updated: 2026-01-30 after initialization*
