# Requirements: genact LLM Training Module

**Defined:** 2026-01-30
**Core Value:** 逼真地模拟大规模 LLM 训练过程，让观看者相信正在进行真正的模型训练

## v1 Requirements

### 初始化 (INIT)

- [ ] **INIT-01**: 显示模型加载信息 (模型名称、参数量、架构)
- [ ] **INIT-02**: 显示 Tokenizer 加载进度
- [ ] **INIT-03**: 显示数据集加载信息 (数据集名称、样本数)
- [ ] **INIT-04**: 显示分布式环境初始化 (NCCL 初始化、GPU 数量、节点数)
- [ ] **INIT-05**: 显示 GPU 状态检查 (64+ GPU 检测确认)

### 训练 (TRAIN)

- [ ] **TRAIN-01**: 显示 Epoch 进度条 (当前/总数, 百分比进度条)
- [ ] **TRAIN-02**: 显示 Step 级别指标 (step 编号, loss, learning rate)
- [ ] **TRAIN-03**: 显示 perplexity 指标 (PPL = e^loss)
- [ ] **TRAIN-04**: 显示速度统计 (tokens/s, samples/s)
- [ ] **TRAIN-05**: 显示时间估算 (elapsed, ETA)
- [ ] **TRAIN-06**: 显示 NCCL 通信日志 (AllReduce 完成时间)
- [ ] **TRAIN-07**: Loss 值随训练逐渐下降 (指数衰减 + 噪声)

### GPU 状态 (GPU)

- [ ] **GPU-01**: 显示各 GPU 显存占用 (已用/总量, 百分比)
- [ ] **GPU-02**: 显示各 GPU 利用率 (百分比)
- [ ] **GPU-03**: 显示各 GPU 温度 (摄氏度)
- [ ] **GPU-04**: 多 GPU 状态网格显示 (64+ GPU 概览表格)

### 验证 (VAL)

- [ ] **VAL-01**: 显示验证阶段开始信息
- [ ] **VAL-02**: 显示验证进度条
- [ ] **VAL-03**: 显示验证集 loss 和 perplexity

### Checkpoint (CKPT)

- [ ] **CKPT-01**: 定期显示 checkpoint 保存信息
- [ ] **CKPT-02**: 显示保存路径和文件大小

### 导出 (EXPORT)

- [ ] **EXPORT-01**: 显示模型导出过程 (格式转换)
- [ ] **EXPORT-02**: 显示训练完成摘要 (总时间、最终 loss、保存位置)

### 数据/显示 (DATA)

- [ ] **DATA-01**: 模型名称随机混合真实名和恶搞名
- [ ] **DATA-02**: 支持多种 GPU 型号名称 (A100, H100 等)
- [ ] **DATA-03**: 数据集名称列表 (真实 + 虚构)

### 技术兼容 (TECH)

- [ ] **TECH-01**: 使用 io::* 函数实现 WASM 兼容
- [ ] **TECH-02**: 实现 Module trait (name, signature, run)
- [ ] **TECH-03**: 检查 appconfig.should_exit() 支持优雅退出
- [ ] **TECH-04**: 通过 cargo clippy -- -D warnings

## v2 Requirements

### 增强显示

- **ENH-01**: 学习率调度器可视化 (warmup, decay 曲线)
- **ENH-02**: 梯度统计 (grad norm)
- **ENH-03**: 混合精度训练指示 (FP16/BF16)
- **ENH-04**: DeepSpeed/FSDP 分片信息

### 高级功能

- **ADV-01**: 多任务训练模拟
- **ADV-02**: 训练中断恢复模拟
- **ADV-03**: 分布式故障恢复模拟

## Out of Scope

| Feature | Reason |
|---------|--------|
| 真实训练功能 | genact 只做模拟，不执行实际计算 |
| 网络请求/API | 保持离线运行 |
| 配置文件解析 | 使用随机生成的配置即可 |
| 交互式控制 | genact 模块是自动运行的 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INIT-01 | Phase 1 | Pending |
| INIT-02 | Phase 1 | Pending |
| INIT-03 | Phase 1 | Pending |
| INIT-04 | Phase 1 | Pending |
| INIT-05 | Phase 1 | Pending |
| TRAIN-01 | Phase 2 | Pending |
| TRAIN-02 | Phase 2 | Pending |
| TRAIN-03 | Phase 2 | Pending |
| TRAIN-04 | Phase 2 | Pending |
| TRAIN-05 | Phase 2 | Pending |
| TRAIN-06 | Phase 2 | Pending |
| TRAIN-07 | Phase 2 | Pending |
| GPU-01 | Phase 2 | Pending |
| GPU-02 | Phase 2 | Pending |
| GPU-03 | Phase 2 | Pending |
| GPU-04 | Phase 2 | Pending |
| VAL-01 | Phase 3 | Pending |
| VAL-02 | Phase 3 | Pending |
| VAL-03 | Phase 3 | Pending |
| CKPT-01 | Phase 3 | Pending |
| CKPT-02 | Phase 3 | Pending |
| EXPORT-01 | Phase 4 | Pending |
| EXPORT-02 | Phase 4 | Pending |
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| TECH-01 | Phase 1 | Pending |
| TECH-02 | Phase 1 | Pending |
| TECH-03 | Phase 1 | Pending |
| TECH-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-30*
*Last updated: 2026-01-30 after initial definition*
