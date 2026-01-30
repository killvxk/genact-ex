# Requirements: genact LLM Training Module

**Defined:** 2026-01-30
**Core Value:** Realistically simulate large-scale LLM training process, making observers believe actual model training is occurring

## v1 Requirements

### Initialization (INIT)

- [ ] **INIT-01**: Display model loading info (model name, parameter count, architecture)
- [ ] **INIT-02**: Display Tokenizer loading progress
- [ ] **INIT-03**: Display dataset loading info (dataset name, sample count)
- [ ] **INIT-04**: Display distributed environment initialization (NCCL init, GPU count, node count)
- [ ] **INIT-05**: Display GPU status check (64+ GPU detection confirmation)

### Training (TRAIN)

- [ ] **TRAIN-01**: Display Epoch progress bar (current/total, percentage progress bar)
- [ ] **TRAIN-02**: Display Step-level metrics (step number, loss, learning rate)
- [ ] **TRAIN-03**: Display perplexity metric (PPL = e^loss)
- [ ] **TRAIN-04**: Display speed statistics (tokens/s, samples/s)
- [ ] **TRAIN-05**: Display time estimates (elapsed, ETA)
- [ ] **TRAIN-06**: Display NCCL communication logs (AllReduce completion time)
- [ ] **TRAIN-07**: Loss value decreases gradually during training (exponential decay + noise)

### GPU Status (GPU)

- [ ] **GPU-01**: Display each GPU's memory usage (used/total, percentage)
- [ ] **GPU-02**: Display each GPU's utilization (percentage)
- [ ] **GPU-03**: Display each GPU's temperature (Celsius)
- [ ] **GPU-04**: Multi-GPU status grid display (64+ GPU overview table)

### Validation (VAL)

- [ ] **VAL-01**: Display validation phase start message
- [ ] **VAL-02**: Display validation progress bar
- [ ] **VAL-03**: Display validation set loss and perplexity

### Checkpoint (CKPT)

- [ ] **CKPT-01**: Periodically display checkpoint save messages
- [ ] **CKPT-02**: Display save path and file size

### Export (EXPORT)

- [ ] **EXPORT-01**: Display model export process (format conversion)
- [ ] **EXPORT-02**: Display training completion summary (total time, final loss, save location)

### Data/Display (DATA)

- [ ] **DATA-01**: Model names randomly mix real names and funny names
- [ ] **DATA-02**: Support multiple GPU model names (A100, H100, etc.)
- [ ] **DATA-03**: Dataset name list (real + fictional)

### Technical Compatibility (TECH)

- [ ] **TECH-01**: Use io::* functions for WASM compatibility
- [ ] **TECH-02**: Implement Module trait (name, signature, run)
- [ ] **TECH-03**: Check appconfig.should_exit() for graceful exit
- [ ] **TECH-04**: Pass cargo clippy -- -D warnings

## v2 Requirements

### Enhanced Display

- **ENH-01**: Learning rate scheduler visualization (warmup, decay curve)
- **ENH-02**: Gradient statistics (grad norm)
- **ENH-03**: Mixed precision training indicator (FP16/BF16)
- **ENH-04**: DeepSpeed/FSDP sharding info

### Advanced Features

- **ADV-01**: Multi-task training simulation
- **ADV-02**: Training resume simulation
- **ADV-03**: Distributed failure recovery simulation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real training functionality | genact only simulates, no actual computation |
| Network requests/API | Maintain offline operation |
| Config file parsing | Use randomly generated config |
| Interactive control | genact modules run automatically |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INIT-01 | Phase 2 | Pending |
| INIT-02 | Phase 2 | Pending |
| INIT-03 | Phase 2 | Pending |
| INIT-04 | Phase 2 | Pending |
| INIT-05 | Phase 2 | Pending |
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
- Unmapped: 0

---
*Requirements defined: 2026-01-30*
*Last updated: 2026-01-30 after roadmap creation*
