# Requirements: genact LLM Training Module

**Defined:** 2026-01-30
**Core Value:** Realistically simulate large-scale LLM training process, making observers believe actual model training is occurring

## v1 Requirements

### Initialization (INIT)

- [x] **INIT-01**: Display model loading info (model name, parameter count, architecture)
- [x] **INIT-02**: Display Tokenizer loading progress
- [x] **INIT-03**: Display dataset loading info (dataset name, sample count)
- [x] **INIT-04**: Display distributed environment initialization (NCCL init, GPU count, node count)
- [x] **INIT-05**: Display GPU status check (64+ GPU detection confirmation)

### Training (TRAIN)

- [x] **TRAIN-01**: Display Epoch progress bar (current/total, percentage progress bar)
- [x] **TRAIN-02**: Display Step-level metrics (step number, loss, learning rate)
- [x] **TRAIN-03**: Display perplexity metric (PPL = e^loss)
- [x] **TRAIN-04**: Display speed statistics (tokens/s, samples/s)
- [x] **TRAIN-05**: Display time estimates (elapsed, ETA)
- [x] **TRAIN-06**: Display NCCL communication logs (AllReduce completion time)
- [x] **TRAIN-07**: Loss value decreases gradually during training (exponential decay + noise)

### GPU Status (GPU)

- [x] **GPU-01**: Display each GPU's memory usage (used/total, percentage)
- [x] **GPU-02**: Display each GPU's utilization (percentage)
- [x] **GPU-03**: Display each GPU's temperature (Celsius)
- [x] **GPU-04**: Multi-GPU status grid display (64+ GPU overview table)

### Validation (VAL)

- [x] **VAL-01**: Display validation phase start message
- [x] **VAL-02**: Display validation progress bar
- [x] **VAL-03**: Display validation set loss and perplexity

### Checkpoint (CKPT)

- [x] **CKPT-01**: Periodically display checkpoint save messages
- [x] **CKPT-02**: Display save path and file size

### Export (EXPORT)

- [x] **EXPORT-01**: Display model export process (format conversion)
- [x] **EXPORT-02**: Display training completion summary (total time, final loss, save location)

### Data/Display (DATA)

- [x] **DATA-01**: Model names randomly mix real names and funny names
- [x] **DATA-02**: Support multiple GPU model names (A100, H100, etc.)
- [x] **DATA-03**: Dataset name list (real + fictional)

### Technical Compatibility (TECH)

- [x] **TECH-01**: Use io::* functions for WASM compatibility
- [x] **TECH-02**: Implement Module trait (name, signature, run)
- [x] **TECH-03**: Check appconfig.should_exit() for graceful exit
- [x] **TECH-04**: Pass cargo clippy -- -D warnings

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
| INIT-01 | Phase 2 | Complete |
| INIT-02 | Phase 2 | Complete |
| INIT-03 | Phase 2 | Complete |
| INIT-04 | Phase 2 | Complete |
| INIT-05 | Phase 2 | Complete |
| TRAIN-01 | Phase 2 | Complete |
| TRAIN-02 | Phase 2 | Complete |
| TRAIN-03 | Phase 2 | Complete |
| TRAIN-04 | Phase 2 | Complete |
| TRAIN-05 | Phase 2 | Complete |
| TRAIN-06 | Phase 2 | Complete |
| TRAIN-07 | Phase 2 | Complete |
| GPU-01 | Phase 2 | Complete |
| GPU-02 | Phase 2 | Complete |
| GPU-03 | Phase 2 | Complete |
| GPU-04 | Phase 2 | Complete |
| VAL-01 | Phase 3 | Complete |
| VAL-02 | Phase 3 | Complete |
| VAL-03 | Phase 3 | Complete |
| CKPT-01 | Phase 3 | Complete |
| CKPT-02 | Phase 3 | Complete |
| EXPORT-01 | Phase 4 | Complete |
| EXPORT-02 | Phase 4 | Complete |
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| TECH-01 | Phase 1 | Complete |
| TECH-02 | Phase 1 | Complete |
| TECH-03 | Phase 1 | Complete |
| TECH-04 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Unmapped: 0

---
*Requirements defined: 2026-01-30*
*Last updated: 2026-01-31 after Phase 4 completion (all v1 requirements complete)*
