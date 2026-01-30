# Roadmap: genact LLM Training Module

## Overview

This roadmap delivers a realistic LLM training simulation module for genact. Starting with data files and module skeleton (Phase 1), we build the core training loop with progress bars and metrics (Phase 2), add validation and checkpoint functionality (Phase 3), and finish with export phase and final polish (Phase 4). Each phase is independently testable with `cargo run -- -m llm_training`.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Foundation** - Data files, module skeleton, basic structure
- [x] **Phase 2: Training Loop** - Core training with progress bars, metrics, GPU status
- [ ] **Phase 3: Validation & Checkpoints** - Mid-training validation and checkpoint saving
- [ ] **Phase 4: Export & Polish** - Final output, summary, clippy compliance

## Phase Details

### Phase 1: Foundation
**Goal**: Establish module structure with data files and basic Module trait implementation
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, TECH-01, TECH-02, TECH-03
**Success Criteria** (what must be TRUE):
  1. Module appears in `genact --list` output
  2. Running `cargo run -- -m llm_training` produces any output (even placeholder)
  3. Data files exist and are loaded via include_str!()
  4. Module compiles for both native and wasm32 targets
  5. Module checks should_exit() for graceful termination
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md - Create data files and module skeleton

### Phase 2: Training Loop
**Goal**: Users see convincing training output with initialization, epoch progress, loss metrics, and GPU status
**Depends on**: Phase 1
**Requirements**: INIT-01, INIT-02, INIT-03, INIT-04, INIT-05, TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, GPU-01, GPU-02, GPU-03, GPU-04
**Success Criteria** (what must be TRUE):
  1. User sees model loading with name, parameter count, and architecture
  2. User sees distributed environment initialization (NCCL, GPU count, nodes)
  3. User sees epoch progress bar advancing with percentage
  4. User sees training loss decreasing realistically over steps
  5. User sees GPU status grid showing memory, utilization, and temperature for 64+ GPUs
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md - Implement initialization phase (model, tokenizer, dataset, distributed)
- [x] 02-02-PLAN.md - Implement training loop with metrics and GPU status

### Phase 3: Validation & Checkpoints
**Goal**: Users see validation phases and checkpoint saving during training
**Depends on**: Phase 2
**Requirements**: VAL-01, VAL-02, VAL-03, CKPT-01, CKPT-02
**Success Criteria** (what must be TRUE):
  1. User sees validation phase start message after training epochs
  2. User sees validation progress bar
  3. User sees validation loss and perplexity metrics
  4. User sees checkpoint save messages with file path and size
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md - Implement validation and checkpoint phases

### Phase 4: Export & Polish
**Goal**: Training completes with export phase and passes all quality checks
**Depends on**: Phase 3
**Requirements**: EXPORT-01, EXPORT-02, TECH-04
**Success Criteria** (what must be TRUE):
  1. User sees model export process with format conversion
  2. User sees training completion summary (total time, final loss, save location)
  3. `cargo clippy -- -D warnings` passes with no errors
  4. Module works correctly in WASM environment
**Plans**: TBD

Plans:
- [ ] 04-01: Implement export phase and final polish

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 1/1 | Complete | 2026-01-30 |
| 2. Training Loop | 2/2 | Complete | 2026-01-30 |
| 3. Validation & Checkpoints | 0/1 | Ready | - |
| 4. Export & Polish | 0/1 | Not started | - |

---
*Roadmap created: 2026-01-30*
*Depth: standard (5-8 phases, using 4 due to cohesive work clustering)*
