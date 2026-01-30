# Project Research Summary

**Project:** genact LLM Training Module
**Domain:** Terminal simulation / Fake activity generator
**Researched:** 2026-01-30
**Confidence:** HIGH

## Executive Summary

This project involves adding a realistic LLM training simulation module to genact, a tool that generates fake terminal activity. The module should simulate multi-GPU distributed training with convincing output including initialization, training loops, validation, checkpointing, and export phases. The research reveals that this is achievable entirely with genact's existing technology stack - no new dependencies are required. All necessary components (async runtime, colors, progress bars, randomization) are already in place.

The recommended approach is a phase-based state machine that follows genact's established Module trait pattern. The simulation should display realistic training metrics (loss curves with proper decay, GPU utilization, distributed coordination messages) that would convince both casual observers and ML practitioners. The key technical challenge is maintaining realism in metric generation - loss curves must follow realistic exponential decay patterns, perplexity must match loss mathematically (PPL = e^loss), and multi-GPU outputs must show proper coordination rather than independent behavior.

The main risks are WASM compatibility issues (using println!() instead of io::print()) and unrealistic metric patterns that expose the simulation as fake. Both are preventable through adherence to existing genact patterns and proper mathematical modeling of training dynamics. The module should complete in 2-4 minutes with natural pacing, include genact's signature humor (funny model names like "ButtGPT-420B"), and allow graceful interruption via should_exit() checks.

## Key Findings

### Recommended Stack

The existing genact stack is entirely sufficient for this module. No new dependencies are required. The module will use:

**Core technologies:**
- **Rust 2024 Edition**: Language - already in use, provides async/await and strong type safety
- **async-std 1.x**: Async runtime - existing dependency, fully WASM compatible
- **yansi 1.x**: Terminal colors - already used by all modules, simple API for colored output
- **progress_string 0.2.x**: Progress bars - proven pattern in julia.rs and download.rs modules
- **rand/rand_distr**: Randomization - already available for realistic metric noise generation
- **chrono 0.4.x**: Timestamps - existing dependency with WASM support via wasmbind feature

All I/O goes through genact's abstraction layer (io::print(), io::csleep()) which handles native/WASM differences transparently. No conditional compilation needed in the module itself.

**Data files needed:**
- `data/llm_models.txt`: ~80 lines with model prefixes, size suffixes, and funny names
- `data/llm_datasets.txt`: ~40 lines with training dataset names
- `data/llm_gpus.txt`: ~25 lines with GPU specs (name, memory, TDP)
- `data/llm_frameworks.txt`: ~20 lines with framework version strings

### Expected Features

The research identified clear categories of features based on real-world LLM training output patterns.

**Must have (table stakes):**
- Distributed environment initialization (NCCL backend, GPU detection, master node)
- Model loading with parameter count and sharding info
- Training loop with epoch progress bars and step counters
- Training loss that decreases realistically over time
- Speed metrics (tokens/second, samples/second)
- Learning rate display with proper scheduling
- GPU memory usage statistics
- Validation phase with val loss and perplexity
- Checkpoint saving messages
- Export phase with final summary

**Should have (differentiators):**
- Color-coded output matching PyTorch/HuggingFace conventions
- Multi-GPU status grid showing all 8-64 GPUs with utilization and temperature
- NCCL communication timing (AllReduce, Broadcast)
- Gradient norm display
- Funny model names mixed with realistic ones (genact signature humor)

**Defer (v2+ or optional):**
- BLEU/ROUGE evaluation metrics (adds complexity, diminishing returns)
- Gradient accumulation details (technical but not visually interesting)
- Advanced error/warning simulation (complexity vs value)
- Loss spike handling (realistic but complex)

### Architecture Approach

The module follows genact's established Module trait pattern with sequential phase execution rather than an explicit state machine. This matches patterns seen in julia.rs, cargo.rs, and other multi-phase modules.

**Major components:**
1. **Phase Functions** - Separate async functions for Init, Training, Validation, Checkpoint, Export phases that execute sequentially
2. **Data Generators** - Functions that generate random model names, realistic metrics with proper mathematical relationships
3. **Output Formatters** - Helper functions for colored terminal output using yansi and progress bars using progress_string
4. **Static Data Files** - Model names, datasets, GPUs loaded via include_str!() following data.rs patterns

The training loop is the most complex component, requiring:
- Epoch/step iteration with proper progress tracking
- Loss decay following exponential curve: `loss = initial * e^(-k*step) + noise + floor`
- Learning rate schedule with warmup and cosine decay
- GPU metrics with per-device variance (not identical)
- Graceful exit checks at natural breakpoints

Data flows from run() → phase functions → formatters → io::print(). The should_exit() check must occur after every phase and within long loops to allow graceful interruption.

### Critical Pitfalls

Based on codebase analysis and domain knowledge, these are the most dangerous mistakes:

1. **Using println!() instead of io::* functions** - Module compiles for native but produces no output in WASM. The io abstraction is mandatory for cross-platform compatibility. Prevention: Add module-level comment, use grep to detect violations, test WASM explicitly.

2. **Missing should_exit() checks in loops** - Module cannot be interrupted, ignores Ctrl+C and --exit-after-time flags. Prevention: Check at start of every loop iteration and after each csleep() call.

3. **Unrealistic loss curve patterns** - Loss decreasing linearly, going negative, or too smoothly exposes the simulation as fake. Prevention: Use exponential decay with noise, ensure loss stays above ~1.0, verify PPL = e^loss mathematically, add occasional variance spikes.

4. **Inconsistent multi-GPU output** - All GPUs showing identical metrics, ranks appearing out of order, or NCCL logs not aligning with gradient sync patterns. Prevention: Model GPUs as coordinated set with per-device variance, ensure NCCL communication appears at proper times (after backward pass).

5. **WASM timing inconsistencies** - Browser setTimeout has 4ms minimum resolution, tab inactivity throttles timers. Prevention: Don't rely on precise timing for displayed metrics, test specifically in WASM environment with trunk serve.

## Implications for Roadmap

Based on research, this module should be implemented in a single focused phase with clear sub-tasks.

### Phase 1: LLM Training Module Foundation (MVP)
**Rationale:** Implement core functionality first to establish patterns and verify WASM compatibility early. This delivers a working simulation that can be refined iteratively.

**Delivers:**
- Working module registered in genact
- All five simulation phases (Init, Training, Validation, Checkpoint, Export)
- Realistic metric generation with proper mathematical models
- WASM compatibility verified

**Addresses (table stakes features):**
- Distributed environment initialization output
- Model loading and dataset info display
- Training loop with progress bar, loss decay, and speed metrics
- Validation phase with proper metrics
- Checkpoint and export messages
- Graceful exit handling

**Avoids (critical pitfalls):**
- Uses io::* functions exclusively (no println!)
- Includes should_exit() checks at all breakpoints
- Implements realistic loss decay formulas from design phase
- Tests WASM explicitly before completion

**Sub-tasks:**
1. Create data files (llm_models.txt, llm_datasets.txt, llm_gpus.txt, llm_frameworks.txt)
2. Add data file loading to src/data.rs
3. Create src/modules/llm_training.rs with Module trait implementation
4. Implement initialization phase output
5. Implement training loop with metric generation
6. Implement validation, checkpoint, and export phases
7. Add colored output formatting
8. Register module in src/modules/mod.rs
9. Test native and WASM builds
10. Add integration test

### Phase 2: Polish and Differentiators (Post-MVP)
**Rationale:** Once core functionality works, add visual polish and humor elements that enhance realism and entertainment value.

**Delivers:**
- Multi-GPU status grid visualization
- NCCL communication timing display
- Gradient norm metrics
- Funny model names (ButtGPT-420B, ChadLLM-Instruct, etc.)

**Uses:**
- yansi for advanced color schemes
- Terminal width detection for responsive layout

**Implementation notes:**
- Non-critical features, can be added incrementally
- Each can be feature-flagged if needed
- Visual testing required for layout on various terminal widths

### Phase Ordering Rationale

Single-phase approach is recommended because:
- Module is self-contained, no external dependencies or integrations
- All patterns exist in codebase (Module trait, phase sequencing, progress bars)
- Splitting into sub-phases adds overhead without benefit
- Early WASM testing prevents late-stage compatibility issues
- MVP includes all table stakes features for realistic simulation

Breaking into Init/Training/Export phases would create artificial boundaries since they must execute sequentially and share state (model config, loss trajectory).

### Research Flags

Phases with standard patterns (skip research-phase):
- **LLM Training Module**: Well-documented domain with clear patterns from existing modules (julia.rs for multi-phase, download.rs for progress bars, cargo.rs for phased output). The mathematical formulas for loss decay are straightforward exponential curves. No API integrations or external systems.

This module does NOT need `/gsd:research-phase` during implementation because:
1. Architecture patterns are proven in codebase
2. Technology stack already exists
3. Output formats well-documented in research
4. No external APIs or integrations
5. Mathematical models for metrics are defined

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All dependencies already in Cargo.toml, verified WASM compatible |
| Features | HIGH | Based on actual PyTorch/HuggingFace training output analysis |
| Architecture | HIGH | Follows existing genact Module trait patterns exactly |
| Pitfalls | HIGH | Based on codebase analysis and ML training domain knowledge |

**Overall confidence:** HIGH

All research is based on direct codebase inspection and established patterns. The domain (terminal output simulation) is straightforward, and real LLM training output is well-documented. No speculative or unverified assumptions.

### Gaps to Address

No critical gaps identified. Minor considerations:

- **Terminal width handling**: Implementation should test various widths (80-200 columns) to ensure GPU status grid adapts gracefully. Existing get_terminal_width() function provides this.

- **Timing calibration**: WASM timing may need adjustment after testing. The csleep() durations in the design are estimates; actual pacing should feel natural in both native and WASM.

- **Data file content curation**: The exact list of model names, datasets, and GPUs can be refined during implementation based on what looks most realistic/funny.

These are implementation details, not research gaps. The core approach is sound.

## Sources

### Primary (HIGH confidence)
- genact codebase: `F:\genact-ex\src\modules\*.rs` - Module patterns, phase sequencing, progress bars
- genact codebase: `F:\genact-ex\src\data.rs` - Data file loading patterns
- genact codebase: `F:\genact-ex\src\io.rs` - I/O abstraction for native/WASM
- genact codebase: `F:\genact-ex\Cargo.toml` - Dependency versions and features
- HuggingFace Trainer docs: https://huggingface.co/docs/transformers/en/trainer - Output format patterns
- PyTorch Distributed docs: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html - NCCL initialization patterns
- NVIDIA NCCL docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html - Communication output format
- NVIDIA GPU specs: https://www.nvidia.com/en-us/data-center/ - A100, H100, H200 specifications

### Secondary (MEDIUM confidence)
- DeepSpeed documentation: https://deepspeed.readthedocs.io/en/latest/ - ZeRO stage output patterns
- Modal LLM training guide: https://modal.com/blog/fine-tuning-llms-hyperparameters-glossary-article - Typical hyperparameter ranges
- Sebastian Raschka LLM training: https://sebastianraschka.com/llms-from-scratch/ch05/10_llm-training-speed/ - Throughput metrics

### Tertiary (LOW confidence)
- General ML training loss curve patterns - verified against standard training logs
- GPU cluster power consumption estimates - approximate based on TDP specs

---
*Research completed: 2026-01-30*
*Ready for roadmap: yes*
