# Domain Pitfalls: LLM Training Simulation Module

**Domain:** Terminal Simulation for LLM Training Output
**Researched:** 2026-01-30
**Confidence:** HIGH (based on codebase analysis + domain knowledge)

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Using `println!()` Instead of `io::*` Functions

**What goes wrong:** Module compiles for native but fails WASM build. The WASM target requires all output to go through `io::print()`, `io::dprint()`, `io::newline()` which route to `write_to_xterm()` in browser environments.

**Why it happens:** Developers familiar with standard Rust default to `println!()`. The compiler doesn't warn about this since both targets compile; only runtime behavior differs.

**Consequences:**
- WASM build produces no visible output
- Native and WASM behavior diverge
- CI passes but production WASM is broken

**Prevention:**
- Never use `println!()`, `print!()`, or `eprintln!()` in module code
- Use grep/clippy custom lint to detect these patterns before commit
- Add module-level comment: `// Note: Use io::* functions, not println!()`

**Detection (Warning Signs):**
- WASM demo page shows no output
- `grep -r "println!" src/modules/` returns matches
- Module works in native but not web demo

**Phase to Address:** Design phase - establish I/O patterns before coding

---

### Pitfall 2: Missing `appconfig.should_exit()` Checks in Loops

**What goes wrong:** Module runs indefinitely and cannot be interrupted. Users cannot Ctrl+C to exit gracefully, and `--exit-after-time` / `--exit-after-modules` flags are ignored.

**Why it happens:** Focus on output realism causes oversight of control flow. Inner loops (epochs, steps, GPU status updates) miss exit checks.

**Consequences:**
- Process cannot be stopped gracefully
- Resource leak if running in background
- Poor UX - user must kill process
- Exit conditions never trigger

**Prevention:**
- Check `appconfig.should_exit()` at start of every loop iteration
- Add check after every `csleep()` call in long-running sections
- Template pattern: `if appconfig.should_exit() { return; }`

**Detection (Warning Signs):**
- Module ignores Ctrl+C during execution
- `--exit-after-modules 1` doesn't stop after module
- Long delays between exit signal and actual termination

**Phase to Address:** Implementation phase - add checks during loop creation

---

### Pitfall 3: Unrealistic Loss Curve Patterns

**What goes wrong:** Training simulation loss values follow mathematically impossible patterns that experts immediately recognize as fake:
- Loss decreasing linearly instead of exponentially
- Loss going negative
- Perplexity values that don't match loss (PPL = e^loss)
- Loss decreasing too smoothly (no noise/variance)
- Loss decreasing during validation higher than training (impossible without bugs)

**Why it happens:** Developer unfamiliar with ML training dynamics generates random numbers without understanding realistic trajectories.

**Consequences:**
- Simulation immediately recognized as fake by anyone familiar with ML
- Defeats purpose of convincing "busy work" display
- Undermines credibility of tool

**Prevention:**
- Start with realistic initial loss (6-10 for cross-entropy on large vocab)
- Use exponential decay with noise: `loss = initial * e^(-k*step) + noise + floor`
- Ensure loss never goes below ~1.0 for language models
- Add occasional spikes (gradient instability simulation)
- Perplexity = e^loss (must be consistent)
- Validation loss tracks training loss with slight lag

**Detection (Warning Signs):**
- Loss values below 0.5 for language models
- Perfectly smooth loss curve (no variance)
- Loss decreasing linearly
- PPL and loss mathematically inconsistent

**Phase to Address:** Design phase - define mathematical models for metrics before implementation

---

### Pitfall 4: Inconsistent Multi-GPU Output Format

**What goes wrong:** Simulated 64-GPU output has timing, rank, or coordination inconsistencies:
- Rank numbers appear out of order illogically
- All GPUs report identical metrics (real training has variance)
- Timestamps don't align with claimed communication
- NCCL AllReduce logs don't match gradient sync patterns

**Why it happens:** Developer generates GPU status independently per GPU without modeling inter-GPU coordination.

**Consequences:**
- Anyone familiar with distributed training spots the fake instantly
- Output looks like template repetition, not real multi-node training

**Prevention:**
- Model GPUs as coordinated set with shared global state
- Introduce per-GPU variance (temperature, memory, utilization)
- NCCL communication should appear between forward/backward passes
- Rank 0 outputs summary; other ranks contribute partial logs
- Global batch timing must be consistent across GPU reports

**Detection (Warning Signs):**
- All 64 GPUs show identical memory usage (should vary by 1-5%)
- NCCL logs appear during forward pass (should be gradient sync only)
- Each GPU reports different loss value (should be identical post-sync)
- Rank ordering appears random instead of sequential

**Phase to Address:** Design phase - document distributed training coordination model

---

### Pitfall 5: WASM Timing Inconsistencies

**What goes wrong:** Timing behaves differently between native and WASM:
- `csleep()` in WASM uses `setTimeout` which has minimum 4ms resolution
- WASM doesn't have access to high-resolution timestamps
- Browser throttles timers when tab is inactive

**Why it happens:** Native implementation uses precise `async_std::task::sleep()` while WASM uses JS Promises with browser limitations.

**Consequences:**
- Training speed metrics (tokens/s, samples/s) show unrealistic values in WASM
- Progress bars move erratically
- ETA calculations are wrong

**Prevention:**
- Don't rely on precise timing for displayed metrics
- Use elapsed wall time for displayed values, not accumulated sleep times
- Accept that WASM timing is approximate
- Test module specifically in WASM environment (trunk serve)

**Detection (Warning Signs):**
- Tokens/second shows 0 or infinity in WASM
- ETA jumps erratically
- Progress bar updates in bursts when tab regains focus

**Phase to Address:** Testing phase - verify WASM behavior explicitly

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt.

### Pitfall 6: Hardcoded Magic Numbers for Training Parameters

**What goes wrong:** Hardcoded values like `batch_size = 4096` or `learning_rate = 0.0001` are scattered throughout code, making the simulation inflexible and unrealistic for different "model sizes."

**Why it happens:** Quick implementation without planning parameter relationships.

**Prevention:**
- Define parameter structs for training configuration
- Generate correlated parameters (larger model = larger batch, more GPUs)
- Use constants with descriptive names: `const BASE_BATCH_SIZE: u32 = 4096;`

**Detection (Warning Signs):**
- Same numbers appear in multiple places
- "7B model" shows same batch size as "70B model"
- Parameter values don't scale with model size

**Phase to Address:** Design phase - define parameter generation strategy

---

### Pitfall 7: Terminal Width Not Respected

**What goes wrong:** Progress bars and multi-column GPU status tables overflow terminal width, causing ugly wrapping or truncation.

**Why it happens:** Module designed for one terminal width; doesn't adapt. The `get_terminal_width()` function is available but not used consistently.

**Prevention:**
- Use `io::get_terminal_width()` to adapt output layout
- Design flexible output formats that degrade gracefully
- Test with narrow terminals (80 columns minimum)

**Detection (Warning Signs):**
- Output wraps mid-line
- Progress bar extends beyond screen
- WASM xterm shows horizontal scrollbar

**Phase to Address:** Implementation phase - build responsive layouts

---

### Pitfall 8: Forgetting `newline()` After `print()`

**What goes wrong:** Output appears on same line or overwrites previous output unexpectedly. Unlike `println!()`, `io::print()` does not append newline.

**Why it happens:** Habit from `println!()` usage in other Rust code.

**Prevention:**
- Always pair `print()` with `newline()` unless doing in-place updates
- Use `erase_line()` before in-place updates
- Review output visually during development

**Detection (Warning Signs):**
- Lines overlap or concatenate incorrectly
- Progress updates appear on new lines instead of in-place

**Phase to Address:** Implementation phase - visual testing

---

### Pitfall 9: Data File Too Large or Too Small

**What goes wrong:**
- Too large: Compile times increase, binary bloats
- Too small: Same model names/GPU names repeat noticeably

**Why it happens:** No guidelines established for data file sizing.

**Prevention:**
- Aim for 50-200 items per list (enough variety without bloat)
- Model names: ~30 realistic + ~20 humorous
- GPU names: Major NVIDIA models (A100, H100, H200, etc.)
- Sample packages: ~100 PyTorch/HuggingFace library names

**Detection (Warning Signs):**
- Same model name appears within 30 seconds of watching
- Binary size increases significantly after data addition
- Compile time noticeably slower

**Phase to Address:** Design phase - spec data file contents

---

### Pitfall 10: Random Number Generator Not Seeded Per Session

**What goes wrong:** Multiple module instances or repeated runs show identical patterns.

**Why it happens:** Using `rng()` creates ThreadRng which is seeded from system entropy, but patterns in random selection can still appear if selection pools are small.

**Consequences:**
- Same "training run" plays out identically
- Users notice repetition quickly

**Prevention:**
- Ensure sufficient variety in data pools
- Use different parameter ranges each run
- Current pattern `let mut rng = rng();` is correct - don't cache RNG across runs

**Detection (Warning Signs):**
- Same model name appears first on multiple runs
- GPU utilization patterns repeat exactly

**Phase to Address:** Testing phase - run multiple times to verify variety

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 11: Inconsistent Color Usage

**What goes wrong:** Colors don't match HuggingFace/PyTorch conventions or are hard to read on dark/light terminals.

**Prevention:**
- Green for success/progress
- Yellow/Orange for warnings
- Red for errors
- Blue/Cyan for informational
- Use `yansi::Paint` consistently
- Test on both dark and light terminal backgrounds

**Phase to Address:** Implementation phase

---

### Pitfall 12: Missing Carriage Return in In-Place Updates

**What goes wrong:** In-place updates (like progress bars) don't return to line start, causing output to shift right.

**Prevention:**
- Use `\r` at start of in-place updates
- Use `erase_line()` before updating if previous content was longer
- Pattern: `erase_line().await; print(new_content).await;`

**Phase to Address:** Implementation phase

---

### Pitfall 13: Clippy Warnings from Unused Imports or Variables

**What goes wrong:** CI fails with `-D warnings` flag even though code is functionally correct.

**Prevention:**
- Run `cargo clippy -- -D warnings` locally before commit
- Use `#[allow(clippy::...)]` sparingly and with justification
- Remove unused imports immediately

**Phase to Address:** Implementation phase - continuous

---

### Pitfall 14: Unicode Characters Not Rendering in WASM Terminal

**What goes wrong:** Special characters (progress bar fills, arrows, etc.) render as boxes or question marks in xterm.js.

**Prevention:**
- Use ASCII-safe progress bars: `=`, `-`, `#`, ` `
- Test Unicode rendering in WASM before committing to fancy characters
- Have ASCII fallback for all visual elements

**Phase to Address:** Testing phase

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Requirements | Underspecifying metric realism | Define exact formulas for loss/PPL curves |
| Design | Ignoring distributed coordination model | Document GPU state synchronization approach |
| Implementation | Using println!() | Code review specifically for I/O functions |
| Implementation | Missing should_exit() | Checklist item for every loop |
| Testing | Only testing native | Run `trunk serve` and verify WASM |
| Testing | Not testing narrow terminals | Test at 80 columns |
| Integration | Breaking existing modules | Run all modules after changes |

## LLM Training Domain-Specific Warnings

| Training Concept | Common Mistake | Realistic Behavior |
|-----------------|----------------|-------------------|
| Loss values | Starting at 0.1 or going negative | Initial: 8-12, floor: 1.5-2.5 |
| Perplexity | Random number unrelated to loss | Always PPL = e^loss |
| Learning rate | Static value | May show warmup/decay schedule |
| Gradient norms | Not shown | Real training often logs grad norms |
| NCCL AllReduce | Appears randomly | Appears after backward pass, before optimizer step |
| GPU memory | All GPUs identical | Per-GPU variance 1-5% |
| Tokens/second | Constant value | Slight variance, dips during eval |
| Checkpoint saving | Instant | Should show file size, path, time taken |
| Validation | Same frequency as training steps | Every N steps (e.g., 1000, 5000) |

## Sources

- genact codebase analysis: `src/io.rs`, `src/modules/mod.rs`, `src/modules/*.rs`
- Project conventions: `.planning/codebase/CONVENTIONS.md`
- Known concerns: `.planning/codebase/CONCERNS.md`
- HuggingFace Trainer documentation: https://huggingface.co/docs/transformers/main_classes/trainer
- PyTorch distributed training: https://docs.pytorch.org/docs/stable/distributed.html
- NCCL communication patterns: https://developer.nvidia.com/blog/enhancing-communication-observability-of-ai-workloads-with-nccl-inspector/
- Training loss patterns: https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves
- Sebastian Raschka LLM training: https://sebastianraschka.com/llms-from-scratch/ch05/10_llm-training-speed/

---

*Pitfalls analysis: 2026-01-30*
