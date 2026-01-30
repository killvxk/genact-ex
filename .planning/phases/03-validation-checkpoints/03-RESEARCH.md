# Phase 3: Validation & Checkpoints - Research

**Researched:** 2026-01-30
**Domain:** PyTorch/HuggingFace validation output patterns, checkpoint saving, early stopping, and warning messages for realistic LLM training simulation
**Confidence:** HIGH

## Summary

Phase 3 extends the existing llm_training module with validation phases and checkpoint saving. The research confirms that validation output follows predictable patterns in real training frameworks: a visual separator marks the transition, metrics are displayed using the same progress bar style as training, and a summary report compares validation vs training metrics. Checkpoint saving in modern LLM training uses safetensors format with clear log messages indicating path and file size. Warning messages (gradient issues, early stopping patience) follow established PyTorch/HuggingFace patterns.

The key insight is that validation is visually distinct but stylistically consistent with training. PyTorch Lightning uses `=== Validation ===` style separators, validation progress bars look identical to training, and metrics appear in the same format (loss, perplexity, accuracy). Checkpoint saving logs include the path and typically show progress for large files. Early stopping displays patience as a counter without immediately stopping (aligns with CONTEXT.md requirement for infinite loop).

**Primary recommendation:** Implement validation as a distinct phase after each epoch using the existing progress bar and logging functions. Add checkpoint saving with progress bar for realism. Integrate warning messages using the established `log_warning()` function with realistic message content from PyTorch training.

## Standard Stack

All necessary dependencies are already available from Phase 2 - no new dependencies needed.

### Core (from existing implementation)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| progress_string | 0.2 | Progress bar rendering | Already used in training loop |
| yansi | 1 | Terminal colors | Already used for INFO/WARNING |
| rand | 0.9 | Random values | Already used throughout |
| rand_distr | 0.5 | Statistical distributions | Already used for loss/GPU variation |
| chrono | 0.4 | Timestamps | Already used in log functions |
| instant | 0.1 | Elapsed time | Already used in training loop |

### Existing Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `log_info()` | llm_training.rs | Timestamp + INFO log |
| `log_warning()` | llm_training.rs | Timestamp + WARNING log |
| `generate_gpu_statuses()` | llm_training.rs | GPU status generation |
| `display_gpu_status_grid()` | llm_training.rs | GPU status display |

### No Additional Dependencies Needed

Phase 3 uses only existing crate dependencies and established module patterns.

## Architecture Patterns

### Recommended Module Structure Extension

```rust
// Add to src/modules/llm_training.rs

/// Run validation phase after each epoch
async fn run_validation(
    appconfig: &AppConfig,
    epoch: u32,
    train_loss: f64,
    best_val_loss: &mut f64,
    total_steps: u32,
) -> f64;

/// Display checkpoint saving with progress bar
async fn save_checkpoint(
    appconfig: &AppConfig,
    step: u32,
    val_loss: f64,
);

/// Generate validation-specific warning messages
fn get_validation_warning(rng: &mut impl Rng) -> Option<&'static str>;
```

### Pattern 1: Validation Phase Separator

**What:** Visual divider marking transition from training to validation
**When to use:** After each epoch's training steps complete, before validation starts
**Example:**

```rust
// Source: Derived from PyTorch Lightning output patterns
// https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html

async fn display_validation_separator() {
    newline().await;
    print(format!(
        "{}",
        Paint::cyan("=============== Validation ===============").bold()
    )).await;
    newline().await;
}
```

### Pattern 2: Validation Progress with Extended Metrics

**What:** Progress bar with validation-specific metrics (loss, perplexity, accuracy, tokens/sec)
**When to use:** During validation step iteration
**Example:**

```rust
// Source: Derived from PyTorch-Ignite tqdm output
// https://docs.pytorch.org/ignite/generated/ignite.handlers.tqdm_logger.html

async fn display_validation_progress(
    step: u32,
    total_steps: u32,
    val_loss: f64,
    val_ppl: f64,
    val_accuracy: f64,
    tokens_per_sec: f64,
    bar: &progress_string::Bar,
) {
    erase_line().await;
    print(format!(
        "Val:   {} | Loss: {:.4} | PPL: {:.2} | Acc: {:.2}% | {:.0}tok/s",
        bar,
        val_loss,
        val_ppl,
        val_accuracy * 100.0,
        tokens_per_sec
    )).await;
    newline().await;
}
```

### Pattern 3: Validation Summary Report

**What:** Multi-line report comparing validation to training metrics
**When to use:** After validation phase completes
**Example:**

```rust
// Source: Derived from PyTorch training loop patterns
// https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html

async fn display_validation_summary(
    train_loss: f64,
    val_loss: f64,
    val_accuracy: f64,
    val_ppl: f64,
    val_time_secs: f64,
) {
    newline().await;
    log_info("Validation Results:").await;
    log_info(&format!("  val_loss:     {:.4}", val_loss)).await;
    log_info(&format!("  val_ppl:      {:.2}", val_ppl)).await;
    log_info(&format!("  val_accuracy: {:.2}%", val_accuracy * 100.0)).await;
    log_info(&format!("  train_loss:   {:.4} (delta: {:+.4})", train_loss, val_loss - train_loss)).await;
    log_info(&format!("  time:         {:.1}s", val_time_secs)).await;
}
```

### Pattern 4: Checkpoint Save with Progress

**What:** Checkpoint saving with path, file size, and progress bar
**When to use:** When validation loss improves (best model strategy)
**Example:**

```rust
// Source: Derived from HuggingFace Trainer checkpoint logging
// https://huggingface.co/docs/transformers/en/main_classes/trainer

async fn save_checkpoint(
    appconfig: &AppConfig,
    step: u32,
    file_size_gb: f32,
) {
    let filename = format!("model-step-{}.safetensors", step);
    log_info(&format!("Saving model checkpoint to ./checkpoints/{}", filename)).await;

    // Progress bar for save operation
    let mut bar = BarBuilder::new()
        .total(100)
        .width(30)
        .full_char('=')
        .include_percent()
        .build();

    for i in 0..=100 {
        bar.replace(i);
        erase_line().await;
        print(format!(
            "  Writing {:.1}GB... {}",
            file_size_gb,
            bar
        )).await;

        csleep(rng.random_range(30..80)).await;

        if appconfig.should_exit() {
            newline().await;
            return;
        }
    }
    newline().await;

    log_info(&format!("Checkpoint saved: {} ({:.1}GB)", filename, file_size_gb)).await;
}
```

### Pattern 5: Early Stopping Patience Display

**What:** Patience counter showing progress toward early stopping (never triggers)
**When to use:** When validation loss does not improve
**Example:**

```rust
// Source: Derived from PyTorch Lightning EarlyStopping
// https://lightning.ai/docs/pytorch/stable/common/early_stopping.html

async fn log_early_stopping_patience(current_patience: u32, max_patience: u32) {
    log_warning(&format!(
        "EarlyStopping: val_loss did not improve. Patience: {}/{}",
        current_patience,
        max_patience
    )).await;
}
```

### Pattern 6: Training Warning Messages

**What:** Realistic warning messages for gradient issues, NaN detection, loss spikes
**When to use:** Randomly during validation (30-40% chance per CONTEXT.md)
**Example:**

```rust
// Source: Derived from PyTorch gradient debugging patterns
// https://discuss.pytorch.org/t/solved-debugging-nans-in-gradients/10532

const VALIDATION_WARNINGS: &[&str] = &[
    "Gradient norm exceeds threshold (norm={:.2}), clipping applied",
    "Detected potential numerical instability in attention scores",
    "Loss spike detected: current={:.4}, moving_avg={:.4}",
    "Memory pressure detected on GPU cluster, consider reducing batch size",
    "Validation accuracy below training accuracy by >10%, possible overfitting",
    "NaN detected in layer outputs, skipping batch",
    "Gradient accumulation buffer near capacity",
];
```

### Anti-Patterns to Avoid

- **Identical train/val loss:** Validation loss should be 5-15% higher than training loss (overfitting simulation)
- **Instant checkpoint saves:** Large models take time to save; show progress bar
- **Missing visual separator:** Validation phase should be visually distinct from training
- **Early stopping triggers:** Per CONTEXT.md, patience displays but never actually stops
- **Uniform warning frequency:** Warnings should appear randomly, not at fixed intervals

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress bars | Manual string building | `progress_string::BarBuilder` | Already used in training |
| Timestamp formatting | Manual datetime | `chrono::Local::now().format()` | Already used in log functions |
| Random noise | Manual random | `rand_distr::Normal` | Already used for loss variation |
| Colored output | ANSI escape codes | `yansi::Paint` | Already used throughout |
| Logging | Custom print | `log_info()`, `log_warning()` | Established in Phase 2 |

**Key insight:** Phase 3 should maximize reuse of Phase 2 patterns. The validation phase is essentially the training loop with different semantics and a summary at the end.

## Common Pitfalls

### Pitfall 1: Validation Loss Too Similar to Training

**What goes wrong:** Validation loss equals training loss, looks unrealistic
**Why it happens:** Using same loss generator without offset
**How to avoid:**
- Add 5-15% to validation loss relative to training loss
- Use `val_loss = train_loss * (1.0 + rng.random_range(0.05..0.15))`
- Add separate noise for per-step variation
**Warning signs:** val_loss always equals train_loss

### Pitfall 2: Checkpoint Saves Too Fast

**What goes wrong:** 2GB+ checkpoint saves instantly, looks fake
**Why it happens:** No progress simulation
**How to avoid:**
- Add progress bar for checkpoint saving
- Simulate 3-8 second save time for GB-scale files
- Show file size in log message
**Warning signs:** Checkpoint saved in <1 second

### Pitfall 3: Warning Messages Appear Simultaneously

**What goes wrong:** All warnings appear at once, looks scripted
**Why it happens:** Checking all warnings in one block
**How to avoid:**
- Random 30-40% chance per validation phase
- Maximum one warning per validation phase
- Vary warning types
**Warning signs:** Multiple warnings in sequence

### Pitfall 4: Early Stopping Counter Resets Wrong

**What goes wrong:** Patience counter doesn't reset when loss improves
**Why it happens:** Forgetting to reset counter on improvement
**How to avoid:**
- Reset patience to 0 when val_loss < best_val_loss
- Only increment when val_loss >= best_val_loss
- Update best_val_loss on improvement
**Warning signs:** Patience keeps increasing even when loss improves

### Pitfall 5: Missing Progress Bar Re-print After Log

**What goes wrong:** Progress bar disappears after warning/log message
**Why it happens:** Not re-printing progress bars after log output (established in Phase 2)
**How to avoid:** Re-print validation progress bar after any log_info/log_warning call
**Warning signs:** Progress bars vanish mid-validation

## Code Examples

Verified patterns from existing codebase and research:

### Complete Validation Phase

```rust
// Source: Derived from Phase 2 patterns + PyTorch Lightning validation format
// F:/genact-ex/src/modules/llm_training.rs

async fn run_validation(
    appconfig: &AppConfig,
    epoch: u32,
    train_loss: f64,
    best_val_loss: &mut f64,
    total_steps: u32,
    patience: &mut u32,
    max_patience: u32,
) -> f64 {
    let mut rng = rng();

    // Validation configuration (per CONTEXT.md: 80-150 steps, 10-20 seconds)
    let val_steps = rng.random_range(80..150);

    // Visual separator
    newline().await;
    print(format!(
        "{}",
        Paint::cyan("=============== Validation ===============").bold()
    )).await;
    newline().await;

    // Validation loss is 5-15% higher than training loss (CONTEXT.md)
    let val_loss_base = train_loss * (1.0 + rng.random_range(0.05..0.15));
    let noise_dist = Normal::new(0.0, 0.02).unwrap();

    // Validation progress bar (reuse training style)
    let mut bar = BarBuilder::new()
        .total(val_steps as usize)
        .width(35)
        .full_char('=')
        .include_percent()
        .build();

    let start_time = Instant::now();
    let mut current_val_loss = val_loss_base;

    for step in 1..=val_steps {
        // Add noise to validation loss
        let noise: f64 = noise_dist.sample(&mut rng);
        current_val_loss = (val_loss_base * (1.0 + noise)).max(0.1);

        let ppl = current_val_loss.exp();
        let accuracy = (1.0 / (1.0 + current_val_loss)).min(0.95);  // Simulated accuracy
        let tokens_per_sec = rng.random_range(900_000.0..1_100_000.0);

        bar.replace(step as usize);

        // Update progress (single line update)
        cursor_up(1).await;
        erase_line().await;
        print(format!(
            "Val:   {} | Loss: {:.4} | PPL: {:.2} | Acc: {:.2}% | {:.0}tok/s",
            bar,
            current_val_loss,
            ppl,
            accuracy * 100.0,
            tokens_per_sec
        )).await;
        newline().await;

        if appconfig.should_exit() {
            return current_val_loss;
        }

        csleep(rng.random_range(80..150)).await;  // ~10-20 sec total
    }

    let val_time = start_time.elapsed().as_secs_f64();
    let final_ppl = current_val_loss.exp();
    let final_accuracy = (1.0 / (1.0 + current_val_loss)).min(0.95);

    // Validation summary
    newline().await;
    log_info("Validation Results:").await;
    log_info(&format!("  val_loss:     {:.4}", current_val_loss)).await;
    log_info(&format!("  val_ppl:      {:.2}", final_ppl)).await;
    log_info(&format!("  val_accuracy: {:.2}%", final_accuracy * 100.0)).await;
    log_info(&format!("  train_loss:   {:.4} (delta: {:+.4})", train_loss, current_val_loss - train_loss)).await;
    log_info(&format!("  time:         {:.1}s", val_time)).await;

    // Random warning (30-40% chance per CONTEXT.md)
    if rng.random_bool(0.35) {
        let warnings = [
            format!("Gradient norm exceeds threshold (norm={:.2}), clipping applied", rng.random_range(5.0..15.0)),
            "Detected potential numerical instability in attention scores".to_string(),
            format!("Loss spike detected: current={:.4}, moving_avg={:.4}", current_val_loss * 1.1, current_val_loss),
            "Validation accuracy below training accuracy by >10%, possible overfitting".to_string(),
        ];
        log_warning(warnings.choose(&mut rng).unwrap()).await;
    }

    // Checkpoint logic
    if current_val_loss < *best_val_loss {
        *best_val_loss = current_val_loss;
        *patience = 0;

        // Save checkpoint with progress bar
        let file_size_gb = rng.random_range(2.0..8.0);
        save_checkpoint(appconfig, total_steps, file_size_gb).await;
    } else {
        *patience += 1;
        // Early stopping warning (never actually stops per CONTEXT.md)
        log_warning(&format!(
            "EarlyStopping: val_loss did not improve. Patience: {}/{}",
            patience,
            max_patience
        )).await;
    }

    current_val_loss
}
```

### Checkpoint Saving with Progress

```rust
// Source: HuggingFace safetensors checkpoint format
// https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/

async fn save_checkpoint(
    appconfig: &AppConfig,
    step: u32,
    file_size_gb: f32,
) {
    let mut rng = rng();
    let filename = format!("model-step-{}.safetensors", step);

    log_info(&format!("Saving model checkpoint to ./checkpoints/{}", filename)).await;

    let mut bar = BarBuilder::new()
        .total(100)
        .width(30)
        .full_char('=')
        .include_percent()
        .build();

    // Print initial progress line
    print(format!("  Writing {:.1}GB... {}", file_size_gb, bar)).await;
    newline().await;

    for i in 1..=100 {
        bar.replace(i);

        cursor_up(1).await;
        erase_line().await;
        print(format!("  Writing {:.1}GB... {}", file_size_gb, bar)).await;
        newline().await;

        if appconfig.should_exit() {
            return;
        }

        // Simulate write speed (~3-8 seconds for full checkpoint)
        csleep(rng.random_range(30..80)).await;
    }

    log_info(&format!("Checkpoint saved: {} ({:.1}GB)", filename, file_size_gb)).await;
}
```

### Warning Message Pool

```rust
// Source: Derived from PyTorch/HuggingFace training warning patterns
// https://discuss.pytorch.org/t/solved-debugging-nans-in-gradients/10532

fn get_random_warning(rng: &mut impl Rng, loss: f64) -> String {
    let warnings = [
        format!("Gradient norm exceeds threshold (norm={:.2}), clipping applied", rng.random_range(5.0..15.0)),
        "Detected potential numerical instability in attention scores".to_string(),
        format!("Loss spike detected: current={:.4}, moving_avg={:.4}", loss * 1.1, loss),
        "Memory pressure detected on GPU cluster, consider reducing batch size".to_string(),
        "Validation accuracy below training accuracy by >10%, possible overfitting".to_string(),
        format!("NaN detected in layer {}, batch skipped", rng.random_range(1..48)),
        "Gradient accumulation buffer near capacity".to_string(),
        format!("Attention pattern anomaly in head {}, layer {}", rng.random_range(0..32), rng.random_range(1..48)),
    ];

    warnings.choose(rng).unwrap().clone()
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `.ckpt` files | `.safetensors` files | 2023-2024 | Modern HuggingFace standard |
| Single checkpoint | Best model + last checkpoint | Standard practice | `save_top_k` pattern |
| Fixed validation | Configurable eval_strategy | Standard | "epoch" or "steps" options |
| Manual early stopping | Callback-based | PyTorch Lightning | Patience-based approach |

**Current conventions:**
- safetensors format is the modern standard for HuggingFace models
- Checkpoint filenames include step/epoch number: `model-step-{N}.safetensors`
- Best model strategy: save only when validation loss improves
- Early stopping uses patience counter (e.g., "Patience: 2/5")

## Open Questions

1. **Exact validation step duration**
   - What we know: CONTEXT.md specifies 10-20 seconds simulated time, 80-150 steps
   - What's unclear: Exact sleep timing per step
   - Recommendation: Calculate `sleep_ms = total_time_ms / num_steps` with some variation

2. **Accuracy calculation**
   - What we know: Should display validation accuracy
   - What's unclear: Relationship between loss and accuracy for realism
   - Recommendation: Use inverse relationship `accuracy = 1 / (1 + loss)` clamped to reasonable range

## Sources

### Primary (HIGH confidence)
- F:/genact-ex/src/modules/llm_training.rs - Existing Phase 2 implementation (lines 1-522)
- F:/genact-ex/.planning/phases/03-validation-checkpoints/03-CONTEXT.md - User decisions
- F:/genact-ex/.planning/phases/02-training-loop/02-RESEARCH.md - Prior research patterns

### Secondary (MEDIUM confidence)
- [PyTorch Lightning Evaluation](https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html) - Validation step patterns
- [PyTorch Lightning Early Stopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html) - Patience display patterns
- [HuggingFace Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) - Checkpoint save logging
- [PyTorch safetensors blog](https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/) - Modern checkpoint format
- [PyTorch-Ignite tqdm logger](https://docs.pytorch.org/ignite/generated/ignite.handlers.tqdm_logger.html) - Progress bar format
- [PyTorch Training Tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) - Loss/validation output format

### Tertiary (LOW confidence)
- [PyTorch Forums - NaN gradients](https://discuss.pytorch.org/t/solved-debugging-nans-in-gradients/10532) - Warning message patterns
- [Neptune.ai gradient clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem) - Gradient warning patterns

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All from existing Phase 2 implementation
- Architecture patterns: HIGH - Extensions of established Phase 2 patterns
- Pitfalls: HIGH - Derived from Phase 2 experience + CONTEXT.md constraints
- Output formats: MEDIUM - Based on research + CONTEXT.md locked decisions

**Research date:** 2026-01-30
**Valid until:** Indefinite for architecture patterns; output format details may evolve
