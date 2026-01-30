# Phase 2: Training Loop - Research

**Researched:** 2026-01-30
**Domain:** Rust implementation of simulated PyTorch/HuggingFace training output with progress bars, metrics, and GPU status displays
**Confidence:** HIGH

## Summary

Phase 2 transforms the placeholder llm_training module into a convincing simulation of large-scale distributed LLM training. The research confirms that all necessary patterns exist within the genact codebase: progress bars via `progress_string`, colored output via `yansi`, cursor manipulation via `io::cursor_up`/`io::erase_line`, and statistical distributions via `rand_distr`. The phase requires implementing three main components: (1) detailed initialization output mimicking PyTorch DDP startup, (2) a training loop with dual progress bars and fluctuating metrics, and (3) a GPU status grid displaying 64+ GPUs grouped by node.

The key insight is that realistic training output follows predictable patterns: initialization logs with timestamps, NCCL backend messages, epoch/step progress bars with tqdm-style formatting, and loss values that decay exponentially with small random fluctuations. The existing `julia.rs` module provides excellent reference patterns for progress bars with cursor manipulation, while `cryptomining.rs` shows how to display multi-GPU status in a loop.

**Primary recommendation:** Follow the established patterns from julia.rs for progress bar updates and cryptomining.rs for GPU status displays. Use `rand_distr::Normal` for loss fluctuation and simple exponential decay for the base loss curve. Structure the module into distinct async functions for each phase (init, train loop, GPU status) following the modular pattern in julia.rs.

## Standard Stack

The existing codebase provides all necessary dependencies - no new dependencies needed.

### Core (from Cargo.toml, already available)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| progress_string | 0.2 | Progress bar rendering | Already used in julia.rs, download.rs |
| yansi | 1 | Terminal colors | Already used in cargo.rs, bootlog.rs |
| rand | 0.9 | Random values | Already used throughout |
| rand_distr | 0.5 | Statistical distributions | Already used in cryptomining.rs, ansible.rs |
| chrono | 0.4 | Timestamps | Already used in cryptomining.rs |
| instant | 0.1 | Elapsed time | Already used in cargo.rs, julia.rs |

### I/O Functions (from crate::io)

| Function | Purpose | Example Usage |
|----------|---------|---------------|
| `print()` | Print without newline | `print(format!("Loss: {:.4}", loss)).await` |
| `newline()` | Print `\r\n` | After each log line |
| `csleep()` | Async sleep respecting speed_factor | `csleep(100).await` |
| `dprint()` | Character-by-character print | For typing effect |
| `cursor_up()` | Move cursor up N lines | `cursor_up(1).await` |
| `erase_line()` | Clear current line | For updating progress bars |
| `get_terminal_width()` | Get terminal width | For responsive layouts |

### No Additional Dependencies Needed

Phase 2 uses only existing crate dependencies. All required functionality is available.

## Architecture Patterns

### Recommended Module Structure

```rust
// src/modules/llm_training.rs - Phase 2 structure

pub struct LlmTraining;

impl Module for LlmTraining {
    async fn run(&self, appconfig: &AppConfig) {
        // Phase 1: Initialization
        run_initialization(appconfig).await;

        // Phase 2: Training Loop
        run_training_loop(appconfig).await;
    }
}

// Separate async functions for each phase
async fn run_initialization(appconfig: &AppConfig) { ... }
async fn display_model_loading(model: &str) { ... }
async fn display_distributed_init(num_gpus: u32, num_nodes: u32) { ... }
async fn display_gpu_detection(gpus: &[GpuStatus]) { ... }

async fn run_training_loop(appconfig: &AppConfig) { ... }
async fn update_epoch_progress(epoch: u32, total: u32, bar: &progress_string::Bar) { ... }
async fn update_step_metrics(step: u32, loss: f64, lr: f64) { ... }
async fn display_gpu_status_grid(gpus: &[GpuStatus]) { ... }

// Helper structs
struct GpuStatus {
    id: u32,
    node: u32,
    memory_used: u64,
    memory_total: u64,
    utilization: f32,
    temperature: u32,
}

struct TrainingState {
    epoch: u32,
    step: u32,
    loss: f64,
    lr: f64,
    start_time: Instant,
}
```

### Pattern 1: Dual Progress Bars (Epoch + Step)

**What:** Two-level progress display showing epoch progress and step progress within epoch
**When to use:** Training loop display
**Example:**

```rust
// Source: Derived from F:/genact-ex/src/modules/julia.rs pattern

// Epoch progress bar (outer loop)
let mut epoch_bar = progress_string::BarBuilder::new()
    .total(total_epochs)
    .width(30)
    .full_char('=')
    .include_numbers()  // Shows "1/3"
    .build();

// Step progress bar (inner loop)
let mut step_bar = progress_string::BarBuilder::new()
    .total(steps_per_epoch)
    .width(40)
    .full_char('=')
    .include_percent()  // Shows "50%"
    .build();

// Update pattern with cursor manipulation
for epoch in 1..=total_epochs {
    epoch_bar.replace(epoch);
    step_bar.replace(0);  // Reset step bar

    for step in 1..=steps_per_epoch {
        step_bar.replace(step);

        // Move cursor up to overwrite previous lines
        cursor_up(2).await;  // Up 2 lines (epoch + step)
        erase_line().await;

        // Print epoch progress
        print(format!("Epoch: {}", epoch_bar)).await;
        newline().await;

        // Print step progress with metrics
        erase_line().await;
        print(format!("Step:  {} | Loss: {:.4}", step_bar, loss)).await;
        newline().await;

        csleep(50).await;
    }
}
```

### Pattern 2: Timestamp + Rank Prefix Logging

**What:** Log lines with PyTorch DDP style timestamps and rank info
**When to use:** All initialization and training log messages
**Example:**

```rust
// Source: Derived from real PyTorch DDP output patterns

use chrono::Local;

async fn log_info(rank: u32, world_size: u32, message: &str) {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] [Rank {}/{}] {} {}",
        timestamp,
        rank,
        world_size,
        Paint::green("INFO").bold(),
        message
    )).await;
    newline().await;
}

async fn log_warning(message: &str) {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] {} {}",
        timestamp,
        Paint::yellow("WARNING").bold(),
        message
    )).await;
    newline().await;
}
```

### Pattern 3: Loss Decay with Noise

**What:** Exponential loss decay with Gaussian noise for realistic training curves
**When to use:** Generating loss values during training
**Example:**

```rust
// Source: Derived from F:/genact-ex/src/modules/cryptomining.rs + research

use rand_distr::{Distribution, Normal};

struct LossGenerator {
    initial_loss: f64,
    decay_rate: f64,
    noise: Normal<f64>,
    step: u32,
}

impl LossGenerator {
    fn new(initial_loss: f64) -> Self {
        Self {
            initial_loss,
            decay_rate: 0.001,  // Controls how fast loss decreases
            noise: Normal::new(0.0, 0.02).unwrap(),  // Small noise
            step: 0,
        }
    }

    fn next(&mut self, rng: &mut ThreadRng) -> f64 {
        self.step += 1;

        // Exponential decay: L(t) = L0 * e^(-k*t)
        let base_loss = self.initial_loss * (-self.decay_rate * self.step as f64).exp();

        // Add Gaussian noise (proportional to current loss for realism)
        let noise_factor = 1.0 + self.noise.sample(rng);

        (base_loss * noise_factor).max(0.01)  // Floor to prevent negative loss
    }
}
```

### Pattern 4: GPU Status Grid Display

**What:** Multi-row table showing GPU stats grouped by node
**When to use:** Periodic GPU status updates (every N steps)
**Example:**

```rust
// Source: Derived from nvidia-smi output format + CONTEXT.md decisions

async fn display_gpu_status_grid(gpus: &[GpuStatus], num_nodes: u32) {
    // Header
    print(format!(
        "{}",
        Paint::cyan("========== GPU Status ==========").bold()
    )).await;
    newline().await;

    // Group by node
    for node in 0..num_nodes {
        print(format!("Node {}: ", node)).await;

        // Show GPUs for this node (typically 8 per node)
        let node_gpus: Vec<_> = gpus.iter()
            .filter(|g| g.node == node)
            .collect();

        for gpu in node_gpus {
            // Color based on utilization/temp
            let color = if gpu.temperature > 80 {
                yansi::Color::Red
            } else if gpu.utilization > 90.0 {
                yansi::Color::Yellow
            } else {
                yansi::Color::Green
            };

            print(format!(
                "[GPU{}: {}% {}C {}GB] ",
                gpu.id % 8,  // Local GPU ID within node
                gpu.utilization as u32,
                gpu.temperature,
                gpu.memory_used / 1024,  // Convert to GB
                color = Paint::new("").fg(color)
            )).await;
        }
        newline().await;
    }
}
```

### Pattern 5: Progress Bar with Inline Metrics (tqdm style)

**What:** Single-line progress bar with metrics inline, tqdm format
**When to use:** Step-level progress display
**Example:**

```rust
// Source: Derived from tqdm output format research

async fn print_tqdm_style_progress(
    step: u32,
    total_steps: u32,
    loss: f64,
    lr: f64,
    tokens_per_sec: f64,
    elapsed_secs: f64,
) {
    let percent = (step as f64 / total_steps as f64 * 100.0) as u32;
    let bar_width = 20;
    let filled = (step as f64 / total_steps as f64 * bar_width as f64) as usize;
    let empty = bar_width - filled;

    let bar = format!("{}{}",
        "=".repeat(filled),
        " ".repeat(empty)
    );

    let eta_secs = if step > 0 {
        (elapsed_secs / step as f64) * (total_steps - step) as f64
    } else {
        0.0
    };

    erase_line().await;
    print(format!(
        "Step: {:>3}%|{}| {}/{} [{}s<{}s, loss={:.4}, lr={:.2e}, {:.0}tok/s]",
        percent,
        bar,
        step,
        total_steps,
        elapsed_secs as u32,
        eta_secs as u32,
        loss,
        lr,
        tokens_per_sec
    )).await;
}
```

### Anti-Patterns to Avoid

- **Static loss values:** Loss should fluctuate with noise, not monotonically decrease
- **Uniform GPU stats:** Each GPU should have slightly different values
- **No cursor management:** Must use `cursor_up` + `erase_line` for updating progress bars
- **Single progress bar:** Real training shows both epoch and step progress
- **Missing timestamps:** PyTorch DDP logs always include timestamps
- **No should_exit checks:** Must check in inner training loop

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Progress bars | Manual `[====  ]` strings | `progress_string::BarBuilder` | Width calculation, edge cases |
| Random noise | Manual `rand() * 0.1` | `rand_distr::Normal` | Proper Gaussian distribution |
| Timestamp formatting | Manual string building | `chrono::Local::now().format()` | Locale handling, format strings |
| Terminal colors | ANSI escape codes | `yansi::Paint` | Cross-platform, readable |
| Terminal width | Hardcoded values | `io::get_terminal_width()` | Responsive layouts |
| Elapsed time | Manual subtraction | `instant::Instant::elapsed()` | Cross-platform, WASM compatible |

**Key insight:** The julia.rs module already solves the hardest problem - multi-line progress bar updates with cursor manipulation. Follow that pattern exactly.

## Common Pitfalls

### Pitfall 1: Progress Bar Flickering

**What goes wrong:** Progress bars flicker or leave artifacts when updating
**Why it happens:** Not erasing line before printing, wrong cursor positioning
**How to avoid:**
```rust
// Correct pattern from julia.rs:
cursor_up(1).await;      // Move to previous line
erase_line().await;      // Clear it completely
print(new_content).await; // Print new content
newline().await;         // Move to next line
```
**Warning signs:** Visual artifacts, duplicate lines, jumping text

### Pitfall 2: Loss Values Too Predictable

**What goes wrong:** Loss decreases perfectly smoothly, looks fake
**Why it happens:** Using linear decay without noise
**How to avoid:**
- Use exponential decay as base
- Add Gaussian noise with `rand_distr::Normal`
- Occasionally add larger spikes (mini-batch variance)
- Scale noise proportionally to current loss value
**Warning signs:** Loss curve is perfectly smooth

### Pitfall 3: GPU Stats All Identical

**What goes wrong:** All 64 GPUs show exactly the same memory/utilization
**Why it happens:** Not adding per-GPU variation
**How to avoid:**
- Generate base values for the cluster
- Add small random offset per GPU using Normal distribution
- Vary temperature based on GPU position (edge GPUs cooler)
**Warning signs:** Every GPU shows exactly "95% 78C 72GB"

### Pitfall 4: Terminal Width Overflow

**What goes wrong:** Long lines wrap unexpectedly, breaking layout
**Why it happens:** Not accounting for terminal width
**How to avoid:**
- Use `io::get_terminal_width()` to get actual width
- Truncate or adapt output based on available space
- Test with narrow terminals (80 columns)
**Warning signs:** Broken formatting on smaller terminals

### Pitfall 5: Blocking Inner Loop

**What goes wrong:** Module doesn't respond to CTRL-C during training
**Why it happens:** `should_exit()` only checked at epoch boundaries
**How to avoid:** Check `should_exit()` inside the step loop, not just epoch loop
```rust
for step in 1..=steps_per_epoch {
    // ... update progress ...

    if appconfig.should_exit() {
        return;  // Exit immediately from step loop
    }
}
```
**Warning signs:** CTRL-C only works between epochs

### Pitfall 6: Cursor Position After Exit

**What goes wrong:** Terminal cursor left in wrong position after early exit
**Why it happens:** Exit happens mid-update before newline
**How to avoid:** Ensure final newline before any return statement
**Warning signs:** Next shell prompt appears in wrong place

## Code Examples

Verified patterns from the existing codebase:

### Complete Initialization Sequence

```rust
// Source: Derived from F:/genact-ex/src/modules/julia.rs structure + CONTEXT.md

use chrono::Local;
use yansi::Paint;
use rand::rng;
use rand::seq::IndexedRandom;

async fn run_initialization(appconfig: &AppConfig) {
    let mut rng = rng();

    // Select random configuration
    let model = LLM_MODELS_LIST.choose(&mut rng).unwrap();
    let gpu = GPU_MODELS_LIST.choose(&mut rng).unwrap();
    let dataset = LLM_DATASETS_LIST.choose(&mut rng).unwrap();

    let num_gpus: u32 = 64;
    let num_nodes: u32 = 8;
    let params_b: f64 = rng.random_range(7.0..405.0);

    // Timestamp format
    let ts = || Local::now().format("%Y-%m-%d %H:%M:%S");

    // Model loading
    print(format!(
        "[{}] {} Loading model: {}",
        ts(),
        Paint::green("INFO").bold(),
        model
    )).await;
    newline().await;
    csleep(200).await;

    print(format!(
        "[{}] {}   Parameters: {:.1}B",
        ts(),
        Paint::green("INFO").bold(),
        params_b
    )).await;
    newline().await;
    csleep(100).await;

    // Model loading progress bar
    let mut bar = progress_string::BarBuilder::new()
        .total(100)
        .width(40)
        .full_char('=')
        .include_percent()
        .build();

    for i in 0..=100 {
        bar.replace(i);
        erase_line().await;
        print(format!(
            "[{}] {} Loading model weights... {}",
            ts(),
            Paint::green("INFO").bold(),
            bar
        )).await;

        csleep(rng.random_range(20..100)).await;

        if appconfig.should_exit() {
            newline().await;
            return;
        }
    }
    newline().await;

    print(format!(
        "[{}] {} Model loaded in {:.1}s",
        ts(),
        Paint::green("INFO").bold(),
        rng.random_range(8.0..25.0)
    )).await;
    newline().await;
    csleep(300).await;

    // Distributed initialization
    print(format!(
        "[{}] {} Initializing distributed environment...",
        ts(),
        Paint::green("INFO").bold()
    )).await;
    newline().await;
    csleep(500).await;

    print(format!(
        "[{}] {} NCCL version: 2.19.3",
        ts(),
        Paint::green("INFO").bold()
    )).await;
    newline().await;

    print(format!(
        "[{}] {} World size: {} GPUs across {} nodes",
        ts(),
        Paint::green("INFO").bold(),
        num_gpus,
        num_nodes
    )).await;
    newline().await;
    csleep(200).await;

    // GPU detection (show a few)
    for node in 0..num_nodes.min(2) {  // Show first 2 nodes
        for local_gpu in 0..8u32 {
            let gpu_id = node * 8 + local_gpu;
            print(format!(
                "[{}] [Rank {}/{}] {} Detected NVIDIA {} on node {}",
                ts(),
                gpu_id,
                num_gpus,
                Paint::green("INFO").bold(),
                gpu,
                node
            )).await;
            newline().await;
            csleep(rng.random_range(10..50)).await;

            if appconfig.should_exit() {
                return;
            }
        }
    }

    print(format!(
        "[{}] {} ... and {} more GPUs",
        ts(),
        Paint::green("INFO").bold(),
        num_gpus - 16
    )).await;
    newline().await;
    csleep(500).await;

    // NCCL communication test
    print(format!(
        "[{}] {} Running NCCL AllReduce test...",
        ts(),
        Paint::green("INFO").bold()
    )).await;
    newline().await;
    csleep(1000).await;

    print(format!(
        "[{}] {} AllReduce test passed ({:.2}ms)",
        ts(),
        Paint::green("INFO").bold(),
        rng.random_range(5.0..50.0)
    )).await;
    newline().await;
    csleep(300).await;

    // Dataset loading
    print(format!(
        "[{}] {} Loading dataset: {}",
        ts(),
        Paint::green("INFO").bold(),
        dataset
    )).await;
    newline().await;
    csleep(500).await;

    let samples = rng.random_range(1_000_000..100_000_000);
    print(format!(
        "[{}] {} Dataset: {} samples, batch_size=2048, micro_batch=4",
        ts(),
        Paint::green("INFO").bold(),
        samples
    )).await;
    newline().await;
    csleep(300).await;

    newline().await;
    print(format!(
        "[{}] {} ======== Training Started ========",
        ts(),
        Paint::green("INFO").bold()
    )).await;
    newline().await;
    newline().await;
}
```

### Training Loop with Metrics

```rust
// Source: Derived from F:/genact-ex/src/modules/julia.rs + CONTEXT.md

use rand_distr::{Distribution, Normal};
use instant::Instant;

async fn run_training_loop(appconfig: &AppConfig) {
    let mut rng = rng();

    let total_epochs = 3u32;
    let steps_per_epoch = rng.random_range(500..2000);

    let mut loss = rng.random_range(8.0..12.0);  // Initial loss
    let decay_rate = 0.002;
    let noise_dist = Normal::new(0.0, 0.03).unwrap();

    let lr_initial = 1e-4;
    let lr = lr_initial;

    let start_time = Instant::now();
    let mut total_steps = 0u32;

    // Epoch progress bar
    let mut epoch_bar = progress_string::BarBuilder::new()
        .total(total_epochs as usize)
        .width(25)
        .full_char('=')
        .include_numbers()
        .build();

    // Step progress bar
    let mut step_bar = progress_string::BarBuilder::new()
        .total(steps_per_epoch as usize)
        .width(35)
        .full_char('=')
        .include_percent()
        .build();

    // Print initial progress (2 lines)
    print(format!("Epoch: {}", epoch_bar)).await;
    newline().await;
    print(format!("Step:  {} | Loss: -.----", step_bar)).await;
    newline().await;

    for epoch in 1..=total_epochs {
        epoch_bar.replace(epoch as usize);
        step_bar.replace(0);

        for step in 1..=steps_per_epoch {
            total_steps += 1;

            // Update loss with decay + noise
            let base_decay = (-decay_rate * total_steps as f64).exp();
            let noise = noise_dist.sample(&mut rng);
            loss = (loss * base_decay * (1.0 + noise)).max(0.1);

            // Perplexity
            let ppl = loss.exp();

            // Tokens per second (simulated)
            let tokens_per_sec = rng.random_range(800_000.0..1_200_000.0);

            // Update step bar
            step_bar.replace(step as usize);

            // Update display
            cursor_up(2).await;

            erase_line().await;
            print(format!("Epoch: {}", epoch_bar)).await;
            newline().await;

            erase_line().await;
            let elapsed = start_time.elapsed().as_secs();
            let eta = if total_steps > 0 {
                (elapsed as f64 / total_steps as f64) *
                ((total_epochs * steps_per_epoch - total_steps) as f64)
            } else {
                0.0
            };

            print(format!(
                "Step:  {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | {:.0}tok/s | {}s<{}s",
                step_bar,
                loss,
                ppl,
                lr,
                tokens_per_sec,
                elapsed,
                eta as u64
            )).await;
            newline().await;

            csleep(rng.random_range(30..100)).await;

            // Periodic detailed log every 10 steps
            if step % 10 == 0 {
                // Don't cursor_up for this, just add new line
                print(format!(
                    "  Step {}/{} | loss={:.4} | ppl={:.2} | grad_norm={:.3}",
                    step,
                    steps_per_epoch,
                    loss,
                    ppl,
                    rng.random_range(0.5..2.0)
                )).await;
                newline().await;
            }

            if appconfig.should_exit() {
                newline().await;
                return;
            }
        }

        // Epoch summary
        newline().await;
        print(format!(
            "  Epoch {} complete | avg_loss={:.4} | time={:.1}s",
            epoch,
            loss,
            start_time.elapsed().as_secs_f64() / epoch as f64
        )).await;
        newline().await;
        newline().await;
    }
}
```

### GPU Status Grid

```rust
// Source: Derived from nvidia-smi format + CONTEXT.md decisions

struct GpuStatus {
    id: u32,
    node: u32,
    memory_used_gb: f32,
    memory_total_gb: f32,
    utilization: f32,
    temperature: u32,
    power_watts: u32,
}

async fn display_gpu_status_grid(gpus: &[GpuStatus], num_nodes: u32) {
    print(Paint::cyan("[ GPU Status Grid ]").bold().to_string()).await;
    newline().await;

    for node in 0..num_nodes {
        let node_gpus: Vec<_> = gpus.iter()
            .filter(|g| g.node == node)
            .collect();

        print(format!("Node {}: ", node)).await;

        for gpu in node_gpus {
            let mem_pct = (gpu.memory_used_gb / gpu.memory_total_gb * 100.0) as u32;

            // Color based on health
            let status_color = if gpu.temperature > 85 || mem_pct > 95 {
                yansi::Color::Red
            } else if gpu.temperature > 75 || mem_pct > 85 {
                yansi::Color::Yellow
            } else {
                yansi::Color::Green
            };

            let status_char = Paint::new("*").fg(status_color);

            print(format!(
                "{} GPU{}: {}% {}C {:.0}/{:.0}GB ",
                status_char,
                gpu.id % 8,
                gpu.utilization as u32,
                gpu.temperature,
                gpu.memory_used_gb,
                gpu.memory_total_gb
            )).await;
        }
        newline().await;
    }
}

fn generate_gpu_statuses(num_gpus: u32, num_nodes: u32, rng: &mut ThreadRng) -> Vec<GpuStatus> {
    let base_util = rng.random_range(90.0..98.0);
    let base_temp = rng.random_range(70..80);
    let base_mem = rng.random_range(70.0..78.0);  // GB used
    let total_mem = 80.0f32;  // 80GB for H100

    let util_noise = Normal::new(0.0, 2.0).unwrap();
    let temp_noise = Normal::new(0.0, 3.0).unwrap();
    let mem_noise = Normal::new(0.0, 1.5).unwrap();

    (0..num_gpus).map(|id| {
        GpuStatus {
            id,
            node: id / 8,
            memory_used_gb: (base_mem + mem_noise.sample(rng) as f32).clamp(60.0, total_mem - 1.0),
            memory_total_gb: total_mem,
            utilization: (base_util + util_noise.sample(rng)).clamp(0.0, 100.0) as f32,
            temperature: (base_temp as f64 + temp_noise.sample(rng)).clamp(50.0, 95.0) as u32,
            power_watts: rng.random_range(350..450),
        }
    }).collect()
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `rand::thread_rng()` | `rand::rng()` | rand 0.9 | Simpler API |
| Manual ANSI codes | `yansi` crate | N/A | Already used |
| `println!()` | `io::print() + io::newline()` | N/A | WASM compatibility |

**Deprecated/outdated:**
- None for this phase - all patterns are current in genact codebase

## Implementation Considerations

### Output Rhythm (CONTEXT.md Decisions)

Per CONTEXT.md, the following is decided:
- Detailed log every 10 steps
- GPU status grid updated every 10 steps alongside detailed log
- Steady output rhythm throughout
- 500-2000 steps per epoch (moderate)
- Multiple epochs simulated
- Occasional WARNING logs for realism

### Color Scheme (CONTEXT.md Decisions)

- INFO: Green (`Paint::green()`)
- WARNING: Yellow (`Paint::yellow()`)
- ERROR: Red (`Paint::red()`)
- Timestamps: Default/dim
- Progress bars: Cyan accent
- GPU status: Green/Yellow/Red based on health

### Discretion Areas (Claude's Choice per CONTEXT.md)

The following are left to implementation discretion:
- Exact output timing/delays
- Exact progress bar style characters
- Specific WARNING message content and frequency
- NCCL communication log format details

## Open Questions

1. **Terminal width handling for GPU grid**
   - What we know: 64 GPUs = 8 nodes x 8 GPUs, each GPU status takes ~25 chars
   - What's unclear: Best layout if terminal is narrow
   - Recommendation: Check `get_terminal_width()`, use compact format if < 100 cols

2. **NCCL log frequency**
   - What we know: Should appear occasionally during training
   - What's unclear: How often is realistic
   - Recommendation: Every 50-100 steps, with random timing

## Sources

### Primary (HIGH confidence)
- F:/genact-ex/src/modules/julia.rs - Progress bar + cursor manipulation patterns (lines 168-451)
- F:/genact-ex/src/modules/cryptomining.rs - Multi-GPU status display pattern (lines 104-114)
- F:/genact-ex/src/modules/download.rs - Progress bar builder usage (lines 61-66)
- F:/genact-ex/src/io.rs - All I/O function signatures
- F:/genact-ex/Cargo.toml - Available dependencies

### Secondary (MEDIUM confidence)
- [PyTorch Distributed Documentation](https://docs.pytorch.org/docs/stable/distributed.html) - DDP output patterns
- [Lambda Labs Multi-Node Guide](https://lambda.ai/blog/multi-node-pytorch-distributed-training-guide) - NCCL setup output
- [NVIDIA SMI Manual](https://docs.nvidia.com/deploy/nvidia-smi/index.html) - GPU status format
- [tqdm Documentation](https://adamoudad.github.io/posts/progress_bar_with_tqdm/) - Progress bar format patterns
- [PyTorch Training Tutorial](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html) - Loss logging format

### Tertiary (LOW confidence)
- Learning curve research papers - Exponential decay + noise patterns (general ML knowledge)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All from existing codebase
- Architecture: HIGH - Patterns derived from julia.rs, cryptomining.rs
- Pitfalls: HIGH - Identified from codebase analysis
- Output format: MEDIUM - Based on research + CONTEXT.md decisions

**Research date:** 2026-01-30
**Valid until:** Indefinite for architecture patterns; output formats may evolve
