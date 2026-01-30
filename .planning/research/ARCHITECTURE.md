# Architecture Patterns: LLM Training Module

**Domain:** genact fake activity module - LLM training simulation
**Researched:** 2026-01-30
**Confidence:** HIGH (based on existing codebase analysis)

## Recommended Architecture

The LLM training module follows genact's established Module trait pattern with a phase-based state machine to simulate realistic multi-stage training workflows.

```
┌─────────────────────────────────────────────────────────────┐
│                    LlmTraining Module                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │ Phase State │   │ Data Files  │   │ Output      │       │
│  │ Machine     │   │ (static)    │   │ Formatters  │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │               │
│         └────────────────┬┼────────────────┘               │
│                          ▼│                                 │
│                  ┌────────┴────────┐                       │
│                  │   run() async   │                       │
│                  └────────┬────────┘                       │
│                           │                                 │
│                           ▼                                 │
│                  ┌─────────────────┐                       │
│                  │ io::print/csleep│                       │
│                  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `LlmTraining` struct | Module trait implementation | `run()` entry point |
| Phase functions | Individual phase logic | Data generators, io module |
| Data generators | Random model names, metrics | Phase functions |
| Static data files | Model names, framework terms | Data generators via `include_str!()` |
| Output formatters | Colored/formatted terminal output | io module |

### Data Flow

```
1. run() called by genact main loop
         │
         ▼
2. Initialize RNG, select random config
   - Model name (from static list + random generation)
   - GPU count (8-64+)
   - Training params (epochs, batch size, lr)
         │
         ▼
3. Execute phases sequentially:

   INIT_PHASE ──► TRAINING_PHASE ──► VALIDATION_PHASE ──► EXPORT_PHASE
       │              │                    │                   │
       ▼              ▼                    ▼                   ▼
   Model load     Epoch loop          Eval metrics        Save model
   Dist setup     Progress bar        Checkpoint          Convert
   NCCL init      Metrics decay       Report              Summary
         │
         ▼
4. Check appconfig.should_exit() at each step
   - Return early if true
   - Otherwise continue to next phase
```

## Phase State Machine Design

### Phase Enum (Recommended Pattern)

```rust
enum TrainingPhase {
    Initialization,
    Training { current_epoch: u32, total_epochs: u32 },
    Validation { epoch: u32 },
    Checkpoint { epoch: u32 },
    Export,
    Complete,
}
```

**Rationale:** Following genact patterns, phases are implemented as sequential async function calls rather than an explicit state machine. The enum above is conceptual - actual implementation uses separate async functions like `julia.rs` does.

### Phase Definitions

#### Phase 1: Initialization
**Duration:** 3-8 seconds simulated
**Output Pattern:**
```
[2026-01-30 14:23:15] Initializing distributed training environment...
[2026-01-30 14:23:15] Setting up NCCL backend (64 GPUs across 8 nodes)
[2026-01-30 14:23:16] Master: node-0, Port: 29500
[2026-01-30 14:23:17] Loading model: GPT-NeoX-420B-Instruct-v2
[2026-01-30 14:23:18] Model size: 420B parameters, sharded across 64 GPUs
[2026-01-30 14:23:19] Loading tokenizer: AutoTokenizer (vocab_size=128256)
[2026-01-30 14:23:20] Dataset: pile-deduped-2024 (1.2T tokens)
```

**Components:**
- `print_distributed_init()` - NCCL/distributed setup messages
- `print_model_loading()` - Model architecture and sharding info
- `print_dataset_info()` - Training data stats

#### Phase 2: Training Loop
**Duration:** 60-80% of module runtime
**Output Pattern:**
```
Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Step 1000/50000 | Loss: 2.3456 | LR: 1.0e-04 | GPU Mem: 78.2GB/80GB
  Speed: 125,432 tokens/s | 8,192 samples/s | ETA: 2h 34m
  [GPU 0-7: 98% util, 72°C] [GPU 8-15: 97% util, 74°C] ...
  [NCCL] AllReduce: 2.3ms | Broadcast: 0.8ms | Barrier: 0.1ms
```

**Components:**
- `run_training_epoch()` - Main epoch loop with progress bar
- `print_step_metrics()` - Per-step loss, LR, memory stats
- `print_gpu_cluster_status()` - Multi-GPU utilization and temps
- `print_nccl_stats()` - Distributed communication metrics

**Metric Decay Pattern:**
```rust
// Loss decays realistically over training
fn calculate_loss(step: u32, total_steps: u32, base_loss: f32) -> f32 {
    let progress = step as f32 / total_steps as f32;
    let decay = (-3.0 * progress).exp();
    let noise = rng.random_range(-0.05..0.05);
    base_loss * (0.3 + 0.7 * decay) + noise
}
```

#### Phase 3: Validation
**Duration:** 5-10 seconds per epoch
**Output Pattern:**
```
[Validation] Epoch 1 | Evaluating on 10,000 samples...
  Val Loss: 2.1234 | Val PPL: 8.37 | Val Acc: 0.892
  BLEU: 32.4 | ROUGE-L: 0.456 | F1: 0.823
```

**Components:**
- `run_validation()` - Validation loop
- `print_eval_metrics()` - Validation results

#### Phase 4: Checkpoint
**Duration:** 2-5 seconds
**Output Pattern:**
```
[Checkpoint] Saving model to checkpoints/epoch_1_step_50000/
  Saving model shards: [████████████████████] 64/64
  Saving optimizer state: 156.3 GB
  Checkpoint saved in 45.2s
```

**Components:**
- `save_checkpoint()` - Checkpoint save simulation

#### Phase 5: Export
**Duration:** 5-15 seconds
**Output Pattern:**
```
[Export] Training complete! Exporting final model...
  Converting to SafeTensors format...
  Exporting: [████████████████████] 100%
  Model saved to: ./models/gpt-neox-420b-finetuned/
  Total training time: 4h 23m 15s
  Final metrics: Loss=0.8234, PPL=2.28
```

**Components:**
- `export_model()` - Model export simulation

## Data File Organization

### Required Static Data Files

Create in `data/` directory:

| File | Content | Size | Purpose |
|------|---------|------|---------|
| `llm_model_names.txt` | Model name components | ~100 lines | Random model name generation |
| `llm_datasets.txt` | Dataset names | ~50 lines | Training data references |
| `llm_frameworks.txt` | Framework versions | ~30 lines | Version strings |

### llm_model_names.txt Structure
```
# Prefixes (realistic)
GPT
LLaMA
Mistral
Falcon
Claude
Gemini
Qwen
DeepSeek

# Suffixes (realistic)
-7B
-13B
-70B
-405B
-Instruct
-Chat
-Base
-v2

# Funny names (genact style)
ButtGPT
ChadLLM
SkyNet-Preview
HAL-9001
T-800-Instruct
```

### llm_datasets.txt Structure
```
pile-deduped
RedPajama-v2
SlimPajama
FineWeb
The-Stack-v2
OpenWebText
C4
CommonCrawl-2024
```

### Data Loading Pattern

Following `data.rs` pattern:

```rust
// In data.rs
static LLM_MODEL_NAMES: &str = include_str!("../data/llm_model_names.txt");
static LLM_DATASETS: &str = include_str!("../data/llm_datasets.txt");

lazy_static::lazy_static! {
    pub static ref LLM_MODEL_NAMES_LIST: Vec<&'static str> =
        LLM_MODEL_NAMES.lines().filter(|l| !l.starts_with('#')).collect();
    pub static ref LLM_DATASETS_LIST: Vec<&'static str> =
        LLM_DATASETS.lines().filter(|l| !l.starts_with('#')).collect();
}
```

## Output Formatting Approach

### Color Scheme (following yansi patterns from existing modules)

| Element | Color | Example |
|---------|-------|---------|
| Timestamps | Magenta | `[2026-01-30 14:23:15]` |
| Phase headers | Cyan/Bold | `[Initialization]` |
| Progress bars | Green | `━━━━━━━━━━` |
| Metrics labels | Blue | `Loss:`, `LR:` |
| Metric values | White/Bold | `2.3456` |
| Warnings | Yellow | `[WARN] OOM on GPU 42` |
| GPU status | Cyan | `[GPU 0-7: 98%]` |
| NCCL stats | Fixed(8) (gray) | `AllReduce: 2.3ms` |

### Output Helper Functions

```rust
// Format a metric line
async fn print_metric(label: &str, value: impl Display) {
    print(format!("  {}: {}", label.blue(), value.bold())).await;
}

// Format GPU cluster status
async fn print_gpu_status(gpus: &[GpuStats]) {
    for chunk in gpus.chunks(8) {
        let status = chunk.iter()
            .map(|g| format!("{}%/{}°C", g.util, g.temp))
            .collect::<Vec<_>>()
            .join(", ");
        print(format!("  {}", status.cyan())).await;
        newline().await;
    }
}

// Format progress bar (using progress_string crate)
fn create_training_progress(total: usize) -> progress_string::Bar {
    progress_string::BarBuilder::new()
        .total(total)
        .width(40)
        .full_char('━')
        .empty_char('─')
        .include_percent()
        .build()
}
```

## Integration with Existing genact Patterns

### Module Registration

In `src/modules/mod.rs`:

```rust
pub mod llm_training;  // Add module declaration

// In ALL_MODULES lazy_static:
all_modules.insert("llm_training", Box::new(llm_training::LlmTraining));
```

### Module Structure Template

```rust
//! Pretend to train a large language model
use async_trait::async_trait;
use instant::Instant;
use rand::seq::IndexedRandom;
use rand::{Rng, rng};
use yansi::Paint;

use crate::args::AppConfig;
use crate::data::{LLM_MODEL_NAMES_LIST, LLM_DATASETS_LIST};
use crate::io::{csleep, cursor_up, erase_line, newline, print};
use crate::modules::Module;

pub struct LlmTraining;

#[async_trait(?Send)]
impl Module for LlmTraining {
    fn name(&self) -> &'static str {
        "llm_training"
    }

    fn signature(&self) -> String {
        "torchrun --nproc_per_node=8 train.py".to_string()
    }

    async fn run(&self, appconfig: &AppConfig) {
        let mut rng = rng();

        // Phase 1: Initialization
        run_initialization(&mut rng).await;
        if appconfig.should_exit() { return; }

        // Phase 2-4: Training loop with validation and checkpoints
        let epochs = rng.random_range(2..5);
        for epoch in 1..=epochs {
            run_training_epoch(&mut rng, epoch, epochs).await;
            if appconfig.should_exit() { return; }

            run_validation(&mut rng, epoch).await;
            if appconfig.should_exit() { return; }

            if epoch < epochs {
                save_checkpoint(&mut rng, epoch).await;
                if appconfig.should_exit() { return; }
            }
        }

        // Phase 5: Export
        export_model(&mut rng).await;
    }
}

// Phase implementations as separate async functions...
```

### Graceful Exit Handling

Following patterns from `cargo.rs`, `download.rs`, `terraform.rs`:

```rust
// Check at natural breakpoints (end of steps, between phases)
for step in 0..total_steps {
    // ... output step metrics ...

    if appconfig.should_exit() {
        // Print summary before exit
        print_training_summary(&metrics).await;
        return;
    }
}
```

### Progress Bar Updates

Following `julia.rs` pattern for in-place updates:

```rust
async fn update_progress_bar(bar: &mut progress_string::Bar, current: usize) {
    cursor_up(1).await;
    erase_line().await;
    bar.replace(current);
    print(format!("{}", bar)).await;
    newline().await;
}
```

## Patterns to Follow

### Pattern 1: Phased Sequential Execution

**What:** Break simulation into distinct phases executed sequentially
**When:** Module simulates multi-stage process
**Example:** See `julia.rs` - registry update, resolving, installing, precompiling
**Why genact does this:** Natural breakpoints for exit checks, realistic feel

### Pattern 2: Randomized Timing

**What:** Use `rng.random_range()` for all delays
**When:** Any `csleep()` call
**Example:**
```rust
csleep(rng.random_range(100..500)).await;  // Variable processing time
```
**Why genact does this:** Prevents robotic/mechanical feel

### Pattern 3: Progress Bar with Cursor Control

**What:** Update progress bar in-place using cursor control
**When:** Showing incremental progress
**Example:** See `julia.rs` `download_artifacts()`, `precompile()`
**Why genact does this:** Clean visual output, realistic progress display

### Pattern 4: Metric Decay with Noise

**What:** Simulate metrics that trend but have variance
**When:** Training loss, performance metrics
**Example:**
```rust
let base_value = start_value * decay_factor;
let noise = rng.random_range(-variance..variance);
base_value + noise
```
**Why genact does this:** Real training metrics are noisy

## Anti-Patterns to Avoid

### Anti-Pattern 1: Blocking Synchronous Output

**What:** Using `println!()` or blocking I/O
**Why bad:** Breaks WASM compatibility, violates genact I/O abstraction
**Instead:** Use `io::print()`, `io::newline()`, `io::dprint()`

### Anti-Pattern 2: Fixed Timing

**What:** Using constant sleep durations
**Why bad:** Looks artificial, predictable
**Instead:** Always use `rng.random_range(min..max)`

### Anti-Pattern 3: Missing Exit Checks

**What:** Long loops without `appconfig.should_exit()` checks
**Why bad:** Prevents graceful termination, poor UX
**Instead:** Check at loop iterations and between phases

### Anti-Pattern 4: Hardcoded Data

**What:** Embedding lists directly in module code
**Why bad:** Harder to maintain, increases binary bloat
**Instead:** Use `data/` files with `include_str!()`

## File Organization Summary

```
src/
├── modules/
│   ├── mod.rs           # Add llm_training module registration
│   └── llm_training.rs  # Main module implementation (~400-600 lines)
│
├── data.rs              # Add LLM_* static data declarations
│
data/
├── llm_model_names.txt  # ~100 lines: model name components
├── llm_datasets.txt     # ~50 lines: dataset names
└── llm_frameworks.txt   # ~30 lines: framework versions (optional)
```

## Scalability Considerations

| Concern | Current Design | Notes |
|---------|---------------|-------|
| GPU count simulation | 8-64 configurable | Random selection per run |
| Epoch duration | 30-120 seconds simulated | Adjusts with speed_factor |
| Output verbosity | Moderate | Balance between realism and noise |
| Memory footprint | Minimal | All data is static strings |

## Sources

- **Project codebase analysis:** `F:\genact-ex\src\modules\*.rs` (HIGH confidence)
- **Module trait definition:** `F:\genact-ex\src\modules\mod.rs` lines 27-32 (HIGH confidence)
- **Data loading patterns:** `F:\genact-ex\src\data.rs` (HIGH confidence)
- **I/O abstraction:** `F:\genact-ex\src\io.rs` (HIGH confidence)
- **Progress bar usage:** `julia.rs`, `download.rs` (HIGH confidence)
- **PyTorch/HuggingFace output format:** Based on real CLI output patterns (MEDIUM confidence - based on training data knowledge)

---

## Quality Gate Checklist

- [x] Fits genact Module trait pattern - struct + async_trait impl with name/signature/run
- [x] Phase transitions clearly defined - 5 phases: Init, Training, Validation, Checkpoint, Export
- [x] Data requirements specified - 2-3 data files in `data/` directory
- [x] Output formatting approach documented - yansi colors, progress_string bars
- [x] Integration points identified - mod.rs registration, data.rs declarations
- [x] Exit handling patterns documented - appconfig.should_exit() at breakpoints
