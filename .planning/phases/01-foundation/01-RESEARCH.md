# Phase 1: Foundation - Research

**Researched:** 2026-01-30
**Domain:** Rust module development for genact with static data files
**Confidence:** HIGH

## Summary

Phase 1 establishes the foundational module structure for the LLM training simulator. This involves creating static data files for model names, GPU names, and dataset names, then implementing a basic Module trait with placeholder output. The research confirms genact uses a well-established pattern: data files in `data/` loaded via `include_str!()` in `src/data.rs`, exposed as `lazy_static` vectors, and consumed by modules that implement the `Module` trait.

The existing codebase provides clear, consistent patterns across 20+ modules. All modules follow the same structure: unit struct, `#[async_trait(?Send)]` implementation, and use of `crate::io::*` functions for cross-platform output. The research shows this is a straightforward implementation task with minimal risk.

**Primary recommendation:** Follow the exact patterns from existing modules (cargo, bootlog, simcity) for data loading and module registration, ensuring WASM compatibility by using `io::*` functions exclusively.

## Standard Stack

The existing codebase defines the stack - no new dependencies needed for Phase 1.

### Core (from Cargo.toml)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| async-trait | 0.1 | Async trait support | Required for Module trait impl |
| lazy_static | 1.5 | Static data initialization | Data file loading pattern |
| rand | 0.9 | Random selection | Choosing random items from lists |
| yansi | 1 | Terminal colors | Colored output formatting |

### Already Available in Crate

| Module | Usage | Example |
|--------|-------|---------|
| `crate::io` | Print/sleep/newline | `io::print()`, `io::csleep()`, `io::newline()` |
| `crate::data` | Static data access | `PACKAGES_LIST`, `SIMCITY_LIST` |
| `crate::generators` | Random utilities | `gen_hex_string()`, `gen_package_version()` |
| `crate::args::AppConfig` | Runtime config | `appconfig.should_exit()` |

### No Additional Dependencies Needed

Phase 1 uses only existing crate dependencies. No new Cargo.toml entries required.

## Architecture Patterns

### Recommended Project Structure

```
data/
├── llm_models.txt          # Model names (real + funny)
├── llm_datasets.txt        # Dataset names (real + fictional)
└── gpu_models.txt          # GPU names (A100, H100, etc.)

src/
├── data.rs                 # Add include_str!() and lazy_static entries
├── modules/
│   ├── mod.rs              # Add pub mod and ALL_MODULES entry
│   └── llm_training.rs     # New module implementation
```

### Pattern 1: Data File Loading

**What:** Static text files loaded at compile time via `include_str!()`, exposed as `Vec<&'static str>` via `lazy_static`
**When to use:** Any module needing random selection from a list of strings
**Example:**

```rust
// Source: F:/genact-ex/src/data.rs (lines 1-25)

// Step 1: Include the file as a static string
static LLM_MODELS: &str = include_str!("../data/llm_models.txt");

// Step 2: Parse into lazy_static vector
lazy_static::lazy_static! {
    pub static ref LLM_MODELS_LIST: Vec<&'static str> = LLM_MODELS.lines().collect();
}
```

### Pattern 2: Module Trait Implementation

**What:** Unit struct implementing `Module` trait with name, signature, and async run
**When to use:** Every genact module
**Example:**

```rust
// Source: F:/genact-ex/src/modules/cargo.rs (lines 14-26)

pub struct LlmTraining;

#[async_trait(?Send)]
impl Module for LlmTraining {
    fn name(&self) -> &'static str {
        "llm_training"
    }

    fn signature(&self) -> String {
        "python train.py --model llama-7b --gpus 64".to_string()
    }

    async fn run(&self, appconfig: &AppConfig) {
        // Implementation here
    }
}
```

### Pattern 3: Module Registration

**What:** Add module to `ALL_MODULES` BTreeMap in `src/modules/mod.rs`
**When to use:** Every new module
**Example:**

```rust
// Source: F:/genact-ex/src/modules/mod.rs (lines 36-57)

// Step 1: Add pub mod declaration (alphabetical order)
pub mod llm_training;

// Step 2: Add to ALL_MODULES BTreeMap
all_modules.insert("llm_training", Box::new(llm_training::LlmTraining));
```

### Pattern 4: Cross-Platform I/O

**What:** Use `crate::io::*` functions instead of `println!()` for WASM compatibility
**When to use:** All output operations
**Example:**

```rust
// Source: F:/genact-ex/src/modules/cargo.rs (lines 43-50)

use crate::io::{csleep, dprint, newline, print};

// Print a line with format
print(format!("Loading model: {model_name}")).await;
newline().await;

// Delayed character-by-character print
dprint("Initializing...", 50).await;
newline().await;

// Sleep respecting speed factor
csleep(500).await;
```

### Pattern 5: Graceful Exit Check

**What:** Check `appconfig.should_exit()` in loops to handle CTRL-C and timeout
**When to use:** Inside every loop that may run for extended time
**Example:**

```rust
// Source: F:/genact-ex/src/modules/cargo.rs (lines 52-54)

for item in items {
    // ... do work ...

    if appconfig.should_exit() {
        return;
    }
}
```

### Anti-Patterns to Avoid

- **Using `println!()`:** Will break WASM build. Always use `io::print()` + `io::newline()`
- **Using `std::thread::sleep()`:** Will break WASM. Always use `io::csleep()`
- **Missing `should_exit()` checks:** Module won't respond to CTRL-C or timeout
- **Non-`?Send` async trait:** Must use `#[async_trait(?Send)]` for WASM compatibility

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Random selection from list | Manual index math | `list.choose(&mut rng)` | Off-by-one errors, unwrap handling |
| Terminal colors | ANSI escape codes | `yansi::Paint` | Cross-platform, readable |
| Formatted output | `format!()` + `println!()` | `io::print(format!(...))` | WASM compatibility |
| Async sleep | `std::thread::sleep` | `io::csleep()` | Respects speed factor, WASM compatible |
| Progress bars | Manual `\r` handling | `progress_string` crate | Already used in julia.rs |

**Key insight:** genact already has utilities for everything Phase 1 needs. The `io` module, `generators` module, and existing data loading patterns handle all cross-platform concerns.

## Common Pitfalls

### Pitfall 1: Forgetting WASM Compatibility

**What goes wrong:** Module compiles for native but fails for WASM target
**Why it happens:** Using `std::io`, `println!()`, or `std::thread::sleep()`
**How to avoid:**
- Only use `crate::io::*` functions
- Test with `RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo check --target wasm32-unknown-unknown`
**Warning signs:** Any `use std::io` or `println!` in the module

### Pitfall 2: Missing Module Registration

**What goes wrong:** Module file exists but doesn't appear in `genact --list`
**Why it happens:** Forgot one of the two registration steps in `mod.rs`
**How to avoid:** Checklist - 1) `pub mod` declaration, 2) `ALL_MODULES.insert()` call
**Warning signs:** `cargo run -- -m llm_training` says "no such module"

### Pitfall 3: Data File Not Found

**What goes wrong:** Compile error about missing file in `include_str!()`
**Why it happens:** File path is relative to source file, typo in path
**How to avoid:** Path is `"../data/filename.txt"` from `src/data.rs`
**Warning signs:** Compile error mentioning file path

### Pitfall 4: Empty Data File Lines

**What goes wrong:** Random selection returns empty strings
**Why it happens:** Trailing newlines or blank lines in data files
**How to avoid:** Ensure data files have no trailing newlines; use `.filter(|s| !s.is_empty())` if needed
**Warning signs:** Output shows empty model/dataset names

### Pitfall 5: Blocking Main Loop

**What goes wrong:** Module doesn't check `should_exit()`, hangs on CTRL-C
**Why it happens:** Forgot exit check in loop
**How to avoid:** Add `if appconfig.should_exit() { return; }` in every loop body
**Warning signs:** Module doesn't respond to CTRL-C until current iteration completes

## Code Examples

Verified patterns from the existing codebase:

### Minimal Module Template (Foundation)

```rust
// Source: Derived from F:/genact-ex/src/modules/bootlog.rs pattern

//! Pretend to train a large language model
use async_trait::async_trait;
use rand::seq::IndexedRandom;
use rand::rng;

use crate::args::AppConfig;
use crate::data::{LLM_MODELS_LIST, GPU_MODELS_LIST, LLM_DATASETS_LIST};
use crate::io::{csleep, newline, print};
use crate::modules::Module;

pub struct LlmTraining;

#[async_trait(?Send)]
impl Module for LlmTraining {
    fn name(&self) -> &'static str {
        "llm_training"
    }

    fn signature(&self) -> String {
        "python train.py --model llama-7b --gpus 64".to_string()
    }

    async fn run(&self, appconfig: &AppConfig) {
        let mut rng = rng();

        // Select random data
        let model = LLM_MODELS_LIST.choose(&mut rng).unwrap_or(&"GPT-Unknown");
        let gpu = GPU_MODELS_LIST.choose(&mut rng).unwrap_or(&"A100");
        let dataset = LLM_DATASETS_LIST.choose(&mut rng).unwrap_or(&"CommonCrawl");

        // Placeholder output
        print(format!("[INFO] Loading model: {model}")).await;
        newline().await;
        csleep(500).await;

        print(format!("[INFO] Detected 64x NVIDIA {gpu}")).await;
        newline().await;
        csleep(300).await;

        print(format!("[INFO] Dataset: {dataset}")).await;
        newline().await;
        csleep(300).await;

        // Simple loop with exit check
        for epoch in 1..=3 {
            print(format!("[TRAIN] Epoch {epoch}/3 - Training...")).await;
            newline().await;
            csleep(1000).await;

            if appconfig.should_exit() {
                return;
            }
        }

        print("[INFO] Training placeholder complete").await;
        newline().await;
    }
}
```

### Data File Format

```text
# llm_models.txt - One name per line, simple format like simcity.txt
GPT-4
LLaMA-70B
Mistral-7B
Claude-3
BERT-Large
OverfittedTransformer-9000
GigaChadLM
TurboSlothAI
HallucinationMaster-XL
```

### Data Loading in data.rs

```rust
// Source: F:/genact-ex/src/data.rs pattern

// Add these static includes
static LLM_MODELS: &str = include_str!("../data/llm_models.txt");
static GPU_MODELS: &str = include_str!("../data/gpu_models.txt");
static LLM_DATASETS: &str = include_str!("../data/llm_datasets.txt");

// Add to lazy_static! block
pub static ref LLM_MODELS_LIST: Vec<&'static str> = LLM_MODELS.lines().collect();
pub static ref GPU_MODELS_LIST: Vec<&'static str> = GPU_MODELS.lines().collect();
pub static ref LLM_DATASETS_LIST: Vec<&'static str> = LLM_DATASETS.lines().collect();
```

### Module Registration in mod.rs

```rust
// Source: F:/genact-ex/src/modules/mod.rs pattern

// Add to module declarations (keep alphabetical)
pub mod llm_training;

// Add to ALL_MODULES in lazy_static! block
all_modules.insert("llm_training", Box::new(llm_training::LlmTraining));
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `rand::thread_rng()` | `rand::rng()` | rand 0.9 | New API, simpler |
| `random()` for ranges | `random_range()` | rand 0.9 | Clearer intent |
| Manual async runtime | `async_std::main` | N/A | Already used |

**Deprecated/outdated:**
- None for this phase - genact uses modern Rust 2024 edition

## Data Content Research

### Model Names (DATA-01: real + funny mix)

**Real model names (verified HIGH confidence):**
- GPT-4, GPT-3.5, GPT-4o (OpenAI)
- LLaMA, LLaMA-2, LLaMA-3 (Meta)
- Claude, Claude-2, Claude-3 (Anthropic)
- Mistral, Mixtral (Mistral AI)
- BERT, RoBERTa, DeBERTa
- Falcon, MPT, Gemma, Gemini
- Qwen, Yi, DeepSeek

**Funny/fictional names (creative):**
- OverfittedTransformer-9000
- GigaChadLM
- HallucinationMaster-XL
- TurboSlothAI
- QuantumVibeNet
- ChonkyBOT-Prime
- MegaBrainGPT
- UltraThinkinator
- SentientToaster-7B
- CopyCat-Turbo

### GPU Names (DATA-02)

**NVIDIA Datacenter GPUs (verified HIGH confidence from official sources):**
- V100 (Volta, 2017)
- T4 (Turing)
- A100 (Ampere, 2020)
- A10, A30, A40
- H100 (Hopper, 2022)
- H200 (2024)
- L4, L40, L40S
- B100, B200 (Blackwell, 2025)

**Memory variants:**
- A100 40GB, A100 80GB
- H100 80GB, H100 NVL
- V100 16GB, V100 32GB

### Dataset Names (DATA-03: real + fictional)

**Real datasets (verified HIGH confidence):**
- ImageNet, ImageNet-21K
- CommonCrawl, C4
- The Pile
- Wikipedia (various languages)
- BookCorpus
- OpenWebText
- RedPajama
- LAION-5B
- MMLU, HellaSwag (benchmarks)

**Fictional datasets (creative):**
- UltraCorpus-2025
- MegaText-Infinity
- InternetDump-Pro
- AllTheBooks-v3
- RandomThoughts-XL
- HumanWisdom-Dataset
- TotalKnowledge-9000

## Open Questions

Things that couldn't be fully resolved:

1. **Optimal data file size**
   - What we know: Other files range from ~10 lines (docker_tags) to ~800K lines (cfiles)
   - What's unclear: Ideal balance for visual variety vs compile time
   - Recommendation: Start with 30-50 items per file; can expand later

2. **Signature format**
   - What we know: Should look like a real training command
   - What's unclear: Best format for "realistic" Python training command
   - Recommendation: Use `"python train.py --model llama-7b --gpus 64"` style

## Sources

### Primary (HIGH confidence)
- F:/genact-ex/src/modules/cargo.rs - Module implementation pattern
- F:/genact-ex/src/modules/bootlog.rs - Simple data loading pattern
- F:/genact-ex/src/modules/mod.rs - Module registration pattern
- F:/genact-ex/src/data.rs - Data file loading pattern
- F:/genact-ex/src/io.rs - Cross-platform I/O functions
- F:/genact-ex/Cargo.toml - Dependency versions

### Secondary (MEDIUM confidence)
- [NVIDIA Data Center GPUs Explained](https://www.bentoml.com/blog/nvidia-data-center-gpus-explained-a100-h200-b200-and-beyond) - GPU naming
- [Wikipedia: List of datasets for machine-learning research](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research) - Dataset names
- [TechTarget: Best LLMs](https://www.techtarget.com/whatis/feature/12-of-the-best-large-language-models) - Model names

### Tertiary (LOW confidence)
- Creative model/dataset names - Author discretion, no external verification needed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Directly from existing codebase
- Architecture: HIGH - Consistent patterns across 20+ modules
- Pitfalls: HIGH - Derived from code analysis and WASM requirements
- Data content: MEDIUM - Mix of verified sources and creative elements

**Research date:** 2026-01-30
**Valid until:** Indefinite for architecture patterns; data content can be updated anytime
