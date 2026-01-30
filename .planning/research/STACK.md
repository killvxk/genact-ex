# Technology Stack: LLM Training Simulation Module

**Project:** genact LLM Training Module
**Researched:** 2026-01-30
**Dimension:** Stack (data files, output patterns, crate verification)

## Recommended Stack

### Core Framework (Existing - No Changes Needed)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Rust | 2024 Edition | Language | Already in use by genact |
| async-std | 1.x | Async runtime | Existing dependency, WASM compatible |
| async-trait | 0.1.x | Async trait support | Required for Module trait pattern |

### Output and Display (Existing - Sufficient)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| yansi | 1.x | Terminal colors | Already used by all modules, WASM compatible |
| progress_string | 0.2.x | Progress bars | Used by julia.rs, download.rs - proven pattern |
| chrono | 0.4.x | Timestamps | Already available, WASM compatible with `wasmbind` feature |

### Randomization (Existing - Sufficient)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| rand | 0.9.x | Core RNG | Already in use everywhere |
| rand_distr | 0.5.x | Statistical distributions | For realistic metric noise (Normal, Exp, ChiSquared) |

**Verdict: No new crates needed.** The existing stack fully supports all simulation requirements.

## Required Data Files

Create these files in `data/` directory following existing patterns.

### 1. llm_models.txt (~80 lines)

**Purpose:** Model name generation for realistic/funny outputs

```text
# Format: One entry per line, comments start with #
# Section: Real model prefixes (use for realistic output)
GPT
GPT-Neo
GPT-NeoX
GPT-J
LLaMA
Llama
Mistral
Mixtral
Falcon
Qwen
DeepSeek
Yi
Gemma
Phi
BLOOM
OPT
Pythia
StableLM
OpenLLaMA
CodeLlama
WizardLM
Vicuna
Alpaca
Orca
SOLAR
InternLM
Baichuan
ChatGLM

# Section: Size suffixes
-1B
-3B
-7B
-8B
-13B
-14B
-30B
-33B
-34B
-65B
-70B
-72B
-180B
-405B
-Nano
-Mini
-Small
-Base
-Large
-XL
-XXL

# Section: Variant suffixes
-Instruct
-Chat
-Base
-Code
-v1
-v2
-v3
-Preview
-Turbo
-Pro
-Ultra
-Plus

# Section: Funny names (genact humor)
ButtGPT
ChadLLM
SkyNet-Preview
HAL-9001
T-800-Chat
Clippy-3000
JARVIS-Lite
Ultron-Beta
GLaDOS-Instruct
SHODAN-v2
MegaMaid-LLM
SpaceBalls-AI
BonziBuddy-GPT
Cortana-Copilot
Tay-Reborn
```

### 2. llm_datasets.txt (~40 lines)

**Purpose:** Training dataset names for init phase output

```text
# Format: One entry per line
# Real datasets
pile-deduped
RedPajama-v2
RedPajama-Data-1T
SlimPajama-627B
FineWeb
FineWeb-Edu
The-Stack-v2
StarCoder-Data
OpenWebText
OpenWebText2
CommonCrawl-2024
C4
mC4
ROOTS
OSCAR
Wikipedia-en
BookCorpus
ArXiv-Papers
GitHub-Code
StackExchange

# Funny datasets
reddit-copypasta
twitter-drama-2024
stackoverflow-rage
4chan-wisdom
linkedin-cringe
tiktok-transcripts
youtube-comments
discord-messages
minecraft-chat-logs
roblox-forums
```

### 3. llm_gpus.txt (~25 lines)

**Purpose:** GPU model names for cluster status display

```text
# Format: name,memory_gb,tdp_watts
# Real NVIDIA datacenter GPUs
NVIDIA A100-SXM4-40GB,40,400
NVIDIA A100-SXM4-80GB,80,400
NVIDIA A100-PCIe-40GB,40,250
NVIDIA A100-PCIe-80GB,80,300
NVIDIA H100-SXM5-80GB,80,700
NVIDIA H100-PCIe-80GB,80,350
NVIDIA H100-NVL-94GB,94,400
NVIDIA H200-SXM-141GB,141,700
NVIDIA L40S-48GB,48,350
NVIDIA L4-24GB,24,72
NVIDIA V100-SXM2-32GB,32,300
NVIDIA V100-PCIe-32GB,32,250

# AMD datacenter GPUs
AMD MI250X-128GB,128,560
AMD MI300X-192GB,192,750

# Funny GPUs (rare appearance)
NVIDIA RTX 4090-Founders,24,450
NVIDIA GeForce-Titan-Cluster,24,320
AMD-Radeon-RX-7900-XTX-Stolen,24,355
Intel-Arc-A770-Enterprise,16,225
```

### 4. llm_frameworks.txt (~20 lines)

**Purpose:** Framework version strings for init output

```text
# Format: framework version
PyTorch 2.5.1
PyTorch 2.4.0
PyTorch 2.3.1
transformers 4.47.0
transformers 4.45.2
transformers 4.44.0
DeepSpeed 0.18.5
DeepSpeed 0.17.0
Megatron-LM 3.0
FSDP 2.0
accelerate 1.2.0
accelerate 1.0.0
bitsandbytes 0.45.0
flash-attn 2.7.0
triton 3.1.0
NCCL 2.29.1
cuDNN 9.5.0
CUDA 12.6
CUDA 12.4
```

## Output Format Patterns

### Phase 1: Initialization Output

```
[2026-01-30 14:23:15] ========================================
[2026-01-30 14:23:15] LLM Training Session Starting
[2026-01-30 14:23:15] ========================================
[2026-01-30 14:23:15] Framework: PyTorch 2.5.1 + transformers 4.47.0
[2026-01-30 14:23:15] DeepSpeed: 0.18.5 (ZeRO Stage 3)
[2026-01-30 14:23:16] NCCL: 2.29.1 | CUDA: 12.6 | cuDNN: 9.5.0
[2026-01-30 14:23:16]
[2026-01-30 14:23:16] Initializing distributed environment...
[2026-01-30 14:23:17]   World size: 64 GPUs across 8 nodes
[2026-01-30 14:23:17]   Master: node-0.cluster.internal:29500
[2026-01-30 14:23:17]   Backend: nccl
[2026-01-30 14:23:18]   NCCL INFO: Using network IB
[2026-01-30 14:23:18]   NCCL INFO: Ring 0: 0->8->16->24->32->40->48->56->0
[2026-01-30 14:23:19]
[2026-01-30 14:23:19] Loading model: Llama-3-70B-Instruct
[2026-01-30 14:23:20]   Architecture: LlamaForCausalLM
[2026-01-30 14:23:20]   Parameters: 70.6B (fp16: 141.2 GB)
[2026-01-30 14:23:21]   Sharding: 64-way tensor parallel
[2026-01-30 14:23:22]   Memory per GPU: 2.2 GB model + 4.8 GB optimizer
[2026-01-30 14:23:23]
[2026-01-30 14:23:23] Loading tokenizer: LlamaTokenizerFast
[2026-01-30 14:23:23]   Vocab size: 128,256 | Max length: 8,192
[2026-01-30 14:23:24]
[2026-01-30 14:23:24] Dataset: RedPajama-v2 (1.2T tokens)
[2026-01-30 14:23:25]   Training samples: 1,200,000,000
[2026-01-30 14:23:25]   Validation samples: 10,000
```

### Phase 2: Training Loop Output

**Per-step output (every N steps):**
```
Epoch 1/3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  15% | Step 7500/50000
  Loss: 2.3456 | PPL: 10.44 | LR: 8.00e-05 | Grad Norm: 1.234
  Throughput: 125,432 tok/s | 8,192 samples/s | ETA: 2h 34m 12s
  GPU Mem: 74.2/80.0 GB (92.8%) | Util: 98% | Temp: 72°C (avg)
```

**Periodic GPU cluster status (every 50-100 steps):**
```
┌─ GPU Cluster Status ─────────────────────────────────────────────┐
│ Node 0 [GPU 0-7]:  98% 72°C | 97% 73°C | 98% 71°C | 96% 74°C    │
│                    97% 72°C | 98% 73°C | 97% 72°C | 98% 71°C    │
│ Node 1 [GPU 8-15]: 97% 74°C | 98% 73°C | 96% 75°C | 98% 72°C    │
│ ...                                                               │
│ Power: 43.2 kW | NVLink: 850 GB/s | InfiniBand: 400 Gb/s        │
└──────────────────────────────────────────────────────────────────┘
```

**NCCL communication stats (periodic):**
```
[NCCL] AllReduce: 2.34ms (1.2GB) | Broadcast: 0.82ms | Barrier: 0.05ms
[NCCL] Ring latency: 125us | Tree latency: 89us | Bandwidth: 285 GB/s
```

### Phase 3: Validation Output

```
[Validation] Epoch 1 complete, evaluating on validation set...
  Evaluating: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 10000/10000
  Val Loss: 2.1234 | Val PPL: 8.37
  Metrics: BLEU=32.4 | ROUGE-L=0.456 | F1=0.823
  Eval time: 45.2s | 221 samples/s
```

### Phase 4: Checkpoint Output

```
[Checkpoint] Saving model checkpoint (epoch 1, step 50000)...
  Saving model shards: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | 64/64
  Saving optimizer state: 156.3 GB
  Saving scheduler state...
  Path: checkpoints/epoch_1_step_50000/
  Checkpoint saved in 45.2s
```

### Phase 5: Export Output

```
[Export] Training complete! Finalizing model...
  Converting ZeRO-3 checkpoint to fp16...
  Merging shards: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
  Exporting to SafeTensors format...

  ========================================
  Training Summary
  ========================================
  Model: Llama-3-70B-Instruct (fine-tuned)
  Total epochs: 3
  Total steps: 150,000
  Final loss: 0.8234
  Final PPL: 2.28
  Training time: 4h 23m 15s
  Output: ./models/llama-3-70b-finetuned/
  ========================================
```

## Metric Generation Patterns

### Loss Decay Formula

```rust
/// Realistic loss decay with noise
fn calculate_loss(step: u32, total_steps: u32, initial_loss: f32, rng: &mut ThreadRng) -> f32 {
    let progress = step as f32 / total_steps as f32;

    // Exponential decay with floor
    let decay = (-3.0 * progress).exp();
    let base_loss = initial_loss * (0.2 + 0.8 * decay);

    // Add realistic noise (more noise early, less late)
    let noise_scale = 0.1 * (1.0 - progress * 0.7);
    let noise: f32 = Normal::new(0.0, noise_scale).unwrap().sample(rng);

    (base_loss + noise).max(0.1)  // Floor at 0.1
}
```

**Typical values:**
- Initial loss: 2.5 - 4.0 (depends on model size)
- Final loss: 0.5 - 1.5
- Loss should generally decrease but have small fluctuations

### Learning Rate Schedule

```rust
/// Cosine decay with warmup
fn calculate_lr(step: u32, total_steps: u32, max_lr: f32, warmup_steps: u32) -> f32 {
    if step < warmup_steps {
        // Linear warmup
        max_lr * (step as f32 / warmup_steps as f32)
    } else {
        // Cosine decay
        let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
        let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        max_lr * (0.01 + 0.99 * decay)  // Decay to 1% of max
    }
}
```

**Typical values:**
- Max LR: 1e-5 to 1e-4
- Min LR: 1e-7 to 1e-6
- Warmup: 5-10% of total steps

### GPU Metrics

```rust
struct GpuMetrics {
    utilization: u8,    // 85-99%
    temperature: u8,    // 65-85°C
    memory_used: f32,   // 85-98% of capacity
    power_draw: u16,    // 80-100% of TDP
}

fn generate_gpu_metrics(rng: &mut ThreadRng, base_util: u8) -> GpuMetrics {
    let noise = rng.random_range(-3i8..4i8);
    GpuMetrics {
        utilization: (base_util as i8 + noise).clamp(85, 99) as u8,
        temperature: rng.random_range(68..82),
        memory_used: rng.random_range(0.88..0.98),
        power_draw: rng.random_range(85..100),
    }
}
```

### Throughput Metrics

**Tokens per second (typical ranges):**
- Small model (7B): 100,000 - 200,000 tok/s on 64 GPUs
- Medium model (70B): 50,000 - 150,000 tok/s on 64 GPUs
- Large model (405B): 20,000 - 80,000 tok/s on 64 GPUs

**Samples per second:**
- Depends on batch size and sequence length
- Typical: batch_size * num_gpus / step_time

## WASM Compatibility Notes

All recommended patterns are WASM compatible:

| Feature | Native | WASM | Notes |
|---------|--------|------|-------|
| `io::print()` | stdout | xterm.js | Already abstracted |
| `io::csleep()` | std::thread::sleep | Promise/setTimeout | Already abstracted |
| `chrono::Local::now()` | System time | JS Date via wasmbind | Feature enabled |
| `rand::rng()` | System entropy | JS crypto via getrandom | Feature enabled |
| `yansi` colors | ANSI codes | ANSI codes (xterm interprets) | Works as-is |
| `progress_string` | Pure Rust | Pure Rust | No platform deps |

**No conditional compilation needed** in the module itself - all platform differences are handled by existing abstractions.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Colors | yansi | colored, termcolor | yansi already in use, simpler API |
| Progress | progress_string | indicatif | indicatif not WASM compatible, progress_string proven |
| Time | chrono | time | chrono already available with WASM features |
| RNG distributions | rand_distr | statrs | rand_distr already in use, sufficient |

## Data File Loading Pattern

Following existing `data.rs` conventions:

```rust
// In src/data.rs - add these declarations

static LLM_MODELS: &str = include_str!("../data/llm_models.txt");
static LLM_DATASETS: &str = include_str!("../data/llm_datasets.txt");
static LLM_GPUS: &str = include_str!("../data/llm_gpus.txt");
static LLM_FRAMEWORKS: &str = include_str!("../data/llm_frameworks.txt");

lazy_static::lazy_static! {
    pub static ref LLM_MODELS_LIST: Vec<&'static str> =
        LLM_MODELS.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .collect();

    pub static ref LLM_DATASETS_LIST: Vec<&'static str> =
        LLM_DATASETS.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .collect();

    pub static ref LLM_GPUS_LIST: Vec<GpuSpec> =
        LLM_GPUS.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .filter_map(|l| {
                let parts: Vec<&str> = l.split(',').collect();
                if parts.len() >= 3 {
                    Some(GpuSpec {
                        name: parts[0],
                        memory_gb: parts[1].parse().ok()?,
                        tdp_watts: parts[2].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

    pub static ref LLM_FRAMEWORKS_LIST: Vec<&'static str> =
        LLM_FRAMEWORKS.lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .collect();
}

pub struct GpuSpec {
    pub name: &'static str,
    pub memory_gb: u32,
    pub tdp_watts: u32,
}
```

## Sources

### HIGH Confidence (verified from existing codebase)
- genact Cargo.toml - crate versions and features: `F:\genact-ex\Cargo.toml`
- genact data.rs patterns: `F:\genact-ex\src\data.rs`
- genact io.rs WASM abstraction: `F:\genact-ex\src\io.rs`
- Existing module patterns: `F:\genact-ex\src\modules\*.rs`

### MEDIUM Confidence (verified from official sources)
- [HuggingFace Trainer documentation](https://huggingface.co/docs/transformers/en/trainer)
- [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/)
- [PyTorch distributed training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [DeepSpeed documentation](https://deepspeed.readthedocs.io/en/latest/)
- [LLM training hyperparameters guide](https://modal.com/blog/fine-tuning-llms-hyperparameters-glossary-article)

### LOW Confidence (general knowledge, verify if critical)
- Typical loss curve patterns for LLMs
- GPU cluster power consumption estimates
- Training throughput ranges

---

## Quality Gate Checklist

- [x] Recommendations fit existing genact architecture - uses only existing crates
- [x] WASM compatibility considered - all patterns verified WASM-safe
- [x] Specific output format examples included - full phase output templates
- [x] Data file specifications complete - 4 files with exact formats
- [x] Metric generation patterns defined - loss decay, LR schedule, GPU stats
- [x] No new dependencies required - existing stack is sufficient
