//! Pretend to train a large language model
use async_trait::async_trait;
use chrono::Local;
use instant::Instant;
use progress_string::BarBuilder;
use rand::Rng;
use rand::rng;
use rand::seq::IndexedRandom;
use rand_distr::{Distribution, Normal};
use yansi::Paint;

use crate::args::AppConfig;
use crate::data::{GPU_MODELS_LIST, LLM_DATASETS_LIST, LLM_MODELS_LIST};
use crate::io::{csleep, cursor_up, erase_line, newline, print};
use crate::modules::Module;

/// Log an INFO message with timestamp
async fn log_info(message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] {} {}",
        ts,
        Paint::green("INFO").bold(),
        message
    ))
    .await;
    newline().await;
}

/// Log an INFO message with timestamp and rank info
async fn log_info_rank(rank: u32, world_size: u32, message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] [Rank {}/{}] {} {}",
        ts,
        rank,
        world_size,
        Paint::green("INFO").bold(),
        message
    ))
    .await;
    newline().await;
}

/// Log a WARNING message with timestamp
async fn log_warning(message: &str) {
    let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
    print(format!(
        "[{}] {} {}",
        ts,
        Paint::yellow("WARNING").bold(),
        message
    ))
    .await;
    newline().await;
}

/// GPU status for monitoring display
struct GpuStatus {
    id: u32,
    node: u32,
    memory_used_gb: f32,
    memory_total_gb: f32,
    utilization: f32,
    temperature: u32,
}

/// Generate GPU statuses with per-GPU variation for realism
fn generate_gpu_statuses(num_gpus: u32, rng: &mut impl Rng) -> Vec<GpuStatus> {
    let base_util: f64 = rng.random_range(90.0..98.0);
    let base_temp: i32 = rng.random_range(70..80);
    let base_mem: f32 = rng.random_range(70.0..78.0);
    let total_mem = 80.0f32;

    let util_noise = Normal::new(0.0, 2.0).unwrap();
    let temp_noise = Normal::new(0.0, 3.0).unwrap();
    let mem_noise = Normal::new(0.0f32, 1.5f32).unwrap();

    (0..num_gpus)
        .map(|id| GpuStatus {
            id,
            node: id / 8,
            memory_used_gb: (base_mem + mem_noise.sample(rng)).clamp(60.0, total_mem - 1.0),
            memory_total_gb: total_mem,
            utilization: ((base_util + util_noise.sample(rng)).clamp(0.0, 100.0)) as f32,
            temperature: ((base_temp as f64 + temp_noise.sample(rng)).clamp(50.0, 95.0)) as u32,
        })
        .collect()
}

/// Display GPU status grid grouped by node with health-colored indicators
async fn display_gpu_status_grid(gpus: &[GpuStatus], num_nodes: u32) {
    print(format!("{}", Paint::cyan("[ GPU Status Grid ]").bold())).await;
    newline().await;

    for node in 0..num_nodes {
        let node_gpus: Vec<_> = gpus.iter().filter(|g| g.node == node).collect();

        print(format!("Node {}: ", node)).await;

        for gpu in node_gpus {
            let mem_pct = (gpu.memory_used_gb / gpu.memory_total_gb * 100.0) as u32;

            // Color based on health (CONTEXT.md decision)
            let status_color = if gpu.temperature > 85 || mem_pct > 95 {
                yansi::Color::Red
            } else if gpu.temperature > 75 || mem_pct > 85 {
                yansi::Color::Yellow
            } else {
                yansi::Color::Green
            };

            let status_char = Paint::new("*").fg(status_color);

            print(format!(
                "{} GPU{}: {}% {}C {:.0}/{:.0}GB  ",
                status_char,
                gpu.id % 8,
                gpu.utilization as u32,
                gpu.temperature,
                gpu.memory_used_gb,
                gpu.memory_total_gb
            ))
            .await;
        }
        newline().await;
    }
}

/// Run the training loop with dual progress bars, metrics, and GPU status
async fn run_training_loop(appconfig: &AppConfig) {
    let mut rng = rng();

    // Configuration
    let total_epochs = 3u32;
    let steps_per_epoch = rng.random_range(500..2000);
    let num_gpus: u32 = 64;
    let num_nodes: u32 = 8;

    // Loss generation parameters (TRAIN-07)
    let mut loss = rng.random_range(8.0..12.0);
    let decay_rate = 0.002;
    let noise_dist = Normal::new(0.0, 0.03).unwrap();

    // Learning rate
    let lr = 1e-4;

    let start_time = Instant::now();
    let mut total_steps = 0u32;

    // Epoch progress bar (TRAIN-01)
    let mut epoch_bar = BarBuilder::new()
        .total(total_epochs as usize)
        .width(25)
        .full_char('=')
        .include_numbers()
        .build();

    // Step progress bar (TRAIN-01)
    let mut step_bar = BarBuilder::new()
        .total(steps_per_epoch as usize)
        .width(35)
        .full_char('=')
        .include_percent()
        .build();

    log_info("======== Training Started ========").await;
    newline().await;

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

            // Update loss with decay + noise (TRAIN-07)
            let base_decay = (-decay_rate * total_steps as f64).exp();
            let noise: f64 = noise_dist.sample(&mut rng);
            loss = (loss * base_decay * (1.0 + noise)).max(0.1);

            // Perplexity (TRAIN-03)
            let ppl = loss.exp();

            // Tokens per second (TRAIN-05)
            let tokens_per_sec = rng.random_range(800_000.0..1_200_000.0);

            // Update step bar
            step_bar.replace(step as usize);

            // Time calculations (TRAIN-06)
            let elapsed = start_time.elapsed().as_secs();
            let eta = if total_steps > 0 {
                (elapsed as f64 / total_steps as f64)
                    * ((total_epochs * steps_per_epoch - total_steps) as f64)
            } else {
                0.0
            };

            // Update display using cursor manipulation (julia.rs pattern)
            cursor_up(2).await;

            erase_line().await;
            print(format!("Epoch: {}", epoch_bar)).await;
            newline().await;

            erase_line().await;
            print(format!(
                "Step:  {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | {:.0}tok/s | {}s<{}s",
                step_bar, loss, ppl, lr, tokens_per_sec, elapsed, eta as u64
            ))
            .await;
            newline().await;

            // Periodic detailed log every 10 steps (CONTEXT.md decision)
            if step % 10 == 0 {
                let grad_norm = rng.random_range(0.5..2.0);
                print(format!(
                    "  Step {}/{} | loss={:.4} | ppl={:.2} | grad_norm={:.3}",
                    step, steps_per_epoch, loss, ppl, grad_norm
                ))
                .await;
                newline().await;

                // GPU status grid (GPU-01 through GPU-04)
                let gpus = generate_gpu_statuses(num_gpus, &mut rng);
                display_gpu_status_grid(&gpus, num_nodes).await;
                newline().await;

                // Re-print progress bars after GPU grid
                print(format!("Epoch: {}", epoch_bar)).await;
                newline().await;
                print(format!(
                    "Step:  {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | {:.0}tok/s | {}s<{}s",
                    step_bar, loss, ppl, lr, tokens_per_sec, elapsed, eta as u64
                ))
                .await;
                newline().await;
            }

            // Occasional NCCL AllReduce log (random, ~every 50-100 steps)
            if rng.random_bool(0.02) {
                let allreduce_ms = rng.random_range(2.0..15.0);
                log_info(&format!(
                    "NCCL AllReduce completed in {:.2}ms",
                    allreduce_ms
                ))
                .await;

                // Re-print progress bars after log
                print(format!("Epoch: {}", epoch_bar)).await;
                newline().await;
                print(format!(
                    "Step:  {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | {:.0}tok/s | {}s<{}s",
                    step_bar, loss, ppl, lr, tokens_per_sec, elapsed, eta as u64
                ))
                .await;
                newline().await;
            }

            // Occasional warning for realism (~0.5% chance)
            if rng.random_bool(0.005) {
                log_warning("GPU memory usage approaching limit").await;

                // Re-print progress bars after warning
                print(format!("Epoch: {}", epoch_bar)).await;
                newline().await;
                print(format!(
                    "Step:  {} | Loss: {:.4} | PPL: {:.2} | LR: {:.2e} | {:.0}tok/s | {}s<{}s",
                    step_bar, loss, ppl, lr, tokens_per_sec, elapsed, eta as u64
                ))
                .await;
                newline().await;
            }

            if appconfig.should_exit() {
                newline().await;
                return;
            }

            csleep(rng.random_range(30..80)).await;
        }

        // Epoch summary
        newline().await;
        let epoch_time = start_time.elapsed().as_secs_f64() / epoch as f64;
        log_info(&format!(
            "Epoch {} complete | avg_loss={:.4} | time={:.1}s",
            epoch, loss, epoch_time
        ))
        .await;
        newline().await;

        // Re-print progress bars for next epoch
        if epoch < total_epochs {
            print(format!("Epoch: {}", epoch_bar)).await;
            newline().await;
            print(format!("Step:  {} | Loss: {:.4}", step_bar, loss)).await;
            newline().await;
        }
    }

    newline().await;
    log_info("======== Training Complete ========").await;
}

/// Run the initialization phase of LLM training simulation
/// Displays model loading, tokenizer, dataset, NCCL init, and GPU detection
async fn run_initialization(appconfig: &AppConfig) {
    let mut rng = rng();

    // Configuration
    let num_gpus: u32 = 64;
    let num_nodes: u32 = 8;

    // Select random model, GPU, and dataset
    let model = LLM_MODELS_LIST.choose(&mut rng).unwrap_or(&"GPT-Unknown");
    let gpu = GPU_MODELS_LIST.choose(&mut rng).unwrap_or(&"A100");
    let dataset = LLM_DATASETS_LIST.choose(&mut rng).unwrap_or(&"CommonCrawl");

    // Generate random architecture details
    let params_b: f64 = rng.random_range(7.0..405.0);
    let num_layers: u32 = rng.random_range(32..128);
    let hidden_size: u32 = rng.random_range(4096..16384);
    let attention_heads: u32 = rng.random_range(32..128);
    let vocab_size: u32 = rng.random_range(32000..128000);
    let samples: u64 = rng.random_range(1_000_000..100_000_000);

    // ===================
    // INIT-01: Model Loading
    // ===================
    log_info(&format!("Loading model: {}", model)).await;
    csleep(200).await;

    if appconfig.should_exit() {
        return;
    }

    log_info(&format!("  Parameters: {:.1}B", params_b)).await;
    csleep(100).await;

    log_info(&format!(
        "  Architecture: {} layers, hidden_size={}, attention_heads={}",
        num_layers, hidden_size, attention_heads
    ))
    .await;
    csleep(100).await;

    if appconfig.should_exit() {
        return;
    }

    // Model loading progress bar
    let mut bar = BarBuilder::new()
        .total(100)
        .width(40)
        .full_char('=')
        .include_percent()
        .build();

    let load_start = std::time::Instant::now();

    for i in 0..=100 {
        bar.replace(i);
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        erase_line().await;
        print(format!(
            "[{}] {} Loading model weights... {}",
            ts,
            Paint::green("INFO").bold(),
            bar
        ))
        .await;

        csleep(rng.random_range(20..80)).await;

        if appconfig.should_exit() {
            newline().await;
            return;
        }
    }
    newline().await;

    let load_time = load_start.elapsed().as_secs_f64();
    log_info(&format!("Model loaded in {:.1}s", load_time)).await;
    csleep(300).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-02: Tokenizer Loading
    // ===================
    let tokenizer_name = if model.contains("Llama") || model.contains("llama") {
        "LlamaTokenizer"
    } else if model.contains("GPT") {
        "GPT2Tokenizer"
    } else {
        "SentencePieceTokenizer"
    };

    log_info(&format!("Loading tokenizer: {}", tokenizer_name)).await;
    csleep(200).await;

    log_info(&format!("  Vocab size: {}", vocab_size)).await;
    csleep(100).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-03: Dataset Loading
    // ===================
    log_info(&format!("Loading dataset: {}", dataset)).await;
    csleep(500).await;

    log_info(&format!(
        "  Dataset: {} samples, batch_size=2048, micro_batch=4",
        samples
    ))
    .await;
    csleep(300).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-04: Distributed Environment
    // ===================
    log_info("Initializing distributed environment...").await;
    csleep(500).await;

    log_info("NCCL version: 2.19.3").await;
    csleep(100).await;

    log_info(&format!(
        "World size: {} GPUs across {} nodes",
        num_gpus, num_nodes
    ))
    .await;
    csleep(200).await;

    if appconfig.should_exit() {
        return;
    }

    // ===================
    // INIT-05: GPU Detection
    // ===================
    // Show first 16 GPUs (2 nodes worth)
    for node in 0..2u32 {
        for local_gpu in 0..8u32 {
            let gpu_id = node * 8 + local_gpu;
            log_info_rank(
                gpu_id,
                num_gpus,
                &format!("Detected NVIDIA {} on node {}", gpu, node),
            )
            .await;
            csleep(rng.random_range(10..50)).await;

            if appconfig.should_exit() {
                return;
            }
        }
    }

    // Show remaining GPUs count
    log_info(&format!("... and {} more GPUs", num_gpus - 16)).await;
    csleep(500).await;

    if appconfig.should_exit() {
        return;
    }

    // NCCL AllReduce test
    log_info("Running NCCL AllReduce test...").await;
    csleep(1000).await;

    let allreduce_time: f64 = rng.random_range(5.0..50.0);
    log_info(&format!("AllReduce test passed ({:.2}ms)", allreduce_time)).await;
    csleep(300).await;

    newline().await;
    log_info("======== Initialization Complete ========").await;
    newline().await;
}

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
        // Phase 2.1: Initialization
        run_initialization(appconfig).await;

        if appconfig.should_exit() {
            return;
        }

        // Phase 2.2: Training Loop
        run_training_loop(appconfig).await;
    }
}
