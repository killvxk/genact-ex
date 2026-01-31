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

/// Get a random validation warning message
fn get_validation_warning(rng: &mut impl Rng, loss: f64) -> String {
    let warnings = [
        format!(
            "Gradient norm exceeds threshold (norm={:.2}), clipping applied",
            rng.random_range(5.0..15.0)
        ),
        "Detected potential numerical instability in attention scores".to_string(),
        format!(
            "Loss spike detected: current={:.4}, moving_avg={:.4}",
            loss * 1.1,
            loss
        ),
        "Memory pressure detected on GPU cluster, consider reducing batch size".to_string(),
        "Validation accuracy below training accuracy by >10%, possible overfitting".to_string(),
        format!(
            "NaN detected in layer {}, batch skipped",
            rng.random_range(1..48)
        ),
        "Gradient accumulation buffer near capacity".to_string(),
    ];

    warnings.choose(rng).unwrap().clone()
}

/// Save checkpoint with progress bar (simulated write)
async fn save_checkpoint(appconfig: &AppConfig, step: u32, file_size_gb: f32) {
    let mut rng = rng();
    let filename = format!("model-step-{}.safetensors", step);

    log_info(&format!(
        "Saving model checkpoint to ./checkpoints/{}",
        filename
    ))
    .await;

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

        // Simulate write speed (~3-8 seconds total)
        csleep(rng.random_range(30..80)).await;
    }

    log_info(&format!(
        "Checkpoint saved: {} ({:.1}GB)",
        filename, file_size_gb
    ))
    .await;
}

/// Run validation phase after each epoch
async fn run_validation(
    appconfig: &AppConfig,
    _epoch: u32,
    train_loss: f64,
    best_val_loss: &mut f64,
    total_steps: u32,
    patience: &mut u32,
    max_patience: u32,
) -> f64 {
    let mut rng = rng();

    // Validation configuration (per CONTEXT.md: 80-150 steps, 10-20 seconds)
    let val_steps = rng.random_range(80..150);

    // Visual separator (per CONTEXT.md: PyTorch Lightning style)
    newline().await;
    print(format!(
        "{}",
        Paint::cyan("=============== Validation ===============").bold()
    ))
    .await;
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

    // Print initial progress line
    print(format!("Val:   {} | Loss: -.----", bar)).await;
    newline().await;

    for step in 1..=val_steps {
        // Add noise to validation loss
        let noise: f64 = noise_dist.sample(&mut rng);
        current_val_loss = (val_loss_base * (1.0 + noise)).max(0.1);

        let ppl = current_val_loss.exp();
        let accuracy = (1.0 / (1.0 + current_val_loss)).min(0.95);
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
        ))
        .await;
        newline().await;

        if appconfig.should_exit() {
            return current_val_loss;
        }

        // ~10-20 sec total: sleep_per_step = 10000-20000ms / 80-150 steps
        csleep(rng.random_range(80..150)).await;
    }

    let val_time = start_time.elapsed().as_secs_f64();
    let final_ppl = current_val_loss.exp();
    let final_accuracy = (1.0 / (1.0 + current_val_loss)).min(0.95);

    // Validation summary (multi-line report per CONTEXT.md)
    newline().await;
    log_info("Validation Results:").await;
    log_info(&format!("  val_loss:     {:.4}", current_val_loss)).await;
    log_info(&format!("  val_ppl:      {:.2}", final_ppl)).await;
    log_info(&format!("  val_accuracy: {:.2}%", final_accuracy * 100.0)).await;
    log_info(&format!(
        "  train_loss:   {:.4} (delta: {:+.4})",
        train_loss,
        current_val_loss - train_loss
    ))
    .await;
    log_info(&format!("  time:         {:.1}s", val_time)).await;

    // Random warning (30-40% chance per CONTEXT.md)
    if rng.random_bool(0.35) {
        log_warning(&get_validation_warning(&mut rng, current_val_loss)).await;
    }

    // Checkpoint logic (best model strategy per CONTEXT.md)
    if current_val_loss < *best_val_loss {
        *best_val_loss = current_val_loss;
        *patience = 0;

        // Save checkpoint with progress bar
        let file_size_gb: f32 = rng.random_range(2.0..8.0);
        save_checkpoint(appconfig, total_steps, file_size_gb).await;
    } else {
        *patience += 1;
        // Early stopping warning (never actually stops per CONTEXT.md)
        log_warning(&format!(
            "EarlyStopping: val_loss did not improve. Patience: {}/{}",
            patience, max_patience
        ))
        .await;
    }

    newline().await;

    current_val_loss
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

/// Run the export phase after training completes
async fn run_export_phase(
    appconfig: &AppConfig,
    model_name: &str,
    params_b: f64,
    train_loss: f64,
    train_time_secs: f64,
    total_epochs: u32,
    total_steps: u32,
) {
    let mut rng = rng();

    // Visual separator
    newline().await;
    print(format!(
        "{}",
        Paint::cyan("=============== Export Phase ===============").bold()
    ))
    .await;
    newline().await;
    newline().await;

    // Calculate model size and shards
    let total_size_gb = params_b * 2.0; // ~2 bytes per param in fp16
    let num_shards = ((total_size_gb / 5.0).ceil() as u32).clamp(1, 8);
    let size_per_shard = total_size_gb / num_shards as f64;

    // Export steps with progress bars
    let export_steps = [
        ("Merging distributed weights...", 3000, 5000),
        ("Optimizing model for inference...", 2000, 3000),
        ("Serializing to SafeTensors format...", 2000, 3000),
        ("Writing model shards...", 1000, 2000),
    ];

    for (step_name, min_ms, max_ms) in export_steps {
        log_info(step_name).await;

        let mut bar = BarBuilder::new()
            .total(100)
            .width(30)
            .full_char('=')
            .include_percent()
            .build();

        // Print initial progress line
        print(format!("  Progress: {}", bar)).await;
        newline().await;

        let step_duration_ms = rng.random_range(min_ms..max_ms);
        let sleep_per_step = step_duration_ms / 100;

        for i in 1..=100 {
            bar.replace(i);

            cursor_up(1).await;
            erase_line().await;
            print(format!("  Progress: {}", bar)).await;
            newline().await;

            if appconfig.should_exit() {
                return;
            }

            csleep(sleep_per_step as u64).await;
        }
    }

    if appconfig.should_exit() {
        return;
    }

    // Shard file exports
    newline().await;
    log_info("Exporting model shards:").await;
    for shard in 1..=num_shards {
        log_info(&format!(
            "  Saved: model-{:05}-of-{:05}.safetensors ({:.1}GB)",
            shard, num_shards, size_per_shard
        ))
        .await;
        csleep(rng.random_range(100..300)).await;

        if appconfig.should_exit() {
            return;
        }
    }

    // Companion files
    newline().await;
    log_info("Saving companion files:").await;
    let companion_files = [
        "config.json",
        "tokenizer.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ];
    for file in companion_files {
        log_info(&format!("  Saved: {}", file)).await;
        csleep(rng.random_range(50..150)).await;

        if appconfig.should_exit() {
            return;
        }
    }

    // Export complete message
    newline().await;
    log_info(&format!(
        "Export complete: {} shards, {:.1}GB total",
        num_shards, total_size_gb
    ))
    .await;

    if appconfig.should_exit() {
        return;
    }

    // Training summary separator
    newline().await;
    print(format!(
        "{}",
        Paint::cyan("============ Training Summary ============").bold()
    ))
    .await;
    newline().await;
    newline().await;

    // Format time
    let hours = (train_time_secs / 3600.0).floor() as u32;
    let minutes = ((train_time_secs % 3600.0) / 60.0).floor() as u32;
    let seconds = (train_time_secs % 60.0).floor() as u32;
    let time_str = if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else {
        format!("{}m {}s", minutes, seconds)
    };

    // Calculate derived metrics
    let final_ppl = train_loss.exp();
    let avg_gpu_util: u32 = rng.random_range(92..98);
    let peak_memory: f64 = rng.random_range(72.0..78.0);

    // Sanitize model name for path (lowercase, replace spaces)
    let model_path = model_name.to_lowercase().replace(' ', "-");

    // Training summary table (ASCII box style)
    print("+-----------------------------------------+".to_string()).await;
    newline().await;
    print("|           Training Summary              |".to_string()).await;
    newline().await;
    print("+-----------------------------------------+".to_string()).await;
    newline().await;
    print(format!("| Total time:     {:>22} |", time_str)).await;
    newline().await;
    print(format!("| Epochs:         {:>22} |", total_epochs)).await;
    newline().await;
    print(format!("| Steps:          {:>22} |", total_steps)).await;
    newline().await;
    print(format!("| Final loss:     {:>22.4} |", train_loss)).await;
    newline().await;
    print(format!("| Final PPL:      {:>22.2} |", final_ppl)).await;
    newline().await;
    print(format!("| Avg GPU util:   {:>21}% |", avg_gpu_util)).await;
    newline().await;
    print(format!("| Peak memory:    {:>20.1}GB |", peak_memory)).await;
    newline().await;
    print(format!(
        "| Model saved:    ./outputs/{}/final/ |",
        &model_path[..model_path.len().min(10)]
    ))
    .await;
    newline().await;
    print(format!("| Total size:     {:>20.1}GB |", total_size_gb)).await;
    newline().await;
    print("+-----------------------------------------+".to_string()).await;
    newline().await;

    // Success message
    newline().await;
    print(format!(
        "{}",
        Paint::green("Training completed successfully!").bold()
    ))
    .await;
    newline().await;
}

/// Run the training loop with dual progress bars, metrics, and GPU status
async fn run_training_loop(appconfig: &AppConfig) -> (f64, u32, u32) {
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

    // Validation state
    let mut best_val_loss = f64::MAX;
    let mut patience: u32 = 0;
    let max_patience: u32 = 5;

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
                return (loss, total_epochs, total_steps);
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

        // Run validation after each epoch
        run_validation(
            appconfig,
            epoch,
            loss,
            &mut best_val_loss,
            total_steps,
            &mut patience,
            max_patience,
        )
        .await;

        if appconfig.should_exit() {
            return (loss, total_epochs, total_steps);
        }

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

    (loss, total_epochs, total_steps)
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
